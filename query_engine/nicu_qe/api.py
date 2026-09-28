"""FastAPI shell over QueryEngine (API_CONTRACT section 3 + v2 section 6).

- GET only; /v1/* requires header X-QE-Token (QE_API_TOKEN) unless QE_ALLOW_NO_TOKEN=1 on loopback.
- TrustedHostMiddleware (127.0.0.1 / localhost) against DNS rebinding.
- Errors are JSON {"error": {"code", "message"}}; no stack traces.
- pymongo is synchronous: every engine call goes through asyncio.to_thread.
- The background ETL loop runs in this process (single uvicorn worker) and holds the meta lease.
- Mongo down: the process keeps running, /health answers 503, the ETL loop retries.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Settings, is_loopback_host
from .engine import QueryEngine, QueryError
from .tz import to_display

log = logging.getLogger("nicu_qe.api")

T_FORMATS = ("iso", "epoch")


class _TimeRenderer:
    """Datetime -> display-tz ISO / epoch, caching the tz offset per 15-minute UTC slot.

    Every real-world tz transition happens on a 15-minute boundary, so the cached offset is exact."""

    __slots__ = ("zone", "epoch", "_cache")

    def __init__(self, tz: str, t_format: str) -> None:
        from zoneinfo import ZoneInfo

        self.zone = ZoneInfo(tz)
        self.epoch = t_format == "epoch"
        self._cache: Dict[int, Any] = {}

    def __call__(self, dt: datetime) -> Any:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        ts = dt.timestamp()
        if self.epoch:
            return ts
        slot = int(ts // 900)
        off = self._cache.get(slot)
        if off is None:
            off = dt.astimezone(self.zone).utcoffset()
            self._cache[slot] = off
        return (dt.astimezone(timezone.utc) + off).replace(tzinfo=timezone(off)).isoformat()


def render(obj: Any, t_format: str, tz: str, _r: Optional[_TimeRenderer] = None) -> Any:
    """JSON-ready copy: datetimes -> display-tz ISO (or epoch seconds), ObjectId -> str."""
    r = _r or _TimeRenderer(tz, t_format)
    if isinstance(obj, datetime):
        return r(obj)
    if isinstance(obj, dict):
        return {str(k): render(v, t_format, tz, r) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [render(v, t_format, tz, r) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, date):
        return obj.isoformat()
    return str(obj)


def respond_series(series_dict: Dict[str, Any], t_format: str, tz: str) -> Response:
    """Fast path for /v1/series: point values are already JSON-safe, only `t` needs formatting."""
    r = _TimeRenderer(tz, t_format)
    d = dict(series_dict)
    points = d.pop("points")
    for p in points:
        p["t"] = r(p["t"])
    out = render(d, t_format, tz, r)
    out["points"] = points
    body = json.dumps(out, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return Response(content=body.encode("utf-8"), status_code=200, media_type="application/json")


def respond(obj: Any, t_format: str, tz: str, status: int = 200) -> Response:
    """Serialize once with json.dumps (FastAPI's jsonable_encoder is too slow for 10k points)."""
    body = json.dumps(render(obj, t_format, tz), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return Response(content=body.encode("utf-8"), status_code=status, media_type="application/json")


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def error_class_only(err: Any) -> Any:
    """Reduce an ETL error string to its exception class name (for unauthenticated /health).

    Formats: meta "mode: ClassName: msg", loop "ClassName: msg", or a fixed phrase.
    """
    if not err or not isinstance(err, str):
        return err
    parts = [p.strip() for p in err.split(":")]
    for p in parts[:2]:
        if _IDENT_RE.fullmatch(p) and p[0].isupper():
            return p
    if err.startswith("lease held"):
        return "LeaseHeld"
    return "Error"


def _err(status: int, code: str, message: str) -> JSONResponse:
    """Uniform error body."""
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(settings: Optional[Settings] = None, engine: Optional[QueryEngine] = None, start_etl: bool = True,
               allowed_hosts: Optional[List[str]] = None, pidfile: Optional[str] = None) -> FastAPI:
    """Build the app. Refuses to build without a token unless no-token mode is allowed on loopback."""
    settings = settings or Settings.from_env()
    settings.validate_for_serve()
    engine = engine or QueryEngine(settings)
    tz = settings.display_tz
    state: Dict[str, Any] = {"loop": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_etl:
            from .etl import Etl, EtlLoop

            loop = EtlLoop(Etl(engine.mongo, settings), settings.etl_interval_s)
            loop.start()
            state["loop"] = loop
        try:
            yield
        finally:
            loop = state.get("loop")
            if loop is not None:
                await asyncio.to_thread(loop.stop, 30.0)
            if pidfile:  # uvicorn re-raises SIGTERM after shutdown, so clean up here
                try:
                    import os

                    os.remove(pidfile)
                except OSError:
                    pass

    app = FastAPI(title="nicu_qe", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    hosts = allowed_hosts or ["127.0.0.1", "localhost"]
    if not is_loopback_host(settings.bind_host) and settings.bind_host not in hosts and settings.bind_host != "0.0.0.0":
        hosts = hosts + [settings.bind_host]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

    @app.exception_handler(QueryError)
    async def _qerr(request: Request, exc: QueryError):
        return _err(exc.status, exc.code, str(exc))

    @app.exception_handler(RequestValidationError)
    async def _verr(request: Request, exc: RequestValidationError):
        msgs = "; ".join(f"{'.'.join(str(x) for x in e.get('loc', []))}: {e.get('msg')}" for e in exc.errors())
        return _err(400, "bad_request", msgs[:500])

    @app.exception_handler(StarletteHTTPException)
    async def _herr(request: Request, exc: StarletteHTTPException):
        code = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _err(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _anyerr(request: Request, exc: Exception):
        from pymongo.errors import ConnectionFailure, ExecutionTimeout, PyMongoError, ServerSelectionTimeoutError

        if isinstance(exc, ExecutionTimeout):
            return _err(504, "timeout", "query exceeded QE_MONGO_MAX_TIME_MS")
        if isinstance(exc, (ServerSelectionTimeoutError, ConnectionFailure)):
            return _err(503, "db_unavailable", "database unavailable")
        if isinstance(exc, PyMongoError):
            log.warning("mongo error: %s", type(exc).__name__)
            return _err(500, "db_error", type(exc).__name__)
        log.exception("unhandled error")
        return _err(500, "internal_error", "internal error")

    async def require_token(x_qe_token: Optional[str] = Header(default=None)) -> None:
        expected = settings.api_token
        if not expected:
            return  # validate_for_serve() guaranteed loopback + QE_ALLOW_NO_TOKEN=1
        if not x_qe_token or not hmac.compare_digest(x_qe_token.encode(), expected.encode()):
            raise StarletteHTTPException(status_code=401, detail="missing or invalid X-QE-Token")

    async def call(fn: Callable, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

    def token_ok(x_qe_token: Optional[str]) -> bool:
        expected = settings.api_token
        return bool(expected and x_qe_token and hmac.compare_digest(x_qe_token.encode(), expected.encode()))

    @app.get("/health")
    async def health(x_qe_token: Optional[str] = Header(default=None)):
        status, body = await call(engine.health)
        full = token_ok(x_qe_token)  # without a valid token, error texts are reduced to the class name
        etl = body.get("etl")
        if isinstance(etl, dict) and not full:
            etl["last_error"] = error_class_only(etl.get("last_error"))
        loop = state.get("loop")
        if loop is not None:
            body["etl_loop"] = {"alive": loop.is_alive(), "holds_lease": loop.holds_lease,
                                "indexes_ready": loop.indexes_ready,
                                "last_error": loop.last_error if full else error_class_only(loop.last_error)}
            if status == 200 and loop.last_error and body.get("status") == "ok":
                body["status"] = "degraded"
        return respond(body, "iso", tz, status)

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_token)])

    @v1.get("/catalog")
    async def catalog(t_format: str = Query("iso", pattern="^(iso|epoch)$")):
        return respond(await call(engine.catalog), t_format, tz)

    @v1.get("/coverage")
    async def coverage(from_: str = Query(..., alias="from"), to: str = Query(...), group: str = "day",
                       source: Optional[str] = None, session_id: Optional[str] = None, camera: Optional[str] = None,
                       device_family: Optional[str] = None, metric: Optional[str] = None,
                       include_analysis_time: bool = False, t_format: str = Query("iso", pattern="^(iso|epoch)$")):
        flt = {"source": source, "session_id": session_id, "camera": camera, "device_family": device_family}
        res = await call(engine.coverage, from_, to, group=group, include_analysis_time=include_analysis_time,
                         metric=metric, **flt)
        return respond(res, t_format, tz)

    @v1.get("/series")
    async def series(metric: str, from_: str = Query(..., alias="from"), to: str = Query(...),
                     step: Optional[str] = None, agg: str = "avg", fill: Optional[str] = None,
                     session_id: Optional[str] = None, camera: Optional[str] = None, source: Optional[str] = None,
                     device_family: Optional[str] = None, llm_model: Optional[str] = None,
                     include_analysis_time: bool = False, exclude_quality: Optional[str] = None,
                     limit: Optional[int] = None, t_format: str = Query("iso", pattern="^(iso|epoch)$")):
        flt = {"session_id": session_id, "camera": camera, "source": source, "device_family": device_family,
               "llm_model": llm_model}
        res = await call(engine.series, metric, from_, to, step=step, agg=agg, fill=fill, limit=limit,
                         include_analysis_time=include_analysis_time, exclude_quality=exclude_quality, **flt)
        return respond_series(res.to_dict(), t_format, tz)

    @v1.get("/latest")
    async def latest(metric: str, session_id: Optional[str] = None, camera: Optional[str] = None,
                     source: Optional[str] = None, device_family: Optional[str] = None,
                     llm_model: Optional[str] = None, max_age: Optional[str] = None,
                     include_analysis_time: bool = False, exclude_quality: Optional[str] = None,
                     t_format: str = Query("iso", pattern="^(iso|epoch)$")):
        flt = {"session_id": session_id, "camera": camera, "source": source, "device_family": device_family,
               "llm_model": llm_model}
        res = await call(engine.latest, metric, max_age=max_age, include_analysis_time=include_analysis_time,
                         exclude_quality=exclude_quality, **flt)
        return respond(res, t_format, tz)

    @v1.get("/sessions")
    async def sessions(from_: Optional[str] = Query(None, alias="from"), to: Optional[str] = None,
                       source: Optional[str] = None, camera: Optional[str] = None, patient_id: Optional[str] = None,
                       t_format: str = Query("iso", pattern="^(iso|epoch)$")):
        res = await call(engine.sessions, from_, to, source=source, camera=camera, patient_id=patient_id)
        return respond(res, t_format, tz)

    @v1.get("/stats")
    async def stats(metric: str, from_: str = Query(..., alias="from"), to: str = Query(...),
                    group_by: str = "none", session_id: Optional[str] = None, camera: Optional[str] = None,
                    source: Optional[str] = None, device_family: Optional[str] = None,
                    llm_model: Optional[str] = None, include_analysis_time: bool = False,
                    exclude_quality: Optional[str] = None, t_format: str = Query("iso", pattern="^(iso|epoch)$")):
        flt = {"session_id": session_id, "camera": camera, "source": source, "device_family": device_family,
               "llm_model": llm_model}
        res = await call(engine.stats, metric, from_, to, group_by=group_by,
                         include_analysis_time=include_analysis_time, exclude_quality=exclude_quality, **flt)
        return respond(res, t_format, tz)

    @v1.get("/unmapped-keys")
    async def unmapped_keys(device_family: Optional[str] = None, since: Optional[str] = None,
                            t_format: str = Query("iso", pattern="^(iso|epoch)$")):
        res = await call(engine.unmapped_keys, device_family=device_family, since=since)
        return respond(res, t_format, tz)

    @v1.get("/aliases")
    async def aliases():
        return await call(engine.aliases)

    app.include_router(v1)
    app.state.engine = engine
    app.state.qe_state = state
    return app
