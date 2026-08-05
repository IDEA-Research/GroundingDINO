"""FastAPI entry point for the Helper Dashboard backend."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .api import (
    anomaly,
    chat,
    dashboard,
    evaluate,
    developer,
    saved_dashboards,
    system_rules as system_rules_api,
)


def _anomaly_tick_interval_s() -> float:
    try:
        return float(os.getenv("ANOMALY_TICK_INTERVAL_S", "15"))
    except ValueError:
        return 15.0


async def _anomaly_loop(app: FastAPI) -> None:
    """Drive the synchronous evaluator `tick` on a real timer.

    This is the ONLY wall-clock driver. It is deliberately thin: it just calls
    the injectable-clock `tick(now)` — all clinical logic lives in the core and
    the service, both of which the golden/integration tests step synchronously
    with an injected clock (pytest_asyncio is not installed). The loop is
    OPT-IN via `ANOMALY_EVALUATOR_ENABLED=1` so it never interferes with the
    existing app or the test suite; rules stay in SHADOW (never page) here.
    """
    # Imported lazily so a missing optional dep never breaks app import.
    from .prometheus.neonatal_sim import Scenario
    from .services.anomaly_data_provider import SimDataProvider
    from .services.anomaly_evaluator_service import AnomalyEvaluatorService
    from .tests_support.default_rules import default_neonatal_rules

    interval = _anomaly_tick_interval_s()
    # Default to the synthetic source: real medical data is not flowing yet.
    provider = SimDataProvider(scenario=Scenario.healthy, step_s=interval)
    service = AnomalyEvaluatorService(
        default_neonatal_rules(),
        provider,
        watchdog_budget_s=max(45.0, interval * 3),
        promoted=False,  # SHADOW: never pages here.
        increment="runtime",  # audit tag: live loop, not a build increment
    )
    app.state.anomaly_service = service
    try:
        while True:
            now = time.time()
            # tick is synchronous (fsync'd audit writes; up to 3x5s webhook
            # retries per pageable event when a dispatcher is wired) — run it
            # in a worker thread so the API stays responsive. Awaiting keeps
            # ticks strictly sequential; errors inside a rule are already
            # caught + audited by the service.
            await asyncio.to_thread(service.tick, now)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:  # graceful shutdown
        raise


def create_app() -> FastAPI:
    # `redirect_slashes=False` matters: FastAPI's default redirect turns
    # `/api/saved_dashboards` into `/api/saved_dashboards/` via a 307 to
    # the ABSOLUTE backend URL (e.g. http://127.0.0.1:8000/...). When
    # the browser called us through the Next.js dev proxy on :3050, the
    # 307 takes the request OUT of the proxy and direct-hits the backend
    # — cross-origin, no CORS Origin header on the redirect target, and
    # the request is blocked. We register both "" and "/" forms on each
    # router instead, so no redirect is ever needed.
    app = FastAPI(
        title="Helper Dashboard",
        description=(
            "Backend for the Helper Dashboard system. User-facing routes "
            "live under /api/chat, /api/dashboard, and /api/evaluate. "
            "/api/developer is internal-only and guarded."
        ),
        version="0.1.0",
        redirect_slashes=False,
    )

    # In dev, the frontend can run on any port (3000 for `next dev`,
    # 3050 for our `next start` wrapper, occasionally 3001-3009 if the
    # primary is taken). An origin REGEX matches whichever the browser
    # is actually using without us hardcoding a list. In prod the
    # operator passes a comma-separated allowlist via the
    # `HELPER_DASHBOARD_CORS_ORIGINS` env var.
    prod_origins_raw = os.getenv("HELPER_DASHBOARD_CORS_ORIGINS", "").strip()
    if prod_origins_raw:
        prod_origins = [o.strip() for o in prod_origins_raw.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=prod_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=False,
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=False,
        )

    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
    app.include_router(evaluate.router, prefix="/api/evaluate", tags=["evaluate"])
    app.include_router(
        saved_dashboards.router, prefix="/api/saved_dashboards",
        tags=["saved_dashboards"],
    )
    # Internal-only, behind a developer gate inside the router.
    app.include_router(developer.router, prefix="/api/developer", tags=["developer"])
    # Anomaly read surface (alerts the decision_flow widget consumes) + the
    # ANOMALY_DEMO-only driver. The /alerts endpoint fails loud (store_unavailable)
    # when no evaluator is wired, so it is always safe to mount.
    app.include_router(anomaly.router, prefix="/api/anomaly", tags=["anomaly"])
    # System-metric rules (CPU/disk/memory/load) — non-clinical, structurally
    # SHADOW, authored via Helper chat. Read surface + validated CRUD.
    app.include_router(
        system_rules_api.router, prefix="/api/system-rules",
        tags=["system-rules"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    Instrumentator().instrument(app).expose(app)

    # Background anomaly evaluator — OPT-IN, shadow-only. Wired via lifecycle
    # events so it is a non-load-bearing timer around the synchronous tick.
    if os.getenv("ANOMALY_EVALUATOR_ENABLED", "0") == "1":

        @app.on_event("startup")
        async def _start_anomaly_loop() -> None:  # pragma: no cover - timer glue
            app.state.anomaly_task = asyncio.create_task(_anomaly_loop(app))

        @app.on_event("shutdown")
        async def _stop_anomaly_loop() -> None:  # pragma: no cover - timer glue
            task = getattr(app.state, "anomaly_task", None)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    # System-rule wall-clock loop — ON by default (SYSTEM_RULES_ENABLED=0 to
    # kill). Reads the real Prometheus and records SHADOW events only; the
    # E7 sim-clock hazard does not apply because both the data and the tick
    # clock are wall-clock here. Tests keep it off via conftest.
    from .services import system_rules as system_rules_svc

    async def _system_rules_loop() -> None:  # pragma: no cover - timer glue
        try:
            interval = float(os.getenv("SYSTEM_RULES_TICK_INTERVAL_S", "30"))
        except ValueError:
            interval = 30.0
        service = system_rules_svc.get_service()
        while True:
            await asyncio.to_thread(service.tick, time.time())
            await asyncio.sleep(interval)

    # The enabled-check runs INSIDE the startup hook (not at create_app
    # time): pytest imports this module at collection, before conftest's
    # SYSTEM_RULES_ENABLED=0 fixture is active — a create_app-time check
    # would bake the loop in and tick against a live Prometheus mid-test.
    @app.on_event("startup")
    async def _start_system_rules_loop() -> None:  # pragma: no cover
        if not system_rules_svc.is_enabled():
            return
        app.state.system_rules_task = asyncio.create_task(_system_rules_loop())

    @app.on_event("shutdown")
    async def _stop_system_rules_loop() -> None:  # pragma: no cover
        task = getattr(app.state, "system_rules_task", None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # Demo driver — OPT-IN via ANOMALY_DEMO=1. Drives the REAL pipeline with an
    # accelerated injected clock so the /api/anomaly/demo/* endpoints can walk
    # the decision_flow widget through healthy -> firing -> signal_lost live,
    # without a five-minute wall-clock wait. Never enabled in a real deployment.
    if os.getenv("ANOMALY_DEMO", "0") == "1":

        @app.on_event("startup")
        async def _start_anomaly_demo() -> None:  # pragma: no cover - demo glue
            from .api.anomaly import DemoDriver

            # Dedicated demo dir — NEVER ANOMALY_ALERT_STATE_DIR (the REAL
            # durable store): DemoDriver wipes its dir per scenario, and an
            # env-var collision would rmtree the real state.json + append-only
            # history audit. DemoDriver additionally refuses the real dir.
            demo_dir = os.getenv(
                "ANOMALY_DEMO_STATE_DIR",
                str(Path(__file__).resolve().parent / "storage" / "alert_state_demo"),
            )
            app.state.anomaly_demo = DemoDriver(app, base_dir=demo_dir)

    return app


app = create_app()
