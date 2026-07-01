"""FastAPI entry point for the Helper Dashboard backend."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .api import chat, dashboard, evaluate, developer, saved_dashboards


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
        promoted=False,  # SHADOW: never pages in INC2.
    )
    app.state.anomaly_service = service
    try:
        while True:
            now = time.time()
            # tick is synchronous + fast; run it directly. Errors inside a
            # rule are already caught + audited by the service.
            service.tick(now)
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

    return app


app = create_app()
