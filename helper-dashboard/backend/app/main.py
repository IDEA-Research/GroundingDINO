"""FastAPI entry point for the Helper Dashboard backend."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .api import chat, dashboard, evaluate, developer, saved_dashboards


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

    return app


app = create_app()
