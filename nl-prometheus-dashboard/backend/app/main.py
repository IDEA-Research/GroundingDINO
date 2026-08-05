from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

load_dotenv()

from .api.agent_routes import router as agent_router
from .api.dashboard_routes import router as dashboard_router
from .api.query_routes import router as query_router
from .prometheus.sample_metrics import render_demo_medical_metrics


def create_app() -> FastAPI:
    app = FastAPI(
        title="Natural Language Prometheus Dashboard API",
        version="0.1.0",
        description="Task-model-driven dashboard generation with Prometheus-backed live widgets.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
        allow_origin_regex=os.getenv("CORS_ORIGIN_REGEX", r"https?://([a-zA-Z0-9.-]+|\d+\.\d+\.\d+\.\d+):5174"),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(agent_router)
    app.include_router(dashboard_router)
    app.include_router(query_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(
            render_demo_medical_metrics(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


app = create_app()
