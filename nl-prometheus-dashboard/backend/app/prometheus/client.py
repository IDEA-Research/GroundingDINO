from __future__ import annotations

import os
from typing import Any

import httpx


class PrometheusQueryError(RuntimeError):
    pass


class PrometheusClient:
    def __init__(self, base_url: str | None = None, timeout_seconds: float = 15.0) -> None:
        self.base_url = (base_url or os.getenv("PROMETHEUS_BASE_URL", "http://localhost:9090")).rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def instant_query(self, query: str, time: float | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"query": query}
        if time is not None:
            params["time"] = time
        return await self._get("/api/v1/query", params)

    async def range_query(
        self,
        *,
        query: str,
        start: float,
        end: float,
        step: int,
    ) -> dict[str, Any]:
        params = {"query": query, "start": start, "end": end, "step": step}
        return await self._get("/api/v1/query_range", params)

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(f"{self.base_url}{path}", params=params)
            response.raise_for_status()
            payload = response.json()

        if payload.get("status") != "success":
            error = payload.get("error") or payload.get("errorType") or "unknown Prometheus error"
            raise PrometheusQueryError(str(error))
        return payload.get("data", {})

