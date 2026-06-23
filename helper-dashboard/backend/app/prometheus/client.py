"""Prometheus HTTP client with deterministic mock fallback.

In the MVP we avoid adding a hard dependency on `requests`. If
`requests` is available we use it; otherwise we fall back to urllib.
When Prometheus is unreachable, `query` returns structured mock data
so the frontend still has something to render — clearly marked as
mock data via the `source: mock` field on the response.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


PROM_URL = os.getenv("HELPER_DASHBOARD_PROM_URL", "http://localhost:9090")
REQUEST_TIMEOUT = float(os.getenv("HELPER_DASHBOARD_PROM_TIMEOUT", "3.0"))


class PrometheusClient:
    """Thin Prometheus HTTP client."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or PROM_URL).rstrip("/")

    # ------------------------------------------------------------
    def probe(self, promql: str) -> tuple[bool, bool | None]:
        """Return `(reachable, returns_data)`.

        `returns_data` is None when the server is unreachable.
        """
        try:
            data = self._query(promql)
        except _Unreachable:
            return False, None
        except Exception:
            return True, False
        result = (data or {}).get("data", {}).get("result", [])
        return True, bool(result)

    # ------------------------------------------------------------
    def query(self, promql: str) -> dict[str, Any]:
        """Run an instant query.

        On failure, returns mock data with `source: mock`. On success,
        returns the parsed Prometheus response with `source: prometheus`.
        """
        try:
            data = self._query(promql)
            data["source"] = "prometheus"
            return data
        except Exception:
            return _mock_response(promql, mode="instant")

    def query_range(
        self, promql: str, range_s: int = 3600, step_s: int = 30
    ) -> dict[str, Any]:
        try:
            data = self._query_range(promql, range_s=range_s, step_s=step_s)
            data["source"] = "prometheus"
            return data
        except Exception:
            return _mock_response(promql, mode="range", range_s=range_s, step_s=step_s)

    # ------------------------------------------------------------
    def _query(self, promql: str) -> dict[str, Any]:
        params = urllib_parse.urlencode({"query": promql})
        return self._get(f"/api/v1/query?{params}")

    def _query_range(self, promql: str, range_s: int, step_s: int) -> dict[str, Any]:
        end = int(time.time())
        start = end - range_s
        params = urllib_parse.urlencode(
            {"query": promql, "start": start, "end": end, "step": step_s}
        )
        return self._get(f"/api/v1/query_range?{params}")

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            with urllib_request.urlopen(url, timeout=REQUEST_TIMEOUT) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib_error.URLError as exc:
            raise _Unreachable(str(exc)) from exc


class _Unreachable(Exception):
    pass


# ------------------------------------------------------------
# Mock response
# ------------------------------------------------------------

def _mock_response(
    promql: str, *, mode: str, range_s: int = 3600, step_s: int = 30
) -> dict[str, Any]:
    """Deterministic mock generator keyed off the query string."""
    now = int(time.time())
    seed = sum(ord(c) for c in promql) % 97
    if mode == "instant":
        value = 40 + seed % 50
        return {
            "status": "success",
            "source": "mock",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"__name__": "mock", "query": promql[:64]},
                        "value": [now, str(float(value))],
                    }
                ],
            },
        }
    # range
    n = max(1, range_s // max(step_s, 1))
    points: list[list[Any]] = []
    for i in range(n):
        t = now - range_s + i * step_s
        v = 50 + 20 * math.sin((i + seed) / 6.0)
        points.append([t, f"{v:.4f}"])
    return {
        "status": "success",
        "source": "mock",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"__name__": "mock", "query": promql[:64]},
                    "values": points,
                }
            ],
        },
    }
