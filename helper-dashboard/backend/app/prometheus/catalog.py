"""Prometheus metric catalog.

Used by the dashboard-spec-agent as ground truth for metric names.
The catalog is loaded from the live Prometheus `/api/v1/label/__name__/values`
endpoint at startup, with a safe fallback set of common metrics when
Prometheus is unreachable.
"""

from __future__ import annotations

import json
from urllib import error as urllib_error
from urllib import request as urllib_request

from .client import PROM_URL, REQUEST_TIMEOUT


FALLBACK_METRICS: list[str] = [
    "up",
    "process_cpu_seconds_total",
    "process_resident_memory_bytes",
    "process_start_time_seconds",
    "node_cpu_seconds_total",
    "node_memory_MemAvailable_bytes",
    "node_memory_MemTotal_bytes",
    "node_filesystem_avail_bytes",
    "node_filesystem_size_bytes",
    "node_network_receive_bytes_total",
    "node_network_transmit_bytes_total",
    "http_requests_total",
    "http_request_duration_seconds",
    "ALERTS",
]


class MetricCatalog:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or PROM_URL).rstrip("/")
        self._cache: list[str] | None = None

    def list_metrics(self) -> list[str]:
        if self._cache is not None:
            return self._cache
        try:
            url = f"{self.base_url}/api/v1/label/__name__/values"
            with urllib_request.urlopen(url, timeout=REQUEST_TIMEOUT) as resp:  # noqa: S310
                raw = json.loads(resp.read().decode("utf-8"))
            values = raw.get("data", []) or []
            if values:
                self._cache = sorted(values)
                return self._cache
        except urllib_error.URLError:
            pass
        except Exception:
            pass
        self._cache = list(FALLBACK_METRICS)
        return self._cache

    def has(self, metric: str) -> bool:
        return metric in self.list_metrics()

    def suggest(self, hint: str, limit: int = 10) -> list[str]:
        hint = hint.lower()
        return [m for m in self.list_metrics() if hint in m.lower()][:limit]
