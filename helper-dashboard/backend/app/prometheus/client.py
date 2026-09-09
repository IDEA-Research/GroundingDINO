"""Prometheus HTTP client — honest failure, never fabricated data.

In the MVP we avoid adding a hard dependency on `requests`, so the
client is built on urllib. When Prometheus is unreachable or a query
fails, `query`/`query_range` raise `PrometheusUnavailable` carrying the
underlying reason; callers translate that into an explicit error state.
There is NO mock fallback on this path: a clinical dashboard must never
render synthetic values as real, and must never render silence — the
API layer turns this error into a loud, visible failure instead.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


PROM_URL = os.getenv("HELPER_DASHBOARD_PROM_URL", "http://localhost:9090")
REQUEST_TIMEOUT = float(os.getenv("HELPER_DASHBOARD_PROM_TIMEOUT", "3.0"))


class PrometheusUnavailable(Exception):
    """Prometheus could not serve the query.

    Raised by `query`/`query_range` instead of any data fallback; the
    message carries the underlying reason (connection refused, timeout,
    bad response, ...). API consumers translate this into an explicit
    error response — never into fabricated data.
    """


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
    def query(self, promql: str, time_s: int | None = None) -> dict[str, Any]:
        """Run an instant query.

        `time_s` (epoch seconds) pins the Prometheus evaluation instant
        (sent as the `time` query arg) so stat cards can be evaluated at
        the same step-aligned moment as chart right edges; None keeps
        Prometheus' own server-side "now".

        On success, returns the parsed Prometheus response with
        `source: prometheus`. On failure, raises `PrometheusUnavailable`
        — never a mock fallback.
        """
        try:
            data = self._query(promql, time_s=time_s)
        except Exception as exc:
            raise PrometheusUnavailable(
                f"instant query failed: {exc}"
            ) from exc
        data["source"] = "prometheus"
        return data

    def query_range(
        self,
        promql: str,
        range_s: int = 3600,
        step_s: int = 30,
        end: int | None = None,
    ) -> dict[str, Any]:
        """Run a range query; raises `PrometheusUnavailable` on failure.

        `end` (epoch seconds) is the shared right edge of the window —
        one dashboard refresh tick sends the same `end` for every
        widget so all charts cover the identical [end-range, end]
        window. None means "now". Either way the end is floored to
        `step_s` before use.
        """
        try:
            data = self._query_range(
                promql, range_s=range_s, step_s=step_s, end=end
            )
        except Exception as exc:
            raise PrometheusUnavailable(
                f"range query failed: {exc}"
            ) from exc
        data["source"] = "prometheus"
        return data

    # ------------------------------------------------------------
    def label_values(
        self,
        label: str,
        *,
        start: int | None = None,
        end: int | None = None,
        match: list[str] | None = None,
    ) -> list[str]:
        """GET /api/v1/label/<label>/values, time-scoped + optional
        series matchers. Raises `PrometheusUnavailable` on failure.
        Callers validate `label` against an allow-list — this method
        never builds PromQL from user input."""
        args: list[tuple[str, Any]] = []
        if start is not None:
            args.append(("start", int(start)))
        if end is not None:
            args.append(("end", int(end)))
        for m in match or []:
            args.append(("match[]", m))
        params = urllib_parse.urlencode(args)
        path = f"/api/v1/label/{urllib_parse.quote(label, safe='')}/values"
        if params:
            path = f"{path}?{params}"
        try:
            data = self._get(path)
        except Exception as exc:
            raise PrometheusUnavailable(
                f"label values query failed: {exc}"
            ) from exc
        values = (data or {}).get("data") or []
        return sorted(str(v) for v in values)

    def targets(self) -> list[dict[str, Any]]:
        """GET /api/v1/targets -> the `activeTargets` list (each with
        `labels.job`, `health`, `lastError`, `lastScrape`). Raises
        `PrometheusUnavailable` on failure."""
        try:
            data = self._get("/api/v1/targets?state=active")
        except Exception as exc:
            raise PrometheusUnavailable(f"targets query failed: {exc}") from exc
        active = ((data or {}).get("data") or {}).get("activeTargets") or []
        return [t for t in active if isinstance(t, dict)]

    # ------------------------------------------------------------
    def _query(self, promql: str, time_s: int | None = None) -> dict[str, Any]:
        query_args: dict[str, Any] = {"query": promql}
        if time_s is not None:
            query_args["time"] = int(time_s)
        params = urllib_parse.urlencode(query_args)
        return self._get(f"/api/v1/query?{params}")

    def _query_range(
        self,
        promql: str,
        range_s: int,
        step_s: int,
        end: int | None = None,
    ) -> dict[str, Any]:
        end = int(end) if end is not None else int(time.time())
        # Defense in depth: always floor the end to the step so the
        # window is step-aligned regardless of the caller. Idempotent
        # when the API layer already aligned it.
        if step_s > 0:
            end -= end % step_s
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
