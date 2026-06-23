"""Prometheus-side validation for individual widget queries.

Runs **best effort**. If Prometheus is unreachable we only do static
checks; we do not fail the dashboard because the user's Prometheus
instance is offline. The Helper/evaluator pipeline surfaces data
problems separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..prometheus.client import PrometheusClient
from ..specs import DashboardSpec


_BAD_PROMQL_PATTERNS = [
    re.compile(r";"),
    re.compile(r"`"),
    re.compile(r"<\s*script", re.IGNORECASE),
    re.compile(r"javascript:", re.IGNORECASE),
]


@dataclass
class QueryCheckResult:
    widget_id: str
    promql: str
    syntax_ok: bool
    reachable: bool
    returns_data: bool | None  # None = unknown
    issues: list[str]


class PrometheusValidator:
    def __init__(self, client: PrometheusClient | None = None) -> None:
        self._client = client or PrometheusClient()

    def validate_dashboard(self, spec: DashboardSpec) -> list[QueryCheckResult]:
        results: list[QueryCheckResult] = []
        for widget in spec.widgets:
            q = widget.query.promql
            issues: list[str] = []
            syntax_ok = True
            for pat in _BAD_PROMQL_PATTERNS:
                if pat.search(q):
                    syntax_ok = False
                    issues.append(f"forbidden token: {pat.pattern!r}")
            reachable = False
            returns_data: bool | None = None
            try:
                reachable, returns_data = self._client.probe(q)
            except Exception as exc:  # pragma: no cover - network
                issues.append(f"probe failed: {exc}")
            results.append(
                QueryCheckResult(
                    widget_id=widget.id,
                    promql=q,
                    syntax_ok=syntax_ok,
                    reachable=reachable,
                    returns_data=returns_data,
                    issues=issues,
                )
            )
        return results
