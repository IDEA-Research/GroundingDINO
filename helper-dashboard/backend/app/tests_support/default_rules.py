"""The default neonatal rSO2 rule set the running service loads.

Kept in ONE place so the FastAPI lifespan loop and the integration tests use
identical, spec-validated rules. Semantics are the locked supervisor decision:
breach = value < 0.80 * avg_over_time(metric[24h]) sustained for 5m, neonatal
only, source must be prometheus, SHADOW by default.
"""

from __future__ import annotations

from ..specs.alert_rule_spec import AlertRuleSpec


def neonatal_rso2_rule(
    *, metric: str = "rso2_left", rule_id: str | None = None
) -> AlertRuleSpec:
    return AlertRuleSpec.model_validate(
        {
            "id": rule_id or f"neo-{metric.replace('_', '-')}-desat",
            "description": f"Neonatal cerebral {metric} desaturation (20% below 24h baseline)",
            "metric": metric,
            "labels": {"patient": "neo-001"},
            "baseline": {"fn": "avg_over_time", "window": "24h"},
            "comparator": "<",
            "ratio": 0.80,
            "for": "5m",
            "severity": "critical",
            "source": "prometheus",
            "mode": "shadow",
        }
    )


def default_neonatal_rules() -> list[AlertRuleSpec]:
    """Both cerebral hemispheres — the running service watches both."""
    return [
        neonatal_rso2_rule(metric="rso2_left"),
        neonatal_rso2_rule(metric="rso2_right"),
    ]
