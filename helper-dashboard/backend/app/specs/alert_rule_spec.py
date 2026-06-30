"""AlertRuleSpec — the clinical anomaly-detection rule contract.

A rule describes ONE breach condition over ONE Prometheus metric:

    breach = value < ratio * avg_over_time(metric[window])   sustained for `for`

The locked semantics (supervisor decision) are:

    ratio      = 0.80   (a 20% relative drop below the 24h baseline)
    window     = "24h"
    comparator = "<"
    for        = "5m"

This is patient-safety-critical software for a neonatal cerebral-oximetry
(rSO2) monitor. The spec is therefore STRICT and FAIL-CLOSED:

- `extra='forbid'` — no unknown keys may smuggle in behaviour.
- `source` MUST be `prometheus`; `source == "mock"` is **rejected at
  validation time** so a rule can never be authored against the silent
  mock fallback in `prometheus/client.py`.
- Free-text fields are scrubbed with `_UNSAFE_PATTERN`; metric/label
  tokens go through the same PromQL token denylist used by `widget_spec`.
- `FORBIDDEN_WIDGET_FIELDS` keys are rejected anywhere they appear so the
  spec_validator's pre-Pydantic scan stays meaningful for alert rules too.

New/edited rules are SHADOW by default at the evaluator layer; this spec
only declares the rule — promotion to paging is a separate, supervisor-
gated step and is intentionally NOT expressible by lowering `severity`.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Reuse the exact security conventions from widget_spec so there is a
# single source of truth for "what is unsafe text".
from .widget_spec import (
    FORBIDDEN_WIDGET_FIELDS,
    QuerySource,
    _UNSAFE_PATTERN,
    _assert_safe_text,
)

__all__ = [
    "AlertRuleSpec",
    "BaselineSpec",
    "AlertSeverity",
    "RuleMode",
    "PROMQL_FORBIDDEN_TOKENS",
]


# PromQL token denylist — mirrors widget_spec.QuerySpec._check_promql and is
# applied to the metric name and every label key/value so a rule cannot
# inject shell / HTML / PromQL-composition characters.
PROMQL_FORBIDDEN_TOKENS: tuple[str, ...] = (";", "`", "<", ">", "$(", "||", "(", ")", "{", "}", "[", "]", " ")


# Identifier shape Prometheus itself enforces for metric/label names.
_PROM_IDENT = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
# Label *values* are freer than names but we still keep them tame and short.
_LABEL_VALUE = re.compile(r"^[a-zA-Z0-9_\-:./]+$")


def _check_prom_token(value: str, field: str) -> str:
    v = value.strip()
    if not v:
        raise ValueError(f"{field} must not be empty")
    for token in PROMQL_FORBIDDEN_TOKENS:
        if token in v:
            raise ValueError(f"{field} contains forbidden token: {token!r}")
    _assert_safe_text(v, field)
    return v


class AlertSeverity(str, Enum):
    """Clinical severity. `critical` may never be silently suppressed."""

    info = "info"
    warning = "warning"
    critical = "critical"


class RuleMode(str, Enum):
    """Paging posture of a rule.

    A NEW or EDITED rule is SHADOW by default and may only be promoted to
    `active` by the supervisor on green goldens. The spec carries the
    declared mode but the evaluator treats anything it has not been told
    to promote as shadow — defence in depth.
    """

    shadow = "shadow"
    active = "active"


class BaselineSpec(BaseModel):
    """The baseline aggregation: avg_over_time(metric[window])."""

    model_config = ConfigDict(extra="forbid")

    # Locked: only avg_over_time is permitted for the clinical baseline.
    fn: Literal["avg_over_time"] = "avg_over_time"
    # Locked: 24h baseline window. Restricted to a small set of safe,
    # well-formed duration literals (no arbitrary strings into PromQL).
    window: Literal["24h"] = "24h"


class AlertRuleSpec(BaseModel):
    """A single clinical anomaly-detection rule.

    breach = value `comparator` (ratio * baseline.fn(metric[baseline.window]))
    sustained for `for_`.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    description: str | None = Field(default=None, max_length=512)

    metric: str = Field(..., min_length=1, max_length=128)
    labels: dict[str, str] = Field(default_factory=dict)

    baseline: BaselineSpec = Field(default_factory=BaselineSpec)

    # Locked clinical semantics. Kept as fields (not hard-coded in the
    # evaluator) so they are auditable in the spec, but constrained so a
    # rule cannot drift from the supervisor's decision.
    comparator: Literal["<"] = "<"
    ratio: float = Field(default=0.80, gt=0.0, lt=1.0)
    for_: str = Field(default="5m", alias="for", pattern=r"^\d{1,4}(s|m|h)$")

    severity: AlertSeverity = AlertSeverity.critical

    # The data SOURCE this rule alerts on. MUST be prometheus.
    source: QuerySource = QuerySource.prometheus

    # Declared paging posture; shadow by default.
    mode: RuleMode = RuleMode.shadow

    @field_validator("metric")
    @classmethod
    def _check_metric(cls, v: str) -> str:
        v = _check_prom_token(v, "metric")
        if not _PROM_IDENT.match(v):
            raise ValueError(
                "metric must be a valid Prometheus identifier "
                "([a-zA-Z_][a-zA-Z0-9_]*)"
            )
        return v

    @field_validator("labels")
    @classmethod
    def _check_labels(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > 16:
            raise ValueError("too many labels (max 16)")
        for key, val in v.items():
            if key in FORBIDDEN_WIDGET_FIELDS:
                raise ValueError(f"label key {key!r} is forbidden")
            if not _PROM_IDENT.match(key):
                raise ValueError(
                    f"label name {key!r} must be a valid Prometheus identifier"
                )
            if not isinstance(val, str):
                raise ValueError(f"label value for {key!r} must be a string")
            if len(val) > 128:
                raise ValueError(f"label value for {key!r} too long (max 128)")
            if not _LABEL_VALUE.match(val):
                raise ValueError(
                    f"label value for {key!r} contains illegal characters"
                )
            _assert_safe_text(val, f"labels.{key}")
        return v

    @field_validator("description")
    @classmethod
    def _safe_description(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _assert_safe_text(v, "description")

    @field_validator("source")
    @classmethod
    def _reject_mock(cls, v: QuerySource) -> QuerySource:
        # FAIL CLOSED: a clinical rule may never alert on the silent mock
        # fallback. This is an inviolable safety invariant — see
        # prometheus/client.py where `source: mock` is written but never
        # read. The evaluator additionally re-checks provenance at runtime.
        if v != QuerySource.prometheus:
            raise ValueError(
                "source must be 'prometheus'; alerting on mock/fake data is "
                "forbidden (fail-closed clinical-safety invariant)"
            )
        return v
