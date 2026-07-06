"""SystemAlertRuleSpec — user-authored SYSTEM-metric alert rules.

A parallel, NON-CLINICAL rule domain (CPU, disk I/O, memory, load on the
Jetson itself), authored through the Helper chat. It deliberately does NOT
touch `alert_rule_spec.py`: the clinical spec's locked semantics (0.80x
relative drop below a 24h baseline, `<` comparator, 5m sustain — LD-3) stay
inviolable. System rules use absolute thresholds instead.

Safety posture (mirrors the clinical domain where it matters):

- **Curated metric catalog, never raw PromQL.** The LLM/user picks a
  `metric_kind` from `SYSTEM_METRIC_CATALOG`; the PromQL expression is a
  hardcoded template in THIS file. No user- or LLM-supplied query text ever
  reaches Prometheus, so there is no injection surface to sanitize.
- **Shadow is structural.** `mode` is a Literal["shadow"] — a spec that asks
  for anything else fails validation. Promotion to a paging rule is not
  expressible here at all; it would be a separate, user-authorized feature
  (LD-6 spirit: no agent can promote by construction).
- **source is locked to "prometheus"** — mock data can never alert (same
  invariant as the clinical domain).
- Threshold and duration are bounded per metric kind so a typo'd request
  ("alert at 9000%") is rejected loudly instead of creating a dead rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .widget_spec import _assert_safe_text


# ---------------------------------------------------------------------------
# Curated metric catalog — the ONLY expressions system rules may evaluate.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogEntry:
    """One selectable system metric. `promql` is a fixed template — the only
    query text that ever reaches Prometheus for this domain."""

    kind: str
    promql: str
    unit: str
    description: str
    min_threshold: float
    max_threshold: float


SYSTEM_METRIC_CATALOG: dict[str, CatalogEntry] = {
    e.kind: e
    for e in (
        CatalogEntry(
            kind="cpu_utilization_pct",
            promql=(
                '100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])))'
            ),
            unit="%",
            description="Overall CPU utilization of the Jetson host",
            min_threshold=1.0,
            max_threshold=100.0,
        ),
        CatalogEntry(
            kind="disk_io_utilization_pct",
            promql="100 * max(rate(node_disk_io_time_seconds_total[5m]))",
            unit="%",
            description="Busy-time of the busiest disk device",
            min_threshold=1.0,
            max_threshold=100.0,
        ),
        CatalogEntry(
            kind="disk_usage_pct",
            promql=(
                '100 * (1 - node_filesystem_avail_bytes{mountpoint="/",fstype!="tmpfs"}'
                ' / node_filesystem_size_bytes{mountpoint="/",fstype!="tmpfs"})'
            ),
            unit="%",
            description="Root filesystem space used",
            min_threshold=1.0,
            max_threshold=100.0,
        ),
        CatalogEntry(
            kind="memory_used_pct",
            promql=(
                "100 * (1 - node_memory_MemAvailable_bytes"
                " / node_memory_MemTotal_bytes)"
            ),
            unit="%",
            description="Memory in use (MemAvailable-based)",
            min_threshold=1.0,
            max_threshold=100.0,
        ),
        CatalogEntry(
            kind="load1",
            promql="node_load1",
            unit="",
            description="1-minute load average",
            min_threshold=0.1,
            max_threshold=64.0,
        ),
    )
}

# Raw gauge whose SAMPLE TIMESTAMP proves the node scrape is alive. Catalog
# expressions are computed vectors (rate/avg), whose timestamp() is the query
# evaluation time — useless for staleness. This probe is a plain series, so
# timestamp() returns its true last-scrape time.
FRESHNESS_PROBE_METRIC = "node_time_seconds"


_DURATION_RE = re.compile(r"^\d{1,3}[smh]$")
_MIN_FOR_S = 60.0
_MAX_FOR_S = 3600.0


def parse_duration_seconds(text: str) -> float:
    unit = text[-1]
    num = float(text[:-1])
    return num * {"s": 1.0, "m": 60.0, "h": 3600.0}[unit]


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class SystemAlertRuleSpec(BaseModel):
    """A single absolute-threshold rule over a curated system metric."""

    id: str = Field(..., min_length=3, max_length=64, pattern=r"^[a-z0-9][a-z0-9\-]+$")
    title: str = Field(..., min_length=1, max_length=120)
    metric_kind: str = Field(..., min_length=1, max_length=64)
    comparator: Literal[">", "<"]
    threshold: float
    for_: str = Field(default="5m", alias="for", max_length=8)
    severity: Literal["info", "warning", "critical"] = "warning"
    # Structural shadow: any other value is a validation error. Promotion is
    # deliberately not expressible in this spec (LD-6 spirit).
    mode: Literal["shadow"] = "shadow"
    # Mock data can never alert — same invariant as the clinical domain.
    source: Literal["prometheus"] = "prometheus"
    created_by: str = Field(default="helper-chat", max_length=64)
    created_at: str = Field(default_factory=_now_iso, max_length=64)
    # Every alerting surface carries the decision-support disclaimer.
    non_diagnostic: bool = True

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("title")
    @classmethod
    def _safe_title(cls, v: str) -> str:
        _assert_safe_text(v, "title")
        return v

    @field_validator("metric_kind")
    @classmethod
    def _known_kind(cls, v: str) -> str:
        if v not in SYSTEM_METRIC_CATALOG:
            raise ValueError(
                f"unknown metric_kind {v!r}; must be one of "
                f"{sorted(SYSTEM_METRIC_CATALOG)}"
            )
        return v

    @field_validator("for_")
    @classmethod
    def _bounded_for(cls, v: str) -> str:
        if not _DURATION_RE.match(v):
            raise ValueError(f"for must look like '90s'/'5m'/'1h', got {v!r}")
        seconds = parse_duration_seconds(v)
        if not (_MIN_FOR_S <= seconds <= _MAX_FOR_S):
            raise ValueError(
                f"for must be between 1m and 1h, got {v!r} ({seconds:.0f}s)"
            )
        return v

    @field_validator("threshold")
    @classmethod
    def _finite_threshold(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("threshold must be a finite number")
        return v

    def model_post_init(self, __context) -> None:  # noqa: D105
        entry = SYSTEM_METRIC_CATALOG[self.metric_kind]
        if not (entry.min_threshold <= self.threshold <= entry.max_threshold):
            raise ValueError(
                f"threshold {self.threshold} out of range for "
                f"{self.metric_kind} ({entry.min_threshold}"
                f"–{entry.max_threshold}{entry.unit})"
            )

    @property
    def catalog_entry(self) -> CatalogEntry:
        return SYSTEM_METRIC_CATALOG[self.metric_kind]

    @property
    def for_seconds(self) -> float:
        return parse_duration_seconds(self.for_)

    def human_summary(self) -> str:
        e = self.catalog_entry
        return (
            f"{e.description} {self.comparator} {self.threshold}{e.unit} "
            f"sustained {self.for_} (severity {self.severity}, SHADOW mode)"
        )
