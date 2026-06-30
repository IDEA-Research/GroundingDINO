"""AnomalyEvaluationReport + AlertEvent — the evaluator's output contract.

The anomaly evaluator core is a pure state machine. On every tick it emits
zero or more `AlertEvent`s describing what happened to a rule, and an
`AnomalyEvaluationReport` snapshot of the rule's current state.

State machine (per rule):

    insufficient_baseline   <24h of history / no usable baseline -> never fires
    signal_lost             data-integrity gate failed (unreachable, no data,
                            source=="mock", or stale) -> NEVER a clinical verdict
    pending                 value breached the threshold; breach timer running
    firing                  breach sustained for `for_` (5m) -> alarmable
    resolved               value recovered above threshold (or breach cleared)

Safety notes encoded here:

- `signal_lost` and `insufficient_baseline` are FIRST-CLASS states, not
  "no anomaly". They are alarmable / surfaced loudly, never swallowed.
- Every event records `would_page` separately from delivery. In SHADOW
  mode a firing event has `would_page=True` but `paged=False`; nothing is
  silently dropped — suppression is always visible in the event.
- Every alerting-surface event carries `non_diagnostic=True` so the
  "decision-support, not a diagnosis" label is structurally present.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AlertState(str, Enum):
    pending = "pending"
    firing = "firing"
    resolved = "resolved"
    signal_lost = "signal_lost"
    insufficient_baseline = "insufficient_baseline"


class SignalLostReason(str, Enum):
    """Why the data-integrity gate failed. None of these are clinical."""

    unreachable = "unreachable"
    no_data = "no_data"
    mock_source = "mock_source"
    stale = "stale"


class AlertEvent(BaseModel):
    """A single transition / observation emitted by the evaluator core."""

    model_config = ConfigDict(extra="forbid")

    # ISO-8601 UTC timestamp from the INJECTED clock (deterministic in tests).
    ts: str = Field(..., min_length=1, max_length=64)
    rule_id: str = Field(..., min_length=1, max_length=64)
    state: AlertState

    # Observed value and the computed baseline / threshold. May be None when
    # the data-integrity gate failed (signal_lost) or baseline is absent.
    value: float | None = None
    baseline: float | None = None
    threshold: float | None = None

    severity: str = "critical"

    # Why this signal was lost, when state == signal_lost.
    signal_lost_reason: SignalLostReason | None = None

    # Paging accounting — suppression is ALWAYS visible, never silent.
    would_page: bool = False  # this event *would* page if active
    paged: bool = False  # this event actually paged (active mode only)
    suppressed_reason: str | None = Field(default=None, max_length=256)

    # Persistent, non-suppressible disclaimer marker.
    non_diagnostic: bool = True

    # Human-readable one-liner for the audit / UI.
    message: str = Field(default="", max_length=512)


class AnomalyEvaluationReport(BaseModel):
    """Snapshot of a rule's state after a tick, plus the events it emitted."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(..., min_length=1, max_length=64)
    ts: str = Field(..., min_length=1, max_length=64)
    state: AlertState

    value: float | None = None
    baseline: float | None = None
    threshold: float | None = None

    # When state == pending, how long the breach has been sustained (seconds),
    # so an operator can see "3m into a 5m window".
    breach_elapsed_s: float | None = None
    for_seconds: float | None = None

    mode: str = "shadow"
    non_diagnostic: bool = True

    events: list[AlertEvent] = Field(default_factory=list, max_length=64)
