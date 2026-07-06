"""Anomaly evaluator CORE — pure, synchronous, unit-testable.

This is the clinical heart of the system. It is deliberately:

- **Pure / synchronous** — no scheduler, no network, no I/O. The caller
  supplies the observation (value, baseline, provenance) and an INJECTED
  clock. This makes every decision deterministic and replayable by the
  golden harness.
- **Fail-closed** — the data-integrity gate runs FIRST on every tick. If it
  fails for any reason, we emit `SIGNAL_LOST`, clear the breach timer, and
  return WITHOUT computing a clinical verdict.
- **Shadow by default** — a firing event records `would_page=True` but only
  sets `paged=True` when the rule has been promoted to `active`. Suppression
  is always visible in the event; nothing is silently dropped.

Order of operations per tick (NEVER reordered):

    1. data-integrity gate:
         reachable ∧ returns_data ∧ source == "prometheus" ∧ fresh
       any failure -> SIGNAL_LOST, clear breach timer, return.
    2. baseline sufficiency:
         baseline present ∧ history covers the full window
       insufficient -> INSUFFICIENT_BASELINE, clear breach timer, return.
    3. breach compare:
         value < ratio * baseline  ->  breach
    4. breach-DURATION state machine:
         not-breaching          -> resolved/normal, clear timer
         breaching, < for_      -> pending (timer running)
         breaching, >= for_     -> firing
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..specs.alert_rule_spec import AlertRuleSpec, RuleMode
from ..specs.anomaly_evaluation_report import (
    AlertEvent,
    AlertState,
    AnomalyEvaluationReport,
    SignalLostReason,
)


def _parse_duration_seconds(text: str) -> float:
    """Parse a Prometheus-style duration like '5m', '30s', '24h'."""
    text = text.strip()
    unit = text[-1]
    num = float(text[:-1])
    if unit == "s":
        return num
    if unit == "m":
        return num * 60.0
    if unit == "h":
        return num * 3600.0
    raise ValueError(f"unsupported duration unit in {text!r}")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@dataclass
class DataStatus:
    """Provenance + freshness of the observation handed to the core.

    The core re-checks these on every tick BEFORE any clinical compare.
    """

    reachable: bool
    returns_data: bool
    source: str  # "prometheus" | "mock" | ...
    sample_ts: float  # epoch seconds of the freshest sample
    # How many seconds of contiguous history back the baseline. Used to
    # decide INSUFFICIENT_BASELINE vs a usable 24h window.
    history_coverage_s: float = 0.0
    # True when the metric selector matched MORE than one series: the value
    # would be an arbitrary pick (possibly the wrong patient). The gate fails
    # closed on it. Defaults False so single-series providers are unaffected.
    ambiguous: bool = False


class AnomalyEvaluatorCore:
    """Per-rule breach-duration state machine.

    One instance tracks ONE rule's breach timer across ticks. Construct with
    a freshness budget (staleness tolerance) and whether the rule is promoted.
    """

    def __init__(
        self,
        rule: AlertRuleSpec,
        *,
        staleness_budget_s: float = 60.0,
        promoted: bool = False,
    ) -> None:
        self.rule = rule
        self.staleness_budget_s = float(staleness_budget_s)
        # Defence in depth: a rule pages only if it is BOTH declared active
        # AND explicitly promoted by the supervisor. Default: shadow.
        self.promoted = bool(promoted) and rule.mode == RuleMode.active

        self._for_seconds = _parse_duration_seconds(rule.for_)
        self._window_seconds = _parse_duration_seconds(rule.baseline.window)

        # Breach-timer state.
        self._breach_started_at: float | None = None
        self._fired: bool = False  # currently in firing state
        self._last_state: AlertState | None = None

    # ------------------------------------------------------------------
    def evaluate(
        self,
        *,
        now: float,
        value: float | None,
        baseline: float | None,
        status: DataStatus,
    ) -> AnomalyEvaluationReport:
        """Evaluate one tick. `now` is the INJECTED clock (epoch seconds)."""
        # --- 1. DATA-INTEGRITY GATE (always first) --------------------
        gate_reason = self._integrity_failure(status, now)
        if gate_reason is not None:
            return self._signal_lost(now, value, gate_reason)

        # --- 2. BASELINE SUFFICIENCY ----------------------------------
        if baseline is None or status.history_coverage_s < self._window_seconds:
            return self._insufficient_baseline(now, value, baseline)

        # --- 3. BREACH COMPARE ----------------------------------------
        threshold = self.rule.ratio * baseline
        # value is guaranteed present here: returns_data was True.
        assert value is not None
        breaching = value < threshold  # comparator is locked to '<'

        # --- 4. BREACH-DURATION STATE MACHINE -------------------------
        if not breaching:
            return self._not_breaching(now, value, baseline, threshold)

        if self._breach_started_at is None:
            self._breach_started_at = now
        elapsed = now - self._breach_started_at

        if elapsed >= self._for_seconds:
            return self._firing(now, value, baseline, threshold, elapsed)
        return self._pending(now, value, baseline, threshold, elapsed)

    # ------------------------------------------------------------------
    # Gate / state helpers
    # ------------------------------------------------------------------
    def _integrity_failure(
        self, status: DataStatus, now: float
    ) -> SignalLostReason | None:
        if not status.reachable:
            return SignalLostReason.unreachable
        if not status.returns_data:
            return SignalLostReason.no_data
        if status.source != "prometheus":
            # FAIL CLOSED on mock/fake data — the silent fallback hazard.
            return SignalLostReason.mock_source
        if status.ambiguous:
            # FAIL CLOSED when the selector matched multiple series — an
            # arbitrary pick could evaluate the wrong patient.
            return SignalLostReason.ambiguous_series
        if (now - status.sample_ts) > self.staleness_budget_s:
            return SignalLostReason.stale
        return None

    def _signal_lost(
        self, now: float, value: float | None, reason: SignalLostReason
    ) -> AnomalyEvaluationReport:
        # Loss of signal is alarmable, never "no anomaly". Clear the breach
        # timer so a stale breach can't silently "complete" while blind.
        self._breach_started_at = None
        self._fired = False
        event = AlertEvent(
            ts=_iso(now),
            rule_id=self.rule.id,
            state=AlertState.signal_lost,
            value=value,
            severity=self.rule.severity.value,
            signal_lost_reason=reason,
            # SIGNAL_LOST is itself a conspicuous, non-suppressible alarm.
            would_page=True,
            paged=self.promoted,
            suppressed_reason=None if self.promoted else "shadow_mode",
            message=f"SIGNAL_LOST ({reason.value}) — data-integrity gate failed; "
            f"no clinical verdict computed",
        )
        return self._report(now, AlertState.signal_lost, value, None, None, [event])

    def _insufficient_baseline(
        self, now: float, value: float | None, baseline: float | None
    ) -> AnomalyEvaluationReport:
        self._breach_started_at = None
        self._fired = False
        event = AlertEvent(
            ts=_iso(now),
            rule_id=self.rule.id,
            state=AlertState.insufficient_baseline,
            value=value,
            baseline=baseline,
            severity=self.rule.severity.value,
            would_page=False,  # not a clinical fire
            paged=False,
            message=(
                f"INSUFFICIENT_BASELINE — <{self.rule.baseline.window} "
                "history; refusing to fire"
            ),
        )
        return self._report(
            now, AlertState.insufficient_baseline, value, baseline, None, [event]
        )

    def _not_breaching(
        self, now: float, value: float, baseline: float, threshold: float
    ) -> AnomalyEvaluationReport:
        events: list[AlertEvent] = []
        # If we were firing or pending and recovered, emit a resolved event.
        if self._fired or self._breach_started_at is not None:
            events.append(
                AlertEvent(
                    ts=_iso(now),
                    rule_id=self.rule.id,
                    state=AlertState.resolved,
                    value=value,
                    baseline=baseline,
                    threshold=threshold,
                    severity=self.rule.severity.value,
                    message="RESOLVED — value recovered above threshold",
                )
            )
        self._breach_started_at = None
        self._fired = False
        state = AlertState.resolved if events else AlertState.resolved
        return self._report(now, state, value, baseline, threshold, events)

    def _pending(
        self,
        now: float,
        value: float,
        baseline: float,
        threshold: float,
        elapsed: float,
    ) -> AnomalyEvaluationReport:
        events: list[AlertEvent] = []
        if self._last_state != AlertState.pending:
            events.append(
                AlertEvent(
                    ts=_iso(now),
                    rule_id=self.rule.id,
                    state=AlertState.pending,
                    value=value,
                    baseline=baseline,
                    threshold=threshold,
                    severity=self.rule.severity.value,
                    would_page=False,
                    message=f"PENDING — breach started; {elapsed:.0f}s/"
                    f"{self._for_seconds:.0f}s into the for-window",
                )
            )
        rpt = self._report(now, AlertState.pending, value, baseline, threshold, events)
        rpt.breach_elapsed_s = elapsed
        rpt.for_seconds = self._for_seconds
        return rpt

    def _firing(
        self,
        now: float,
        value: float,
        baseline: float,
        threshold: float,
        elapsed: float,
    ) -> AnomalyEvaluationReport:
        events: list[AlertEvent] = []
        # Emit a firing event on the transition into firing (edge), so we
        # page once rather than every tick — but suppression stays visible.
        if not self._fired:
            paged = self.promoted
            events.append(
                AlertEvent(
                    ts=_iso(now),
                    rule_id=self.rule.id,
                    state=AlertState.firing,
                    value=value,
                    baseline=baseline,
                    threshold=threshold,
                    severity=self.rule.severity.value,
                    would_page=True,
                    paged=paged,
                    suppressed_reason=None if paged else "shadow_mode",
                    # The stated criterion interpolates the rule's ACTUAL
                    # ratio — hardcoded text would lie for a non-default rule.
                    message=f"FIRING — value {value:.2f} < threshold "
                    f"{threshold:.2f} ({self.rule.ratio:.2f}x baseline) "
                    f"sustained {elapsed:.0f}s",
                )
            )
        self._fired = True
        rpt = self._report(now, AlertState.firing, value, baseline, threshold, events)
        rpt.breach_elapsed_s = elapsed
        rpt.for_seconds = self._for_seconds
        return rpt

    # ------------------------------------------------------------------
    def _report(
        self,
        now: float,
        state: AlertState,
        value: float | None,
        baseline: float | None,
        threshold: float | None,
        events: list[AlertEvent],
    ) -> AnomalyEvaluationReport:
        self._last_state = state
        return AnomalyEvaluationReport(
            rule_id=self.rule.id,
            ts=_iso(now),
            state=state,
            value=value,
            baseline=baseline,
            threshold=threshold,
            mode="active" if self.promoted else "shadow",
            events=events,
        )
