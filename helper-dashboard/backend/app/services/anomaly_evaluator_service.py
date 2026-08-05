"""Background anomaly evaluator service — the continuous loop around the core.

This wraps the pure `AnomalyEvaluatorCore` (INC1) in a runnable service:

    - a `tick(now)` function that is driven by an INJECTED CLOCK. It fetches
      one observation per rule via a `DataProvider`, runs the core, and
      persists durable state via `AlertStateStore`. Because `pytest_asyncio`
      is not installed, `tick` is a plain synchronous function the tests step
      by hand; the FastAPI lifespan task (see main.py) merely calls it on a
      real timer.

    - a HEARTBEAT + WATCHDOG: every tick stamps `last_tick_monotonic` /
      `last_tick_wall`. `watchdog(now_monotonic)` reports whether the loop is
      healthy or STALLED (a crashed / hung tick is detectable rather than
      silently "no anomaly"). Absence-of-series and source=="mock" are
      surfaced as alarmable SIGNAL_LOST via the core, and the watchdog tracks
      how many consecutive ticks were degraded.

Shadow discipline (INC2): rules stay in SHADOW. A firing event records
`would_page=True` / `paged=False`; NOTHING is paged here (the notifier is
INC3). Suppression stays visible in the persisted event, never silent.

Rehydration: on construction the service rehydrates each core's breach timer
from the durable store, so a process restart in the middle of a 5m breach
does NOT reset the clock — the breach continues counting from `breach_start`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from ..specs.alert_rule_spec import AlertRuleSpec
from ..specs.anomaly_evaluation_report import (
    AlertEvent,
    AlertState,
    AnomalyEvaluationReport,
)
from . import anomaly_build_audit as audit
from . import anomaly_lifecycle_audit as lifecycle
from .alert_state_store import AlertStateStore
from .anomaly_core import AnomalyEvaluatorCore
from .anomaly_data_provider import DataProvider
from .anomaly_notification_dispatch import DispatchOutcome, NotificationDispatcher


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# States that mean "the loop is producing a degraded, non-clinical verdict".
_DEGRADED_STATES = frozenset(
    {AlertState.signal_lost, AlertState.insufficient_baseline}
)


@dataclass
class Heartbeat:
    """Liveness record for the watchdog.

    `last_tick_monotonic` is used to detect a STALLED loop (a crashed/hung
    tick). `last_tick_wall` is the injected-clock time of the last tick, for
    the audit/UI. `consecutive_degraded` counts back-to-back degraded ticks
    so a persistently-blind loop is loud, not quiet.
    """

    started_wall: float | None = None
    last_tick_monotonic: float | None = None
    last_tick_wall: float | None = None
    tick_count: int = 0
    error_count: int = 0
    consecutive_degraded: int = 0
    last_error: str | None = None


@dataclass
class WatchdogVerdict:
    healthy: bool
    stalled: bool
    seconds_since_last_tick: float | None
    degraded: bool
    reason: str = ""


@dataclass
class TickResult:
    """What one tick produced across all rules (for tests + audit)."""

    now: float
    reports: dict[str, AnomalyEvaluationReport] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    # Per-rule notification-dispatch outcome (INC3). Empty when no dispatcher
    # is wired. Delivery is out of band and never gates the clinical verdict.
    dispatch: dict[str, DispatchOutcome] = field(default_factory=dict)


class AnomalyEvaluatorService:
    """Drives one or more rules through the core on an injected clock."""

    def __init__(
        self,
        rules: list[AlertRuleSpec],
        provider: DataProvider,
        *,
        store: AlertStateStore | None = None,
        staleness_budget_s: float = 60.0,
        watchdog_budget_s: float = 45.0,
        promoted: bool = False,
        monotonic: Callable[[], float] | None = None,
        # Audit tag for records this service produces. Pass a build increment
        # ("INC5") only from an actual build loop; live wiring passes
        # "runtime" / "demo" so runtime records never claim build activity.
        increment: str = "runtime",
        dispatcher: NotificationDispatcher | None = None,
    ) -> None:
        self.rules = list(rules)
        self.provider = provider
        self.store = store or AlertStateStore()
        # Optional out-of-band notifier wiring (INC3). When absent, the loop
        # behaves exactly as INC2 (record-only). Delivery NEVER gates or
        # delays the clinical verdict; it runs after the verdict is persisted.
        self.dispatcher = dispatcher
        self.staleness_budget_s = float(staleness_budget_s)
        # Watchdog budget defaults to 3x a 15s tick — a missed tick is caught
        # quickly but a single slow tick does not false-alarm.
        self.watchdog_budget_s = float(watchdog_budget_s)
        self.promoted = bool(promoted)
        self._monotonic = monotonic  # injectable for tests; else time.monotonic
        self.increment = increment

        self.heartbeat = Heartbeat()
        self._cores: dict[str, AnomalyEvaluatorCore] = {}
        for rule in self.rules:
            core = AnomalyEvaluatorCore(
                rule,
                staleness_budget_s=self.staleness_budget_s,
                promoted=self.promoted,
            )
            self._rehydrate(core, rule)
            self._cores[rule.id] = core
            # Clinical lifecycle: a rule entering the running set is an audited
            # activation event (shadow by default; paging is separate).
            lifecycle.record_rule_activated(
                rule, promoted=core.promoted, increment=self.increment
            )

        if self.store.load_error:
            # Surface a corrupt-state load loudly; do NOT pretend healthy.
            audit.append(
                increment=self.increment,
                stage="observe",
                action="run",
                target="AlertStateStore.load",
                summary="Durable alert-state snapshot was unreadable on startup",
                reasoning="Fail-closed: corrupt state is surfaced, not reset to healthy",
                evidence={"load_error": self.store.load_error},
                outcome="degraded_startup",
            )

    # ------------------------------------------------------------------
    def _rehydrate(self, core: AnomalyEvaluatorCore, rule: AlertRuleSpec) -> None:
        """Restore a breach-in-progress timer so a restart never resets 5m."""
        st = self.store.get(rule.id)
        if st.breach_start_ts is not None:
            # Reach into the core's private timer deliberately: this is the
            # ONE place allowed to seed it, and only from durable state.
            core._breach_started_at = st.breach_start_ts  # noqa: SLF001
            core._fired = bool(st.fired)  # noqa: SLF001
            if st.state:
                try:
                    core._last_state = AlertState(st.state)  # noqa: SLF001
                except ValueError:
                    core._last_state = None  # noqa: SLF001

    # ------------------------------------------------------------------
    def _mono(self) -> float:
        if self._monotonic is not None:
            return self._monotonic()
        import time

        return time.monotonic()

    # ------------------------------------------------------------------
    def tick(self, now: float, *, mono: float | None = None) -> TickResult:
        """Evaluate every rule once at injected clock time `now`.

        `mono` is an injectable monotonic reading for the watchdog; if omitted
        the service reads its monotonic source. Any per-rule failure is caught
        and recorded (the loop must not die because one rule threw), but a
        failure to PERSIST a clinical event propagates from the store and is
        counted as an error — it is never swallowed into a false green.
        """
        if self.heartbeat.started_wall is None:
            self.heartbeat.started_wall = now

        result = TickResult(now=now)
        any_degraded = False

        for rule in self.rules:
            core = self._cores[rule.id]
            # Snapshot the core's edge state: evaluate() consumes a firing
            # edge (sets _fired) BEFORE persist/audit/dispatch run. If any of
            # those fail, the edge must be rolled back so the next tick
            # re-emits the firing event and the page is re-attempted —
            # otherwise a one-tick disk hiccup at the firing edge would
            # silently swallow the page for the whole episode (at-least-once:
            # a duplicate page on retry is acceptable; a lost page is not).
            pre_breach = core._breach_started_at  # noqa: SLF001
            pre_fired = core._fired  # noqa: SLF001
            pre_last_state = core._last_state  # noqa: SLF001
            report = None
            try:
                obs = self.provider.observe(
                    rule.metric, now=now, labels=rule.labels
                )
                report = core.evaluate(
                    now=now,
                    value=obs.value,
                    baseline=obs.baseline,
                    status=obs.status,
                )
                # Persist durable state, seeding the breach-start from the
                # core's current timer so a restart can rehydrate it.
                self.store.record_report(
                    report,
                    breach_start_ts=core._breach_started_at,  # noqa: SLF001
                )
                result.reports[rule.id] = report
                if report.state in _DEGRADED_STATES:
                    any_degraded = True
                self._audit_report(rule, report)
                # Clinical lifecycle audit: every transition captured (INC5).
                for ev in report.events:
                    lifecycle.record_transition(
                        ev, mode=report.mode, increment=self.increment
                    )

                # Out-of-band delivery (INC3) runs AFTER the clinical verdict
                # is persisted, so a notifier hiccup can never lose or delay a
                # clinical record. A dispatch failure is caught below and made
                # visible; it does not corrupt the clinical state.
                if self.dispatcher is not None:
                    outcome = self.dispatcher.dispatch_report(report)
                    result.dispatch[rule.id] = outcome
                    # Reconcile expected vs actual paging: a failed attempt is
                    # a DELIVERY_FAILED (channel failure); a paged event with
                    # no receipt at all is a MISSED must-fire — the
                    # top-severity clinical event.
                    self._audit_delivery_lifecycle(rule, report, outcome)
                    # A non-signal_lost verdict ends the loss episode: re-arm
                    # its dedup so the NEXT genuine loss pages again (a quiet
                    # recovery emits no events, so this must be state-driven).
                    if report.state != AlertState.signal_lost:
                        self.dispatcher.rearm_signal_lost(rule.id)
            except Exception as exc:  # noqa: BLE001
                # A tick that raises must be VISIBLE (watchdog + audit), not
                # a silent "no anomaly". Record and keep the loop alive.
                # Roll the edge back ONLY when this tick consumed a FIRING
                # edge that never finished persist+dispatch, so the next tick
                # re-emits and re-pages it. A blanket rollback is itself a
                # hazard: restoring _fired=True after a failed RESOLVED edge
                # would swallow the next episode's firing event entirely.
                if (
                    report is not None
                    and report.state == AlertState.firing
                    and not pre_fired
                ):
                    core._breach_started_at = pre_breach  # noqa: SLF001
                    core._fired = pre_fired  # noqa: SLF001
                    core._last_state = pre_last_state  # noqa: SLF001
                # Symmetric duty on the exception path: a verdict that is NOT
                # signal_lost proves the signal recovered, so the loss-episode
                # dedup must re-arm even when this tick failed — the normal
                # re-arm at the end of the try block was skipped, and every
                # later degraded tick reports signal_lost again, so it would
                # never run and the NEXT loss episode would stay muted.
                # Worst case on retry is a duplicate page, which the design
                # accepts; a muted episode is not.
                if (
                    self.dispatcher is not None
                    and report is not None
                    and report.state != AlertState.signal_lost
                ):
                    self.dispatcher.rearm_signal_lost(rule.id)
                self.heartbeat.error_count += 1
                self.heartbeat.last_error = f"{type(exc).__name__}: {exc}"
                result.errors[rule.id] = self.heartbeat.last_error
                any_degraded = True
                audit.append(
                    increment=self.increment,
                    stage="observe",
                    action="run",
                    target=f"tick/{rule.id}",
                    summary="Rule evaluation raised during tick",
                    reasoning="Fail-loud: a throwing tick is a watchdog event, not 'no anomaly'",
                    evidence={"error": self.heartbeat.last_error, "now": now},
                    outcome="tick_error",
                )

        # Heartbeat / watchdog bookkeeping.
        self.heartbeat.tick_count += 1
        self.heartbeat.last_tick_wall = now
        self.heartbeat.last_tick_monotonic = (
            mono if mono is not None else self._mono()
        )
        if any_degraded:
            self.heartbeat.consecutive_degraded += 1
        else:
            self.heartbeat.consecutive_degraded = 0

        return result

    # ------------------------------------------------------------------
    def watchdog(self, now_monotonic: float | None = None) -> WatchdogVerdict:
        """Report loop liveness. A stalled/crashed tick is detectable here."""
        mono = now_monotonic if now_monotonic is not None else self._mono()
        last = self.heartbeat.last_tick_monotonic
        if last is None:
            # Never ticked yet — not healthy, but not "stalled" (no baseline).
            return WatchdogVerdict(
                healthy=False,
                stalled=False,
                seconds_since_last_tick=None,
                degraded=False,
                reason="no ticks recorded yet",
            )
        since = mono - last
        stalled = since > self.watchdog_budget_s
        degraded = self.heartbeat.consecutive_degraded > 0
        healthy = (not stalled) and self.heartbeat.error_count == 0
        reason = ""
        if stalled:
            reason = (
                f"STALLED: {since:.1f}s since last tick "
                f"(> {self.watchdog_budget_s:.0f}s budget)"
            )
        elif self.heartbeat.error_count:
            reason = f"errors observed: {self.heartbeat.error_count}"
        elif degraded:
            reason = (
                f"degraded (signal_lost/insufficient_baseline) for "
                f"{self.heartbeat.consecutive_degraded} consecutive tick(s)"
            )
        return WatchdogVerdict(
            healthy=healthy,
            stalled=stalled,
            seconds_since_last_tick=since,
            degraded=degraded,
            reason=reason,
        )

    # ------------------------------------------------------------------
    def _audit_report(
        self, rule: AlertRuleSpec, report: AnomalyEvaluationReport
    ) -> None:
        """Audit any state transition (events) — shadow would-fires included."""
        if not report.events:
            return
        for ev in report.events:
            audit.append(
                increment=self.increment,
                stage="observe",
                action="run",
                target=f"rule/{rule.id}",
                summary=f"{ev.state.value}: {ev.message}",
                reasoning=(
                    "shadow would-fire recorded, never paged (notifier is INC3)"
                    if ev.would_page and not ev.paged
                    else "lifecycle transition persisted to alert-state store"
                ),
                evidence={
                    "state": ev.state.value,
                    "value": ev.value,
                    "threshold": ev.threshold,
                    "would_page": ev.would_page,
                    "paged": ev.paged,
                    "suppressed_reason": ev.suppressed_reason,
                    "signal_lost_reason": (
                        ev.signal_lost_reason.value
                        if ev.signal_lost_reason
                        else None
                    ),
                    "mode": report.mode,
                    "ts": ev.ts,
                },
                outcome=ev.state.value,
            )

    # ------------------------------------------------------------------
    def _audit_delivery_lifecycle(
        self,
        rule: AlertRuleSpec,
        report: AnomalyEvaluationReport,
        outcome: DispatchOutcome,
    ) -> None:
        """Record delivery / would-fire / suppression / failure per pageable event.

        Reconciles the report's pageable events against what the dispatcher
        actually did. A promoted (paged=True) pageable event whose delivery was
        attempted but failed/never acked is a DELIVERY_FAILED (channel
        failure); one with no receipt at all — neither a delivery attempt nor
        a legitimate dedup — is a MISSED must-fire, the top-severity
        clinical-safety event.
        """
        delivered_keys = {r.dedup_key for r in outcome.delivered}
        acked_keys = {r.dedup_key for r in outcome.delivered if r.acked}
        deduped_keys = set(outcome.deduped)
        would_fire_keys = set(outcome.would_fire_only)
        refused_keys = set(outcome.refused)

        for ev in report.events:
            if ev.state not in (AlertState.firing, AlertState.signal_lost):
                continue
            key = self._dedup_key(ev)

            # Shadow would-fire: recorded, never paged (visible, not silent).
            if not ev.paged:
                if key in would_fire_keys or ev.would_page:
                    lifecycle.record_would_fire(ev, increment=self.increment)
                continue

            # Promoted + this event actually paged: reconcile delivery.
            receipt = next(
                (r for r in outcome.delivered if r.dedup_key == key), None
            )
            if receipt is not None and receipt.acked:
                lifecycle.record_delivery(
                    ev,
                    channel=receipt.channel,
                    masked_target=self._masked_target(),
                    delivered=receipt.delivered,
                    acked=receipt.acked,
                    attempts=receipt.attempts,
                    error=receipt.error,
                    increment=self.increment,
                )
            elif key in deduped_keys:
                # A legitimate page-once dedup: recorded as suppressed, NOT a
                # miss (the first edge already delivered).
                lifecycle.record_suppressed(
                    ev.rule_id,
                    state=ev.state.value,
                    severity=ev.severity,
                    reason="deduped_page_once",
                    increment=self.increment,
                )
            elif key in refused_keys:
                # Suppression of a paging alert was REFUSED loudly upstream.
                lifecycle.record_suppressed(
                    ev.rule_id,
                    state=ev.state.value,
                    severity=ev.severity,
                    reason="suppression_refused",
                    increment=self.increment,
                )
            elif receipt is not None:
                # Delivery was ATTEMPTED but failed / never acked: a channel
                # failure, recorded distinctly so a webhook outage never
                # masquerades as the worst-case clinical MISSED signal. Still
                # loud, still fail-closed, error + attempts captured.
                lifecycle.record_delivery_failed(
                    ev,
                    channel=receipt.channel,
                    masked_target=self._masked_target(),
                    delivered=receipt.delivered,
                    acked=receipt.acked,
                    attempts=receipt.attempts,
                    error=receipt.error,
                    increment=self.increment,
                )
            else:
                # Paged event with NO delivery accounting at all -> the
                # pipeline lost track of a page: MISSED must-fire (top sev).
                lifecycle.record_missed(
                    ev.rule_id,
                    state=ev.state.value,
                    severity=ev.severity,
                    detail=f"no delivery receipt for paged event key={key}",
                    increment=self.increment,
                )

    @staticmethod
    def _dedup_key(event: AlertEvent) -> str:
        # MUST match the dispatcher's key exactly, or delivery reconciliation
        # would misclassify delivered pages as MISSED — one shared derivation.
        from .anomaly_notifier import event_dedup_key

        return event_dedup_key(event)

    def _masked_target(self) -> str:
        if self.dispatcher is not None and self.dispatcher.notifier is not None:
            try:
                return self.dispatcher.notifier.masked_target()
            except Exception:  # noqa: BLE001
                return "****"
        return "****"

    # ------------------------------------------------------------------
    def acknowledge(self, rule_id: str, *, by: str) -> None:
        """Record a human ack of a firing alert (durable + lifecycle audit)."""
        self.store.acknowledge(rule_id, by=by)
        lifecycle.record_ack(rule_id, by=by, increment=self.increment)

    # ------------------------------------------------------------------
    def status(self) -> dict:
        """A compact status object for the /health surface + break-glass."""
        wd = self.watchdog()
        return {
            "increment": self.increment,
            "promoted": self.promoted,
            "rules": [r.id for r in self.rules],
            "heartbeat": {
                "tick_count": self.heartbeat.tick_count,
                "error_count": self.heartbeat.error_count,
                "last_tick_wall": self.heartbeat.last_tick_wall,
                "consecutive_degraded": self.heartbeat.consecutive_degraded,
                "last_error": self.heartbeat.last_error,
            },
            "watchdog": {
                "healthy": wd.healthy,
                "stalled": wd.stalled,
                "degraded": wd.degraded,
                "seconds_since_last_tick": wd.seconds_since_last_tick,
                "reason": wd.reason,
            },
            "store_load_error": self.store.load_error,
            "non_diagnostic": True,
        }
