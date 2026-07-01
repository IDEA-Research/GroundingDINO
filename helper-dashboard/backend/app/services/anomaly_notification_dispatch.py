"""NotificationDispatcher — wire firing transitions to the Notifier (INC3).

This is the POLICY layer between the evaluator and the channel. It consumes
the `AlertEvent`s an `AnomalyEvaluationReport` emits and decides, per event,
whether to deliver an out-of-band notification. It NEVER re-derives a clinical
verdict — the core already did that. Its job is delivery discipline:

Rules it enforces (all inviolable):

- **Shadow records, never pages.** A firing event whose ``paged`` flag is
  False (the rule is not promoted) is recorded as a would-fire only — no
  delivery is attempted. Promotion out of shadow is what flips ``paged`` in
  the core; the dispatcher trusts that single source of truth and additionally
  refuses to deliver anything the core marked as suppressed/shadow.
- **Page once per firing (dedup + hysteresis).** A firing EDGE delivers once;
  a flapping value that re-crosses the threshold does not spam. Dedup is keyed
  on the event's ``dedup_key`` (rule + state + transition ts). A resolved
  event clears the firing dedup key so the NEXT genuine firing pages again.
- **SIGNAL_LOST raises its own distinct page.** Loss of signal is a separate,
  conspicuous notification (its own dedup key + banner), never folded into a
  clinical firing and never silenced.
- **A display/severity filter can NEVER silence a firing.** There is no
  severity threshold that drops a page. Attempting to suppress a *critical*
  alert is refused outright (:class:`SuppressionRefused`) and audited.
- **At-least-once + ack, every attempt audited (masked).** Delivery goes
  through the `Notifier`; the receipt (delivered/acked/attempts) is audited
  with the channel's MASKED target — never the raw webhook token.

The dispatcher keeps only a small in-memory set of already-delivered dedup
keys for the current process; durable page accounting lives in the
`AlertStateStore` (paged_count / would_page_count), which the evaluator
service already maintains.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..specs.anomaly_evaluation_report import (
    AlertEvent,
    AlertState,
    AnomalyEvaluationReport,
)
from . import anomaly_build_audit as audit
from .anomaly_notifier import (
    DeliveryReceipt,
    NotificationMessage,
    Notifier,
)

# States that, when they FIRE as a transition edge, warrant an out-of-band
# page for a promoted rule. SIGNAL_LOST is its own distinct page.
_PAGEABLE_STATES = frozenset({AlertState.firing, AlertState.signal_lost})


class SuppressionRefused(RuntimeError):
    """Raised when something tries to silence a critical/firing page.

    A display or severity filter may never drop a page. This is the loud
    failure that enforces "no silent suppression".
    """


@dataclass
class DispatchOutcome:
    """What the dispatcher did with ONE report's events."""

    delivered: list[DeliveryReceipt] = field(default_factory=list)
    would_fire_only: list[str] = field(default_factory=list)  # dedup keys (shadow)
    deduped: list[str] = field(default_factory=list)  # already delivered
    refused: list[str] = field(default_factory=list)  # suppression refused

    @property
    def delivery_count(self) -> int:
        return len(self.delivered)


class NotificationDispatcher:
    """Turns evaluator events into at-most-one delivery per firing edge."""

    def __init__(
        self,
        notifier: Notifier,
        *,
        increment: str = "INC3",
    ) -> None:
        self.notifier = notifier
        self.increment = increment
        # dedup keys we have already delivered in this process lifetime.
        self._delivered_keys: set[str] = set()
        # Track the last firing dedup key per rule so a resolve can clear it,
        # re-arming the NEXT genuine firing (hysteresis).
        self._active_firing_key: dict[str, str] = {}

    # ------------------------------------------------------------------
    def dispatch_report(self, report: AnomalyEvaluationReport) -> DispatchOutcome:
        """Process every event in a report; deliver at most once per edge."""
        outcome = DispatchOutcome()
        for event in report.events:
            self._dispatch_event(event, outcome)
        return outcome

    # ------------------------------------------------------------------
    def _dispatch_event(self, event: AlertEvent, outcome: DispatchOutcome) -> None:
        # A resolved event re-arms the rule: clear its firing dedup key so a
        # future genuine firing pages again (this is the hysteresis boundary).
        if event.state == AlertState.resolved:
            key = self._active_firing_key.pop(event.rule_id, None)
            if key is not None:
                self._delivered_keys.discard(key)
            return

        if event.state not in _PAGEABLE_STATES:
            # pending / insufficient_baseline never page.
            return

        message = NotificationMessage.from_event(event)

        # ---- SHADOW: record would-fire only, NEVER deliver ----------------
        # The core marks a non-promoted firing with paged=False and a
        # suppressed_reason of "shadow_mode". We trust that single source of
        # truth: if the event did not actually page, we do not deliver.
        if not event.paged:
            # Defence in depth: shadow suppression of a CRITICAL alert is
            # allowed ONLY because it is shadow-by-default policy, and it is
            # ALWAYS visible (would_page recorded). It is never silent.
            if event.would_page:
                outcome.would_fire_only.append(message.dedup_key)
                self._audit_would_fire(event, message)
            return

        # ---- ACTIVE (promoted): this event actually pages -----------------
        # No display/severity filter may drop it. If anything upstream tried
        # to mark a firing/critical event as suppressed while STILL paging,
        # that is an inconsistent, unsafe state — refuse loudly.
        if event.suppressed_reason:
            outcome.refused.append(message.dedup_key)
            self._audit_suppression_refused(event, message)
            raise SuppressionRefused(
                f"refusing to suppress a paging {event.severity} "
                f"{event.state.value} for rule {event.rule_id!r}: "
                f"suppressed_reason={event.suppressed_reason!r}"
            )

        # ---- DEDUP / HYSTERESIS: page once per firing edge ---------------
        if message.dedup_key in self._delivered_keys:
            outcome.deduped.append(message.dedup_key)
            self._audit_deduped(event, message)
            return

        # ---- DELIVER (at-least-once + ack) -------------------------------
        receipt = self.notifier.deliver(message)
        outcome.delivered.append(receipt)
        if event.state == AlertState.firing:
            self._active_firing_key[event.rule_id] = message.dedup_key
        # Mark delivered even if not acked so a within-process retry is not
        # duplicated; a NON-acked receipt is surfaced in the audit as a
        # failed delivery for the operator to act on (at-least-once means we
        # tried; ack tells us whether it landed).
        self._delivered_keys.add(message.dedup_key)
        self._audit_delivery(event, message, receipt)

    # ------------------------------------------------------------------
    # Public helper: an explicit attempt to suppress a critical alert is
    # ALWAYS refused. This is the "display filter can never silence a page"
    # guard exposed as an API so callers cannot route around it.
    # ------------------------------------------------------------------
    def refuse_suppression(self, event: AlertEvent) -> None:
        if event.severity == "critical" or event.state in _PAGEABLE_STATES:
            self._audit_suppression_refused(event, NotificationMessage.from_event(event))
            raise SuppressionRefused(
                f"suppression of a {event.severity} {event.state.value} alert "
                f"for rule {event.rule_id!r} is forbidden"
            )

    # ------------------------------------------------------------------
    # Audit helpers — every attempt masked, never the raw webhook token.
    # ------------------------------------------------------------------
    def _audit_delivery(
        self, event: AlertEvent, message: NotificationMessage, receipt: DeliveryReceipt
    ) -> None:
        audit.append(
            increment=self.increment,
            stage="observe",
            action="run",
            target=f"notify/{event.rule_id}",
            summary=(
                f"delivered {event.state.value} to {self.notifier.channel_name} "
                f"(acked={receipt.acked})"
            ),
            reasoning=(
                "promoted rule paged once per firing edge; TEST TUNNEL only, "
                "NOT a clinician notification; delivery attempt recorded (masked)"
            ),
            evidence={
                "dedup_key": message.dedup_key,
                "state": event.state.value,
                "severity": event.severity,
                "channel": receipt.channel,
                "masked_target": self.notifier.masked_target(),
                "delivered": receipt.delivered,
                "acked": receipt.acked,
                "attempts": receipt.attempts,
                "error": receipt.error,
                "banner": message.banner,
                "non_diagnostic": True,
            },
            outcome="delivered" if receipt.ok else "delivery_unacked",
        )

    def _audit_would_fire(self, event: AlertEvent, message: NotificationMessage) -> None:
        audit.append(
            increment=self.increment,
            stage="observe",
            action="run",
            target=f"notify/{event.rule_id}",
            summary=f"SHADOW would-fire recorded for {event.state.value} (not paged)",
            reasoning="rule not promoted; record would-fire, never deliver; visible not silent",
            evidence={
                "dedup_key": message.dedup_key,
                "state": event.state.value,
                "severity": event.severity,
                "would_page": event.would_page,
                "paged": event.paged,
                "suppressed_reason": event.suppressed_reason,
                "non_diagnostic": True,
            },
            outcome="would_fire_shadow",
        )

    def _audit_deduped(self, event: AlertEvent, message: NotificationMessage) -> None:
        audit.append(
            increment=self.increment,
            stage="observe",
            action="run",
            target=f"notify/{event.rule_id}",
            summary=f"deduped {event.state.value} (already paged this firing edge)",
            reasoning="page once per firing; a flap must not spam; suppression is logged not silent",
            evidence={
                "dedup_key": message.dedup_key,
                "state": event.state.value,
                "severity": event.severity,
            },
            outcome="deduped",
        )

    def _audit_suppression_refused(
        self, event: AlertEvent, message: NotificationMessage
    ) -> None:
        audit.append(
            increment=self.increment,
            stage="decide",
            action="decide",
            target=f"notify/{event.rule_id}",
            summary=f"REFUSED suppression of a {event.severity} {event.state.value}",
            reasoning="no silent suppression: a display/severity filter can never silence a page",
            evidence={
                "dedup_key": message.dedup_key,
                "state": event.state.value,
                "severity": event.severity,
                "suppressed_reason": event.suppressed_reason,
            },
            outcome="suppression_refused",
        )
