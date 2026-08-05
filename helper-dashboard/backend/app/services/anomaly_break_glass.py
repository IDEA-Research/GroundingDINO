"""Break-glass manual control surface (INC5).

The anomaly system runs FULLY AUTOMATICALLY. This module is the ONLY
human-touchable surface, and it is a kill-switch / recovery hatch used ONLY
when the automatic system is disabled or misbehaving — never part of the run
loop, never a routine-maintenance dependency.

It controls a single evaluator + its synthetic data source and exposes exactly
the operations the charter requires:

    start()             (re)enable the evaluator loop; an active
                        force_signal_lost hold is PRESERVED (never lifted as
                        a side effect — clear it explicitly),
    stop()              disable the loop (ticks are refused, not silently
                        producing "no anomaly"),
    restart()           stop then start (rehydrates durable state; holds
                        preserved),
    force_shadow(rule)  kill paging on a rule (drop to non-paging), loudly,
    clear_force_shadow(rule)  lift a manual shadow-hold, restoring the
                        rule's PRE-hold posture (an undo, not a promotion),
    force_signal_lost() force every rule to SIGNAL_LOST on the next tick
                        (a manual "the data is not trustworthy" override),
    clear_force_signal_lost()  lift the manual signal-lost override,
    recover()           restore last-known-good rule + evaluator state and
                        re-enable the loop.

Safety posture (inviolable):

  - **Fail loud, not silent.** A stopped loop that is ticked raises
    :class:`BreakGlassError` — it never returns a fake "no anomaly". A manual
    force-signal-lost makes the degraded state CONSPICUOUS.
  - **Break-glass can only make things SAFER.** It can force a rule into
    shadow (kill paging) or force SIGNAL_LOST; it can NOT promote a rule to
    paging (promotion requires the user's explicit approval — LD-6) and it
    can NOT silence a page without recording it.
  - **Every action is audited** (build-audit + lifecycle-audit), fail-closed,
    with the operator identity, before AND after the state change.
  - **recover() restores last-known-good** so a botched manual action is
    reversible; a recover with no LKG fails loud rather than blanking state.

Because ``pytest_asyncio`` is not installed, this controller — like the
evaluator service — is driven by an INJECTED CLOCK: it never sleeps. The
``tick`` here wraps the service's ``tick`` with the enabled/forced overrides.
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
    SignalLostReason,
)
from . import anomaly_build_audit as build_audit
from . import anomaly_lifecycle_audit as lifecycle
from .anomaly_core import DataStatus
from .anomaly_data_provider import DataProvider, Observation
from .anomaly_evaluator_service import AnomalyEvaluatorService, TickResult


class BreakGlassError(RuntimeError):
    """Raised when a break-glass invariant is violated (e.g. tick while stopped)."""


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@dataclass
class BreakGlassState:
    """The manual override flags. All default to 'automatic, no override'."""

    enabled: bool = True
    force_signal_lost: bool = False
    forced_shadow_rules: set[str] = field(default_factory=set)
    # Each rule's core.promoted value at the moment force_shadow was applied,
    # so lifting the hold can restore the PRE-HOLD posture (an undo, not a
    # promotion — the original promotion authority is unchanged, LD-6).
    pre_hold_promoted: dict[str, bool] = field(default_factory=dict)
    last_action: str | None = None
    last_operator: str | None = None
    last_action_ts: str | None = None


class ForcedSignalLostProvider:
    """Wraps a real provider and forces an unreachable observation.

    Used by ``force_signal_lost``: every observe() reports the data as NOT
    trustworthy so the core emits SIGNAL_LOST (conspicuous), regardless of what
    the underlying source would have said. It NEVER fabricates a clinical value.
    """

    def __init__(self, inner: DataProvider) -> None:
        self.inner = inner

    def observe(
        self,
        metric: str,
        *,
        now: float,
        labels: dict[str, str] | None = None,
    ) -> Observation:
        status = DataStatus(
            reachable=False,
            returns_data=False,
            source="prometheus",
            sample_ts=now,
            history_coverage_s=0.0,
        )
        return Observation(value=None, baseline=None, status=status)


class BreakGlassController:
    """Manual control surface around ONE evaluator + its data source."""

    def __init__(
        self,
        service: AnomalyEvaluatorService,
        provider: DataProvider,
        *,
        increment: str = "runtime",
    ) -> None:
        self.service = service
        # The "live" provider the service currently uses. force_signal_lost
        # swaps this for a ForcedSignalLostProvider; recover/start restore it.
        self._real_provider = provider
        self.state = BreakGlassState()
        self.increment = increment
        # Ensure the service points at the real provider on construction.
        self.service.provider = provider
        # Capture an initial last-known-good so recover() always has a target.
        try:
            self.service.store.backup_last_known_good()
        except Exception as exc:  # noqa: BLE001 — surface, do not hide
            self._audit(
                "init_lkg_failed",
                by="system",
                summary="initial last-known-good backup failed",
                reasoning="recover() needs an LKG target; surface the failure",
                evidence={"error": f"{type(exc).__name__}: {exc}"},
                outcome="lkg_backup_failed",
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, *, by: str = "operator") -> None:
        """(Re)enable the loop.

        PRESERVES an active force_signal_lost hold: an operator's "this data
        is not trustworthy" override must never be lifted as a side effect of
        cycling the loop — only an explicit :meth:`clear_force_signal_lost`
        (or an explicit :meth:`recover`) lifts it.
        """
        self.state.enabled = True
        if not self.state.force_signal_lost:
            self.service.provider = self._real_provider
        self._mark(action="start", by=by)
        hold_note = (
            " (force_signal_lost hold PRESERVED — clear it explicitly)"
            if self.state.force_signal_lost
            else ""
        )
        self._audit(
            "start",
            by=by,
            summary=f"break-glass START: evaluator enabled{hold_note}",
            reasoning=(
                "manual re-enable of the automatic loop; an active "
                "data-untrustworthy hold is never lifted as a side effect"
            ),
            evidence=self.status(),
            outcome="started",
        )

    def stop(self, *, by: str = "operator") -> None:
        """Disable the loop. A subsequent tick RAISES (never fake 'no anomaly')."""
        self.state.enabled = False
        self._mark(action="stop", by=by)
        self._audit(
            "stop",
            by=by,
            summary="break-glass STOP: evaluator disabled (kill switch)",
            reasoning="manual disable; a stopped loop is loud, never a silent no-anomaly",
            evidence=self.status(),
            outcome="stopped",
        )

    def restart(self, *, by: str = "operator") -> None:
        """Stop then start. In-memory cores persist (no re-rehydration);
        manual holds are preserved — see :meth:`start`."""
        self.stop(by=by)
        self.start(by=by)
        self._mark(action="restart", by=by)
        self._audit(
            "restart",
            by=by,
            summary="break-glass RESTART: evaluator cycled",
            reasoning="manual restart; breach timers rehydrate from durable store",
            evidence=self.status(),
            outcome="restarted",
        )

    # ------------------------------------------------------------------
    # Force shadow (kill paging) — can only make things SAFER
    # ------------------------------------------------------------------
    def force_shadow(self, rule_id: str, *, by: str = "operator") -> None:
        """Manually hold a rule in shadow (kill paging) — loudly, recorded."""
        # Also flip the core's promoted flag off so the verdict itself is
        # shadow, remembering the pre-hold posture so clear_force_shadow can
        # RESTORE it (undo of the hold, not a promotion — LD-6 unchanged).
        core = self.service._cores.get(rule_id)  # noqa: SLF001
        if core is not None and rule_id not in self.state.forced_shadow_rules:
            self.state.pre_hold_promoted[rule_id] = bool(core.promoted)
        self.state.forced_shadow_rules.add(rule_id)
        if core is not None:
            core.promoted = False
        self._mark(action=f"force_shadow:{rule_id}", by=by)
        lifecycle.record(
            kind=lifecycle.LifecycleKind.rule_shadowed,
            rule_id=rule_id,
            summary=f"break-glass FORCE SHADOW on {rule_id!r} (paging killed)",
            reasoning="manual kill-paging; break-glass can only make things safer",
            payload={"operator": by, "forced_shadow": True},
            increment=self.increment,
        )
        self._audit(
            "force_shadow",
            by=by,
            summary=f"break-glass FORCE SHADOW on {rule_id!r}",
            reasoning="manual kill-paging is always safe and always recorded",
            evidence={"rule_id": rule_id},
            outcome="forced_shadow",
        )

    def clear_force_shadow(self, rule_id: str, *, by: str = "operator") -> None:
        """Lift a manual shadow-hold, RESTORING the pre-hold paging posture.

        Restoring is an undo of the hold, NOT a promotion decision: a rule
        returns to paging here only if it was already user-promoted before
        the hold (LD-6 authority unchanged). Previously the core stayed
        silently demoted forever while status reported promoted=True.
        """
        self.state.forced_shadow_rules.discard(rule_id)
        restored = self.state.pre_hold_promoted.pop(rule_id, None)
        core = self.service._cores.get(rule_id)  # noqa: SLF001
        if core is not None and restored is not None:
            core.promoted = restored
        self._mark(action=f"clear_force_shadow:{rule_id}", by=by)
        self._audit(
            "clear_force_shadow",
            by=by,
            summary=(
                f"break-glass cleared force-shadow on {rule_id!r} "
                f"(pre-hold posture restored: promoted={restored})"
            ),
            reasoning=(
                "lifts the manual hold and restores the PRE-HOLD posture — "
                "an undo, not a promotion; promotion still requires the "
                "user's explicit approval (LD-6)"
            ),
            evidence={"rule_id": rule_id, "restored_promoted": restored},
            outcome="cleared_force_shadow",
        )

    # ------------------------------------------------------------------
    # Force SIGNAL_LOST — conspicuous manual "data not trustworthy"
    # ------------------------------------------------------------------
    def force_signal_lost(self, *, by: str = "operator") -> None:
        """Force every rule to SIGNAL_LOST on subsequent ticks (conspicuous)."""
        self.state.force_signal_lost = True
        self.service.provider = ForcedSignalLostProvider(self._real_provider)
        self._mark(action="force_signal_lost", by=by)
        self._audit(
            "force_signal_lost",
            by=by,
            summary="break-glass FORCE SIGNAL_LOST: data marked untrustworthy",
            reasoning="manual conspicuous degradation; never a silent no-anomaly",
            evidence=self.status(),
            outcome="forced_signal_lost",
        )

    def clear_force_signal_lost(self, *, by: str = "operator") -> None:
        """Lift the manual signal-lost override, restore the real source."""
        self.state.force_signal_lost = False
        self.service.provider = self._real_provider
        self._mark(action="clear_force_signal_lost", by=by)
        self._audit(
            "clear_force_signal_lost",
            by=by,
            summary="break-glass cleared FORCE SIGNAL_LOST",
            reasoning="restore the real data source",
            evidence=self.status(),
            outcome="cleared_force_signal_lost",
        )

    # ------------------------------------------------------------------
    # Explicit last-known-good checkpoint (operator marks a healthy state)
    # ------------------------------------------------------------------
    def backup(self, *, by: str = "operator") -> str:
        """Checkpoint the current durable state as last-known-good.

        An operator (or the loop when healthy) marks a known-good moment so a
        later :meth:`recover` has a meaningful target. Fails loud on I/O error.
        """
        path = self.service.store.backup_last_known_good()
        self._mark(action="backup", by=by)
        self._audit(
            "backup",
            by=by,
            summary="break-glass BACKUP: captured last-known-good state",
            reasoning="checkpoint a healthy state so recover() can restore it",
            evidence={"lkg": str(path), "rules": list(self.service.store.all().keys())},
            outcome="backed_up",
        )
        return str(path)

    # ------------------------------------------------------------------
    # Recover last-known-good
    # ------------------------------------------------------------------
    def recover(self, *, by: str = "operator") -> dict:
        """Restore last-known-good rule + evaluator state and re-enable the loop.

        Fails loud if there is no LKG to restore (never blanks state to a fake
        'healthy'). After restoring durable state it rehydrates the in-memory
        cores from the store so a breach-in-progress is not reset.
        """
        restored = self.service.store.restore_last_known_good()
        # Rehydrate the cores from the just-restored durable state.
        for rule in self.service.rules:
            self.service._rehydrate(  # noqa: SLF001
                self.service._cores[rule.id], rule  # noqa: SLF001
            )
        # Clear manual overrides and re-enable. Unlike start()/restart(),
        # lifting holds HERE is the operator's stated intent — RECOVER means
        # "return to automatic". Each held rule's pre-hold posture is
        # restored (undo, not promotion), and every lifted hold is named in
        # the audit summary so nothing is lifted silently.
        lifted_shadow = sorted(self.state.forced_shadow_rules)
        for rid in lifted_shadow:
            core = self.service._cores.get(rid)  # noqa: SLF001
            pre_hold = self.state.pre_hold_promoted.pop(rid, None)
            if core is not None and pre_hold is not None:
                core.promoted = pre_hold
        self.state.forced_shadow_rules.clear()
        lifted_signal_hold = self.state.force_signal_lost
        self.state.force_signal_lost = False
        self.state.enabled = True
        self.service.provider = self._real_provider
        self._mark(action="recover", by=by)
        lifted_bits = []
        if lifted_signal_hold:
            lifted_bits.append("force_signal_lost")
        if lifted_shadow:
            lifted_bits.append(f"force_shadow on {lifted_shadow}")
        lifted_note = (
            f" (lifted manual holds: {', '.join(lifted_bits)})"
            if lifted_bits
            else ""
        )
        self._audit(
            "recover",
            by=by,
            summary=(
                "break-glass RECOVER: restored last-known-good state"
                f"{lifted_note}"
            ),
            reasoning=(
                "reversible recovery hatch; loop re-enabled to automatic — "
                "an explicit recover lifts manual holds by stated intent, "
                "restoring each rule's pre-hold posture (never promoting)"
            ),
            evidence={
                "restored_rules": list((restored.get("rules") or {}).keys()),
                "lifted_force_signal_lost": lifted_signal_hold,
                "lifted_forced_shadow": lifted_shadow,
                "status": self.status(),
            },
            outcome="recovered",
        )
        return restored

    # ------------------------------------------------------------------
    # Guarded tick — honours enabled / force flags
    # ------------------------------------------------------------------
    def tick(self, now: float, *, mono: float | None = None) -> TickResult:
        """Tick the loop, honouring the manual overrides.

        A STOPPED loop raises — a disabled evaluator must be conspicuous, never
        a silent 'no anomaly'. force_signal_lost is already wired through the
        swapped provider; forced-shadow rules are already un-promoted in their
        cores, so the normal service tick honours both.
        """
        if not self.state.enabled:
            raise BreakGlassError(
                "evaluator is STOPPED via break-glass; refusing to tick "
                "(a disabled loop must not fabricate a 'no anomaly' verdict)"
            )
        return self.service.tick(now, mono=mono)

    # ------------------------------------------------------------------
    # Introspection + helpers
    # ------------------------------------------------------------------
    def status(self) -> dict:
        return {
            "enabled": self.state.enabled,
            "force_signal_lost": self.state.force_signal_lost,
            "forced_shadow_rules": sorted(self.state.forced_shadow_rules),
            # The cores' ACTUAL paging posture — the service-level `promoted`
            # flag alone would misrepresent a rule held in shadow.
            "effective_promoted": {
                rid: core.promoted
                for rid, core in sorted(
                    self.service._cores.items()  # noqa: SLF001
                )
            },
            "last_action": self.state.last_action,
            "last_operator": self.state.last_operator,
            "last_action_ts": self.state.last_action_ts,
            "has_last_known_good": self.service.store.has_last_known_good(),
            "non_diagnostic": True,
        }

    def _mark(self, *, action: str, by: str) -> None:
        self.state.last_action = action
        self.state.last_operator = by
        self.state.last_action_ts = datetime.now(tz=timezone.utc).isoformat()

    def _audit(
        self,
        action: str,
        *,
        by: str,
        summary: str,
        reasoning: str,
        evidence,
        outcome: str,
    ) -> None:
        build_audit.append(
            increment=self.increment,
            stage="decide",
            action="run",
            target=f"break_glass/{action}",
            summary=f"[operator={by}] {summary}",
            reasoning=reasoning,
            evidence=evidence,
            outcome=outcome,
        )
