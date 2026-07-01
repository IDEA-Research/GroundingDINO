"""Alert-lifecycle audit — the clinical evidence trail (INC5).

The build-audit (`anomaly_build_audit`) records how the AGENT thought and
edited. This module records the CLINICAL lifecycle of an alert rule so a human
can later reconstruct, end-to-end:

    - rule ACTIVATION / EDIT / PROMOTION (a rule entering shadow or active),
    - every STATE TRANSITION the evaluator produced (pending / firing /
      resolved / signal_lost / insufficient_baseline),
    - DELIVERY of a page to a channel (masked target, acked?), and a
      would-fire that was recorded in shadow but never paged,
    - ACK of a firing alert by a human,
    - SUPPRESSED / WITHHELD events (every dedup/shadow/refusal is logged — no
      silent suppression is ever allowed), and
    - a MISSED must-fire — the top-severity clinical-safety event: a firing
      the system should have paged but did not deliver.

Fail-closed posture (inviolable): a write failure on a CLINICAL lifecycle
event (firing / signal_lost / delivered / missed / suppressed-critical) RAISES
via :class:`LifecycleAuditError`. Losing the record that an alert fired — or
that a page was withheld — is worse than crashing loudly. This mirrors the
`AlertStateStore` and `anomaly_build_audit` posture.

The records go to a dedicated JSONL stream so the clinical trail is not diluted
by build-loop noise:

    backend/app/storage/anomaly_lifecycle_audit/<UTC-date>.jsonl

and, so the two views stay reconcilable, a mirror line is ALSO appended to the
build-audit (which itself fails closed). If EITHER sink fails on a clinical
event, the caller sees the failure.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from ..specs.anomaly_evaluation_report import AlertEvent, AlertState
from . import anomaly_build_audit as build_audit

_LIFECYCLE_DIR = (
    Path(__file__).resolve().parent.parent / "storage" / "anomaly_lifecycle_audit"
)


class LifecycleAuditError(RuntimeError):
    """Raised when a clinical lifecycle event cannot be durably audited."""


class LifecycleKind(str, Enum):
    """The lifecycle events we record. `missed` is the top-severity failure."""

    rule_activated = "rule_activated"
    rule_edited = "rule_edited"
    rule_promoted = "rule_promoted"
    rule_shadowed = "rule_shadowed"
    transition = "transition"
    delivered = "delivered"
    would_fire = "would_fire"
    acked = "acked"
    suppressed = "suppressed"
    missed = "missed"


# Kinds whose LOSS must fail loud (a swallowed write here is a safety hazard).
_CLINICAL_KINDS = frozenset(
    {
        LifecycleKind.delivered,
        LifecycleKind.missed,
        LifecycleKind.rule_activated,
        LifecycleKind.rule_promoted,
    }
)

# Transition states that are themselves clinical (their loss must fail loud).
_CLINICAL_TRANSITION_STATES = frozenset(
    {AlertState.firing, AlertState.signal_lost}
)


def _lifecycle_dir() -> Path:
    d = Path(os.getenv("ANOMALY_LIFECYCLE_AUDIT_DIR", str(_LIFECYCLE_DIR)))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _is_clinical(kind: LifecycleKind, payload: dict[str, Any]) -> bool:
    if kind in _CLINICAL_KINDS:
        return True
    if kind == LifecycleKind.transition:
        state = payload.get("state")
        if state in {s.value for s in _CLINICAL_TRANSITION_STATES}:
            return True
    if kind == LifecycleKind.suppressed and payload.get("severity") == "critical":
        # Suppressing a critical alert is a safety-relevant act; its record
        # must not be lost.
        return True
    return False


def record(
    *,
    kind: LifecycleKind,
    rule_id: str,
    summary: str,
    reasoning: str,
    payload: dict[str, Any] | None = None,
    increment: str = "INC5",
) -> None:
    """Append one lifecycle record to the clinical stream AND the build-audit.

    A write failure on a clinical event raises :class:`LifecycleAuditError`.
    A non-clinical event still surfaces its failure (a broken audit sink is a
    defect), but only the clinical ones are singled out as the top-severity
    "we lost a clinical record" case.
    """
    payload = dict(payload or {})
    record_obj = {
        "ts": _now_iso(),
        "increment": increment,
        "kind": kind.value,
        "rule_id": rule_id,
        "summary": summary,
        "reasoning": reasoning,
        "non_diagnostic": True,
        "payload": payload,
    }
    line = json.dumps(record_obj, ensure_ascii=False, default=str)
    clinical = _is_clinical(kind, payload)

    # 1) Dedicated clinical stream (fail-closed).
    try:
        path = _lifecycle_dir() / f"{datetime.now(tz=timezone.utc):%Y-%m-%d}.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception as exc:  # noqa: BLE001
        msg = (
            f"lifecycle-audit write FAILED for rule {rule_id!r} kind={kind.value}: "
            f"{type(exc).__name__}: {exc}"
        )
        raise LifecycleAuditError(msg) from exc

    # 2) Mirror into the build-audit so the two views reconcile. The build
    #    audit itself fails closed; a clinical event that cannot be mirrored
    #    must also surface.
    try:
        build_audit.append(
            increment=increment,
            stage="observe" if kind != LifecycleKind.missed else "decide",
            action="run" if kind != LifecycleKind.missed else "decide",
            target=f"lifecycle/{rule_id}",
            summary=summary,
            reasoning=reasoning,
            evidence={"kind": kind.value, **payload},
            outcome=kind.value,
        )
    except Exception as exc:  # noqa: BLE001
        if clinical:
            raise LifecycleAuditError(
                f"lifecycle build-audit mirror FAILED for a clinical event "
                f"({kind.value}, rule {rule_id!r}): {type(exc).__name__}: {exc}"
            ) from exc
        raise


# ---------------------------------------------------------------------------
# Convenience recorders for the common lifecycle events.
# ---------------------------------------------------------------------------
def record_rule_activated(rule, *, promoted: bool, increment: str = "INC5") -> None:
    """A rule entering the running set (shadow by default)."""
    record(
        kind=LifecycleKind.rule_activated,
        rule_id=rule.id,
        summary=(
            f"rule {rule.id!r} activated in "
            f"{'ACTIVE (paging)' if promoted else 'SHADOW (non-paging)'} mode"
        ),
        reasoning=(
            "shadow by default; promotion to paging is a separate, "
            "supervisor-gated act on green goldens"
        ),
        payload={
            "metric": rule.metric,
            "ratio": rule.ratio,
            "for": rule.for_,
            "window": rule.baseline.window,
            "severity": rule.severity.value,
            "declared_mode": rule.mode.value,
            "promoted": bool(promoted),
        },
        increment=increment,
    )


def record_transition(event: AlertEvent, *, mode: str, increment: str = "INC5") -> None:
    """One evaluator state transition — every one is captured, none silent."""
    record(
        kind=LifecycleKind.transition,
        rule_id=event.rule_id,
        summary=f"{event.state.value}: {event.message}",
        reasoning=(
            "shadow would-fire recorded, never paged"
            if event.would_page and not event.paged
            else "lifecycle transition persisted"
        ),
        payload={
            "state": event.state.value,
            "value": event.value,
            "baseline": event.baseline,
            "threshold": event.threshold,
            "severity": event.severity,
            "would_page": event.would_page,
            "paged": event.paged,
            "suppressed_reason": event.suppressed_reason,
            "signal_lost_reason": (
                event.signal_lost_reason.value if event.signal_lost_reason else None
            ),
            "mode": mode,
            "ts": event.ts,
        },
        increment=increment,
    )


def record_delivery(
    event: AlertEvent,
    *,
    channel: str,
    masked_target: str,
    delivered: bool,
    acked: bool,
    attempts: int,
    error: str | None,
    increment: str = "INC5",
) -> None:
    """A page delivered (or attempted) to a channel.

    A delivered-but-un-acked page is NOT a success; it is surfaced. A page that
    was expected but never delivered is a MISSED must-fire (recorded separately
    via :func:`record_missed`).
    """
    record(
        kind=LifecycleKind.delivered,
        rule_id=event.rule_id,
        summary=(
            f"delivered {event.state.value} to {channel} "
            f"(delivered={delivered}, acked={acked})"
        ),
        reasoning=(
            "TEST TUNNEL only, NOT a clinician notification; delivery attempt "
            "recorded with masked target; at-least-once + ack"
        ),
        payload={
            "state": event.state.value,
            "severity": event.severity,
            "channel": channel,
            "masked_target": masked_target,
            "delivered": delivered,
            "acked": acked,
            "attempts": attempts,
            "error": error,
        },
        increment=increment,
    )


def record_would_fire(event: AlertEvent, *, increment: str = "INC5") -> None:
    """A firing/signal_lost that would have paged but is in shadow."""
    record(
        kind=LifecycleKind.would_fire,
        rule_id=event.rule_id,
        summary=f"SHADOW would-fire recorded for {event.state.value} (not paged)",
        reasoning="rule not promoted; would-fire visible, never silent",
        payload={
            "state": event.state.value,
            "severity": event.severity,
            "would_page": event.would_page,
            "paged": event.paged,
        },
        increment=increment,
    )


def record_ack(rule_id: str, *, by: str, increment: str = "INC5") -> None:
    """A human acknowledged a firing alert (never silences a re-fire)."""
    record(
        kind=LifecycleKind.acked,
        rule_id=rule_id,
        summary=f"firing alert for {rule_id!r} acknowledged by {by!r}",
        reasoning="ack records human awareness only; a new firing edge re-arms",
        payload={"acked_by": by},
        increment=increment,
    )


def record_suppressed(
    rule_id: str,
    *,
    state: str,
    severity: str,
    reason: str,
    increment: str = "INC5",
) -> None:
    """A deduped / shadow / withheld page — ALWAYS logged, never silent."""
    record(
        kind=LifecycleKind.suppressed,
        rule_id=rule_id,
        summary=f"{state} for {rule_id!r} withheld ({reason})",
        reasoning="no silent suppression: every withheld page is recorded",
        payload={"state": state, "severity": severity, "reason": reason},
        increment=increment,
    )


def record_missed(
    rule_id: str,
    *,
    state: str,
    severity: str,
    detail: str,
    increment: str = "INC5",
) -> None:
    """TOP-SEVERITY: a page that should have been delivered was not.

    This is the worst clinical-safety outcome. It always fails closed — if it
    cannot be recorded, the caller crashes rather than silently swallowing a
    missed page.
    """
    record(
        kind=LifecycleKind.missed,
        rule_id=rule_id,
        summary=f"MISSED must-fire for {rule_id!r}: {state} not delivered",
        reasoning=(
            "a missed real desaturation is the worst outcome; recorded as the "
            "top-severity clinical-safety event"
        ),
        payload={"state": state, "severity": severity, "detail": detail},
        increment=increment,
    )


def read_records(day: str | None = None) -> list[dict[str, Any]]:
    """Read back the lifecycle stream for a day (default: today) — for tests."""
    d = _lifecycle_dir()
    day = day or f"{datetime.now(tz=timezone.utc):%Y-%m-%d}"
    path = d / f"{day}.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out
