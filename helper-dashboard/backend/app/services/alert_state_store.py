"""Durable, fail-closed alert-state + lifecycle store.

The pure `AnomalyEvaluatorCore` (INC1) holds its breach timer only in memory:
if the process restarts, that timer is lost and a breach in progress could
silently "restart" its 5m clock. That is a clinical hazard. This store makes
the lifecycle DURABLE across ticks and across process restarts:

    - `breach_start` per rule (when the current sustained breach began),
    - the last known `AlertState` per rule,
    - an append-only lifecycle history (pending / firing / resolved /
      signal_lost / insufficient_baseline / acked),
    - ack accounting (who/when a firing alert was acknowledged).

Fail-closed posture (mirrors `anomaly_build_audit` and `extend_audit`):

    - Writes are atomic (write-temp + os.replace) and fsync'd, so a crash
      mid-write cannot leave a torn record that reads back as "no breach".
    - A write failure on a CLINICAL event (firing / signal_lost) RAISES —
      it is never swallowed. Losing the record that an alert fired is worse
      than crashing loudly.
    - On load, an unreadable / corrupt state file is treated as UNKNOWN and
      surfaced (not silently reset to "healthy"), so recovery is deliberate.

The store is deliberately a plain JSON snapshot + a JSONL history sidecar,
not a database — no new dependency, easy for a human to inspect during a
break-glass recovery, and trivially last-known-good restorable (INC5).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..specs.anomaly_evaluation_report import (
    AlertEvent,
    AlertState,
    AnomalyEvaluationReport,
)

# Clinical states whose LOSS must fail loud, not be swallowed.
_CLINICAL_STATES = frozenset(
    {AlertState.firing, AlertState.signal_lost}
)

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "storage" / "alert_state"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class AlertStateStoreError(RuntimeError):
    """Raised when a clinical state write cannot be durably persisted."""


@dataclass
class RuleState:
    """Durable per-rule lifecycle snapshot.

    `breach_start_ts` is in the evaluator's INJECTED-clock domain (epoch
    seconds), so it survives a restart and the breach timer can be rehydrated
    into the core rather than silently restarting the 5m window.
    """

    rule_id: str
    state: str = AlertState.resolved.value
    breach_start_ts: float | None = None
    last_value: float | None = None
    last_baseline: float | None = None
    last_threshold: float | None = None
    last_tick_ts: float | None = None  # injected-clock ts of last evaluation
    fired: bool = False  # currently in a firing edge (page-once accounting)
    would_page_count: int = 0
    paged_count: int = 0
    signal_lost_reason: str | None = None
    # Ack accounting — a firing alert can be acknowledged; ack never silences
    # a future re-fire, it only records human awareness for the audit/UI.
    acked: bool = False
    acked_by: str | None = None
    acked_ts: str | None = None
    updated_at: str = field(default_factory=_now_iso)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "RuleState":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in data.items() if k in allowed}
        return cls(**clean)


class AlertStateStore:
    """File-backed, fail-closed alert lifecycle store.

    One store instance serves all rules for one evaluator. State is a single
    JSON snapshot (`state.json`) plus an append-only `history.jsonl` sidecar
    that records every emitted `AlertEvent` for the audit / UI timeline.
    """

    def __init__(self, base_dir: str | os.PathLike[str] | None = None) -> None:
        self.base_dir = Path(
            base_dir
            if base_dir is not None
            else os.getenv("ANOMALY_ALERT_STATE_DIR", str(_DEFAULT_DIR))
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_path = self.base_dir / "state.json"
        self._history_path = self.base_dir / "history.jsonl"
        self._states: dict[str, RuleState] = {}
        self._load_error: str | None = None
        self._load()

    # ------------------------------------------------------------------
    # Load / persist
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._snapshot_path.exists():
            self._states = {}
            return
        try:
            raw = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
            states: dict[str, RuleState] = {}
            for rule_id, blob in (raw.get("rules") or {}).items():
                states[rule_id] = RuleState.from_json(blob)
            self._states = states
        except Exception as exc:  # noqa: BLE001 — surface, never silently reset
            # Fail-closed: do NOT silently reset to "healthy". Record the
            # corruption so recovery (break-glass) is a deliberate act.
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._states = {}

    @property
    def load_error(self) -> str | None:
        """Non-None when the on-disk snapshot was unreadable/corrupt."""
        return self._load_error

    def _atomic_write_snapshot(self) -> None:
        payload = {
            "version": 1,
            "updated_at": _now_iso(),
            "rules": {rid: st.to_json() for rid, st in self._states.items()},
        }
        data = json.dumps(payload, ensure_ascii=False, indent=2)
        # Atomic replace: write to a temp file in the same dir then os.replace.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.base_dir), prefix=".state-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self._snapshot_path)
        except Exception:
            # Clean up the temp file; re-raise so the caller fails loud.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _append_history(self, event: AlertEvent) -> None:
        line = json.dumps(
            {"logged_at": _now_iso(), **event.model_dump()},
            ensure_ascii=False,
            default=str,
        )
        with open(self._history_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    def get(self, rule_id: str) -> RuleState:
        return self._states.get(rule_id) or RuleState(rule_id=rule_id)

    def all(self) -> dict[str, RuleState]:
        return dict(self._states)

    def breach_start(self, rule_id: str) -> float | None:
        st = self._states.get(rule_id)
        return st.breach_start_ts if st else None

    # ------------------------------------------------------------------
    # Write API — called by the evaluator service after each tick.
    # ------------------------------------------------------------------
    def record_report(
        self, report: AnomalyEvaluationReport, *, breach_start_ts: float | None
    ) -> None:
        """Persist the durable state implied by one evaluator report.

        `breach_start_ts` is the core's current breach-timer start (or None);
        we store it so a restart can rehydrate the timer instead of resetting
        the 5m clock. Fails loud on any write error for a clinical event.
        """
        rule_id = report.rule_id
        st = self._states.get(rule_id) or RuleState(rule_id=rule_id)

        st.state = report.state.value
        st.breach_start_ts = breach_start_ts
        st.last_value = report.value
        st.last_baseline = report.baseline
        st.last_threshold = report.threshold
        st.last_tick_ts = _tick_epoch(report.ts)
        st.updated_at = _now_iso()

        is_clinical = False
        for ev in report.events:
            if ev.state in _CLINICAL_STATES:
                is_clinical = True
            if ev.would_page:
                st.would_page_count += 1
            if ev.paged:
                st.paged_count += 1
            if ev.state == AlertState.firing:
                st.fired = True
                # A fresh firing edge clears any prior ack — a NEW alert must
                # be re-acknowledged; ack never suppresses a re-fire.
                st.acked = False
                st.acked_by = None
                st.acked_ts = None
            if ev.state == AlertState.resolved:
                st.fired = False
            if ev.state == AlertState.signal_lost:
                st.signal_lost_reason = (
                    ev.signal_lost_reason.value if ev.signal_lost_reason else None
                )
            else:
                st.signal_lost_reason = None

        self._states[rule_id] = st

        # History first (append-only), then the atomic snapshot. Any failure
        # on a clinical event surfaces as AlertStateStoreError.
        try:
            for ev in report.events:
                self._append_history(ev)
            self._atomic_write_snapshot()
        except Exception as exc:  # noqa: BLE001
            msg = (
                f"alert-state persist FAILED for rule {rule_id!r} "
                f"state={report.state.value}: {type(exc).__name__}: {exc}"
            )
            if is_clinical:
                # Losing a clinical record is worse than crashing.
                raise AlertStateStoreError(msg) from exc
            # Non-clinical (pending/insufficient_baseline) still re-raises:
            # a store that cannot write is broken and must be visible.
            raise AlertStateStoreError(msg) from exc

    # ------------------------------------------------------------------
    def acknowledge(self, rule_id: str, *, by: str) -> RuleState:
        """Record a human ack of a firing alert. Never silences a re-fire."""
        st = self._states.get(rule_id) or RuleState(rule_id=rule_id)
        st.acked = True
        st.acked_by = by
        st.acked_ts = _now_iso()
        st.updated_at = _now_iso()
        self._states[rule_id] = st
        self._atomic_write_snapshot()
        return st

    def snapshot_dict(self) -> dict[str, Any]:
        """Return the full durable snapshot (for last-known-good backup)."""
        return {
            "version": 1,
            "updated_at": _now_iso(),
            "load_error": self._load_error,
            "rules": {rid: st.to_json() for rid, st in self._states.items()},
        }


def _tick_epoch(ts_iso: str) -> float | None:
    """Best-effort epoch seconds from an ISO timestamp for last_tick_ts."""
    try:
        return datetime.fromisoformat(ts_iso).timestamp()
    except Exception:
        return None
