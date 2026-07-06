"""System-metric alert rules — store, provider, core, and service.

The NON-CLINICAL sibling of the neonatal anomaly pipeline: absolute-threshold
rules over a curated catalog of host metrics (CPU, disk, memory, load),
authored through Helper chat, evaluated on a wall clock against the real
Prometheus, always in SHADOW.

It deliberately REUSES the clinical building blocks that are rule-agnostic —
`DataStatus`, `AlertEvent`/`AnomalyEvaluationReport`, and a second
`AlertStateStore` instance in its OWN directory — and reimplements only what
differs (absolute-threshold compare instead of the locked relative-baseline
compare). No clinical file is modified.

Safety posture carried over 1:1:

- Data-integrity gate FIRST on every tick: unreachable / no-data /
  source!="prometheus" / stale  =>  SIGNAL_LOST, never a verdict.
  (Source-flag branching copied from `anomaly_data_provider`, per the
  CLAUDE.md rule for new consumers of `prometheus/client.py`.)
- Shadow is structural: the spec cannot express an active rule, the core
  never sets `paged=True`, and suppression is recorded on every event.
- No silent suppression; every event lands in the durable state store's
  append-only history.
- Every surface carries `non_diagnostic=True`.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..prometheus.client import PrometheusClient
from ..specs.anomaly_evaluation_report import (
    AlertEvent,
    AlertState,
    AnomalyEvaluationReport,
    SignalLostReason,
)
from ..specs.system_rule_spec import (
    FRESHNESS_PROBE_METRIC,
    SystemAlertRuleSpec,
)
from .alert_state_store import AlertStateStore
from .anomaly_core import DataStatus

_DEFAULT_RULES_DIR = (
    Path(__file__).resolve().parent.parent / "storage" / "system_rules"
)
_DEFAULT_STATE_DIR = (
    Path(__file__).resolve().parent.parent / "storage" / "system_alert_state"
)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Rule store — one JSON file per rule, spec-validated on load.
# ---------------------------------------------------------------------------


class SystemRuleStore:
    def __init__(self, base_dir: str | os.PathLike[str] | None = None) -> None:
        self.base_dir = Path(
            base_dir
            if base_dir is not None
            else os.getenv("SYSTEM_RULES_DIR", str(_DEFAULT_RULES_DIR))
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.load_errors: list[str] = []

    def _path(self, rule_id: str) -> Path:
        return self.base_dir / f"{rule_id}.json"

    def save(self, rule: SystemAlertRuleSpec) -> Path:
        p = self._path(rule.id)
        p.write_text(
            rule.model_dump_json(by_alias=True, indent=2), encoding="utf-8"
        )
        return p

    def load_all(self) -> list[SystemAlertRuleSpec]:
        """Load every rule; invalid files are surfaced in `load_errors`,
        never silently dropped AND never allowed to evaluate."""
        self.load_errors = []
        rules: list[SystemAlertRuleSpec] = []
        for p in sorted(self.base_dir.glob("*.json")):
            try:
                rules.append(
                    SystemAlertRuleSpec.model_validate_json(
                        p.read_text(encoding="utf-8")
                    )
                )
            except Exception as exc:  # noqa: BLE001 — surfaced, not swallowed
                self.load_errors.append(f"{p.name}: {type(exc).__name__}: {exc}")
        return rules

    def delete(self, rule_id: str) -> bool:
        p = self._path(rule_id)
        if p.exists():
            p.unlink()
            return True
        return False


# ---------------------------------------------------------------------------
# Provider — catalog expression + honest provenance/freshness.
# ---------------------------------------------------------------------------


class SystemMetricProvider:
    """Evaluate a catalog expression and report provenance honestly.

    Freshness: catalog expressions are computed vectors (rate()/avg()), whose
    timestamp() is just the query-evaluation time — structurally unable to
    trip a staleness gate. Instead we probe `timestamp(node_time_seconds)`,
    a raw node_exporter gauge whose sample timestamp is its true last-scrape
    time. If node scrapes stop, the probe goes stale (and after Prometheus's
    5m staleness window the expression itself returns no data) — either way
    the gate fails closed. Unconfirmable freshness = -inf (infinitely stale).
    """

    def __init__(
        self,
        client: PrometheusClient | None = None,
        *,
        staleness_budget_s: float = 120.0,
    ) -> None:
        self.client = client or PrometheusClient()
        self.staleness_budget_s = float(staleness_budget_s)

    def observe(self, rule: SystemAlertRuleSpec) -> tuple[float | None, DataStatus]:
        expr = rule.catalog_entry.promql  # hardcoded template, never user text
        resp = self.client.query(expr)

        # Source-flag branching — copied from anomaly_data_provider: a mock
        # fallback is reported honestly and the core refuses to alert on it.
        src = str(resp.get("source", "mock"))
        reachable = src == "prometheus"

        value, _n = _instant_scalar(resp)
        returns_data = value is not None

        sample_ts = float("-inf")
        if reachable and returns_data:
            ts_resp = self.client.query(f"timestamp({FRESHNESS_PROBE_METRIC})")
            if str(ts_resp.get("source", "mock")) == "prometheus":
                ts_val, ts_n = _instant_scalar(ts_resp)
                if ts_val is not None and ts_n >= 1:
                    sample_ts = float(ts_val)

        status = DataStatus(
            reachable=reachable,
            returns_data=returns_data,
            source=src,
            sample_ts=sample_ts,
            # Absolute-threshold rules have no baseline stage; coverage is
            # not consulted by SystemThresholdCore.
            history_coverage_s=0.0,
        )
        return value, status


def _instant_scalar(resp: dict[str, Any]) -> tuple[float | None, int]:
    try:
        results = resp.get("data", {}).get("result", [])
        if not results:
            return None, 0
        pair = results[0].get("value")
        if not pair or len(pair) < 2:
            return None, len(results)
        return float(pair[1]), len(results)
    except Exception:
        return None, 0


# ---------------------------------------------------------------------------
# Core — gate -> absolute-threshold compare -> duration state machine.
# ---------------------------------------------------------------------------


class SystemThresholdCore:
    """Per-rule state machine. Same order of operations as the clinical
    core (gate first, always), with an absolute-threshold compare and NO
    baseline stage. Never pages: shadow is structural in this domain."""

    def __init__(
        self,
        rule: SystemAlertRuleSpec,
        *,
        staleness_budget_s: float = 120.0,
    ) -> None:
        self.rule = rule
        self.staleness_budget_s = float(staleness_budget_s)
        self._for_seconds = rule.for_seconds
        self._breach_started_at: float | None = None
        self._fired = False
        self._last_state: AlertState | None = None

    @property
    def breach_started_at(self) -> float | None:
        return self._breach_started_at

    def evaluate(
        self, *, now: float, value: float | None, status: DataStatus
    ) -> AnomalyEvaluationReport:
        # --- 1. DATA-INTEGRITY GATE (always first) --------------------
        reason = self._integrity_failure(status, now)
        if reason is not None:
            return self._signal_lost(now, value, reason)

        # --- 2. THRESHOLD COMPARE -------------------------------------
        assert value is not None  # returns_data was True
        threshold = self.rule.threshold
        breaching = (
            value > threshold if self.rule.comparator == ">" else value < threshold
        )

        # --- 3. DURATION STATE MACHINE --------------------------------
        if not breaching:
            return self._not_breaching(now, value, threshold)
        if self._breach_started_at is None:
            self._breach_started_at = now
        elapsed = now - self._breach_started_at
        if elapsed >= self._for_seconds:
            return self._firing(now, value, threshold, elapsed)
        return self._pending(now, value, threshold, elapsed)

    # ------------------------------------------------------------------
    def _integrity_failure(
        self, status: DataStatus, now: float
    ) -> SignalLostReason | None:
        if not status.reachable:
            return SignalLostReason.unreachable
        if not status.returns_data:
            return SignalLostReason.no_data
        if status.source != "prometheus":
            return SignalLostReason.mock_source
        if (now - status.sample_ts) > self.staleness_budget_s:
            return SignalLostReason.stale
        return None

    def _event(self, now: float, state: AlertState, **kw: Any) -> AlertEvent:
        return AlertEvent(
            ts=_iso(now),
            rule_id=self.rule.id,
            state=state,
            severity=self.rule.severity,
            **kw,
        )

    def _signal_lost(
        self, now: float, value: float | None, reason: SignalLostReason
    ) -> AnomalyEvaluationReport:
        self._breach_started_at = None
        self._fired = False
        ev = self._event(
            now,
            AlertState.signal_lost,
            value=value,
            signal_lost_reason=reason,
            would_page=True,
            paged=False,
            suppressed_reason="shadow_mode",
            message=(
                f"SIGNAL_LOST ({reason.value}) — data-integrity gate failed; "
                "no verdict computed"
            ),
        )
        return self._report(now, AlertState.signal_lost, value, None, [ev])

    def _not_breaching(
        self, now: float, value: float, threshold: float
    ) -> AnomalyEvaluationReport:
        events: list[AlertEvent] = []
        if self._fired or self._breach_started_at is not None:
            events.append(
                self._event(
                    now,
                    AlertState.resolved,
                    value=value,
                    threshold=threshold,
                    message="RESOLVED — value back within threshold",
                )
            )
        self._breach_started_at = None
        self._fired = False
        return self._report(now, AlertState.resolved, value, threshold, events)

    def _pending(
        self, now: float, value: float, threshold: float, elapsed: float
    ) -> AnomalyEvaluationReport:
        events: list[AlertEvent] = []
        if self._last_state != AlertState.pending:
            events.append(
                self._event(
                    now,
                    AlertState.pending,
                    value=value,
                    threshold=threshold,
                    message=(
                        f"PENDING — breach started; {elapsed:.0f}s/"
                        f"{self._for_seconds:.0f}s into the for-window"
                    ),
                )
            )
        rpt = self._report(now, AlertState.pending, value, threshold, events)
        rpt.breach_elapsed_s = elapsed
        rpt.for_seconds = self._for_seconds
        return rpt

    def _firing(
        self, now: float, value: float, threshold: float, elapsed: float
    ) -> AnomalyEvaluationReport:
        events: list[AlertEvent] = []
        if not self._fired:
            events.append(
                self._event(
                    now,
                    AlertState.firing,
                    value=value,
                    threshold=threshold,
                    would_page=True,
                    paged=False,  # shadow is structural in this domain
                    suppressed_reason="shadow_mode",
                    message=(
                        f"FIRING — value {value:.2f} {self.rule.comparator} "
                        f"threshold {threshold:.2f} sustained {elapsed:.0f}s"
                    ),
                )
            )
        self._fired = True
        rpt = self._report(now, AlertState.firing, value, threshold, events)
        rpt.breach_elapsed_s = elapsed
        rpt.for_seconds = self._for_seconds
        return rpt

    def _report(
        self,
        now: float,
        state: AlertState,
        value: float | None,
        threshold: float | None,
        events: list[AlertEvent],
    ) -> AnomalyEvaluationReport:
        self._last_state = state
        return AnomalyEvaluationReport(
            rule_id=self.rule.id,
            ts=_iso(now),
            state=state,
            value=value,
            baseline=None,
            threshold=threshold,
            mode="shadow",
            events=events,
        )


# ---------------------------------------------------------------------------
# Service — rules + cores + durable state, ticked by the wall-clock loop.
# ---------------------------------------------------------------------------


class SystemRuleService:
    def __init__(
        self,
        *,
        rule_store: SystemRuleStore | None = None,
        provider: SystemMetricProvider | None = None,
        state_store: AlertStateStore | None = None,
        staleness_budget_s: float = 120.0,
    ) -> None:
        self.rule_store = rule_store or SystemRuleStore()
        self.provider = provider or SystemMetricProvider(
            staleness_budget_s=staleness_budget_s
        )
        self.state_store = state_store or AlertStateStore(
            base_dir=os.getenv("SYSTEM_ALERT_STATE_DIR", str(_DEFAULT_STATE_DIR))
        )
        self.staleness_budget_s = float(staleness_budget_s)
        self._lock = threading.Lock()
        self._cores: dict[str, SystemThresholdCore] = {}
        self.last_tick_wall: float | None = None
        self.last_tick_errors: dict[str, str] = {}
        for rule in self.rule_store.load_all():
            self._cores[rule.id] = self._new_core(rule)

    def _new_core(self, rule: SystemAlertRuleSpec) -> SystemThresholdCore:
        return SystemThresholdCore(
            rule, staleness_budget_s=self.staleness_budget_s
        )

    # ------------------------------------------------------------------
    def add_rule(self, rule: SystemAlertRuleSpec) -> None:
        """Persist + hot-add a validated rule. Overwriting an existing id
        resets its breach timer (a changed rule must re-earn its fire)."""
        with self._lock:
            self.rule_store.save(rule)
            self._cores[rule.id] = self._new_core(rule)

    def remove_rule(self, rule_id: str) -> bool:
        with self._lock:
            self._cores.pop(rule_id, None)
            return self.rule_store.delete(rule_id)

    def rules(self) -> list[SystemAlertRuleSpec]:
        with self._lock:
            return [core.rule for core in self._cores.values()]

    # ------------------------------------------------------------------
    def tick(self, now: float | None = None) -> dict[str, AnomalyEvaluationReport]:
        """Evaluate every rule once. Per-rule failures are recorded and
        never kill the loop (mirrors the clinical service's posture)."""
        now = time.time() if now is None else float(now)
        out: dict[str, AnomalyEvaluationReport] = {}
        errors: dict[str, str] = {}
        with self._lock:
            cores = dict(self._cores)
        for rule_id, core in cores.items():
            try:
                value, status = self.provider.observe(core.rule)
                report = core.evaluate(now=now, value=value, status=status)
                self.state_store.record_report(
                    report, breach_start_ts=core.breach_started_at
                )
                out[rule_id] = report
            except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
                errors[rule_id] = f"{type(exc).__name__}: {exc}"
        self.last_tick_wall = now
        self.last_tick_errors = errors
        return out

    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "domain": "system",
            "mode": "shadow",  # structural — see SystemAlertRuleSpec.mode
            "rules": [r.model_dump(by_alias=True) for r in self.rules()],
            "rule_load_errors": self.rule_store.load_errors,
            "state": {
                rid: st.to_json() for rid, st in self.state_store.all().items()
            },
            "state_load_error": self.state_store.load_error,
            "last_tick_wall": self.last_tick_wall,
            "last_tick_errors": self.last_tick_errors,
            "non_diagnostic": True,
        }

    def alerts(self) -> list[dict[str, Any]]:
        return self.state_store.history_records()


# ---------------------------------------------------------------------------
# Process-wide singleton — shared by the API router, the wall-clock loop in
# main.py, and the chat orchestrator.
# ---------------------------------------------------------------------------

_SINGLETON: SystemRuleService | None = None
_SINGLETON_LOCK = threading.Lock()


def get_service() -> SystemRuleService:
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = SystemRuleService()
        return _SINGLETON


def reset_service_for_tests() -> None:
    global _SINGLETON
    with _SINGLETON_LOCK:
        _SINGLETON = None


def is_enabled() -> bool:
    """Wall-clock loop gate. Default ON — the loop only reads Prometheus
    and records shadow events. `SYSTEM_RULES_ENABLED=0` is the kill
    switch; the test suite keeps it off via conftest."""
    return os.getenv("SYSTEM_RULES_ENABLED", "1") not in (
        "0", "false", "False", "no", "off",
    )
