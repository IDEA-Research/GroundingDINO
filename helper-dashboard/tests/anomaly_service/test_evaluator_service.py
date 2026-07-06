"""Integration suite for the INC2 background evaluator service.

The service wraps the INC1 pure core in a runnable loop with:
  - an injectable-clock `tick(now)` (pytest_asyncio is absent, so we step the
    clock BY HAND — no wall-clock sleeps),
  - a durable, fail-closed alert-state store (breach_start + lifecycle),
  - a heartbeat + watchdog that makes a stalled/crashed tick detectable.

These tests assert the FULL lifecycle with accelerated/injected time:

  (1) normal -> breach -> firing@5m -> resolved,
  (2) mid-trace real -> mock  =>  SIGNAL_LOST (never a clinical fire),
  (3) durable-restart rehydration: a breach in progress does NOT reset its 5m
      clock across a simulated process restart,
  (4) watchdog: a stalled tick is detected; a fresh loop is healthy,
  (5) shadow discipline: a firing event would-page but never pages here,
  (6) a throwing rule is fail-loud (audited + watchdog), not silent green.

A MISSED must-fire is the top-severity failure and blocks the increment.
"""

from __future__ import annotations

import os

import pytest

from app.prometheus.neonatal_sim import Scenario
from app.services.alert_state_store import AlertStateStore, AlertStateStoreError
from app.services.anomaly_core import DataStatus
from app.services.anomaly_data_provider import Observation, SimDataProvider
from app.services.anomaly_evaluator_service import AnomalyEvaluatorService
from app.specs.anomaly_evaluation_report import AlertState, SignalLostReason
from app.tests_support.default_rules import neonatal_rso2_rule

STEP_S = 30.0
FOR_S = 300.0  # 5m
N_TICKS = 40


def _store(tmp_path, name="state"):
    return AlertStateStore(base_dir=tmp_path / name)


def _service(provider, tmp_path, *, name="state", **kw):
    return AnomalyEvaluatorService(
        [neonatal_rso2_rule(metric="rso2_left")],
        provider,
        store=_store(tmp_path, name),
        staleness_budget_s=60.0,
        watchdog_budget_s=45.0,
        monotonic=lambda: 0.0,  # overridden per-tick via mono=
        **kw,
    )


def _run(service, rule_id, *, n=N_TICKS, step=STEP_S, start=0.0):
    """Step the injected clock synchronously; return per-tick states."""
    states = []
    for i in range(n):
        now = start + i * step
        result = service.tick(now, mono=now)
        rep = result.reports.get(rule_id)
        states.append((now, rep.state if rep else None))
    return states


# ---------------------------------------------------------------------------
# (1) FULL LIFECYCLE: normal -> breach -> firing@5m -> resolved
# ---------------------------------------------------------------------------
def test_lifecycle_normal_breach_firing_resolved(tmp_path):
    rule = neonatal_rso2_rule(metric="rso2_left")

    # A trace that is healthy, then desaturates, then recovers. We build it by
    # composing three sim providers around a shared injected clock via a small
    # switching provider.
    healthy = SimDataProvider(
        scenario=Scenario.healthy, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS
    )
    desat = SimDataProvider(
        scenario=Scenario.rso2_desat, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS
    )

    class Phased:
        """healthy for t<T0, desat for T0<=t<T1, healthy again after."""

        def __init__(self, t0, t1):
            self.t0, self.t1 = t0, t1

        def observe(self, metric, *, now, labels=None):
            if self.t0 <= now < self.t1:
                return desat.observe(metric, now=now)
            return healthy.observe(metric, now=now)

    # Breach begins at t=60s and lasts through t=60+300+90 so firing occurs at
    # t=60+300=360s, then recovers.
    t0, t1 = 60.0, 60.0 + FOR_S + 120.0
    service = AnomalyEvaluatorService(
        [rule], Phased(t0, t1), store=_store(tmp_path), monotonic=lambda: 0.0
    )
    states = _run(service, rule.id, n=N_TICKS)

    # No fire before 5m of sustained breach (breach starts at t0=60).
    for now, st in states:
        if now < t0 + FOR_S:
            assert st != AlertState.firing, f"fired too early at t={now}"

    fired_ts = [now for now, st in states if st == AlertState.firing]
    assert fired_ts, "MUST-FIRE MISSED: sustained desaturation never fired"
    assert fired_ts[0] == pytest.approx(t0 + FOR_S), (
        f"fired at t={fired_ts[0]}, expected exactly {t0 + FOR_S} (breach+5m)"
    )

    # Must go through pending before firing.
    assert any(st == AlertState.pending for _, st in states), "no pending phase"

    # After recovery the rule resolves.
    resolved_after = [
        now for now, st in states if st == AlertState.resolved and now >= t1
    ]
    assert resolved_after, "did not RESOLVE after value recovered"

    # Durable store reflects the final resolved state and cleared breach timer.
    final = service.store.get(rule.id)
    assert final.state == AlertState.resolved.value
    assert final.breach_start_ts is None


# ---------------------------------------------------------------------------
# (2) mid-trace real -> mock  =>  SIGNAL_LOST, never a clinical fire
# ---------------------------------------------------------------------------
def test_mid_trace_real_to_mock_yields_signal_lost(tmp_path):
    rule = neonatal_rso2_rule(metric="rso2_left")
    provider = SimDataProvider(
        scenario=Scenario.source_degraded, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS
    )
    service = AnomalyEvaluatorService(
        [rule], provider, store=_store(tmp_path), monotonic=lambda: 0.0
    )
    states = _run(service, rule.id, n=N_TICKS)

    lost = [st for _, st in states if st == AlertState.signal_lost]
    assert lost, "expected SIGNAL_LOST once the source degrades to mock"
    assert not any(st == AlertState.firing for _, st in states), (
        "FAIL-OPEN: fired on mock/fake data — the data-integrity gate did not "
        "run first (top-severity safety failure)"
    )
    st = service.store.get(rule.id)
    assert st.signal_lost_reason == SignalLostReason.mock_source.value
    # SIGNAL_LOST is conspicuous / non-suppressible: it counts as would_page.
    assert st.would_page_count >= 1
    # Shadow: it never actually paged.
    assert st.paged_count == 0


def test_unreachable_and_stale_yield_signal_lost(tmp_path):
    for scenario, reason, name in (
        (Scenario.unreachable, SignalLostReason.unreachable, "unreach"),
        (Scenario.stale, SignalLostReason.stale, "stale"),
    ):
        rule = neonatal_rso2_rule(metric="rso2_left")
        provider = SimDataProvider(
            scenario=scenario, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS
        )
        service = AnomalyEvaluatorService(
            [rule], provider, store=_store(tmp_path, name), monotonic=lambda: 0.0
        )
        states = _run(service, rule.id, n=N_TICKS)
        assert any(st == AlertState.signal_lost for _, st in states), (
            f"{scenario} did not SIGNAL_LOST"
        )
        assert not any(st == AlertState.firing for _, st in states), (
            f"FAIL-OPEN on {scenario}"
        )
        assert service.store.get(rule.id).signal_lost_reason == reason.value


# ---------------------------------------------------------------------------
# (3) durable-restart rehydration: breach timer survives a process restart
# ---------------------------------------------------------------------------
def test_restart_rehydrates_breach_timer(tmp_path):
    rule = neonatal_rso2_rule(metric="rso2_left")
    state_dir = tmp_path / "shared"

    # Phase A: breach starts at t=0, run only up to t=150 (2.5m, < 5m).
    provA = SimDataProvider(
        scenario=Scenario.rso2_desat, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS
    )
    svcA = AnomalyEvaluatorService(
        [rule], provA, store=AlertStateStore(base_dir=state_dir), monotonic=lambda: 0.0
    )
    statesA = _run(svcA, rule.id, n=6)  # t=0..150
    assert all(st != AlertState.firing for _, st in statesA), "fired before 5m"
    assert svcA.store.get(rule.id).breach_start_ts == 0.0, "breach_start not durable"

    # Phase B: NEW store + NEW service from the SAME dir (simulated restart).
    provB = SimDataProvider(
        scenario=Scenario.rso2_desat, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS
    )
    svcB = AnomalyEvaluatorService(
        [rule], provB, store=AlertStateStore(base_dir=state_dir), monotonic=lambda: 0.0
    )
    # Continue at t=300. Because breach_start rehydrated to 0.0, elapsed=300s
    # >= 5m and it MUST fire — a restart did NOT reset the 5m window.
    result = svcB.tick(300.0, mono=300.0)
    assert result.reports[rule.id].state == AlertState.firing, (
        "restart RESET the breach timer — a real desaturation would have its "
        "5m clock silently restarted (clinical hazard)"
    )


# ---------------------------------------------------------------------------
# (4) watchdog / heartbeat
# ---------------------------------------------------------------------------
def test_watchdog_detects_stall_and_reports_healthy(tmp_path):
    rule = neonatal_rso2_rule(metric="rso2_left")
    provider = SimDataProvider(
        scenario=Scenario.healthy, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS
    )
    service = AnomalyEvaluatorService(
        [rule], provider, store=_store(tmp_path), monotonic=lambda: 0.0
    )

    # Before any tick: not healthy, not stalled (no baseline yet).
    wd0 = service.watchdog(now_monotonic=0.0)
    assert wd0.healthy is False
    assert wd0.stalled is False

    # Tick a few times at 30s cadence.
    for i in range(5):
        service.tick(i * STEP_S, mono=i * STEP_S)
    last = 4 * STEP_S

    # 5s after last tick: healthy, not stalled.
    wd_fresh = service.watchdog(now_monotonic=last + 5.0)
    assert wd_fresh.stalled is False
    assert wd_fresh.healthy is True

    # 100s after last tick (> 45s budget): STALLED and unhealthy.
    wd_stalled = service.watchdog(now_monotonic=last + 100.0)
    assert wd_stalled.stalled is True
    assert wd_stalled.healthy is False
    assert "STALLED" in wd_stalled.reason


def test_watchdog_flags_persistent_degradation(tmp_path):
    rule = neonatal_rso2_rule(metric="rso2_left")
    provider = SimDataProvider(
        scenario=Scenario.unreachable, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS
    )
    service = AnomalyEvaluatorService(
        [rule], provider, store=_store(tmp_path), monotonic=lambda: 0.0
    )
    # Run past the unreachable onset so several consecutive ticks are degraded.
    _run(service, rule.id, n=N_TICKS)
    wd = service.watchdog(now_monotonic=(N_TICKS - 1) * STEP_S)
    assert wd.degraded is True
    assert service.heartbeat.consecutive_degraded > 0


# ---------------------------------------------------------------------------
# (5) shadow discipline: firing would-page but never pages in INC2
# ---------------------------------------------------------------------------
def test_shadow_never_pages(tmp_path):
    rule = neonatal_rso2_rule(metric="rso2_left")
    provider = SimDataProvider(
        scenario=Scenario.rso2_desat, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS
    )
    service = AnomalyEvaluatorService(
        [rule], provider, store=_store(tmp_path), monotonic=lambda: 0.0, promoted=False
    )
    _run(service, rule.id, n=15)  # long enough to fire
    st = service.store.get(rule.id)
    assert st.would_page_count >= 1, "a firing edge must record would_page"
    assert st.paged_count == 0, "SHADOW rule paged — invariant violated"


# ---------------------------------------------------------------------------
# (6) fail-loud: a throwing rule is visible, not a silent green
# ---------------------------------------------------------------------------
def test_throwing_provider_is_fail_loud(tmp_path):
    rule = neonatal_rso2_rule(metric="rso2_left")

    class Boom:
        def observe(self, metric, *, now, labels=None):
            raise RuntimeError("provider exploded")

    service = AnomalyEvaluatorService(
        [rule], Boom(), store=_store(tmp_path), monotonic=lambda: 0.0
    )
    result = service.tick(0.0, mono=0.0)
    # The loop survives, but the error is recorded (never swallowed to green).
    assert rule.id in result.errors
    assert service.heartbeat.error_count == 1
    wd = service.watchdog(now_monotonic=0.0)
    assert wd.healthy is False


# ---------------------------------------------------------------------------
# store-level fail-closed: clinical persist failure RAISES
# ---------------------------------------------------------------------------
def test_store_load_error_is_surfaced_not_reset(tmp_path):
    # A corrupt snapshot must be surfaced, never silently reset to healthy.
    d = tmp_path / "corrupt"
    d.mkdir()
    (d / "state.json").write_text("{ this is not json", encoding="utf-8")
    store = AlertStateStore(base_dir=d)
    assert store.load_error is not None
    # A fresh rule reads back as UNKNOWN/resolved default, but the load error
    # remains visible for a deliberate recovery.
    assert store.get("neo-rso2-left-desat").breach_start_ts is None
