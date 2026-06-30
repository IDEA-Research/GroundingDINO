"""Behavioural golden suite for the neonatal rSO2 anomaly evaluator.

These are the inviolable behavioural gates for Increment 1:

  (a) known-anomaly trace MUST fire at t = 5m, severity critical
  (b) known-normal trace MUST NOT fire
  (c) real -> mock degradation MUST yield SIGNAL_LOST, never a fire
  (d) <24h history MUST yield INSUFFICIENT_BASELINE, never a fire

Plus the unreachable / stale signal-loss variants and the spec-level
fail-closed checks (source=='mock' rejected at validation).

A MISSED MUST-FIRE is the top-severity failure and blocks the increment.
"""

from __future__ import annotations

import pytest

from app.prometheus.neonatal_sim import NeonatalSim, Scenario
from app.specs.alert_rule_spec import AlertRuleSpec
from app.specs.anomaly_evaluation_report import (
    AlertState,
    SignalLostReason,
)

from .harness import baseline_rule, replay


# 30s step, 5m for-window => fires on the 11th tick (t=300s) at the latest.
STEP_S = 30.0
N_TICKS = 30  # 15 minutes of trace at 30s steps


# ---------------------------------------------------------------------------
# (a) known-anomaly trace MUST fire at t = 5m
# ---------------------------------------------------------------------------
def test_golden_anomaly_must_fire_at_5m():
    rule = baseline_rule()
    sim = NeonatalSim(scenario=Scenario.rso2_desat, start_ts=0.0, step_s=STEP_S)
    samples = sim.series(rule.metric, n=N_TICKS)

    result = replay(rule, samples)

    assert result.fired(), (
        "MUST-FIRE MISSED: sustained 25% rSO2 desaturation did not fire "
        "(top-severity failure)"
    )
    fire = result.first_event(AlertState.firing)
    assert fire is not None
    # for_=5m, step=30s, breach from t=0 -> firing the first tick at which
    # elapsed >= 300s, i.e. t == 300s exactly.
    assert fire.severity == "critical"
    fire_ts_seconds = result.ts[
        next(i for i, ev_state in enumerate(result.states) if ev_state == AlertState.firing)
    ]
    assert fire_ts_seconds == pytest.approx(300.0), (
        f"fired at t={fire_ts_seconds}s, expected exactly 300s (5m)"
    )
    # Shadow by default: would-page but not actually paged.
    assert fire.would_page is True
    assert fire.paged is False
    assert fire.suppressed_reason == "shadow_mode"
    assert fire.non_diagnostic is True


def test_golden_anomaly_does_not_fire_before_5m():
    rule = baseline_rule()
    sim = NeonatalSim(scenario=Scenario.rso2_desat, start_ts=0.0, step_s=STEP_S)
    samples = sim.series(rule.metric, n=N_TICKS)

    result = replay(rule, samples)

    # Every tick strictly before 300s must NOT be in firing.
    for state, ts in zip(result.states, result.ts):
        if ts < 300.0:
            assert state != AlertState.firing, f"fired too early at t={ts}s"
            # but it should be pending once the breach timer starts (t>=0)
    assert result.has_state(AlertState.pending), "expected a pending phase"


# ---------------------------------------------------------------------------
# (b) known-normal trace MUST NOT fire
# ---------------------------------------------------------------------------
def test_golden_normal_must_not_fire():
    rule = baseline_rule()
    sim = NeonatalSim(scenario=Scenario.healthy, start_ts=0.0, step_s=STEP_S)
    samples = sim.series(rule.metric, n=N_TICKS)

    result = replay(rule, samples)

    assert not result.fired(), "FALSE ALARM: healthy trace fired"
    assert not result.has_state(AlertState.pending), "healthy trace went pending"
    assert not result.has_state(AlertState.signal_lost)


# ---------------------------------------------------------------------------
# (c) real -> mock degradation MUST yield SIGNAL_LOST, never a fire
# ---------------------------------------------------------------------------
def test_golden_source_degraded_signal_lost_not_fire():
    rule = baseline_rule()
    sim = NeonatalSim(scenario=Scenario.source_degraded, start_ts=0.0, step_s=STEP_S)
    samples = sim.series(rule.metric, n=N_TICKS)

    result = replay(rule, samples)

    assert result.has_state(AlertState.signal_lost), "expected SIGNAL_LOST on mock"
    sl = result.first_event(AlertState.signal_lost)
    assert sl is not None
    assert sl.signal_lost_reason == SignalLostReason.mock_source
    # SIGNAL_LOST is itself conspicuous / non-suppressible.
    assert sl.would_page is True
    # The mock samples are driven BELOW threshold (a numeric breach); the
    # only thing preventing a fire is the data-integrity gate. This proves
    # the gate — not the numbers — is load-bearing.
    assert not result.fired(), "FAIL-OPEN: fired on mock/fake data"


def test_golden_unreachable_signal_lost_not_fire():
    rule = baseline_rule()
    sim = NeonatalSim(scenario=Scenario.unreachable, start_ts=0.0, step_s=STEP_S)
    samples = sim.series(rule.metric, n=N_TICKS)

    result = replay(rule, samples)

    sl = result.first_event(AlertState.signal_lost)
    assert sl is not None
    assert sl.signal_lost_reason == SignalLostReason.unreachable
    assert not result.fired()


def test_golden_stale_signal_lost_not_fire():
    rule = baseline_rule()
    sim = NeonatalSim(scenario=Scenario.stale, start_ts=0.0, step_s=STEP_S)
    samples = sim.series(rule.metric, n=N_TICKS)

    # staleness budget 60s; data freezes after a third of the trace.
    result = replay(rule, samples, staleness_budget_s=60.0)

    sl = result.first_event(AlertState.signal_lost)
    assert sl is not None
    assert sl.signal_lost_reason == SignalLostReason.stale
    assert not result.fired()


# ---------------------------------------------------------------------------
# (d) <24h history MUST yield INSUFFICIENT_BASELINE, never a fire
# ---------------------------------------------------------------------------
def test_golden_short_history_insufficient_baseline_not_fire():
    rule = baseline_rule()
    # Even a desaturating trace must NOT fire if the baseline window is not
    # covered by enough history.
    sim = NeonatalSim(scenario=Scenario.rso2_desat, start_ts=0.0, step_s=STEP_S)
    samples = sim.series(rule.metric, n=N_TICKS)

    result = replay(rule, samples, history_coverage_s=12 * 3600.0)  # only 12h

    assert result.has_state(AlertState.insufficient_baseline)
    ib = result.first_event(AlertState.insufficient_baseline)
    assert ib is not None
    assert ib.would_page is False  # not a clinical fire
    assert not result.fired(), "fired without a sufficient baseline"


# ---------------------------------------------------------------------------
# Spec-level fail-closed safety: source=='mock' rejected at VALIDATION
# ---------------------------------------------------------------------------
def test_rule_rejects_mock_source_at_validation():
    with pytest.raises(Exception) as exc:
        AlertRuleSpec.model_validate(
            {
                "id": "bad-rule",
                "metric": "rso2_left",
                "source": "mock",
            }
        )
    assert "mock" in str(exc.value).lower() or "prometheus" in str(exc.value).lower()


def test_rule_rejects_extra_fields():
    with pytest.raises(Exception):
        AlertRuleSpec.model_validate(
            {
                "id": "bad-rule",
                "metric": "rso2_left",
                "evil_extra": "x",
            }
        )


def test_rule_rejects_promql_injection_in_metric():
    for bad in ["rso2;drop", "rso2`x`", "rso2(1)", "rso2 left", "rso2<1"]:
        with pytest.raises(Exception):
            AlertRuleSpec.model_validate({"id": "r", "metric": bad})


def test_rule_locked_semantics_defaults():
    rule = baseline_rule()
    assert rule.ratio == 0.80
    assert rule.baseline.window == "24h"
    assert rule.baseline.fn == "avg_over_time"
    assert rule.comparator == "<"
    assert rule.for_ == "5m"
    # Shadow by default.
    assert rule.mode.value == "shadow"
