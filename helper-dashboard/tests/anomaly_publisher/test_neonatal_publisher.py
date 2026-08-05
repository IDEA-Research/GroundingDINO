"""INC6 CHECK: the synthetic neonatal metrics publisher.

Proves (all against an ISOLATED CollectorRegistry so the process-wide DEFAULT
registry is never touched by the suite):

  1. The publisher registers the expected gauges, each carrying the `patient`
     label, so `rso2_left{patient="neo-001"}` resolves on /metrics.
  2. One update tick sets `rso2_left` to a plausible neonatal value (~72) from
     the sim (healthy scenario), and the heartbeat/freshness gauge advances.
  3. The `rso2_desat` scenario drives `rso2_left` below 0.80x its 24h baseline,
     i.e. into the rule's breach territory — the whole point of pointing
     Prometheus at real (synthetic) series.
  4. Registration is idempotent: two publishers on the same registry do not
     raise "Duplicated timeseries".
  5. Only `prometheus`-sourced (real synthetic) series are published — never
     mock — preserving the source==mock safety gate.
"""

from __future__ import annotations

import time

from prometheus_client import CollectorRegistry, generate_latest

from app.prometheus.neonatal_publisher import (
    NeonatalPublisher,
    _HEARTBEAT_METRIC,
    _PHYS_METRICS,
)
from app.prometheus.neonatal_sim import NeonatalSim, Scenario


def _gauge_value(registry: CollectorRegistry, name: str, patient: str) -> float:
    val = registry.get_sample_value(name, {"patient": patient})
    assert val is not None, f"{name}{{patient={patient!r}}} not present on /metrics"
    return val


def test_registers_expected_gauges_with_patient_label() -> None:
    reg = CollectorRegistry()
    pub = NeonatalPublisher(registry=reg, scenario=Scenario.healthy)

    # Even before any tick, the collectors exist so a scrape has the series.
    exposition = generate_latest(reg).decode("utf-8")
    for metric in _PHYS_METRICS:
        assert metric in exposition
    assert _HEARTBEAT_METRIC in exposition

    # After a tick, each physiological gauge carries the patient label.
    pub.update_once()
    for metric in _PHYS_METRICS:
        assert reg.get_sample_value(metric, {"patient": "neo-001"}) is not None


def test_healthy_tick_sets_rso2_left_to_plausible_neonatal_value() -> None:
    reg = CollectorRegistry()
    pub = NeonatalPublisher(registry=reg, scenario=Scenario.healthy)

    written = pub.update_once()
    rso2 = _gauge_value(reg, "rso2_left", "neo-001")

    # Baseline centre is 72.0; healthy wander stays within a few percent.
    assert 68.0 <= rso2 <= 76.0, rso2
    assert abs(written["rso2_left"] - rso2) < 1e-9
    # A plausible neonatal cerebral rSO2 range.
    assert 55.0 <= rso2 <= 85.0


def test_heartbeat_advances_and_tracks_wall_clock() -> None:
    reg = CollectorRegistry()
    pub = NeonatalPublisher(registry=reg, scenario=Scenario.healthy)

    before = time.time()
    pub.update_once()
    hb1 = _gauge_value(reg, _HEARTBEAT_METRIC, "neo-001")
    assert hb1 >= before - 1.0  # freshness == real wall clock

    time.sleep(0.01)
    pub.update_once()
    hb2 = _gauge_value(reg, _HEARTBEAT_METRIC, "neo-001")
    assert hb2 >= hb1  # heartbeat is monotonic across ticks


def test_rso2_desat_scenario_drops_below_080_of_baseline() -> None:
    reg = CollectorRegistry()
    pub = NeonatalPublisher(registry=reg, scenario=Scenario.rso2_desat)

    pub.update_once()
    rso2 = _gauge_value(reg, "rso2_left", "neo-001")

    # The baseline the rule compares against is the healthy 24h average.
    baseline = NeonatalSim(scenario=Scenario.healthy).baseline("rso2_left")
    threshold = 0.80 * baseline
    assert rso2 < threshold, (rso2, threshold)
    # And it must remain a physiologically plausible (not impossible) value.
    assert rso2 > 0.0


def test_registration_is_idempotent_on_same_registry() -> None:
    reg = CollectorRegistry()
    # Two publishers sharing a registry must not raise "Duplicated timeseries".
    pub_a = NeonatalPublisher(registry=reg, scenario=Scenario.healthy)
    pub_b = NeonatalPublisher(registry=reg, scenario=Scenario.rso2_desat)

    pub_a.update_once()
    pub_b.update_once()

    # Both operate; the last writer wins on the shared gauge.
    assert reg.get_sample_value("rso2_left", {"patient": "neo-001"}) is not None


def test_publisher_only_emits_real_synthetic_prometheus_samples() -> None:
    # The healthy sim never marks a sample source=="mock"; the publisher only
    # ever reads sample.value from a prometheus-sourced series. Assert the sim
    # contract the publisher relies on, so a future regression that leaks mock
    # data into the publisher path is caught here.
    sim = NeonatalSim(scenario=Scenario.healthy)
    for s in sim.series("rso2_left", n=5):
        assert s.source == "prometheus"
        assert s.reachable and s.has_data


def test_start_stop_is_idempotent() -> None:
    reg = CollectorRegistry()
    pub = NeonatalPublisher(registry=reg, scenario=Scenario.healthy, interval_s=1.0)
    try:
        assert pub.start() is True
        assert pub.start() is False  # second start is a no-op
    finally:
        pub.stop()
        pub.stop()  # safe to stop twice


def test_sample_at_matches_series_for_every_scenario() -> None:
    """sample_at(metric, i, n) must be byte-identical to series(metric, n)[i].

    The publisher's per-tick path relies on this equivalence (it switched off
    full-series regeneration, which grew O(uptime^2)); a divergence would
    silently change the synthetic trace shape the goldens were built on.
    """
    for scenario in Scenario:
        sim = NeonatalSim(scenario=scenario)
        for metric in ("rso2_left", "rso2_right", "spo2", "hr"):
            full = sim.series(metric, n=30)
            for i in (0, 9, 10, 11, 29):  # spans the n//3 scenario boundary
                assert sim.sample_at(metric, i, n=30) == full[i], (
                    f"{scenario.value}/{metric}[{i}] diverged from series()"
                )
