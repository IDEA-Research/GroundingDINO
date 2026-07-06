"""PrometheusDataProvider safety gates (A_DIAGNOSIS E12).

Before these fixes the provider was unsafe to wire into a live loop:

  (i)   its staleness signal was the query-EVALUATION timestamp (~now), so
        frozen data passed as fresh and SignalLostReason.stale was
        structurally unreachable,
  (ii)  the rule's validated labels were never sent — an unlabeled selector
        returns EVERY series of the metric and the first was picked
        arbitrarily (wrong-patient hazard),
  (iii) history coverage was asserted as a 24h constant, structurally
        defeating the INSUFFICIENT_BASELINE gate on live data.

All fixes fail CLOSED: unavailable sub-queries yield maximal staleness or
zero coverage, and a multi-series answer is an integrity failure
(ambiguous_series), never a first-pick. NO test here touches a network —
the client is an injected fake.
"""

from __future__ import annotations

from app.services.anomaly_core import AnomalyEvaluatorCore, DataStatus
from app.services.anomaly_data_provider import PrometheusDataProvider
from app.specs.anomaly_evaluation_report import AlertState, SignalLostReason
from app.tests_support.default_rules import neonatal_rso2_rule

NOW = 1_000_000.0
DAY_S = 24 * 3600.0


class FakePromClient:
    """Table-driven fake: maps query string -> canned response."""

    def __init__(self, table: dict) -> None:
        self.table = dict(table)
        self.queries: list[str] = []

    def query(self, q: str) -> dict:
        self.queries.append(q)
        if q in self.table:
            return self.table[q]
        return {"source": "prometheus", "data": {"result": []}}


def vec(*pairs, source="prometheus"):
    """Build an instant-vector response; pairs are (eval_ts, value)."""
    return {
        "source": source,
        "data": {
            "result": [
                {"metric": {}, "value": [ts, str(v)]} for ts, v in pairs
            ]
        },
    }


SEL = 'rso2_left{patient="neo-001"}'
LABELS = {"patient": "neo-001"}


def _healthy_table(*, sample_ts=NOW - 10.0, coverage_samples=DAY_S / 15.0):
    return {
        SEL: vec((NOW, 45.0)),
        f"avg_over_time({SEL}[24h])": vec((NOW, 60.0)),
        f"timestamp({SEL})": vec((NOW, sample_ts)),
        f"count_over_time({SEL}[24h])": vec((NOW, coverage_samples)),
    }


def test_labels_are_sent_in_every_query():
    """(ii) the rule's labels select exactly one series — never the bare metric."""
    client = FakePromClient(_healthy_table())
    p = PrometheusDataProvider(client, scrape_interval_s=15.0)
    p.observe("rso2_left", now=NOW, labels=LABELS)
    assert client.queries, "no queries issued"
    assert all(SEL in q for q in client.queries), (
        f"a query used the bare metric name (wrong-patient hazard): "
        f"{client.queries}"
    )


def test_healthy_single_series_passes_all_gates():
    client = FakePromClient(_healthy_table())
    p = PrometheusDataProvider(client, scrape_interval_s=15.0)
    obs = p.observe("rso2_left", now=NOW, labels=LABELS)
    assert obs.value == 45.0
    assert obs.baseline == 60.0
    assert obs.status.reachable and obs.status.returns_data
    assert obs.status.ambiguous is False
    assert obs.status.sample_ts == NOW - 10.0
    assert obs.status.history_coverage_s == DAY_S  # capped at the window


def test_multi_series_is_ambiguous_and_gates_to_signal_lost():
    """(ii) >1 series => integrity failure, never an arbitrary first-pick."""
    table = _healthy_table()
    table[SEL] = vec((NOW, 45.0), (NOW, 82.0))  # two patients/probes
    p = PrometheusDataProvider(FakePromClient(table), scrape_interval_s=15.0)
    obs = p.observe("rso2_left", now=NOW, labels=LABELS)
    assert obs.status.ambiguous is True

    core = AnomalyEvaluatorCore(neonatal_rso2_rule(metric="rso2_left"))
    rpt = core.evaluate(
        now=NOW, value=obs.value, baseline=obs.baseline, status=obs.status
    )
    assert rpt.state == AlertState.signal_lost
    assert rpt.events[0].signal_lost_reason == SignalLostReason.ambiguous_series


def test_frozen_sample_trips_the_staleness_gate():
    """(i) sample_ts comes from timestamp(<sel>) — a frozen series whose
    last scrape was an hour ago must trip `stale`, even though the instant
    vector's eval-ts is ~now."""
    table = _healthy_table(sample_ts=NOW - 3600.0)
    p = PrometheusDataProvider(FakePromClient(table), scrape_interval_s=15.0)
    obs = p.observe("rso2_left", now=NOW, labels=LABELS)
    assert obs.status.sample_ts == NOW - 3600.0

    core = AnomalyEvaluatorCore(
        neonatal_rso2_rule(metric="rso2_left"), staleness_budget_s=60.0
    )
    rpt = core.evaluate(
        now=NOW, value=obs.value, baseline=obs.baseline, status=obs.status
    )
    assert rpt.state == AlertState.signal_lost
    assert rpt.events[0].signal_lost_reason == SignalLostReason.stale


def test_unavailable_timestamp_query_fails_to_infinitely_stale():
    """An unconfirmable sample time is -inf — stale in ANY consumer's budget.

    A finite sentinel keyed to the provider's own budget would pass a core
    configured with a larger staleness budget (verifier-found defect).
    """
    table = _healthy_table()
    del table[f"timestamp({SEL})"]  # falls back to empty result
    p = PrometheusDataProvider(
        FakePromClient(table), staleness_budget_s=60.0, scrape_interval_s=15.0
    )
    obs = p.observe("rso2_left", now=NOW, labels=LABELS)
    assert obs.status.sample_ts == float("-inf")

    core = AnomalyEvaluatorCore(
        neonatal_rso2_rule(metric="rso2_left"),
        staleness_budget_s=3600.0,  # far larger than the provider's budget
    )
    rpt = core.evaluate(
        now=NOW, value=obs.value, baseline=obs.baseline, status=obs.status
    )
    assert rpt.state == AlertState.signal_lost
    assert rpt.events[0].signal_lost_reason == SignalLostReason.stale


HEARTBEAT = "neonatal_sim_last_update_timestamp_seconds"
HB_SEL = f'{HEARTBEAT}{{patient="neo-001"}}'


def test_frozen_publisher_behind_live_endpoint_trips_via_heartbeat():
    """(i) A dead publisher thread behind a live /metrics endpoint: Prometheus
    re-stamps the frozen value on every scrape, so timestamp() stays ~now —
    only the measurement-time heartbeat catches it (verifier-found defect)."""
    table = _healthy_table(sample_ts=NOW - 7.0)  # scrape looks fresh
    table[HB_SEL] = vec((NOW, NOW - 3600.0))  # last real measurement: 1h ago
    p = PrometheusDataProvider(
        FakePromClient(table),
        scrape_interval_s=15.0,
        freshness_metric=HEARTBEAT,
    )
    obs = p.observe("rso2_left", now=NOW, labels=LABELS)
    assert obs.status.sample_ts == NOW - 3600.0  # the OLDER of the two

    core = AnomalyEvaluatorCore(
        neonatal_rso2_rule(metric="rso2_left"), staleness_budget_s=60.0
    )
    rpt = core.evaluate(
        now=NOW, value=obs.value, baseline=obs.baseline, status=obs.status
    )
    assert rpt.state == AlertState.signal_lost
    assert rpt.events[0].signal_lost_reason == SignalLostReason.stale


def test_configured_but_unavailable_heartbeat_fails_closed():
    table = _healthy_table(sample_ts=NOW - 7.0)  # no heartbeat entry at all
    p = PrometheusDataProvider(
        FakePromClient(table),
        scrape_interval_s=15.0,
        freshness_metric=HEARTBEAT,
    )
    obs = p.observe("rso2_left", now=NOW, labels=LABELS)
    assert obs.status.sample_ts == float("-inf"), (
        "a configured heartbeat that cannot be confirmed must fail CLOSED"
    )


def test_fresh_heartbeat_passes():
    table = _healthy_table(sample_ts=NOW - 7.0)
    table[HB_SEL] = vec((NOW, NOW - 5.0))  # measurement 5s ago
    p = PrometheusDataProvider(
        FakePromClient(table),
        scrape_interval_s=15.0,
        freshness_metric=HEARTBEAT,
    )
    obs = p.observe("rso2_left", now=NOW, labels=LABELS)
    assert obs.status.sample_ts == NOW - 7.0  # older of scrape/measurement


def test_coverage_is_measured_not_asserted():
    """(iii) 20 minutes of samples => ~20 minutes of coverage, and the core
    reports INSUFFICIENT_BASELINE instead of trusting a 24h constant."""
    table = _healthy_table(coverage_samples=80.0)  # 80 x 15s = 1200s
    p = PrometheusDataProvider(FakePromClient(table), scrape_interval_s=15.0)
    obs = p.observe("rso2_left", now=NOW, labels=LABELS)
    assert obs.status.history_coverage_s == 1200.0

    core = AnomalyEvaluatorCore(neonatal_rso2_rule(metric="rso2_left"))
    rpt = core.evaluate(
        now=NOW, value=obs.value, baseline=obs.baseline, status=obs.status
    )
    assert rpt.state == AlertState.insufficient_baseline


def test_unavailable_count_query_fails_to_zero_coverage():
    table = _healthy_table()
    del table[f"count_over_time({SEL}[24h])"]
    p = PrometheusDataProvider(FakePromClient(table), scrape_interval_s=15.0)
    obs = p.observe("rso2_left", now=NOW, labels=LABELS)
    assert obs.status.history_coverage_s == 0.0


def test_mock_fallback_is_reported_as_mock():
    table = {
        SEL: vec((NOW, 45.0), source="mock"),
        f"avg_over_time({SEL}[24h])": vec((NOW, 60.0), source="mock"),
    }
    p = PrometheusDataProvider(FakePromClient(table), scrape_interval_s=15.0)
    obs = p.observe("rso2_left", now=NOW, labels=LABELS)
    assert obs.status.source == "mock"
    assert obs.status.reachable is False


def test_half_mock_baseline_taints_the_observation():
    table = _healthy_table()
    table[f"avg_over_time({SEL}[24h])"] = vec((NOW, 60.0), source="mock")
    p = PrometheusDataProvider(FakePromClient(table), scrape_interval_s=15.0)
    obs = p.observe("rso2_left", now=NOW, labels=LABELS)
    assert obs.status.source == "mock", "half-mock observation must not pass"


def test_label_values_are_quote_escaped():
    sel = PrometheusDataProvider._label_selector(
        "rso2_left", {"note": 'a"b\\c'}
    )
    assert sel == 'rso2_left{note="a\\"b\\\\c"}'
