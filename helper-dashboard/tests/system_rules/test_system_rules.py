"""System-metric alert rules — spec, core, service, API, and chat flow.

The NON-CLINICAL sibling of the anomaly pipeline. These tests never touch
a live Prometheus: providers are stubbed and the conftest autouse fixture
redirects both stores to tmp_path and disables the wall-clock loop.
"""

from __future__ import annotations

import pytest

from app.services.system_rules import (
    SystemMetricProvider,
    SystemRuleService,
    SystemRuleStore,
    SystemThresholdCore,
)
from app.specs.anomaly_evaluation_report import AlertState
from app.specs.system_rule_spec import (
    SYSTEM_METRIC_CATALOG,
    SystemAlertRuleSpec,
)
from app.services.anomaly_core import DataStatus


def _rule(**overrides) -> SystemAlertRuleSpec:
    base = dict(
        id="sys-cpu-high",
        title="CPU high",
        metric_kind="cpu_utilization_pct",
        comparator=">",
        threshold=90.0,
    )
    base.update(overrides)
    return SystemAlertRuleSpec(**{("for" if k == "for_" else k): v
                                   for k, v in base.items()})


def _status(**overrides) -> DataStatus:
    base = dict(
        reachable=True,
        returns_data=True,
        source="prometheus",
        sample_ts=1000.0,
        history_coverage_s=0.0,
    )
    base.update(overrides)
    return DataStatus(**base)


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_spec_accepts_every_catalog_kind():
    for kind, entry in SYSTEM_METRIC_CATALOG.items():
        r = SystemAlertRuleSpec(
            id=f"sys-{kind.replace('_', '-')}",
            title=f"t {kind}",
            metric_kind=kind,
            comparator=">",
            threshold=entry.min_threshold,
        )
        assert r.mode == "shadow"
        assert r.source == "prometheus"


def test_spec_rejects_unknown_kind():
    with pytest.raises(Exception):
        _rule(metric_kind="rm_dash_rf_slash")


def test_spec_mode_is_structurally_shadow():
    with pytest.raises(Exception):
        _rule(mode="active")


def test_spec_rejects_out_of_range_threshold():
    with pytest.raises(Exception):
        _rule(threshold=9000.0)


def test_spec_rejects_out_of_range_for():
    with pytest.raises(Exception):
        _rule(for_="10s")
    with pytest.raises(Exception):
        _rule(for_="2h")


def test_spec_rejects_extra_fields():
    with pytest.raises(Exception):
        SystemAlertRuleSpec(
            id="sys-x", title="t", metric_kind="load1", comparator=">",
            threshold=1.0, raw_promql="up",
        )


# ---------------------------------------------------------------------------
# Threshold core — gate first, then compare, then duration machine
# ---------------------------------------------------------------------------


def test_core_gate_runs_first_mock_source_never_fires():
    core = SystemThresholdCore(_rule(), staleness_budget_s=60)
    # Value is way over threshold, but the source is mock: SIGNAL_LOST.
    rpt = core.evaluate(
        now=1000.0, value=99.0, status=_status(source="mock"),
    )
    assert rpt.state == AlertState.signal_lost
    assert rpt.events and rpt.events[0].paged is False


def test_core_stale_data_is_signal_lost():
    core = SystemThresholdCore(_rule(), staleness_budget_s=60)
    rpt = core.evaluate(
        now=2000.0, value=99.0, status=_status(sample_ts=1000.0),
    )
    assert rpt.state == AlertState.signal_lost
    assert rpt.events[0].signal_lost_reason.value == "stale"


def test_core_fires_only_after_sustained_for():
    core = SystemThresholdCore(_rule(for_="5m"), staleness_budget_s=3600)
    t0 = 10_000.0
    r1 = core.evaluate(now=t0, value=95.0, status=_status(sample_ts=t0))
    assert r1.state == AlertState.pending
    r2 = core.evaluate(
        now=t0 + 200, value=95.0, status=_status(sample_ts=t0 + 200),
    )
    assert r2.state == AlertState.pending
    r3 = core.evaluate(
        now=t0 + 301, value=95.0, status=_status(sample_ts=t0 + 301),
    )
    assert r3.state == AlertState.firing
    fire = [e for e in r3.events if e.state == AlertState.firing]
    # Shadow is structural: would_page recorded, paged never.
    assert fire and fire[0].would_page is True and fire[0].paged is False
    assert fire[0].suppressed_reason == "shadow_mode"


def test_core_resolves_on_recovery():
    core = SystemThresholdCore(_rule(for_="5m"), staleness_budget_s=3600)
    t0 = 10_000.0
    core.evaluate(now=t0, value=95.0, status=_status(sample_ts=t0))
    rpt = core.evaluate(
        now=t0 + 60, value=50.0, status=_status(sample_ts=t0 + 60),
    )
    assert rpt.state == AlertState.resolved


def test_core_below_comparator():
    core = SystemThresholdCore(
        _rule(id="sys-load-low", metric_kind="load1", comparator="<",
              threshold=0.5),
        staleness_budget_s=3600,
    )
    rpt = core.evaluate(now=1.0e4, value=0.2, status=_status(sample_ts=1.0e4))
    assert rpt.state == AlertState.pending


# ---------------------------------------------------------------------------
# Store round-trip + service tick
# ---------------------------------------------------------------------------


def test_rule_store_round_trip(tmp_path):
    store = SystemRuleStore(base_dir=tmp_path)
    store.save(_rule())
    loaded = store.load_all()
    assert len(loaded) == 1 and loaded[0].id == "sys-cpu-high"
    assert store.load_errors == []


def test_rule_store_surfaces_corrupt_files(tmp_path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    store = SystemRuleStore(base_dir=tmp_path)
    assert store.load_all() == []
    assert store.load_errors and "bad.json" in store.load_errors[0]


class _StubProvider:
    def __init__(self, value: float, *, source: str = "prometheus"):
        self.value = value
        self.source = source
        self.now = 10_000.0

    def observe(self, rule):
        return self.value, _status(
            source=self.source, sample_ts=self.now,
            reachable=self.source == "prometheus",
        )


def test_service_tick_records_durable_state(tmp_path):
    from app.services.alert_state_store import AlertStateStore

    provider = _StubProvider(95.0)
    svc = SystemRuleService(
        rule_store=SystemRuleStore(base_dir=tmp_path / "rules"),
        provider=provider,
        state_store=AlertStateStore(base_dir=tmp_path / "state"),
        staleness_budget_s=3600,
    )
    svc.add_rule(_rule(for_="5m"))
    reports = svc.tick(provider.now)
    assert reports["sys-cpu-high"].state == AlertState.pending
    st = svc.state_store.get("sys-cpu-high")
    assert st.state == AlertState.pending.value
    # Sustain past the for-window: firing lands in the durable history.
    provider.now += 400
    reports = svc.tick(provider.now)
    assert reports["sys-cpu-high"].state == AlertState.firing
    events = [
        r for r in svc.state_store.history_records()
        if r.get("state") == "firing"
    ]
    assert events and events[0]["paged"] is False


def test_service_status_is_shadow_and_non_diagnostic(tmp_path):
    from app.services.alert_state_store import AlertStateStore

    svc = SystemRuleService(
        rule_store=SystemRuleStore(base_dir=tmp_path / "rules"),
        provider=_StubProvider(10.0),
        state_store=AlertStateStore(base_dir=tmp_path / "state"),
    )
    svc.add_rule(_rule())
    s = svc.status()
    assert s["mode"] == "shadow"
    assert s["non_diagnostic"] is True
    assert len(s["rules"]) == 1


# ---------------------------------------------------------------------------
# Provider source-flag branching (CLAUDE.md rule for prometheus/client users)
# ---------------------------------------------------------------------------


def test_provider_reports_mock_source_honestly(monkeypatch):
    class _MockClient:
        def query(self, expr):
            return {"source": "mock", "data": {"result": [
                {"value": [0, "99"]}
            ]}}

    provider = SystemMetricProvider(client=_MockClient())
    value, status = provider.observe(_rule())
    assert status.source == "mock"
    assert status.reachable is False
    # And the core turns that into SIGNAL_LOST, never a verdict.
    core = SystemThresholdCore(_rule(), staleness_budget_s=3600)
    rpt = core.evaluate(now=1000.0, value=value, status=status)
    assert rpt.state == AlertState.signal_lost


# ---------------------------------------------------------------------------
# Chat flow — mock runtime end-to-end
# ---------------------------------------------------------------------------


def test_chat_creates_shadow_rule_end_to_end(monkeypatch):
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)

    from app.helper.orchestrator import Orchestrator
    from app.services import system_rules

    orch = Orchestrator()
    res = orch.handle_user_message(
        session_id="s-alert-rule",
        message="alert me when cpu is above 90% for 5 minutes",
    )
    assert res["intent_type"] == "SystemAlertRule"
    reply = res["user_reply"]
    assert "SHADOW" in reply
    assert "diagnosis" in reply  # decision-support disclaimer present

    rules = system_rules.get_service().rules()
    assert len(rules) == 1
    r = rules[0]
    assert r.metric_kind == "cpu_utilization_pct"
    assert r.comparator == ">"
    assert r.threshold == 90.0
    assert r.for_ == "5m"
    assert r.mode == "shadow"


def test_chat_disk_below_variant(monkeypatch):
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)

    from app.helper.orchestrator import Orchestrator
    from app.services import system_rules

    orch = Orchestrator()
    res = orch.handle_user_message(
        session_id="s-alert-rule-2",
        message="notify me if disk io is above 80% for 10 minutes",
    )
    assert res["intent_type"] == "SystemAlertRule"
    r = system_rules.get_service().rules()[0]
    assert r.metric_kind == "disk_io_utilization_pct"
    assert r.threshold == 80.0
    assert r.for_ == "10m"


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_api_create_list_and_alerts():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        r = client.post("/api/system-rules/rules", json={
            "id": "sys-mem-high", "title": "Memory high",
            "metric_kind": "memory_used_pct", "comparator": ">",
            "threshold": 85.0, "for": "5m",
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["rule"]["mode"] == "shadow"
        assert body["non_diagnostic"] is True

        # mode=active must be rejected by validation, not silently coerced.
        r2 = client.post("/api/system-rules/rules", json={
            "id": "sys-bad", "title": "t",
            "metric_kind": "memory_used_pct", "comparator": ">",
            "threshold": 85.0, "mode": "active",
        })
        assert r2.status_code == 422

        st = client.get("/api/system-rules").json()
        assert st["mode"] == "shadow"
        assert any(x["id"] == "sys-mem-high" for x in st["rules"])

        alerts = client.get("/api/system-rules/alerts").json()
        assert alerts["non_diagnostic"] is True

        cat = client.get("/api/system-rules/catalog").json()
        assert {c["kind"] for c in cat["catalog"]} == set(SYSTEM_METRIC_CATALOG)
