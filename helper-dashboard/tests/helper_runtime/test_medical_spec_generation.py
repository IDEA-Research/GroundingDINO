"""Catalog-backed spec generation through the mock runtime.

The clinician's "幫我盯心率和血氧" must produce widgets that query REAL
patient metrics with the mandatory `max by (patient_id)` aggregation —
never http_requests_total defaults, never silent fallbacks. The live
catalog is stubbed suite-wide in tests/conftest.py (Tier 0, no network).
"""

from __future__ import annotations

import json

import pytest

from app.helper.orchestrator import Orchestrator
from app.helper.runtime import MockHelperRuntime, _resolve_metric_hints
from app.prometheus import catalog as catalog_mod
from app.prometheus.catalog import MetricCatalog
from app.prometheus.client import PrometheusUnavailable
from app.services.spec_validator import SpecValidator


def _chat(message: str) -> dict:
    return MockHelperRuntime().invoke(
        "helper-chat-agent", {"message": message}
    )


def _generate(intent: dict) -> dict:
    out = MockHelperRuntime().invoke(
        "dashboard-spec-agent", {"intent": intent}
    )
    assert out["type"] == "DashboardSpec"
    return out["spec"]


def _all_promql(spec: dict) -> list[str]:
    return [
        (w.get("query") or {}).get("promql", "")
        for w in spec.get("widgets") or []
    ]


# ---------------------------------------------------------------------------
# 幫我盯心率和血氧 — the end-to-end medical path
# ---------------------------------------------------------------------------


def test_chinese_vitals_message_classifies_as_dashboard_intent():
    intent = _chat("幫我盯心率和血氧")
    assert intent["type"] == "DashboardIntent"
    hints = intent["requirements"]["metrics_hints"]
    assert "patient_heart_rate_bpm" in hints
    assert "patient_spo2_percentage" in hints
    assert intent["requirements"]["metric_catalog_available"] is True


def test_chinese_vitals_generate_uses_patient_metrics_with_aggregation():
    intent = _chat("幫我盯心率和血氧")
    spec = _generate(intent)
    promql = _all_promql(spec)
    assert any(
        "max by (patient_id) (patient_heart_rate_bpm)" in q for q in promql
    ), promql
    assert any(
        "max by (patient_id) (patient_spo2_percentage)" in q for q in promql
    ), promql
    # Never the old fabricated default.
    assert "http_requests_total" not in json.dumps(spec)
    SpecValidator().validate_dashboard(spec)  # raises on failure


def test_english_vitals_equivalent():
    intent = _chat("watch heart rate and spo2")
    spec = _generate(intent)
    blob = json.dumps(spec)
    assert "max by (patient_id) (patient_heart_rate_bpm)" in blob
    assert "max by (patient_id) (patient_spo2_percentage)" in blob


def test_vitals_trio_expands_to_three_metrics():
    intent = _chat("幫我盯生命徵象")
    spec = _generate(intent)
    blob = json.dumps(spec)
    for metric in (
        "patient_heart_rate_bpm",
        "patient_spo2_percentage",
        "patient_respiration_rate_per_minute",
    ):
        assert f"max by (patient_id) ({metric})" in blob


# ---------------------------------------------------------------------------
# System metrics still resolve to real names (catalog-verified)
# ---------------------------------------------------------------------------


def test_watch_cpu_and_memory_uses_real_system_metrics():
    intent = _chat("watch cpu and memory")
    assert intent["type"] == "DashboardIntent"
    hints = intent["requirements"]["metrics_hints"]
    assert "process_cpu_seconds_total" in hints
    assert "node_cpu_seconds_total" in hints
    assert "process_resident_memory_bytes" in hints
    spec = _generate(intent)
    blob = json.dumps(spec)
    assert "http_requests_total" not in blob
    assert "node_" in blob or "process_" in blob
    SpecValidator().validate_dashboard(spec)


def test_system_hint_absent_from_catalog_is_dropped(monkeypatch):
    # cpu synonyms are presence-verified: a catalog without the node/
    # process cpu metrics must not produce those hints.
    monkeypatch.setattr(
        MetricCatalog, "_fetch_names", lambda self: ["up"]
    )
    catalog_mod.get_catalog().invalidate()
    hints, available = _resolve_metric_hints("watch cpu")
    assert available is True
    assert hints == []


def test_exact_metric_name_mention_is_honored():
    hints, available = _resolve_metric_hints("watch node_load1 for me")
    assert available is True
    assert "node_load1" in hints


# ---------------------------------------------------------------------------
# No match → explicit placeholder, no fabricated metrics
# ---------------------------------------------------------------------------


def test_no_match_yields_labeled_placeholder_not_http_defaults():
    intent = _chat("幫我盯外面的天氣")
    assert intent["type"] == "DashboardIntent"
    assert intent["requirements"]["metrics_hints"] == []
    spec = _generate(intent)
    blob = json.dumps(spec)
    assert "http_requests_total" not in blob
    widgets = spec["widgets"]
    assert len(widgets) == 1
    w = widgets[0]
    assert w["query"]["promql"] == "vector(0)"
    assert "No matching metric" in w["description"]
    assert "intentionally reads 0" in w["description"]
    SpecValidator().validate_dashboard(spec)


# ---------------------------------------------------------------------------
# Catalog unavailable → explicit degradation, never silent defaults
# ---------------------------------------------------------------------------


def _break_catalog(monkeypatch):
    def _raise(self):
        raise PrometheusUnavailable("catalog down (test)")

    monkeypatch.setattr(MetricCatalog, "_fetch_names", _raise)
    catalog_mod.get_catalog().invalidate()


def test_catalog_unavailable_is_flagged_in_intent(monkeypatch):
    _break_catalog(monkeypatch)
    intent = _chat("幫我盯心率和血氧")
    assert intent["type"] == "DashboardIntent"
    assert intent["requirements"]["metrics_hints"] == []
    assert intent["requirements"]["metric_catalog_available"] is False


def test_catalog_unavailable_note_lands_in_spec(monkeypatch):
    _break_catalog(monkeypatch)
    intent = _chat("幫我盯心率和血氧")
    spec = _generate(intent)
    assert "metric catalog unavailable" in spec["description"]
    blob = json.dumps(spec)
    assert "http_requests_total" not in blob
    assert spec["widgets"][0]["query"]["promql"] == "vector(0)"
    assert "unavailable" in spec["widgets"][0]["description"]
    SpecValidator().validate_dashboard(spec)


# ---------------------------------------------------------------------------
# prometheus-agent suggestions come from the catalog
# ---------------------------------------------------------------------------


def test_prometheus_agent_suggests_from_catalog():
    out = MockHelperRuntime().invoke(
        "prometheus-agent", {"question": "心率"}
    )
    assert out["type"] == "PrometheusQueryReport"
    metrics = [
        s["metric"] for s in out["report"]["metric_suggestions"]
    ]
    assert "patient_heart_rate_bpm" in metrics


def test_prometheus_agent_reports_catalog_unavailable(monkeypatch):
    _break_catalog(monkeypatch)
    out = MockHelperRuntime().invoke(
        "prometheus-agent", {"question": "心率"}
    )
    assert out["report"]["metric_suggestions"] == []
    assert "metric catalog unavailable" in out["report"]["notes"]


# ---------------------------------------------------------------------------
# Golden api-observability widgets: annotate absent metrics, keep shape
# ---------------------------------------------------------------------------


def test_api_observability_annotates_missing_metrics(monkeypatch):
    monkeypatch.setattr(
        MetricCatalog,
        "_fetch_names",
        lambda self: ["up", "process_cpu_seconds_total",
                      "process_resident_memory_bytes"],
    )
    catalog_mod.get_catalog().invalidate()
    from app.helper.runtime import _api_observability_widgets

    widgets = {w["id"]: w for w in _api_observability_widgets()}
    assert len(widgets) == 8  # composition keeps its shape
    assert "(metric not present)" in widgets["w-request-rate"]["description"]
    assert "(metric not present)" not in widgets["w-cpu"]["description"]
    assert "(metric not present)" not in widgets["w-availability"]["description"]


def test_api_observability_no_annotation_when_catalog_down(monkeypatch):
    # Unknown is not absent: unreachable catalog → no annotation.
    _break_catalog(monkeypatch)
    from app.helper.runtime import _api_observability_widgets

    widgets = _api_observability_widgets()
    assert len(widgets) == 8
    assert all(
        "(metric not present)" not in (w.get("description") or "")
        for w in widgets
    )


# ---------------------------------------------------------------------------
# Orchestrator ships the catalog to the runtime (LLM prompt ground truth)
# ---------------------------------------------------------------------------


class _CaptureRuntime:
    def __init__(self):
        self.calls = []

    def invoke_operation(self, operation, args, **kwargs):
        self.calls.append((operation, args))
        return {"type": "DashboardSpec", "spec": {}}


def _bare_orchestrator() -> tuple[Orchestrator, _CaptureRuntime]:
    orch = Orchestrator.__new__(Orchestrator)  # no stores, no side effects
    rt = _CaptureRuntime()
    orch._runtime = rt
    orch._catalog = catalog_mod.get_catalog()
    return orch, rt


def test_dispatch_injects_metric_catalog_for_generate():
    orch, rt = _bare_orchestrator()
    orch._dispatch(
        {"type": "DashboardIntent", "requirements": {}},
        current_dashboard=None,
    )
    operation, args = rt.calls[0]
    assert operation == "generate_dashboard"
    mc = args["metric_catalog"]
    assert mc["available"] is True
    assert "patient_heart_rate_bpm" in mc["metrics"]
    assert mc["fetched_at"]


def test_metric_catalog_args_reports_unavailable(monkeypatch):
    _break_catalog(monkeypatch)
    orch, _rt = _bare_orchestrator()
    mc = orch._metric_catalog_args()
    assert mc["available"] is False
    assert mc["metrics"] == []


def test_metric_catalog_args_caps_at_500_keeping_clinical(monkeypatch):
    many = [f"zz_system_metric_{i:04d}" for i in range(600)]
    many += ["patient_heart_rate_bpm", "ventilator_fio2_percentage"]
    monkeypatch.setattr(
        MetricCatalog, "_fetch_names", lambda self: sorted(many)
    )
    catalog_mod.get_catalog().invalidate()
    orch, _rt = _bare_orchestrator()
    mc = orch._metric_catalog_args()
    assert mc["available"] is True
    assert len(mc["metrics"]) == 500
    assert "patient_heart_rate_bpm" in mc["metrics"]
    assert "ventilator_fio2_percentage" in mc["metrics"]
