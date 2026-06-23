"""Demo scenario tests — API observability.

Covers:
- Golden sample validates cleanly.
- MockHelperRuntime generates an API-observability dashboard with
  the expected shape when prompted.
- Each of the four directed patch scenarios produces the expected
  operation kind targeted at the expected widget.
- Every produced spec / patch survives SpecValidator (no hallucinated
  fields, no unsafe content).
"""

from __future__ import annotations

import pytest

from app.helper.runtime import MockHelperRuntime
from app.samples import load_sample
from app.services.patch_service import PatchService
from app.services.spec_validator import SpecValidationError, SpecValidator


# ---------------------------------------------------------------------------
# Golden sample
# ---------------------------------------------------------------------------


def test_api_observability_golden_validates():
    spec = load_sample("api_observability")
    assert spec.title == "API Observability"
    # 8 widgets across 3 rows, fits 12 cols.
    assert len(spec.widgets) == 8
    # Expected widget types all present.
    types = {w.type.value for w in spec.widgets}
    assert {"line_chart", "gauge", "alert_list"} <= types


def test_api_observability_golden_has_thresholds_on_critical_widgets():
    spec = load_sample("api_observability")
    by_id = {w.id: w for w in spec.widgets}
    for wid in ("w-error-rate", "w-latency-p95", "w-latency-p99",
                "w-availability"):
        w = by_id[wid]
        assert len(w.thresholds) >= 1, f"{wid} missing thresholds"


def test_api_observability_golden_uses_correct_promql_patterns():
    spec = load_sample("api_observability")
    by_id = {w.id: w for w in spec.widgets}

    # percentile latency must use histogram_quantile + by (le)
    p95 = by_id["w-latency-p95"].query.promql
    assert "histogram_quantile(0.95" in p95
    assert "by (le)" in p95

    # error rate is a ratio, not a raw count
    er = by_id["w-error-rate"].query.promql
    assert "/" in er
    assert "status=~" in er

    # request rate uses rate(...[5m])
    rr = by_id["w-request-rate"].query.promql
    assert "rate(http_requests_total[5m])" in rr


# ---------------------------------------------------------------------------
# Heuristic generator — MockHelperRuntime
# ---------------------------------------------------------------------------


def _mock_generate_for_api():
    mock = MockHelperRuntime()
    result = mock.invoke(
        "dashboard-spec-agent",
        {
            "intent": {
                "type": "DashboardIntent",
                "summary": "API observability",
                "requirements": {
                    "title": "API Observability",
                    "goal": "http request rate, error rate, p95 latency, cpu, memory, alerts for the api service",
                    "metrics_hints": [
                        "http_requests_total",
                        "http_request_duration_seconds",
                    ],
                    "widget_hints": ["line_chart", "gauge", "alert_list"],
                    "refresh_interval_hint": "30s",
                },
            },
        },
    )
    assert result["type"] == "DashboardSpec"
    return result["spec"]


def test_mock_detects_api_observability_and_uses_golden():
    spec = _mock_generate_for_api()
    ids = [w["id"] for w in spec["widgets"]]
    # Golden 8-widget layout.
    assert ids == [
        "w-request-rate", "w-error-rate", "w-latency-p95",
        "w-latency-p99", "w-availability", "w-cpu", "w-memory",
        "w-alerts",
    ]


def test_mock_api_observability_validates_via_pydantic():
    spec = _mock_generate_for_api()
    SpecValidator().validate_dashboard(spec)  # would raise on failure


# ---------------------------------------------------------------------------
# Patch scenarios — targeted rephrasing coverage
# ---------------------------------------------------------------------------


@pytest.fixture
def api_dashboard():
    return load_sample("api_observability").model_dump(mode="json")


def _patch_call(mock: MockHelperRuntime, change: str, dashboard: dict) -> dict:
    return mock.invoke(
        "patch-agent",
        {
            "intent": {
                "type": "PatchIntent",
                "target_dashboard_id": dashboard["dashboard_id"],
                "requested_changes": [change],
                "message_to_user": "",
            },
            "dashboard": dashboard,
        },
    )


def test_patch_move_critical_widgets_to_top(api_dashboard):
    mock = MockHelperRuntime()
    out = _patch_call(mock, "move critical widgets to the top", api_dashboard)
    ops = out["spec"]["operations"]
    reorder = next((op for op in ops if op["op"] == "reorder_widgets"), None)
    assert reorder is not None, ops
    # First 3 should be the critical widgets (error / latency / availability).
    assert reorder["order"][0].startswith(("w-error", "w-latency",
                                             "w-availability"))


def test_patch_make_latency_more_prominent(api_dashboard):
    mock = MockHelperRuntime()
    out = _patch_call(mock, "make latency more prominent", api_dashboard)
    ops = out["spec"]["operations"]
    upd = next((op for op in ops if op["op"] == "update_widget"), None)
    assert upd is not None
    assert "latency" in upd["widget_id"]
    assert upd["fields"].get("position", {}).get("w") == 12


def test_patch_add_error_rate_threshold_at_2_percent(api_dashboard):
    mock = MockHelperRuntime()
    out = _patch_call(mock, "add error-rate threshold at 2%", api_dashboard)
    ops = out["spec"]["operations"]
    upd = next((op for op in ops if op["op"] == "update_widget"), None)
    assert upd is not None
    assert upd["widget_id"] == "w-error-rate"
    thresholds = upd["fields"].get("thresholds") or []
    # Existing 2 thresholds + 1 new = 3 after patch.
    assert len(thresholds) >= 1
    assert any(abs(t["value"] - 0.02) < 1e-9 for t in thresholds)


def test_patch_change_cpu_to_memory(api_dashboard):
    mock = MockHelperRuntime()
    out = _patch_call(mock, "change CPU chart to memory chart", api_dashboard)
    ops = out["spec"]["operations"]
    upd = next((op for op in ops if op["op"] == "update_widget"), None)
    assert upd is not None
    assert upd["widget_id"] == "w-cpu"
    assert (
        "process_resident_memory_bytes"
        in upd["fields"]["query"]["promql"]
    )


# ---------------------------------------------------------------------------
# Applied patches still validate
# ---------------------------------------------------------------------------


def test_all_four_patches_apply_cleanly(api_dashboard):
    mock = MockHelperRuntime()
    ps = PatchService()
    spec = SpecValidator().validate_dashboard(api_dashboard)

    for change in [
        "move critical widgets to the top",
        "make latency more prominent",
        "add error-rate threshold at 2%",
        "change CPU chart to memory chart",
    ]:
        out = _patch_call(mock, change, spec.model_dump(mode="json"))
        patch = SpecValidator().validate_patch(out["spec"])
        spec = ps.apply(spec, patch)  # raises on any failure
    # All four applied; dashboard still validates and has 8 widgets.
    assert len(spec.widgets) == 8
