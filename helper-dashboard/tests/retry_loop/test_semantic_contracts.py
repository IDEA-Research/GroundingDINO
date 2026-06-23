"""Tests for semantic contracts used by the retry loop."""

from __future__ import annotations

import pytest

from app.helper.semantic_contracts import (
    AddThresholdContract,
    AnyValidGenerateContract,
    AnyValidPatchContract,
    ApiObservabilityGenerateContract,
    ChangeChartContract,
    MakeProminentContract,
    MoveCriticalToTopContract,
    RemoveWidgetContract,
    contract_for_generate,
    contract_for_patch,
)


def _patch(ops: list[dict]) -> dict:
    return {
        "type": "PatchSpec",
        "spec": {
            "patch_id": "p1",
            "reason": "r",
            "target_dashboard_id": "d",
            "created_by": "patch-agent",
            "operations": ops,
        },
    }


def _dashboard(widgets: list[dict]) -> dict:
    return {
        "type": "DashboardSpec",
        "spec": {
            "dashboard_id": "d",
            "title": "T",
            "description": "",
            "layout": {"columns": 12, "row_height": 40},
            "variables": [],
            "widgets": widgets,
            "refresh_interval": "30s",
        },
    }


def _args(requested: str, dashboard: dict | None = None) -> dict:
    return {
        "intent": {"requested_changes": [requested]},
        "dashboard": dashboard or {"dashboard_id": "d", "widgets": []},
    }


# ---------------------------------------------------------------------------
# AnyValidPatchContract
# ---------------------------------------------------------------------------


def test_any_valid_patch_accepts_valid_output():
    c = AnyValidPatchContract()
    r = c.check(_patch([{"op": "update_dashboard", "fields": {"title": "X"}}]),
                _args("rename"))
    assert r.ok
    assert "update_dashboard" in r.op_summary[0]


def test_any_valid_patch_rejects_unknown_operation():
    c = AnyValidPatchContract()
    r = c.check(_patch([{"op": "shell_exec", "cmd": "ls"}]), _args("whatever"))
    assert not r.ok
    assert any("not in the allowed set" in e for e in r.errors)


def test_any_valid_patch_rejects_unsupported_widget_type():
    # `sankey` is intentionally not in WidgetType. If the auto-extend
    # pipeline ever adds it, pick a fresh name that isn't cached yet.
    c = AnyValidPatchContract()
    r = c.check(
        _patch([{"op": "add_widget", "widget": {"id": "w1", "type": "sankey"}}]),
        _args("add a sankey"),
    )
    assert not r.ok
    assert any("allowed widget types" in e for e in r.errors)


def test_any_valid_patch_rejects_wrong_envelope():
    c = AnyValidPatchContract()
    r = c.check({"type": "DashboardSpec"}, _args("x"))
    assert not r.ok


# ---------------------------------------------------------------------------
# MakeProminentContract
# ---------------------------------------------------------------------------


def test_make_prominent_accepts_update_on_target_with_big_width():
    c = MakeProminentContract("latency")
    out = _patch([{
        "op": "update_widget",
        "widget_id": "w-latency-p95",
        "fields": {"position": {"x": 0, "y": 0, "w": 12, "h": 6}},
    }])
    r = c.check(out, _args("make latency more prominent"))
    assert r.ok


def test_make_prominent_rejects_if_no_update_widget():
    c = MakeProminentContract("latency")
    out = _patch([{"op": "reorder_widgets", "order": ["a", "b"]}])
    r = c.check(out, _args("make latency more prominent"))
    assert not r.ok
    assert any("update_widget" in e and "latency" in e for e in r.errors)


def test_make_prominent_rejects_if_target_is_wrong_widget():
    c = MakeProminentContract("latency")
    out = _patch([{
        "op": "update_widget",
        "widget_id": "w-cpu",
        "fields": {"position": {"x": 0, "y": 0, "w": 12, "h": 6}},
    }])
    r = c.check(out, _args("make latency more prominent"))
    assert not r.ok


def test_make_prominent_rejects_if_not_actually_prominent():
    c = MakeProminentContract("latency")
    out = _patch([{
        "op": "update_widget",
        "widget_id": "w-latency-p95",
        "fields": {"position": {"x": 6, "y": 6, "w": 3, "h": 4}},
    }])
    r = c.check(out, _args("make latency more prominent"))
    assert not r.ok
    assert any("more prominent" in e for e in r.errors)


# ---------------------------------------------------------------------------
# MoveCriticalToTopContract
# ---------------------------------------------------------------------------


_SAMPLE_DASHBOARD = {
    "dashboard_id": "d",
    "widgets": [
        {"id": "w-request-rate", "type": "line_chart", "title": "request rate"},
        {"id": "w-error-rate", "type": "line_chart", "title": "error rate"},
        {"id": "w-latency-p95", "type": "line_chart", "title": "latency p95"},
        {"id": "w-cpu", "type": "line_chart", "title": "cpu"},
        {"id": "w-alerts", "type": "alert_list", "title": "alerts"},
    ],
}


def test_move_critical_accepts_when_critical_first():
    c = MoveCriticalToTopContract()
    out = _patch([{
        "op": "reorder_widgets",
        "order": ["w-error-rate", "w-latency-p95", "w-request-rate",
                   "w-cpu", "w-alerts"],
    }])
    r = c.check(out, _args("move critical to top", _SAMPLE_DASHBOARD))
    assert r.ok


def test_move_critical_rejects_when_non_critical_first():
    c = MoveCriticalToTopContract()
    out = _patch([{
        "op": "reorder_widgets",
        "order": ["w-cpu", "w-request-rate", "w-error-rate",
                   "w-latency-p95", "w-alerts"],
    }])
    r = c.check(out, _args("move critical to top", _SAMPLE_DASHBOARD))
    assert not r.ok


def test_move_critical_rejects_when_no_reorder_op():
    c = MoveCriticalToTopContract()
    out = _patch([{"op": "update_dashboard", "fields": {"title": "x"}}])
    r = c.check(out, _args("move critical", _SAMPLE_DASHBOARD))
    assert not r.ok


# ---------------------------------------------------------------------------
# AddThresholdContract
# ---------------------------------------------------------------------------


def test_add_threshold_accepts_correct_value():
    c = AddThresholdContract(target_keyword="error", expected_value=0.02,
                              raw_value_label="2%")
    out = _patch([{
        "op": "update_widget",
        "widget_id": "w-error-rate",
        "fields": {"thresholds": [{"value": 0.02, "color": "#f59e0b",
                                     "label": "warn"}]},
    }])
    r = c.check(out, _args("add error threshold 2%"))
    assert r.ok


def test_add_threshold_rejects_wrong_value():
    c = AddThresholdContract(target_keyword="error", expected_value=0.02,
                              raw_value_label="2%")
    out = _patch([{
        "op": "update_widget",
        "widget_id": "w-error-rate",
        "fields": {"thresholds": [{"value": 0.05, "color": "#f59e0b"}]},
    }])
    r = c.check(out, _args("add error threshold 2%"))
    assert not r.ok


def test_add_threshold_rejects_wrong_target():
    c = AddThresholdContract(target_keyword="error", expected_value=0.02,
                              raw_value_label="2%")
    out = _patch([{
        "op": "update_widget",
        "widget_id": "w-cpu",
        "fields": {"thresholds": [{"value": 0.02, "color": "#f59e0b"}]},
    }])
    r = c.check(out, _args("add error threshold 2%"))
    assert not r.ok


# ---------------------------------------------------------------------------
# ChangeChartContract
# ---------------------------------------------------------------------------


def test_change_chart_accepts_cpu_to_memory():
    c = ChangeChartContract(
        from_keyword="cpu", to_keyword="memory",
        expected_promql_substrings=["process_resident_memory_bytes"],
    )
    out = _patch([{
        "op": "update_widget",
        "widget_id": "w-cpu",
        "fields": {"query": {"promql": "process_resident_memory_bytes"}},
    }])
    r = c.check(out, _args("change cpu chart to memory chart"))
    assert r.ok


def test_change_chart_rejects_when_promql_unchanged():
    c = ChangeChartContract(
        from_keyword="cpu", to_keyword="memory",
        expected_promql_substrings=["process_resident_memory_bytes"],
    )
    out = _patch([{
        "op": "update_widget",
        "widget_id": "w-cpu",
        "fields": {"query": {"promql": "rate(process_cpu_seconds_total[5m])"}},
    }])
    r = c.check(out, _args("change cpu chart to memory chart"))
    assert not r.ok


# ---------------------------------------------------------------------------
# RemoveWidgetContract
# ---------------------------------------------------------------------------


def test_remove_widget_accepts_matching_id():
    c = RemoveWidgetContract(target_keyword="memory")
    out = _patch([{"op": "remove_widget", "widget_id": "w-memory"}])
    r = c.check(out, _args("remove the memory widget"))
    assert r.ok


def test_remove_widget_rejects_wrong_id():
    c = RemoveWidgetContract(target_keyword="memory")
    out = _patch([{"op": "remove_widget", "widget_id": "w-cpu"}])
    r = c.check(out, _args("remove the memory widget"))
    assert not r.ok


# ---------------------------------------------------------------------------
# Generate contracts
# ---------------------------------------------------------------------------


def test_any_valid_generate_requires_widgets():
    c = AnyValidGenerateContract()
    assert not c.check(_dashboard([]), {}).ok
    assert c.check(
        _dashboard([{
            "id": "w1", "type": "line_chart", "title": "x",
            "query": {"promql": "up"}, "position": {}, "encoding": {},
            "thresholds": [], "options": {},
        }]), {},
    ).ok


def test_any_valid_generate_rejects_bad_widget_type():
    # `sankey` is intentionally not in WidgetType. If the auto-extend
    # pipeline ever adds it, pick a fresh name that isn't cached yet.
    c = AnyValidGenerateContract()
    out = _dashboard([{"id": "w1", "type": "sankey", "title": "x"}])
    assert not c.check(out, {}).ok


def test_api_obs_requires_min_widgets_and_promql_patterns():
    c = ApiObservabilityGenerateContract(min_widgets=6)
    # Below threshold.
    out = _dashboard([
        {"id": f"w{i}", "type": "line_chart", "title": f"x{i}",
         "query": {"promql": "up"}}
        for i in range(3)
    ])
    assert not c.check(out, {}).ok

    # 6 widgets but no percentile, no threshold, no request rate.
    out = _dashboard([
        {"id": f"w{i}", "type": "line_chart", "title": f"x{i}",
         "query": {"promql": "up"}}
        for i in range(6)
    ])
    r = c.check(out, {})
    assert not r.ok
    msg = " ".join(r.errors)
    assert "percentile" in msg.lower() or "histogram_quantile" in msg
    assert "threshold" in msg.lower() or "request" in msg.lower()


def test_api_obs_accepts_well_formed_dashboard():
    c = ApiObservabilityGenerateContract(min_widgets=6)
    widgets = [
        {"id": "w-req", "type": "line_chart", "title": "r",
         "query": {"promql": "rate(http_requests_total[5m])"},
         "thresholds": []},
        {"id": "w-err", "type": "line_chart", "title": "err",
         "query": {"promql": "sum(rate(http_requests_total{status=~\"5..\"}[5m]))"},
         "thresholds": [{"value": 0.01, "color": "#f59e0b"}]},
        {"id": "w-p95", "type": "line_chart", "title": "p95",
         "query": {"promql": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))"},
         "thresholds": []},
        {"id": "w-up", "type": "gauge", "title": "up",
         "query": {"promql": "avg(up)"}, "thresholds": []},
        {"id": "w-cpu", "type": "line_chart", "title": "cpu",
         "query": {"promql": "rate(process_cpu_seconds_total[5m])"},
         "thresholds": []},
        {"id": "w-alerts", "type": "alert_list", "title": "alerts",
         "query": {"promql": "ALERTS"}, "thresholds": []},
    ]
    r = c.check(_dashboard(widgets), {})
    assert r.ok, r.errors


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def test_contract_for_patch_routes_make_prominent():
    c = contract_for_patch(["make latency more prominent"], _SAMPLE_DASHBOARD)
    assert c.name == "make_prominent"


def test_contract_for_patch_routes_reorder():
    c = contract_for_patch(["move critical widgets to the top"], _SAMPLE_DASHBOARD)
    assert c.name == "move_critical_to_top"


def test_contract_for_patch_routes_threshold():
    c = contract_for_patch(["add error-rate threshold at 2%"], _SAMPLE_DASHBOARD)
    assert c.name == "add_threshold"


def test_contract_for_patch_routes_swap():
    c = contract_for_patch(["change CPU chart to memory chart"], _SAMPLE_DASHBOARD)
    assert c.name == "change_chart"


def test_contract_for_generate_routes_api_obs():
    reqs = {
        "title": "API Observability",
        "goal": "http request rate, p95 latency, error rate",
        "metrics_hints": ["http_requests_total"],
    }
    c = contract_for_generate(reqs)
    assert c.name == "api_observability_generate"


def test_contract_for_generate_default():
    c = contract_for_generate({"title": "whatever"})
    assert c.name == "any_valid_generate"


# ---------------------------------------------------------------------------
# Feedback messages are deterministic and scrub secrets
# ---------------------------------------------------------------------------


def test_feedback_is_deterministic():
    c = MakeProminentContract("latency")
    args = _args("make latency more prominent")
    m1 = c.feedback_message(["err1", "err2"], args)
    m2 = c.feedback_message(["err1", "err2"], args)
    assert m1 == m2


def test_feedback_does_not_echo_secrets_passed_in_args():
    # Contract feedback uses requested_changes and its own static
    # strings — it must not include arbitrary args content.
    c = MakeProminentContract("latency")
    args = {
        "intent": {"requested_changes": ["make latency more prominent"]},
        "_prior_errors": ["OPENROUTER_API_KEY=sk-or-XYZ"],
    }
    msg = c.feedback_message(["sample error"], args)
    assert "sk-or-" not in msg
    assert "OPENROUTER_API_KEY" not in msg
