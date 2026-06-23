"""Tests for AgentValidationRetryLoop."""

from __future__ import annotations

import json
import os

import pytest

from app.helper.retry_loop import (
    AgentValidationRetryLoop,
    AttemptLog,
    resolve_max_attempts,
)
from app.helper.semantic_contracts import (
    AddThresholdContract,
    AnyValidPatchContract,
    ApiObservabilityGenerateContract,
    MakeProminentContract,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubRuntime:
    """Replays a queue of pre-set responses per operation."""

    def __init__(self, script: list[dict], runtime_used: str = "opencode"):
        self._q = list(script)
        self._runtime_used = runtime_used
        self.calls: list[tuple[str, dict]] = []

    def invoke_operation(self, operation, args, *, developer=False):
        self.calls.append((operation, dict(args)))
        if not self._q:
            raise AssertionError("stub runtime ran out of scripted responses")
        r = dict(self._q.pop(0))
        r.setdefault("runtime_used", self._runtime_used)
        return r


def _patch_result(ops: list[dict]) -> dict:
    return {
        "type": "PatchSpec",
        "spec": {
            "patch_id": "p1", "reason": "r",
            "target_dashboard_id": "d",
            "created_by": "patch-agent",
            "operations": ops,
        },
    }


def _dashboard_result(widgets: list[dict]) -> dict:
    return {
        "type": "DashboardSpec",
        "spec": {
            "dashboard_id": "d", "title": "T", "description": "",
            "layout": {"columns": 12, "row_height": 40},
            "variables": [], "widgets": widgets,
            "refresh_interval": "30s",
        },
    }


def _valid_update_on_latency() -> dict:
    return {
        "op": "update_widget",
        "widget_id": "w-latency-p95",
        "fields": {"position": {"x": 0, "y": 0, "w": 12, "h": 6}},
    }


# ---------------------------------------------------------------------------
# Directive-enumerated tests (the 7)
# ---------------------------------------------------------------------------


def test_first_attempt_bad_second_good_accepted_on_attempt_2():
    stub = _StubRuntime([
        _patch_result([{"op": "reorder_widgets", "order": ["a", "b"]}]),
        _patch_result([_valid_update_on_latency()]),
    ])
    loop = AgentValidationRetryLoop(stub, max_attempts=3, persist_logs=False)
    outcome = loop.run(
        "patch_dashboard",
        {"intent": {"requested_changes": ["make latency more prominent"]}},
        contract=MakeProminentContract("latency"),
    )
    assert outcome.accepted is not None
    assert outcome.attempts_used == 2
    assert outcome.attempts[0].accepted is False
    assert outcome.attempts[1].accepted is True
    # Feedback was injected into attempt 2.
    args_of_call_2 = stub.calls[1][1]
    assert args_of_call_2.get("_retry_attempt") == 2
    assert "_prior_feedback_message" in args_of_call_2


def test_all_attempts_invalid_returns_none_with_log():
    stub = _StubRuntime([
        _patch_result([{"op": "reorder_widgets", "order": ["a", "b"]}]),
        _patch_result([{"op": "reorder_widgets", "order": ["c", "d"]}]),
        _patch_result([{"op": "reorder_widgets", "order": ["e", "f"]}]),
    ])
    loop = AgentValidationRetryLoop(stub, max_attempts=3, persist_logs=False)
    outcome = loop.run(
        "patch_dashboard",
        {"intent": {"requested_changes": ["make latency more prominent"]}},
        contract=MakeProminentContract("latency"),
    )
    assert outcome.accepted is None
    assert outcome.attempts_used == 3
    for a in outcome.attempts:
        assert a.accepted is False
        assert a.errors  # every attempt has errors


def test_valid_first_attempt_no_retry():
    stub = _StubRuntime([_patch_result([_valid_update_on_latency()])])
    loop = AgentValidationRetryLoop(stub, max_attempts=3, persist_logs=False)
    outcome = loop.run(
        "patch_dashboard",
        {"intent": {"requested_changes": ["make latency more prominent"]}},
        contract=MakeProminentContract("latency"),
    )
    assert outcome.attempts_used == 1
    assert outcome.accepted is not None
    # The stub was called exactly once.
    assert len(stub.calls) == 1


def test_retry_does_not_bypass_schema_validation():
    """Even if the semantic contract would accept it, an output with
    a forbidden field must still fail through Pydantic."""
    forbidden = {
        "type": "PatchSpec",
        "spec": {
            "patch_id": "p1", "reason": "r",
            "target_dashboard_id": "d", "created_by": "patch-agent",
            "operations": [{
                "op": "add_widget",
                "widget": {
                    "id": "w1", "type": "line_chart", "title": "x",
                    "description": "",
                    "query": {"source": "prometheus", "promql": "up",
                               "query_type": "instant"},
                    "position": {"x": 0, "y": 0, "w": 6, "h": 6},
                    "encoding": {}, "thresholds": [],
                    # Forbidden:
                    "raw_html": "<script>x</script>",
                    "options": {},
                },
            }],
        },
    }
    stub = _StubRuntime([forbidden, forbidden, forbidden])
    loop = AgentValidationRetryLoop(stub, max_attempts=3, persist_logs=False)
    outcome = loop.run(
        "patch_dashboard",
        {"intent": {"requested_changes": ["add a chart"]}},
        contract=AnyValidPatchContract(),
    )
    assert outcome.accepted is None
    # Every attempt failed at the schema layer.
    for a in outcome.attempts:
        assert any("raw_html" in e or "forbidden" in e.lower()
                    or "Extra inputs" in e for e in a.errors), a.errors


def test_retry_cannot_accept_unsupported_widget_type():
    bad = _patch_result([{
        "op": "add_widget",
        "widget": {"id": "w1", "type": "heatmap", "title": "x"},
    }])
    stub = _StubRuntime([bad, bad, bad])
    loop = AgentValidationRetryLoop(stub, max_attempts=3, persist_logs=False)
    outcome = loop.run(
        "patch_dashboard",
        {"intent": {"requested_changes": ["add a heatmap"]}},
        contract=AnyValidPatchContract(),
    )
    assert outcome.accepted is None
    assert outcome.attempts_used == 3


def test_make_latency_more_prominent_accepted_on_attempt_3():
    stub = _StubRuntime([
        _patch_result([{"op": "reorder_widgets",
                         "order": ["a", "b", "c"]}]),
        _patch_result([{"op": "update_widget", "widget_id": "w-cpu",
                         "fields": {"position": {"x": 0, "y": 0,
                                                    "w": 12, "h": 6}}}]),
        _patch_result([_valid_update_on_latency()]),
    ])
    loop = AgentValidationRetryLoop(stub, max_attempts=3, persist_logs=False)
    outcome = loop.run(
        "patch_dashboard",
        {"intent": {"requested_changes": ["make latency more prominent"]}},
        contract=MakeProminentContract("latency"),
    )
    assert outcome.accepted is not None
    assert outcome.attempts_used == 3


def test_attempt_logs_mask_secrets(monkeypatch, tmp_path):
    """If a retry log somehow inherits secret-looking strings, they
    must be masked in the serialized artifact."""
    # Force the CLI to "fail" with a secret-looking stderr by having
    # the stub raise something that contains a key-like string.
    from app.helper.runtime import RuntimeError_

    class _LeakyRuntime:
        def invoke_operation(self, operation, args, *, developer=False):
            raise RuntimeError_(
                "runtime failure: OPENROUTER_API_KEY=sk-or-v1-secret123"
            )

    # Redirect persistence to tmp.
    import app.helper.retry_loop as rl
    monkeypatch.setattr(rl, "_RETRY_LOGS_DIR", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-secret123")
    monkeypatch.setenv("HELPER_DASHBOARD_PERSIST_RETRY_LOGS", "1")

    loop = AgentValidationRetryLoop(
        _LeakyRuntime(), max_attempts=2, persist_logs=True,
    )
    outcome = loop.run(
        "patch_dashboard",
        {"intent": {"requested_changes": ["anything"]}},
        contract=AnyValidPatchContract(),
    )
    # Find the written log file.
    logs = list(tmp_path.glob("*.json"))
    assert logs, "persisted log file expected"
    dump = logs[0].read_text()
    assert "sk-or-" not in dump, dump
    assert "secret123" not in dump
    # The in-memory log also scrubs.
    for a in outcome.attempts:
        for err in a.errors:
            assert "sk-or-" not in err
            assert "secret123" not in err


# ---------------------------------------------------------------------------
# Decision 6 — deterministic provider short-circuit
# ---------------------------------------------------------------------------


def test_mock_runtime_short_circuits_after_one_failed_attempt():
    stub = _StubRuntime(
        [_patch_result([{"op": "reorder_widgets", "order": ["a", "b"]}]),
         _patch_result([_valid_update_on_latency()])],
        runtime_used="mock",
    )
    loop = AgentValidationRetryLoop(stub, max_attempts=3, persist_logs=False)
    outcome = loop.run(
        "patch_dashboard",
        {"intent": {"requested_changes": ["make latency more prominent"]}},
        contract=MakeProminentContract("latency"),
    )
    # Short-circuit after attempt 1 failed and runtime_used=mock.
    assert outcome.attempts_used == 1
    assert outcome.accepted is None


def test_mock_runtime_accepts_on_first_attempt_does_not_retry():
    stub = _StubRuntime(
        [_patch_result([_valid_update_on_latency()])],
        runtime_used="mock",
    )
    loop = AgentValidationRetryLoop(stub, max_attempts=3, persist_logs=False)
    outcome = loop.run(
        "patch_dashboard",
        {"intent": {"requested_changes": ["make latency more prominent"]}},
        contract=MakeProminentContract("latency"),
    )
    assert outcome.attempts_used == 1
    assert outcome.accepted is not None


# ---------------------------------------------------------------------------
# Max-attempts clamping
# ---------------------------------------------------------------------------


def test_max_attempts_clamped_low(monkeypatch):
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "0")
    assert resolve_max_attempts() == 1


def test_max_attempts_clamped_high(monkeypatch):
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "99")
    assert resolve_max_attempts() == 5


def test_max_attempts_default_when_unset(monkeypatch):
    monkeypatch.delenv("HELPER_AGENT_MAX_ATTEMPTS", raising=False)
    assert resolve_max_attempts() == 3


def test_max_attempts_bad_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "not-a-number")
    assert resolve_max_attempts() == 3


# ---------------------------------------------------------------------------
# Generate contract — retry path for API observability
# ---------------------------------------------------------------------------


def test_generate_retry_accepts_once_api_obs_constraints_met():
    too_small = _dashboard_result([
        {"id": "w1", "type": "line_chart", "title": "x",
         "description": "",
         "query": {"source": "prometheus", "promql": "up",
                    "query_type": "instant"},
         "position": {"x": 0, "y": 0, "w": 6, "h": 6},
         "encoding": {}, "thresholds": [], "options": {}},
    ])
    good_widgets = [
        {"id": "w-req", "type": "line_chart", "title": "r",
         "description": "",
         "query": {"source": "prometheus",
                    "promql": "rate(http_requests_total[5m])",
                    "query_type": "range", "range": "1h", "step": "30s"},
         "position": {"x": 0, "y": 0, "w": 6, "h": 6},
         "encoding": {}, "thresholds": [], "options": {}},
        {"id": "w-err", "type": "line_chart", "title": "err",
         "description": "",
         "query": {"source": "prometheus",
                    "promql": 'sum(rate(http_requests_total{status=~"5.."}[5m]))',
                    "query_type": "range", "range": "1h", "step": "30s"},
         "position": {"x": 6, "y": 0, "w": 6, "h": 6},
         "encoding": {},
         "thresholds": [{"value": 0.01, "color": "#f59e0b"}], "options": {}},
        {"id": "w-p95", "type": "line_chart", "title": "p95",
         "description": "",
         "query": {"source": "prometheus",
                    "promql": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
                    "query_type": "range", "range": "1h", "step": "30s"},
         "position": {"x": 0, "y": 6, "w": 6, "h": 6},
         "encoding": {}, "thresholds": [], "options": {}},
        {"id": "w-up", "type": "gauge", "title": "up",
         "description": "",
         "query": {"source": "prometheus", "promql": "avg(up)",
                    "query_type": "instant"},
         "position": {"x": 0, "y": 12, "w": 3, "h": 4},
         "encoding": {}, "thresholds": [], "options": {}},
        {"id": "w-cpu", "type": "line_chart", "title": "cpu",
         "description": "",
         "query": {"source": "prometheus",
                    "promql": "rate(process_cpu_seconds_total[5m])",
                    "query_type": "range", "range": "1h", "step": "30s"},
         "position": {"x": 3, "y": 12, "w": 5, "h": 4},
         "encoding": {}, "thresholds": [], "options": {}},
        {"id": "w-alerts", "type": "alert_list", "title": "alerts",
         "description": "",
         "query": {"source": "prometheus", "promql": "ALERTS",
                    "query_type": "instant"},
         "position": {"x": 0, "y": 16, "w": 12, "h": 4},
         "encoding": {}, "thresholds": [], "options": {}},
    ]
    stub = _StubRuntime([too_small, _dashboard_result(good_widgets)])
    loop = AgentValidationRetryLoop(stub, max_attempts=3, persist_logs=False)
    outcome = loop.run(
        "generate_dashboard",
        {"intent": {"requirements": {"title": "API Observability",
                                       "goal": "api request rate p95"}}},
        contract=ApiObservabilityGenerateContract(min_widgets=6),
    )
    assert outcome.accepted is not None
    assert outcome.attempts_used == 2


# ---------------------------------------------------------------------------
# Terminal fallback envelope (M1)
# ---------------------------------------------------------------------------
#
# When a specialist agent emits a DeveloperTicket (allowed in
# EXPECTED_OUTPUT_TYPES for generate/patch operations), the retry loop
# must treat it as a deliberate "I can't satisfy this with the toolkit"
# signal and accept on the first attempt — not waste retries on the
# semantic contract that expects DashboardSpec/PatchSpec. Without this
# the orchestrator would never see the ticket, and a synthesized empty
# DashboardSpec would be returned instead.


def _developer_ticket(widget_type: str, source_agent: str) -> dict:
    return {
        "type": "DeveloperTicket",
        "source_agent": source_agent,
        "severity": "medium",
        "summary": f"{widget_type} widget type not in toolkit",
        "user_visible_effect": f"user asked for a {widget_type}",
        "technical_evidence": "WidgetType enum allows: line_chart, ...",
        "requested_action": f"extend widget toolkit with {widget_type}",
        "safety_notes": "",
    }


def test_generate_dashboard_developer_ticket_accepted_on_first_attempt():
    stub = _StubRuntime([
        _developer_ticket("pie_chart", "dashboard-spec-agent"),
    ])
    loop = AgentValidationRetryLoop(stub, max_attempts=3, persist_logs=False)
    outcome = loop.run(
        "generate_dashboard",
        {"intent": {"requirements": {"title": "Mem", "goal": "pie chart memory"}}},
        contract=ApiObservabilityGenerateContract(min_widgets=6),
    )
    assert outcome.accepted is not None
    assert outcome.accepted["type"] == "DeveloperTicket"
    assert outcome.accepted["requested_action"] == "extend widget toolkit with pie_chart"
    assert outcome.attempts_used == 1
    assert outcome.attempts[0].accepted is True
    assert outcome.attempts[0].output_type == "DeveloperTicket"
    # No semantic-contract retry triggered — runtime was called exactly once.
    assert len(stub.calls) == 1


def test_patch_dashboard_developer_ticket_accepted_on_first_attempt():
    stub = _StubRuntime([
        _developer_ticket("bar_chart", "patch-agent"),
    ])
    loop = AgentValidationRetryLoop(stub, max_attempts=3, persist_logs=False)
    outcome = loop.run(
        "patch_dashboard",
        {"intent": {"requested_changes": ["change cpu to bar chart"]}},
        contract=AnyValidPatchContract(),
    )
    assert outcome.accepted is not None
    assert outcome.accepted["type"] == "DeveloperTicket"
    assert outcome.attempts_used == 1
    assert outcome.attempts[0].accepted is True
    assert len(stub.calls) == 1


def test_developer_ticket_fallback_skips_semantic_contract_errors():
    """A DeveloperTicket envelope must NOT have any contract errors
    attached to its attempt log — it short-circuits the contract step."""
    stub = _StubRuntime([
        _developer_ticket("heatmap", "dashboard-spec-agent"),
    ])
    loop = AgentValidationRetryLoop(stub, max_attempts=3, persist_logs=False)
    outcome = loop.run(
        "generate_dashboard",
        {"intent": {"requirements": {"title": "X", "goal": "heatmap"}}},
        contract=ApiObservabilityGenerateContract(min_widgets=6),
    )
    assert outcome.attempts[0].errors == []
    assert outcome.attempts[0].op_summary == ["fallback_envelope=DeveloperTicket"]
