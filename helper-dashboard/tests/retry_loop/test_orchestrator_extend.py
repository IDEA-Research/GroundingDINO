"""Orchestrator extend-then-redispatch flow.

The widget toolkit is a cache of pre-built widgets, not a fence on
what can be built. When the specialist agent emits a DeveloperTicket
whose `requested_action` parses as a toolkit-extension request, the
orchestrator should:
  1. Invoke `rescue_extend` (Big guy in tool-using mode) — by default,
     no opt-in flag required.
  2. On a resolved DeveloperReport, re-dispatch the same intent ONCE.
  3. If the re-dispatch produces a clean DashboardSpec/PatchSpec,
     route through the normal handler; the user sees the new dashboard.
  4. If the second attempt still fails, fall through to an honest
     "tried to add X but the build didn't pass" reply — never the old
     "X isn't in the toolkit" framing.

Safety still applies — `extend_gate.py` enforces denylist, prompt-
injection check, daily quota, and audit log inside ExtendRunner.run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.helper.extend_runner import ExtendOutcome
from app.helper.orchestrator import Orchestrator, _PENDING_SAVE


def _reset_storage():
    root = Path("backend/app/storage")
    for sub in ("dashboards", "tickets", "saved_dashboards", "retry_logs"):
        d = root / sub
        if d.exists():
            for p in d.glob("*.json"):
                p.unlink()
    _PENDING_SAVE.clear()


class _ScriptedRuntime:
    def __init__(self):
        self.queues: dict[str, list[dict]] = {}
        self.calls: list[tuple[str, dict]] = []

    def enqueue(self, operation: str, result: dict):
        self.queues.setdefault(operation, []).append(result)

    def invoke_operation(self, operation, args, *, developer=False):
        self.calls.append((operation, dict(args)))
        q = self.queues.get(operation) or []
        if not q:
            return {"type": "UserResponse", "message": "noop",
                    "runtime_used": "mock"}
        r = dict(q.pop(0))
        r.setdefault("runtime_used", "opencode")
        return r


def _intent_dashboard() -> dict:
    return {
        "type": "DashboardIntent",
        "summary": "s",
        "requirements": {
            "title": "Mem by host", "goal": "pie chart of memory",
            "metrics_hints": [],
            "widget_hints": ["pie_chart"],
            "refresh_interval_hint": "30s",
        },
        "clarification_needed": False,
        "message_to_user": "on it",
        "runtime_used": "opencode",
    }


def _developer_ticket_extend_pie() -> dict:
    return {
        "type": "DeveloperTicket",
        "source_agent": "dashboard-spec-agent",
        "severity": "medium",
        "summary": "pie_chart widget type not in toolkit",
        "user_visible_effect": "user asked for pie chart",
        "technical_evidence": "WidgetType enum allows: line_chart, ...",
        "requested_action": "extend widget toolkit with pie_chart",
        "safety_notes": "",
    }


def _developer_ticket_generic() -> dict:
    """A non-extend DeveloperTicket (some unrelated bug)."""
    return {
        "type": "DeveloperTicket",
        "source_agent": "dashboard-spec-agent",
        "severity": "high",
        "summary": "promql parser crashed",
        "user_visible_effect": "no dashboard",
        "technical_evidence": "stack trace",
        "requested_action": "fix the promql parser",
        "safety_notes": "",
    }


def _valid_dashboard_spec() -> dict:
    return {
        "type": "DashboardSpec",
        "spec": {
            "dashboard_id": "d", "title": "T", "description": "",
            "layout": {"columns": 12, "row_height": 40},
            "variables": [],
            "widgets": [{
                "id": "w1", "type": "line_chart", "title": "x",
                "description": "",
                "query": {"source": "prometheus", "promql": "up",
                          "query_type": "instant"},
                "position": {"x": 0, "y": 0, "w": 6, "h": 6},
                "encoding": {}, "thresholds": [], "options": {},
            }],
            "refresh_interval": "30s",
        },
    }


# ---------------------------------------------------------------------------
# Happy path: extend ticket → extend ok → re-dispatch → fresh dashboard
# ---------------------------------------------------------------------------


def test_extend_ticket_triggers_redispatch_and_succeeds(monkeypatch):
    _reset_storage()
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "2")

    orch = Orchestrator()
    rt = _ScriptedRuntime()
    # 1) classify
    rt.enqueue("user_message", _intent_dashboard())
    # 2) first generate → DeveloperTicket(extend pie_chart)
    rt.enqueue("generate_dashboard", _developer_ticket_extend_pie())
    # 3) rescue_extend → resolved (we stub the ExtendRunner directly
    #    so this queue doesn't need a real entry)
    # 4) second generate → valid dashboard
    rt.enqueue("generate_dashboard", _valid_dashboard_spec())

    orch._runtime = rt
    orch._retry._runtime = rt

    # Stub the extend runner to short-circuit (no real subprocess).
    def _fake_run(extend, *, user_intent="", original_args=None):
        return ExtendOutcome(report={
            "type": "DeveloperReport", "status": "resolved",
            "summary": "added pie_chart",
            "actions_taken": ["wrote PieChartWidget.tsx"],
        })
    orch._extend_runner.run = _fake_run  # type: ignore[method-assign]

    res = orch.handle_user_message(
        session_id="ext-1",
        message="pie chart of memory by host",
    )

    assert res["intent_type"] == "DashboardSpec"
    assert res["dashboard"] is not None
    # Two generate calls + one user_message classification.
    op_seq = [op for op, _ in rt.calls]
    assert op_seq.count("generate_dashboard") == 2


# ---------------------------------------------------------------------------
# Safety gate refusal → honest reply, no "isn't supported" framing
# ---------------------------------------------------------------------------


def test_safety_gate_refusal_yields_honest_reply(monkeypatch):
    """When a safety gate refuses (denylist, prompt-injection, quota),
    the reply should name the widget the user asked for and the gate
    reason — never say it's "not in the toolkit"."""
    _reset_storage()
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "2")

    orch = Orchestrator()
    rt = _ScriptedRuntime()
    rt.enqueue("user_message", _intent_dashboard())
    rt.enqueue("generate_dashboard", _developer_ticket_extend_pie())
    orch._runtime = rt
    orch._retry._runtime = rt

    # Simulate the gate refusing (e.g. quota exhausted).
    def _gate_refused(extend, *, user_intent="", original_args=None):
        return ExtendOutcome(error="gate refused (quota): daily quota exhausted: 50/50")
    orch._extend_runner.run = _gate_refused  # type: ignore[method-assign]

    res = orch.handle_user_message(
        session_id="ext-gate-refused",
        message="pie chart please",
    )
    assert res["intent_type"] == "DeveloperTicket"
    # Reply must name the widget AND not use the "isn't in the toolkit" line.
    reply = res["user_reply"]
    assert "pie_chart" in reply
    assert "not in" not in reply.lower()
    assert "isn't" not in reply.lower() or "didn't" in reply.lower()


# ---------------------------------------------------------------------------
# Extend fails → no redispatch, ticket reply
# ---------------------------------------------------------------------------


def test_extend_failure_falls_through_to_ticket_reply(monkeypatch):
    _reset_storage()
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "2")

    orch = Orchestrator()
    rt = _ScriptedRuntime()
    rt.enqueue("user_message", _intent_dashboard())
    rt.enqueue("generate_dashboard", _developer_ticket_extend_pie())
    orch._runtime = rt
    orch._retry._runtime = rt

    def _fake_run(extend, *, user_intent="", original_args=None):
        return ExtendOutcome(error="LLM crashed")
    orch._extend_runner.run = _fake_run  # type: ignore[method-assign]

    res = orch.handle_user_message(
        session_id="ext-fail",
        message="pie please",
    )
    # No re-dispatch happened → we're still on the original DeveloperTicket.
    assert res["intent_type"] == "DeveloperTicket"
    # Only one generate_dashboard call (no retry-after-extend).
    op_seq = [op for op, _ in rt.calls]
    assert op_seq.count("generate_dashboard") == 1
    # Reply should reflect that we tried — not say "isn't in the toolkit".
    reply = res["user_reply"]
    assert "pie_chart" in reply
    assert "tried" in reply.lower() or "build didn't pass" in reply.lower()


# ---------------------------------------------------------------------------
# Non-extend ticket: extend path is bypassed entirely
# ---------------------------------------------------------------------------


def test_non_extend_ticket_does_not_trigger_extend(monkeypatch):
    _reset_storage()
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "2")

    orch = Orchestrator()
    rt = _ScriptedRuntime()
    rt.enqueue("user_message", _intent_dashboard())
    rt.enqueue("generate_dashboard", _developer_ticket_generic())
    orch._runtime = rt
    orch._retry._runtime = rt

    def _should_not_be_called(*a, **kw):
        raise AssertionError("extend runner was called for non-extend ticket")
    orch._extend_runner.run = _should_not_be_called  # type: ignore[method-assign]

    res = orch.handle_user_message(
        session_id="generic",
        message="something",
    )
    assert res["intent_type"] == "DeveloperTicket"


# ---------------------------------------------------------------------------
# Re-dispatch yields ANOTHER extend ticket → no infinite loop
# ---------------------------------------------------------------------------


def test_extend_does_not_recurse_on_second_extend_ticket(monkeypatch):
    """If the re-dispatched generate STILL emits an extend ticket (for
    a different widget type, say), the orchestrator must not extend
    again — it returns the ticket as a normal DeveloperTicket reply."""
    _reset_storage()
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "2")

    orch = Orchestrator()
    rt = _ScriptedRuntime()
    rt.enqueue("user_message", _intent_dashboard())
    rt.enqueue("generate_dashboard", _developer_ticket_extend_pie())
    # Second attempt — still an extend ticket, for a different type.
    second = dict(_developer_ticket_extend_pie())
    second["requested_action"] = "extend widget toolkit with sankey"
    rt.enqueue("generate_dashboard", second)
    orch._runtime = rt
    orch._retry._runtime = rt

    call_count = {"n": 0}
    def _fake_run(extend, *, user_intent="", original_args=None):
        call_count["n"] += 1
        return ExtendOutcome(report={
            "type": "DeveloperReport", "status": "resolved",
            "summary": "ok",
        })
    orch._extend_runner.run = _fake_run  # type: ignore[method-assign]

    res = orch.handle_user_message(
        session_id="ext-nested",
        message="pie chart and sankey diagram",
    )
    # Extend was called exactly once, not twice.
    assert call_count["n"] == 1
    # Final response is a DeveloperTicket (for sankey).
    assert res["intent_type"] == "DeveloperTicket"
