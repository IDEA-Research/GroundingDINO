"""Review loop tests.

Covers the synchronous pre-output review flow end-to-end, using a
stubbed evaluator so we control the "rendered" state and a stubbed
runtime so we control Helper's and Big guy's decisions.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from app.helper.review_loop import (
    HELPER_MAX_ATTEMPTS,
    ReviewLoop,
    ReviewOutcome,
    is_enabled,
)
from app.helper.runtime import OpenCodeRuntime
from app.services.browser_evaluator import BrowserEvaluator
from app.services.spec_validator import SpecValidator
from app.specs import DashboardSpec
from app.specs.evaluation_report import BrowserEvaluationReport


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------


def _spec(widgets: list[dict] | None = None) -> DashboardSpec:
    return SpecValidator().validate_dashboard({
        "dashboard_id": "d",
        "title": "Demo",
        "description": "",
        "layout": {"columns": 12, "row_height": 40},
        "variables": [],
        "widgets": widgets if widgets is not None else [
            {
                "id": "w1", "type": "line_chart", "title": "x",
                "description": "",
                "query": {"source": "prometheus", "promql": "up",
                          "query_type": "range", "range": "1h", "step": "30s"},
                "position": {"x": 0, "y": 0, "w": 6, "h": 6},
                "encoding": {}, "thresholds": [], "options": {},
            },
        ],
        "refresh_interval": "30s",
    })


class _StubEvaluator:
    """Yields a pre-set sequence of reports, one per `.evaluate()` call."""

    def __init__(self, reports: list[dict]):
        self._reports = list(reports)

    def evaluate(self, spec: DashboardSpec) -> BrowserEvaluationReport:
        payload = self._reports.pop(0) if self._reports else {
            "page_loaded": True, "widgets_rendered": [], "missing_widgets": [],
            "console_errors": [], "layout_errors": [], "prometheus_errors": [],
            "recommendation": "clean",
        }
        payload.setdefault("dashboard_id", spec.dashboard_id)
        return BrowserEvaluationReport.model_validate(payload)


class _StubRuntime:
    """Replays a queue of (operation, result) pairs."""

    def __init__(self, script: list[tuple[str, dict]]):
        self._script = list(script)
        self.calls: list[tuple[str, dict]] = []

    def invoke_operation(self, operation, args, *, developer=False):
        self.calls.append((operation, args))
        op, result = self._script.pop(0)
        assert op == operation, f"expected {op!r}, got {operation!r}"
        return dict(result)


# ---------------------------------------------------------------------------
# feature flag
# ---------------------------------------------------------------------------


def test_is_enabled_default_off_in_mock(monkeypatch):
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    assert not is_enabled()


def test_is_enabled_default_on_in_opencode(monkeypatch):
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE", "opencode")
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    assert is_enabled()


def test_is_enabled_default_on_in_auto(monkeypatch):
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE", "auto")
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    assert is_enabled()


def test_is_enabled_env_forces_on(monkeypatch):
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE", "mock")
    monkeypatch.setenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", "1")
    assert is_enabled()


def test_is_enabled_env_forces_off(monkeypatch):
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE", "opencode")
    monkeypatch.setenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", "0")
    assert not is_enabled()


# ---------------------------------------------------------------------------
# happy path: first render approves
# ---------------------------------------------------------------------------


def test_review_approves_on_first_render():
    spec = _spec()
    ev = _StubEvaluator([
        {"page_loaded": True, "widgets_rendered": ["w1"],
         "missing_widgets": [], "console_errors": [],
         "layout_errors": [], "prometheus_errors": [],
         "recommendation": "clean"},
    ])
    rt = _StubRuntime([
        ("review_rendered",
         {"type": "ReviewDecision", "decision": "approve",
          "rationale": "looks good"}),
    ])
    loop = ReviewLoop(rt, evaluator=ev)
    out = loop.run(spec, user_intent="cpu dashboard")
    assert out.kind == "approved"
    assert out.dashboard.dashboard_id == "d"
    assert not out.patches_applied
    assert [t["stage"] for t in out.trail] == ["render", "review"]


# ---------------------------------------------------------------------------
# Helper patches once, second render approves
# ---------------------------------------------------------------------------


def test_helper_patches_once_then_approves():
    spec = _spec(widgets=[
        {"id": "a", "type": "line_chart", "title": "a", "description": "",
         "query": {"source": "prometheus", "promql": "up",
                   "query_type": "range", "range": "1h", "step": "30s"},
         "position": {"x": 0, "y": 0, "w": 6, "h": 6},
         "encoding": {}, "thresholds": [], "options": {}},
        {"id": "b", "type": "stat_card", "title": "b", "description": "",
         "query": {"source": "prometheus", "promql": "up",
                   "query_type": "instant"},
         "position": {"x": 6, "y": 0, "w": 3, "h": 4},
         "encoding": {}, "thresholds": [], "options": {}},
    ])
    ev = _StubEvaluator([
        # first render: 'b' missing
        {"page_loaded": True, "widgets_rendered": ["a"],
         "missing_widgets": ["b"], "console_errors": [],
         "layout_errors": [], "prometheus_errors": [], "recommendation": "x"},
        # second render (after removing 'b'): clean
        {"page_loaded": True, "widgets_rendered": ["a"],
         "missing_widgets": [], "console_errors": [],
         "layout_errors": [], "prometheus_errors": [], "recommendation": "x"},
    ])
    rt = _StubRuntime([
        ("review_rendered", {
            "type": "ReviewDecision", "decision": "patch",
            "rationale": "remove broken widget",
            "patch": {
                "patch_id": "p1", "reason": "auto",
                "target_dashboard_id": "d", "created_by": "helper-review-agent",
                "operations": [{"op": "remove_widget", "widget_id": "b"}],
            },
        }),
        ("review_rendered", {
            "type": "ReviewDecision", "decision": "approve",
            "rationale": "clean after patch",
        }),
    ])
    out = ReviewLoop(rt, evaluator=ev).run(spec, user_intent="x")
    assert out.kind == "approved"
    assert len(out.patches_applied) == 1
    # Only widget 'a' remains.
    assert [w.id for w in out.dashboard.widgets] == ["a"]


# ---------------------------------------------------------------------------
# Helper exhausts retries → escalate → Big guy ticket
# ---------------------------------------------------------------------------


def test_helper_escalates_then_big_guy_files_ticket():
    spec = _spec()
    # Three renders return console errors; Helper escalates right away
    # on attempt 1 (page_loaded but console errors) by returning
    # decision="escalate". Big guy files a ticket.
    ev = _StubEvaluator([
        {"page_loaded": True, "widgets_rendered": ["w1"],
         "missing_widgets": [], "console_errors": ["TypeError"],
         "layout_errors": [], "prometheus_errors": [], "recommendation": "x"},
        {"page_loaded": True, "widgets_rendered": ["w1"],
         "missing_widgets": [], "console_errors": ["TypeError"],
         "layout_errors": [], "prometheus_errors": [], "recommendation": "x"},
    ])
    rt = _StubRuntime([
        ("review_rendered", {
            "type": "ReviewDecision", "decision": "escalate",
            "rationale": "console error smells like code bug",
        }),
        ("rescue_review", {
            "type": "RescueDecision", "kind": "ticket",
            "rationale": "code-level bug",
            "ticket": {
                "ticket_id": "tkt-1",
                "source_agent": "big-guy-developer-agent",
                "severity": "high",
                "summary": "widget-toolkit crash",
                "user_visible_effect": "broken render",
                "technical_evidence": {"console_errors": ["TypeError"]},
                "requested_action": "fix widget",
                "safety_notes": "",
            },
        }),
    ])
    out = ReviewLoop(rt, evaluator=ev).run(spec, user_intent="x")
    assert out.kind == "ticket"
    assert out.ticket is not None
    assert out.ticket.ticket_id == "tkt-1"
    # LD-1/LD-2: the ticket outcome must carry the validated draft so
    # the orchestrator can deliver it instead of blocking on a human.
    assert out.dashboard is not None
    assert out.dashboard.dashboard_id == spec.dashboard_id


# ---------------------------------------------------------------------------
# Helper escalates → Big guy patches → confirmation render approves
# ---------------------------------------------------------------------------


def test_big_guy_patches_and_confirms():
    spec = _spec()
    ev = _StubEvaluator([
        # initial render has issues
        {"page_loaded": True, "widgets_rendered": [],
         "missing_widgets": ["w1"], "console_errors": [],
         "layout_errors": [], "prometheus_errors": [], "recommendation": "x"},
        # rescue render (before rescue decision)
        {"page_loaded": True, "widgets_rendered": [],
         "missing_widgets": ["w1"], "console_errors": [],
         "layout_errors": [], "prometheus_errors": [], "recommendation": "x"},
        # confirmation render after Big guy patch — clean
        {"page_loaded": True, "widgets_rendered": ["w2"],
         "missing_widgets": [], "console_errors": [],
         "layout_errors": [], "prometheus_errors": [], "recommendation": "x"},
    ])
    new_widget = {
        "id": "w2", "type": "stat_card", "title": "New", "description": "",
        "query": {"source": "prometheus", "promql": "up",
                  "query_type": "instant"},
        "position": {"x": 6, "y": 0, "w": 3, "h": 4},
        "encoding": {}, "thresholds": [], "options": {},
    }
    rt = _StubRuntime([
        ("review_rendered", {
            "type": "ReviewDecision", "decision": "escalate",
            "rationale": "can't fix",
        }),
        ("rescue_review", {
            "type": "RescueDecision", "kind": "patch",
            "rationale": "use a simpler widget",
            "patch": {
                "patch_id": "p2", "reason": "replace", "target_dashboard_id": "d",
                "created_by": "big-guy-developer-agent",
                "operations": [
                    {"op": "remove_widget", "widget_id": "w1"},
                    {"op": "add_widget", "widget": new_widget},
                ],
            },
        }),
    ])
    out = ReviewLoop(rt, evaluator=ev).run(spec, user_intent="x")
    assert out.kind == "approved"
    assert [w.id for w in out.dashboard.widgets] == ["w2"]
    assert len(out.patches_applied) == 1


# ---------------------------------------------------------------------------
# Helper escalates → Big guy asks user → orchestrator returns clarify
# ---------------------------------------------------------------------------


def test_big_guy_asks_user():
    spec = _spec()
    ev = _StubEvaluator([
        {"page_loaded": True, "widgets_rendered": [],
         "missing_widgets": ["w1"], "console_errors": [],
         "layout_errors": [], "prometheus_errors": [], "recommendation": "x"},
        {"page_loaded": True, "widgets_rendered": [],
         "missing_widgets": ["w1"], "console_errors": [],
         "layout_errors": [], "prometheus_errors": [], "recommendation": "x"},
    ])
    rt = _StubRuntime([
        ("review_rendered", {
            "type": "ReviewDecision", "decision": "escalate",
            "rationale": "unclear",
        }),
        ("rescue_review", {
            "type": "RescueDecision", "kind": "ask_user",
            "rationale": "intent ambiguous",
            "questions": ["Which service?", "What time range?"],
        }),
    ])
    out = ReviewLoop(rt, evaluator=ev).run(spec, user_intent="x")
    assert out.kind == "clarify"
    assert out.questions == ["Which service?", "What time range?"]


# ---------------------------------------------------------------------------
# Helper hits max attempts → escalate
# ---------------------------------------------------------------------------


def test_helper_hits_max_attempts_then_escalates():
    assert HELPER_MAX_ATTEMPTS == 3
    spec = _spec()
    # 3 review_rendered calls all return `patch` with a no-op patch;
    # on the 4th render the loop escalates automatically (attempt > max).
    # We only provide 3 patch reviews + 1 rescue.
    noop_patch = {
        "patch_id": "noop", "reason": "noop", "target_dashboard_id": "d",
        "created_by": "helper-review-agent",
        "operations": [{"op": "update_dashboard", "fields": {"description": "x"}}],
    }
    reports = [
        {"page_loaded": True, "widgets_rendered": ["w1"],
         "missing_widgets": [], "console_errors": [],
         "layout_errors": [], "prometheus_errors": [],
         "recommendation": "x"},
    ] * 4
    ev = _StubEvaluator(reports)
    rt = _StubRuntime([
        ("review_rendered", {
            "type": "ReviewDecision", "decision": "patch",
            "rationale": "try 1", "patch": noop_patch,
        }),
        ("review_rendered", {
            "type": "ReviewDecision", "decision": "patch",
            "rationale": "try 2", "patch": noop_patch,
        }),
        ("review_rendered", {
            "type": "ReviewDecision", "decision": "patch",
            "rationale": "try 3", "patch": noop_patch,
        }),
        ("rescue_review", {
            "type": "RescueDecision", "kind": "ask_user",
            "rationale": "too many retries",
            "questions": ["Tell me more."],
        }),
    ])
    out = ReviewLoop(rt, evaluator=ev).run(spec, user_intent="x")
    assert out.kind == "clarify"
    # 3 helper attempts + 1 rescue render + 1 rescue call → at least
    # 4 renders.
    render_events = [t for t in out.trail if t["stage"] == "render"]
    assert len(render_events) == HELPER_MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# Invalid patch from Helper → retry with feedback, then escalate (Fix 3)
# ---------------------------------------------------------------------------


_BAD_PATCH = {
    "type": "ReviewDecision", "decision": "patch",
    "rationale": "bad patch",
    "patch": {
        "patch_id": "bad",
        "reason": "bad",
        "target_dashboard_id": "d",
        "created_by": "helper-review-agent",
        "operations": [{"op": "shell_exec", "cmd": "rm -rf /"}],
    },
}


def test_helper_invalid_patch_retried_with_feedback_then_escalates():
    """The helper loop must retry up to HELPER_MAX_ATTEMPTS times when
    its patch is rejected by Pydantic, surfacing the errors as
    `_prior_feedback_message` to the next attempt — not immediately
    escalating to rescue. After exhausting attempts, rescue takes over
    and degrades to ask_user."""
    spec = _spec()
    # 3 render reports for the 3 helper attempts, then 1 for rescue.
    clean_report = {
        "page_loaded": True, "widgets_rendered": ["w1"],
        "missing_widgets": [], "console_errors": [],
        "layout_errors": [], "prometheus_errors": [],
        "recommendation": "x",
    }
    ev = _StubEvaluator([clean_report] * 4)
    rt = _StubRuntime([
        ("review_rendered", _BAD_PATCH),
        ("review_rendered", _BAD_PATCH),
        ("review_rendered", _BAD_PATCH),
        ("rescue_review", {
            "type": "RescueDecision", "kind": "ask_user",
            "rationale": "bailout", "questions": ["what do you want?"],
        }),
    ])
    out = ReviewLoop(rt, evaluator=ev).run(spec, user_intent="x")
    assert out.kind == "clarify"

    # Each helper attempt after the first must have received feedback.
    review_calls = [args for (op, args) in rt.calls if op == "review_rendered"]
    assert len(review_calls) == 3
    assert "_prior_feedback_message" not in review_calls[0]
    assert "_prior_feedback_message" in review_calls[1]
    assert "_prior_feedback_message" in review_calls[2]
    # Trail records the validation failures.
    stages = [t.get("stage") for t in out.trail]
    assert stages.count("patch_validation_failed") == 3


def test_helper_invalid_patch_then_valid_patch_succeeds():
    """If the second helper attempt produces a valid patch after
    receiving feedback, the loop accepts on attempt 2 without ever
    reaching rescue."""
    spec = _spec()
    clean_report = {
        "page_loaded": True, "widgets_rendered": ["w1"],
        "missing_widgets": [], "console_errors": [],
        "layout_errors": [], "prometheus_errors": [],
        "recommendation": "x",
    }
    ev = _StubEvaluator([clean_report] * 3)
    good_patch = {
        "type": "ReviewDecision", "decision": "patch",
        "rationale": "fixed",
        "patch": {
            "patch_id": "p1", "reason": "r",
            "target_dashboard_id": "d", "created_by": "helper-review-agent",
            "operations": [{
                "op": "update_dashboard",
                "fields": {"title": "Fixed"},
            }],
        },
    }
    rt = _StubRuntime([
        ("review_rendered", _BAD_PATCH),
        ("review_rendered", good_patch),
        ("review_rendered", {"type": "ReviewDecision",
                              "decision": "approve",
                              "rationale": "ok"}),
    ])
    out = ReviewLoop(rt, evaluator=ev).run(spec, user_intent="x")
    assert out.kind == "approved"
    # No rescue call.
    assert all(op != "rescue_review" for (op, _) in rt.calls)


# ---------------------------------------------------------------------------
# Rescue retry with feedback (Fix B) + graceful degrade (Fix C)
# ---------------------------------------------------------------------------


_BAD_TICKET_DICT = {
    # missing required fields, includes wrong-named ones from the
    # title/component/description hallucination that broke the live walk-through
    "title": "Something broke",
    "component": "renderer",
    "description": "the dashboard didn't load",
}


def test_rescue_bad_ticket_retries_with_feedback_then_clarifies():
    """First rescue attempt returns a malformed DeveloperTicket. The
    loop must retry rescue once with Pydantic errors injected as
    `_prior_feedback_message`. If the second attempt is also bad,
    degrade to clarify — never crash to RuntimeError."""
    spec = _spec()
    clean_report = {
        "page_loaded": True, "widgets_rendered": ["w1"],
        "missing_widgets": [], "console_errors": [],
        "layout_errors": [], "prometheus_errors": [],
        "recommendation": "x",
    }
    ev = _StubEvaluator([clean_report] * 5)
    rt = _StubRuntime([
        # Helper escalates immediately.
        ("review_rendered", {"type": "ReviewDecision",
                              "decision": "escalate",
                              "rationale": "bail"}),
        # Rescue attempt 1: bad ticket.
        ("rescue_review", {
            "type": "RescueDecision", "kind": "ticket",
            "rationale": "x", "ticket": dict(_BAD_TICKET_DICT),
        }),
        # Rescue attempt 2: still bad.
        ("rescue_review", {
            "type": "RescueDecision", "kind": "ticket",
            "rationale": "x", "ticket": dict(_BAD_TICKET_DICT),
        }),
    ])
    out = ReviewLoop(rt, evaluator=ev).run(spec, user_intent="x")

    # Degraded to clarify, not failed.
    assert out.kind == "clarify"
    stages = [t.get("stage") for t in out.trail]
    assert stages.count("rescue_ticket_validation_failed") == 2
    # The second rescue call must have received feedback.
    rescue_calls = [args for (op, args) in rt.calls if op == "rescue_review"]
    assert len(rescue_calls) == 2
    assert "_prior_feedback_message" not in rescue_calls[0]
    assert "_prior_feedback_message" in rescue_calls[1]
    msg = rescue_calls[1]["_prior_feedback_message"]
    assert "DeveloperTicket" in msg
    assert "title" in msg  # the framing names the hallucinated keys


def test_rescue_bad_ticket_then_good_ticket_succeeds():
    """If rescue attempt 2 returns a well-formed ticket after feedback,
    we should return kind=ticket with the parsed DeveloperTicket."""
    spec = _spec()
    clean_report = {
        "page_loaded": True, "widgets_rendered": ["w1"],
        "missing_widgets": [], "console_errors": [],
        "layout_errors": [], "prometheus_errors": [],
        "recommendation": "x",
    }
    ev = _StubEvaluator([clean_report] * 5)
    good_ticket = {
        "ticket_id": "tkt-1",
        "source_agent": "big-guy-developer-agent",
        "severity": "medium",
        "summary": "renderer issue",
        "user_visible_effect": "dashboard didn't render",
        "requested_action": "investigate renderer",
    }
    rt = _StubRuntime([
        ("review_rendered", {"type": "ReviewDecision",
                              "decision": "escalate",
                              "rationale": "bail"}),
        ("rescue_review", {
            "type": "RescueDecision", "kind": "ticket",
            "rationale": "x", "ticket": dict(_BAD_TICKET_DICT),
        }),
        ("rescue_review", {
            "type": "RescueDecision", "kind": "ticket",
            "rationale": "y", "ticket": good_ticket,
        }),
    ])
    out = ReviewLoop(rt, evaluator=ev).run(spec, user_intent="x")
    assert out.kind == "ticket"
    assert out.ticket is not None
    assert out.ticket.ticket_id == "tkt-1"
    # LD-1/LD-2: the ticket outcome must carry the validated draft so
    # the orchestrator can deliver it instead of blocking on a human.
    assert out.dashboard is not None
    assert out.dashboard.dashboard_id == spec.dashboard_id


def test_rescue_runtime_error_degrades_to_clarify():
    """A RuntimeError_ from `rescue_review` invocation must not
    propagate as failed/RuntimeError — degrade to clarify."""
    from app.helper.runtime import RuntimeError_

    spec = _spec()
    clean_report = {
        "page_loaded": True, "widgets_rendered": ["w1"],
        "missing_widgets": [], "console_errors": [],
        "layout_errors": [], "prometheus_errors": [],
        "recommendation": "x",
    }
    ev = _StubEvaluator([clean_report, clean_report])

    class _CrashingRuntime:
        def __init__(self):
            self.calls = []

        def invoke_operation(self, op, args, *, developer=False):
            self.calls.append((op, args))
            if op == "review_rendered":
                return {"type": "ReviewDecision", "decision": "escalate",
                        "rationale": "bail"}
            raise RuntimeError_("provider down")

    rt = _CrashingRuntime()
    out = ReviewLoop(rt, evaluator=ev).run(spec, user_intent="x")  # type: ignore[arg-type]
    assert out.kind == "clarify"
    stages = [t.get("stage") for t in out.trail]
    assert "rescue_runtime_error" in stages
