"""Orchestrator integration with retry loop.

Confirms that:
- User-originated generate/patch flow through the validation-retry
  loop.
- When retry exhausts all attempts, a DeveloperTicket is filed and
  the user gets a safe error.
- When retry accepts on attempt > 1, the response includes a warning
  indicating the retry count.
- Review loop dispatch does NOT stack validation-retry (Decision 2c).
"""

from __future__ import annotations

import os

import pytest

from app.helper.orchestrator import Orchestrator, _PENDING_SAVE
from app.helper.retry_loop import AgentValidationRetryLoop


def _reset_storage():
    from pathlib import Path
    root = Path("backend/app/storage")
    for sub in ("dashboards", "tickets", "saved_dashboards", "retry_logs"):
        d = root / sub
        if d.exists():
            for p in d.glob("*.json"):
                p.unlink()
    _PENDING_SAVE.clear()


class _ScriptedRuntime:
    """Runtime stub that replays a per-operation queue."""

    def __init__(self):
        self.queues: dict[str, list[dict]] = {}
        self.calls: list[tuple[str, dict]] = []

    def enqueue(self, operation: str, result: dict):
        self.queues.setdefault(operation, []).append(result)

    def invoke_operation(self, operation, args, *, developer=False):
        self.calls.append((operation, dict(args)))
        q = self.queues.get(operation) or []
        if not q:
            # Fall back to a passthrough UserResponse for operations
            # we didn't script.
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
            "title": "T", "goal": "g",
            "metrics_hints": [],
            "widget_hints": ["line_chart"],
            "refresh_interval_hint": "30s",
        },
        "clarification_needed": False,
        "message_to_user": "on it",
        "runtime_used": "opencode",
    }


def _intent_patch(did: str) -> dict:
    return {
        "type": "PatchIntent",
        "target_dashboard_id": did,
        "requested_changes": ["make latency more prominent"],
        "message_to_user": "ok",
        "runtime_used": "opencode",
    }


def _dashboard_spec_result(widgets: list[dict]) -> dict:
    return {
        "type": "DashboardSpec",
        "spec": {
            "dashboard_id": "d", "title": "T", "description": "",
            "layout": {"columns": 12, "row_height": 40},
            "variables": [], "widgets": widgets,
            "refresh_interval": "30s",
        },
    }


def _valid_simple_dashboard() -> dict:
    return _dashboard_spec_result([
        {"id": "w1", "type": "line_chart", "title": "x",
         "description": "",
         "query": {"source": "prometheus",
                    "promql": "rate(http_requests_total[5m])",
                    "query_type": "range", "range": "1h", "step": "30s"},
         "position": {"x": 0, "y": 0, "w": 6, "h": 6},
         "encoding": {}, "thresholds": [], "options": {}},
    ])


def _valid_latency_update() -> dict:
    return {
        "type": "PatchSpec",
        "spec": {
            "patch_id": "p1", "reason": "r",
            "target_dashboard_id": "d", "created_by": "patch-agent",
            "operations": [{
                "op": "update_widget",
                "widget_id": "w-latency-p95",
                "fields": {"position": {"x": 0, "y": 0, "w": 12, "h": 6}},
            }],
        },
    }


# ---------------------------------------------------------------------------
# Generate: accepted on first attempt
# ---------------------------------------------------------------------------


def test_generate_accepts_on_first_attempt(monkeypatch):
    _reset_storage()
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "3")

    orch = Orchestrator()
    rt = _ScriptedRuntime()
    rt.enqueue("user_message", _intent_dashboard())
    rt.enqueue("generate_dashboard", _valid_simple_dashboard())

    orch._runtime = rt
    orch._retry._runtime = rt

    res = orch.handle_user_message(
        session_id="s1",
        message="show me a CPU dashboard",
    )
    assert res["intent_type"] == "DashboardSpec"
    assert res["dashboard"] is not None
    assert "validation retry" not in " ".join(res["warnings"])


# ---------------------------------------------------------------------------
# Generate: retry accepts on attempt 2 -> warning surfaced
# ---------------------------------------------------------------------------


def test_generate_retry_accepts_on_second_attempt(monkeypatch):
    _reset_storage()
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "3")

    orch = Orchestrator()
    rt = _ScriptedRuntime()
    rt.enqueue("user_message", _intent_dashboard())
    # Attempt 1: empty widgets (fails any-valid-generate).
    rt.enqueue("generate_dashboard", _dashboard_spec_result([]))
    # Attempt 2: one valid widget.
    rt.enqueue("generate_dashboard", _valid_simple_dashboard())

    orch._runtime = rt
    orch._retry._runtime = rt

    res = orch.handle_user_message(
        session_id="s2",
        message="show me a dashboard",
    )
    assert res["intent_type"] == "DashboardSpec"
    assert res["dashboard"] is not None
    joined = " ".join(res["warnings"])
    assert "validation retry" in joined
    assert "2 attempts" in joined


# ---------------------------------------------------------------------------
# Generate: all attempts fail -> DeveloperTicket filed, safe reply
# ---------------------------------------------------------------------------


def test_generate_all_attempts_fail_files_ticket(monkeypatch):
    _reset_storage()
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "2")

    orch = Orchestrator()
    rt = _ScriptedRuntime()
    rt.enqueue("user_message", _intent_dashboard())
    bad = _dashboard_spec_result([
        {"id": "w1", "type": "heatmap", "title": "x"},
    ])
    for _ in range(2):
        rt.enqueue("generate_dashboard", bad)

    orch._runtime = rt
    orch._retry._runtime = rt

    res = orch.handle_user_message(
        session_id="s3",
        message="show me a dashboard",
    )
    assert res["dashboard"] is None
    assert "logged a diagnostic" in res["user_reply"].lower()
    # Ticket on disk.
    from pathlib import Path
    tickets = list(
        Path("backend/app/storage/tickets").glob("*.json"),
    )
    assert tickets


# ---------------------------------------------------------------------------
# Patch: retry exhausts -> ticket
# ---------------------------------------------------------------------------


def test_patch_all_attempts_fail_files_ticket(monkeypatch):
    _reset_storage()
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "2")

    orch = Orchestrator()

    # Seed a current dashboard so PatchIntent routes correctly.
    from app.specs import DashboardSpec
    spec = DashboardSpec.model_validate({
        "dashboard_id": "d", "title": "T", "description": "",
        "layout": {"columns": 12, "row_height": 40},
        "variables": [],
        "widgets": [
            {"id": "w-latency-p95", "type": "line_chart",
             "title": "latency p95", "description": "",
             "query": {"source": "prometheus", "promql": "up",
                        "query_type": "instant"},
             "position": {"x": 0, "y": 0, "w": 6, "h": 6},
             "encoding": {}, "thresholds": [], "options": {}},
        ],
        "refresh_interval": "30s",
    })
    orch._store.save_dashboard(spec)

    rt = _ScriptedRuntime()
    rt.enqueue("user_message", _intent_patch("d"))
    # Both attempts return a PatchSpec with only reorder (fails
    # make_prominent contract).
    reorder_only = {
        "type": "PatchSpec",
        "spec": {
            "patch_id": "p1", "reason": "r",
            "target_dashboard_id": "d", "created_by": "patch-agent",
            "operations": [{"op": "reorder_widgets",
                             "order": ["w-latency-p95"]}],
        },
    }
    rt.enqueue("patch_dashboard", reorder_only)
    rt.enqueue("patch_dashboard", reorder_only)

    orch._runtime = rt
    orch._retry._runtime = rt

    res = orch.handle_user_message(
        session_id="s4",
        message="make latency more prominent",
        current_dashboard_id="d",
    )
    assert res["patch"] is None
    assert "logged a diagnostic" in res["user_reply"].lower()


# ---------------------------------------------------------------------------
# Patch: retry accepts on attempt 2 -> warning, patch applied
# ---------------------------------------------------------------------------


def test_patch_retry_accepts_on_second_attempt(monkeypatch):
    _reset_storage()
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "3")

    orch = Orchestrator()

    from app.specs import DashboardSpec
    spec = DashboardSpec.model_validate({
        "dashboard_id": "d", "title": "T", "description": "",
        "layout": {"columns": 12, "row_height": 40},
        "variables": [],
        "widgets": [
            {"id": "w-latency-p95", "type": "line_chart",
             "title": "latency p95", "description": "",
             "query": {"source": "prometheus", "promql": "up",
                        "query_type": "instant"},
             "position": {"x": 0, "y": 0, "w": 6, "h": 6},
             "encoding": {}, "thresholds": [], "options": {}},
        ],
        "refresh_interval": "30s",
    })
    orch._store.save_dashboard(spec)

    rt = _ScriptedRuntime()
    rt.enqueue("user_message", _intent_patch("d"))
    # Attempt 1: reorder-only (semantic fail for make_prominent).
    rt.enqueue("patch_dashboard", {
        "type": "PatchSpec",
        "spec": {
            "patch_id": "p1", "reason": "r",
            "target_dashboard_id": "d", "created_by": "patch-agent",
            "operations": [{"op": "reorder_widgets",
                             "order": ["w-latency-p95"]}],
        },
    })
    # Attempt 2: valid update on latency widget.
    rt.enqueue("patch_dashboard", _valid_latency_update())

    orch._runtime = rt
    orch._retry._runtime = rt

    res = orch.handle_user_message(
        session_id="s5",
        message="make latency more prominent",
        current_dashboard_id="d",
    )
    assert res["patch"] is not None
    assert res["dashboard"] is not None
    joined = " ".join(res["warnings"])
    assert "validation retry" in joined
    assert "2 attempts" in joined


# ---------------------------------------------------------------------------
# Decision 2c: review loop uses runtime directly, not _handle_patch_spec
# ---------------------------------------------------------------------------


def test_review_loop_does_not_wrap_with_retry(monkeypatch):
    """ReviewLoop calls runtime.invoke_operation directly, so its
    inner review_rendered / rescue_review / generated patches are
    NOT retried by AgentValidationRetryLoop. This prevents the
    combinatorial explosion."""
    from app.helper.review_loop import ReviewLoop
    import inspect
    src = inspect.getsource(ReviewLoop)
    assert "AgentValidationRetryLoop" not in src
    assert "runtime.invoke_operation" in src.replace(" ", "") \
        or "self._runtime.invoke_operation" in src


# ---------------------------------------------------------------------------
# LD-1/LD-2: review outcome "ticket" delivers the dashboard, no human wait
# ---------------------------------------------------------------------------


def test_review_ticket_outcome_still_delivers_dashboard(monkeypatch):
    """A rescue `ticket` is a diagnostic record, not a hand-off to
    humans. The Python-validated draft must still be delivered with
    an honest caveat; the reply must never claim a team was
    notified."""
    _reset_storage()
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    # Force pre-output review ON (off by default in mock mode).
    monkeypatch.setenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", "1")
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "2")

    orch = Orchestrator()
    rt = _ScriptedRuntime()
    rt.enqueue("user_message", _intent_dashboard())
    rt.enqueue("generate_dashboard", _valid_simple_dashboard())
    orch._runtime = rt
    orch._retry._runtime = rt

    from app.helper.review_loop import ReviewOutcome
    from app.specs.developer_ticket import DeveloperTicket

    def _fake_review_run(draft, *, user_intent):
        ticket = DeveloperTicket(
            ticket_id="tkt-render-flake",
            source_agent="big-guy-developer-agent",
            severity="high",
            summary="render check could not confirm widgets",
            user_visible_effect="widgets may look empty",
            requested_action="investigate evaluator race",
        )
        return ReviewOutcome(
            kind="ticket", ticket=ticket, dashboard=draft,
            trail=[{"stage": "rescue", "kind": "ticket"}],
        )

    orch._review.run = _fake_review_run  # type: ignore[method-assign]

    res = orch.handle_user_message(
        session_id="s-ticket-deliver",
        message="show me a line chart of rSO2 for the last hour",
    )
    # Delivered, not withheld behind a ticket.
    assert res["dashboard"] is not None
    assert res["intent_type"] == "DashboardSpec"
    # Honest messaging: diagnostic logged, no humans invoked.
    reply = res["user_reply"].lower()
    assert "logged a diagnostic" in reply
    assert "team" not in reply
    assert "notified" not in reply
    joined = " ".join(res["warnings"])
    assert "diagnostic logged" in joined
    # The diagnostic record is still persisted for the automated
    # pipeline. Glob the default tickets dir (module constant, never
    # mutated) rather than a cwd-relative path so this holds no matter
    # where pytest runs from.
    from app.services import dashboard_store as _ds
    tickets = list(_ds._TICKETS_DIR.glob("*.json"))
    assert any("tkt-render-flake" in p.name for p in tickets)


# ---------------------------------------------------------------------------
# Review ticket outcome schedules a background Big-guy auto-fix
# ---------------------------------------------------------------------------


def test_review_ticket_outcome_schedules_auto_fix(monkeypatch):
    """When a rescue ticket is persisted, the orchestrator hands it to
    the auto-fix pipeline and the caveat tells the user a background
    fix has started. (auto_fix itself is stubbed — its internals are
    covered in tests/auto_fix/.)"""
    _reset_storage()
    monkeypatch.delenv("HELPER_DASHBOARD_OPENCODE", raising=False)
    monkeypatch.setenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", "1")
    monkeypatch.setenv("HELPER_AGENT_MAX_ATTEMPTS", "2")

    orch = Orchestrator()
    rt = _ScriptedRuntime()
    rt.enqueue("user_message", _intent_dashboard())
    rt.enqueue("generate_dashboard", _valid_simple_dashboard())
    orch._runtime = rt
    orch._retry._runtime = rt

    from app.helper import auto_fix as auto_fix_mod
    from app.helper.review_loop import ReviewOutcome
    from app.specs.developer_ticket import DeveloperTicket

    scheduled: list[str] = []

    class _FakeThread:
        pass

    def _fake_schedule(ticket, *, runtime, store, user_intent=""):
        scheduled.append(ticket.ticket_id)
        return _FakeThread()

    monkeypatch.setattr(auto_fix_mod, "schedule_auto_fix", _fake_schedule)

    def _fake_review_run(draft, *, user_intent):
        ticket = DeveloperTicket(
            ticket_id="tkt-schedule-fix",
            source_agent="big-guy-developer-agent",
            severity="high",
            summary="render check could not confirm widgets",
            user_visible_effect="widgets may look empty",
            requested_action="investigate evaluator race",
        )
        return ReviewOutcome(
            kind="ticket", ticket=ticket, dashboard=draft,
            trail=[{"stage": "rescue", "kind": "ticket"}],
        )

    orch._review.run = _fake_review_run  # type: ignore[method-assign]

    res = orch.handle_user_message(
        session_id="s-ticket-autofix",
        message="show me a line chart of rSO2 for the last hour",
    )
    assert scheduled == ["tkt-schedule-fix"]
    assert res["dashboard"] is not None
    assert "automatic background fix" in res["user_reply"]
