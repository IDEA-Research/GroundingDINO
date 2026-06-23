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
    assert "team has been notified" in res["user_reply"].lower()
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
    assert "team has been notified" in res["user_reply"].lower()


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
