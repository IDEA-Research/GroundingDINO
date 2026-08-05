"""Tests for ExtendRunner (the wrapper that invokes rescue_extend).

ExtendRunner runs the safety gates (denylist, prompt-injection, quota,
audit) before invoking the runtime. Gate behavior itself is covered in
test_gate.py — these tests just reset quota and isolate audit writes.
"""

from __future__ import annotations

import pytest

from app.helper.extend_gate import reset_quota_for_tests
from app.helper.extend_request import ExtendRequest as ParsedExtendRequest
from app.helper.extend_runner import ExtendOutcome, ExtendRunner
from app.helper.runtime import RuntimeError_


@pytest.fixture(autouse=True)
def _reset_quota_and_isolate_audit(monkeypatch, tmp_path):
    """Reset the in-memory quota AND redirect audit writes into
    tmp_path so pytest never leaves rep-test / xxxxxx entries in the
    production backend/app/storage/extend_audit/ directory (Bug C-4
    from the 2026-05-26 live demo)."""
    import app.helper.extend_gate as _gate
    monkeypatch.setattr(_gate, "_AUDIT_DIR", tmp_path / "audit")
    reset_quota_for_tests()
    yield
    reset_quota_for_tests()


class _StubRuntime:
    """Replay a single canned response or raise on invoke_operation."""

    def __init__(self, response=None, raise_exc=None):
        self._response = response
        self._raise = raise_exc
        self.calls: list[tuple[str, dict]] = []

    def invoke_operation(self, operation, args, *, developer=False):
        self.calls.append((operation, dict(args)))
        if self._raise:
            raise self._raise
        return self._response


def _parsed(widget_type: str = "pie_chart") -> ParsedExtendRequest:
    return ParsedExtendRequest(
        widget_type=widget_type,
        source_agent="dashboard-spec-agent",
        summary="pie_chart not in toolkit",
        user_visible_effect="user asked for pie",
    )


def _developer_report(status: str = "resolved", **extras) -> dict:
    return {
        "type": "DeveloperReport",
        "report_id": "rep-test",
        "summary": "extended",
        "actions_taken": ["wrote PieChartWidget.tsx"],
        "tests_run": ["pytest"],
        "status": status,
        **extras,
    }


def test_run_returns_ok_outcome_on_resolved_report():
    runtime = _StubRuntime(response=_developer_report())
    runner = ExtendRunner(runtime)
    outcome = runner.run(_parsed())
    assert outcome.ok is True
    assert outcome.report is not None
    assert outcome.report["status"] == "resolved"


def test_run_outcome_not_ok_on_in_progress_status():
    runtime = _StubRuntime(response=_developer_report(status="in_progress"))
    runner = ExtendRunner(runtime)
    outcome = runner.run(_parsed())
    # report is present but not resolved → ok=False
    assert outcome.report is not None
    assert outcome.ok is False


def test_run_outcome_not_ok_on_rejected_status():
    runtime = _StubRuntime(response=_developer_report(status="rejected"))
    runner = ExtendRunner(runtime)
    outcome = runner.run(_parsed())
    assert outcome.ok is False


def test_run_returns_error_on_runtime_exception():
    runtime = _StubRuntime(raise_exc=RuntimeError_("subprocess died"))
    runner = ExtendRunner(runtime)
    outcome = runner.run(_parsed())
    assert outcome.report is None
    assert outcome.error is not None
    assert "subprocess died" in outcome.error


def test_run_returns_error_on_non_dict_response():
    runtime = _StubRuntime(response="not a dict")
    runner = ExtendRunner(runtime)
    outcome = runner.run(_parsed())
    assert outcome.report is None
    assert "non-dict" in (outcome.error or "")


def test_run_returns_error_on_wrong_type():
    runtime = _StubRuntime(response={"type": "DashboardSpec"})  # wrong envelope
    runner = ExtendRunner(runtime)
    outcome = runner.run(_parsed())
    assert outcome.report is None
    assert "unexpected type" in (outcome.error or "")


def test_run_passes_widget_type_and_rationale_to_runtime():
    runtime = _StubRuntime(response=_developer_report())
    runner = ExtendRunner(runtime)
    runner.run(_parsed("bar_chart"), user_intent="show bar chart of cpu")
    op, args = runtime.calls[0]
    assert op == "rescue_extend"
    assert args["extend"]["widget_type"] == "bar_chart"
    assert "show bar chart" in args["user_intent"]


def test_run_truncates_long_user_intent():
    runtime = _StubRuntime(response=_developer_report())
    runner = ExtendRunner(runtime)
    long_msg = "x" * 5000
    runner.run(_parsed(), user_intent=long_msg)
    _, args = runtime.calls[0]
    assert len(args["user_intent"]) == 1024


def test_run_validates_widget_type_via_pydantic():
    """Even if a caller hands us a ParsedExtendRequest with a bad name,
    the runner's Pydantic re-validation must reject it before any
    rescue_extend invocation happens."""
    runtime = _StubRuntime(response=_developer_report())
    runner = ExtendRunner(runtime)
    bad = ParsedExtendRequest(
        widget_type="PieChart",  # uppercase — Pydantic regex will reject
        source_agent="x", summary="y", user_visible_effect="z",
    )
    outcome = runner.run(bad)
    assert outcome.report is None
    assert "extend request invalid" in (outcome.error or "")
    # And the runtime was never called.
    assert runtime.calls == []
