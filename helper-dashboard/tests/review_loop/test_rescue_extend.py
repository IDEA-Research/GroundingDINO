"""M4: review_loop kind=extend handling.

When Big guy in rescue_review emits RescueDecision(kind="extend"), the
loop must invoke rescue_extend, then return a graceful clarify
message (full re-render after extend is a future milestone).
"""

from __future__ import annotations

from typing import Any

from app.helper.extend_runner import ExtendOutcome, ExtendRunner
from app.helper.review_loop import ReviewLoop, ReviewOutcome
from app.services.spec_validator import SpecValidator
from app.specs import DashboardSpec
from app.specs.evaluation_report import BrowserEvaluationReport


def _spec() -> DashboardSpec:
    return SpecValidator().validate_dashboard({
        "dashboard_id": "d",
        "title": "Demo",
        "description": "",
        "layout": {"columns": 12, "row_height": 40},
        "variables": [],
        "widgets": [
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
    def __init__(self, reports):
        self._reports = list(reports)

    def evaluate(self, spec):
        payload = self._reports.pop(0) if self._reports else {
            "page_loaded": True, "widgets_rendered": [], "missing_widgets": [],
            "console_errors": [], "layout_errors": [], "prometheus_errors": [],
            "recommendation": "clean",
        }
        payload.setdefault("dashboard_id", spec.dashboard_id)
        return BrowserEvaluationReport.model_validate(payload)


class _StubRuntime:
    def __init__(self, script):
        self._script = list(script)
        self.calls: list[tuple[str, dict]] = []

    def invoke_operation(self, operation, args, *, developer=False):
        self.calls.append((operation, args))
        op, result = self._script.pop(0)
        assert op == operation, f"expected {op!r}, got {operation!r}"
        return dict(result)


class _StubExtendRunner:
    """Returns a canned ExtendOutcome and records the call."""

    def __init__(self, outcome: ExtendOutcome):
        self._outcome = outcome
        self.calls: list[tuple] = []

    def run(self, extend, *, user_intent="", original_args=None):
        self.calls.append((extend.widget_type, user_intent))
        return self._outcome


# ---------------------------------------------------------------------------
# Happy path: extend resolves → clarify with "added widget, please retry"
# ---------------------------------------------------------------------------


def test_rescue_extend_kind_resolved_returns_clarify_with_widget_hint():
    runtime = _StubRuntime([
        ("review_rendered", {"decision": "escalate",
                              "rationale": "missing widget"}),
        ("rescue_review", {
            "kind": "extend",
            "rationale": "user wants pie chart but toolkit lacks it",
            "extend": {"widget_type": "pie_chart",
                       "rationale": "pie chart is the right viz"},
        }),
    ])
    eval_ = _StubEvaluator([{
        "page_loaded": True, "widgets_rendered": ["w1"],
        "missing_widgets": [], "console_errors": [],
        "layout_errors": [], "prometheus_errors": [],
        "recommendation": "ok",
    }])
    extend = _StubExtendRunner(ExtendOutcome(report={
        "type": "DeveloperReport", "status": "resolved",
        "summary": "added", "actions_taken": ["wrote PieChartWidget.tsx"],
    }))

    loop = ReviewLoop(
        runtime,  # type: ignore[arg-type]
        evaluator=eval_,  # type: ignore[arg-type]
        extend_runner=extend,  # type: ignore[arg-type]
    )
    outcome = loop.run(_spec(), user_intent="give me a pie chart of memory")

    assert outcome.kind == "clarify"
    assert outcome.questions is not None
    assert any("pie_chart" in q for q in outcome.questions)
    assert extend.calls[0][0] == "pie_chart"

    # Trail should include both extend stages.
    stages = [t.get("stage") for t in outcome.trail]
    assert "rescue_extend_start" in stages
    assert "rescue_extend_resolved" in stages


# ---------------------------------------------------------------------------
# extend fails → clarify with fallback message
# ---------------------------------------------------------------------------


def test_rescue_extend_kind_failed_falls_back_to_clarify():
    runtime = _StubRuntime([
        ("review_rendered", {"decision": "escalate", "rationale": "bad"}),
        ("rescue_review", {
            "kind": "extend",
            "rationale": "x",
            "extend": {"widget_type": "heatmap", "rationale": "y"},
        }),
    ])
    eval_ = _StubEvaluator([{
        "page_loaded": True, "widgets_rendered": ["w1"],
        "missing_widgets": [], "console_errors": [],
        "layout_errors": [], "prometheus_errors": [],
        "recommendation": "ok",
    }])
    extend = _StubExtendRunner(ExtendOutcome(error="subprocess died"))

    loop = ReviewLoop(
        runtime, evaluator=eval_, extend_runner=extend,  # type: ignore[arg-type]
    )
    outcome = loop.run(_spec(), user_intent="heatmap please")

    assert outcome.kind == "clarify"
    assert any("couldn't finish" in q for q in (outcome.questions or []))
    stages = [t.get("stage") for t in outcome.trail]
    assert "rescue_extend_failed" in stages


# ---------------------------------------------------------------------------
# extend without widget_type → invalid, clarify
# ---------------------------------------------------------------------------


def test_rescue_extend_kind_missing_widget_type_falls_back():
    runtime = _StubRuntime([
        ("review_rendered", {"decision": "escalate", "rationale": "bad"}),
        ("rescue_review", {
            "kind": "extend",
            "rationale": "x",
            "extend": {},  # no widget_type
        }),
    ])
    eval_ = _StubEvaluator([{
        "page_loaded": True, "widgets_rendered": ["w1"],
        "missing_widgets": [], "console_errors": [],
        "layout_errors": [], "prometheus_errors": [],
        "recommendation": "ok",
    }])
    extend = _StubExtendRunner(ExtendOutcome(report={"type": "DeveloperReport"}))

    loop = ReviewLoop(
        runtime, evaluator=eval_, extend_runner=extend,  # type: ignore[arg-type]
    )
    outcome = loop.run(_spec(), user_intent="x")
    assert outcome.kind == "clarify"
    stages = [t.get("stage") for t in outcome.trail]
    assert "rescue_extend_invalid" in stages
    # ExtendRunner must not have been called.
    assert extend.calls == []
