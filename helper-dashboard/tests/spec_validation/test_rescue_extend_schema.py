"""Tests for the M2 schema additions: ExtendRequest model,
RescueDecision.kind="extend", and the rescue_extend operation entry.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.helper.runtime import (
    EXPECTED_OUTPUT_TYPES,
    OPERATIONS,
    USER_OPERATIONS,
)
from app.specs import ExtendRequest, RescueDecision


# ---------------------------------------------------------------------------
# ExtendRequest schema
# ---------------------------------------------------------------------------


def test_extend_request_minimal_valid():
    req = ExtendRequest(
        widget_type="pie_chart",
        rationale="user asked for pie chart of memory by host",
    )
    assert req.widget_type == "pie_chart"
    assert req.component_hint is None


def test_extend_request_with_component_hint():
    req = ExtendRequest(
        widget_type="heatmap",
        rationale="cohort analysis needs intensity grid",
        component_hint="use recharts Treemap or a custom svg grid",
    )
    assert req.component_hint == "use recharts Treemap or a custom svg grid"


def test_extend_request_rejects_uppercase_widget_type():
    with pytest.raises(ValidationError):
        ExtendRequest(widget_type="PieChart", rationale="x")


def test_extend_request_rejects_hyphen_widget_type():
    with pytest.raises(ValidationError):
        ExtendRequest(widget_type="pie-chart", rationale="x")


def test_extend_request_rejects_widget_type_too_short():
    with pytest.raises(ValidationError):
        ExtendRequest(widget_type="pi", rationale="x")


def test_extend_request_rejects_widget_type_too_long():
    with pytest.raises(ValidationError):
        ExtendRequest(widget_type="x" * 33, rationale="x")


def test_extend_request_rejects_widget_type_starting_with_digit():
    with pytest.raises(ValidationError):
        ExtendRequest(widget_type="3d_scatter", rationale="x")


def test_extend_request_rejects_extra_fields():
    """ExtendRequest must be a closed shape — LLMs sometimes hallucinate
    extra fields like `code`, `script`, `path`. They must be rejected
    by Pydantic before they reach the runtime gate."""
    with pytest.raises(ValidationError):
        ExtendRequest.model_validate({
            "widget_type": "pie_chart",
            "rationale": "x",
            "code": "import os; os.system('rm -rf /')",  # injection attempt
        })


def test_extend_request_requires_rationale():
    with pytest.raises(ValidationError):
        ExtendRequest(widget_type="pie_chart", rationale="")


# ---------------------------------------------------------------------------
# RescueDecision with kind=extend
# ---------------------------------------------------------------------------


def test_rescue_decision_extend_kind_parses():
    raw = {
        "kind": "extend",
        "rationale": "missing pie_chart widget caused render failure",
        "extend": {
            "widget_type": "pie_chart",
            "rationale": "user explicitly requested pie",
        },
    }
    dec = RescueDecision.model_validate(raw)
    assert dec.kind == "extend"
    assert dec.extend is not None
    assert dec.extend.widget_type == "pie_chart"


def test_rescue_decision_extend_with_invalid_widget_type_rejected():
    """An invalid nested ExtendRequest must invalidate the whole
    RescueDecision, not silently strip the bad field."""
    raw = {
        "kind": "extend",
        "rationale": "x",
        "extend": {"widget_type": "PieChart", "rationale": "x"},
    }
    with pytest.raises(ValidationError):
        RescueDecision.model_validate(raw)


def test_rescue_decision_extend_without_extend_field_is_currently_allowed():
    """The extend field is Optional in the model — M4's review_loop
    handler is responsible for checking it's present when kind==extend.
    This test pins the current behavior so the contract doesn't change
    by accident; if M4 tightens this, update both places together."""
    raw = {"kind": "extend", "rationale": "x"}
    dec = RescueDecision.model_validate(raw)
    assert dec.extend is None


def test_rescue_decision_extend_with_extra_fields_rejected():
    raw = {
        "kind": "extend",
        "rationale": "x",
        "extend": {"widget_type": "pie_chart", "rationale": "x"},
        "shell_command": "ls",  # not a field
    }
    with pytest.raises(ValidationError):
        RescueDecision.model_validate(raw)


def test_existing_rescue_decision_kinds_still_work():
    """M2 adds 'extend' to the kind Literal — must not break existing
    patch / ask_user / ticket shapes."""
    RescueDecision.model_validate({
        "kind": "ask_user",
        "rationale": "need clarification",
        "questions": ["which time range?"],
    })
    RescueDecision.model_validate({
        "kind": "patch",
        "rationale": "fix is straightforward",
        "patch": {
            "patch_id": "p1", "reason": "r",
            "target_dashboard_id": "d", "created_by": "big-guy",
            "operations": [{
                "op": "update_dashboard",
                "fields": {"title": "X"},
            }],
        },
    })


# ---------------------------------------------------------------------------
# runtime.py OPERATIONS entry
# ---------------------------------------------------------------------------


def test_rescue_extend_operation_registered():
    assert "rescue_extend" in OPERATIONS
    op = OPERATIONS["rescue_extend"]
    assert op["agent"] == "big-guy-developer-agent"
    assert op["command"] == "rescue-extend"


def test_rescue_extend_in_user_operations():
    """User chat must be able to (indirectly) reach rescue_extend via
    the review loop. The M5 env flag is what stops the loop from
    invoking it when auto-extend is disabled — the allow-list itself
    permits the call."""
    assert "rescue_extend" in USER_OPERATIONS


def test_rescue_extend_expected_output_is_developer_report():
    assert EXPECTED_OUTPUT_TYPES["rescue_extend"] == frozenset({"DeveloperReport"})
