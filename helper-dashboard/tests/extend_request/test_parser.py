"""Tests for the extend-toolkit ticket parser.

The parser recognizes the canonical `requested_action` string emitted
by `dashboard-spec-agent` / `patch-agent` when the user asks for a
widget type that's not in the toolkit. M4 will route these through
the rescue_extend path; M1 only exercises detection.
"""

from __future__ import annotations

from app.helper.extend_request import ExtendRequest, parse_extend_ticket


def _ticket(action: str, **extras) -> dict:
    """Build a DeveloperTicket-shaped dict with a custom requested_action."""
    base = {
        "type": "DeveloperTicket",
        "source_agent": "dashboard-spec-agent",
        "severity": "medium",
        "summary": "pie_chart widget type not in toolkit",
        "user_visible_effect": "user asked for a pie chart of memory by host",
        "technical_evidence": "WidgetType enum allows: line_chart, ...",
        "requested_action": action,
        "safety_notes": "",
    }
    base.update(extras)
    return base


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_canonical_extend_request_parses():
    raw = _ticket("extend widget toolkit with pie_chart")
    got = parse_extend_ticket(raw)
    assert isinstance(got, ExtendRequest)
    assert got.widget_type == "pie_chart"
    assert got.source_agent == "dashboard-spec-agent"


def test_extend_request_is_case_insensitive_on_verb():
    # The verb is permissive — agents may capitalize in prose.
    raw = _ticket("Extend Widget Toolkit With pie_chart")
    got = parse_extend_ticket(raw)
    assert got is not None
    assert got.widget_type == "pie_chart"


def test_extend_request_tolerates_surrounding_whitespace():
    raw = _ticket("   extend widget toolkit with bar_chart   ")
    got = parse_extend_ticket(raw)
    assert got is not None
    assert got.widget_type == "bar_chart"


def test_extend_request_normalizes_widget_type_to_lowercase():
    # If an agent slips uppercase chars in the widget_type slot, we
    # accept the spelling and normalize. The strict snake_case check
    # still applies to the lowered form.
    raw = _ticket("extend widget toolkit with Heatmap")
    got = parse_extend_ticket(raw)
    assert got is not None
    assert got.widget_type == "heatmap"


def test_extend_request_preserves_metadata():
    raw = _ticket(
        "extend widget toolkit with sankey",
        source_agent="patch-agent",
        summary="sankey not in toolkit",
        user_visible_effect="user wanted a sankey flow",
    )
    got = parse_extend_ticket(raw)
    assert got is not None
    assert got.source_agent == "patch-agent"
    assert got.summary == "sankey not in toolkit"
    assert got.user_visible_effect == "user wanted a sankey flow"


# ---------------------------------------------------------------------------
# Negative paths — must return None, not raise
# ---------------------------------------------------------------------------


def test_none_input_returns_none():
    assert parse_extend_ticket(None) is None


def test_non_dict_input_returns_none():
    assert parse_extend_ticket("not a dict") is None  # type: ignore[arg-type]
    assert parse_extend_ticket(42) is None  # type: ignore[arg-type]


def test_missing_requested_action_returns_none():
    raw = {"type": "DeveloperTicket", "source_agent": "x"}
    assert parse_extend_ticket(raw) is None


def test_non_string_requested_action_returns_none():
    raw = _ticket(action="")
    raw["requested_action"] = 123  # type: ignore[assignment]
    assert parse_extend_ticket(raw) is None


def test_unrelated_action_returns_none():
    # A real bug-report ticket, not an extension request.
    raw = _ticket("fix the broken legend on line_chart")
    assert parse_extend_ticket(raw) is None


def test_action_with_extra_words_returns_none():
    # The pattern is strict on shape: anything before/after the
    # widget_type slot disqualifies. We do not want to match
    # "please extend widget toolkit with X eventually".
    raw = _ticket("please extend widget toolkit with pie_chart eventually")
    assert parse_extend_ticket(raw) is None


def test_widget_type_too_short_returns_none():
    raw = _ticket("extend widget toolkit with pi")  # 2 chars, min is 3
    assert parse_extend_ticket(raw) is None


def test_widget_type_too_long_returns_none():
    raw = _ticket("extend widget toolkit with " + "x" * 33)
    assert parse_extend_ticket(raw) is None


def test_widget_type_starting_with_digit_returns_none():
    raw = _ticket("extend widget toolkit with 3dscatter")
    assert parse_extend_ticket(raw) is None


def test_widget_type_with_hyphen_returns_none():
    # Hyphens aren't snake_case; reject so the downstream Python enum
    # generator never sees an illegal identifier.
    raw = _ticket("extend widget toolkit with pie-chart")
    assert parse_extend_ticket(raw) is None


def test_widget_type_with_space_returns_none():
    raw = _ticket("extend widget toolkit with pie chart")
    assert parse_extend_ticket(raw) is None
