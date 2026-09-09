"""Threshold.direction + the shared resolver (specs/threshold_semantics).

The frontend helper (frontend/lib/threshold.ts) mirrors this logic; the
cases below are the shared fixture set, including the live SpO2 ladder
that motivated the field.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.specs.threshold_semantics import (
    DIRECTION_RULES_DOC,
    active_threshold,
    color_severity,
    resolve_direction,
)
from app.specs.widget_spec import WidgetSpec, WidgetThreshold

LIVE_SPO2 = [
    {"value": 94, "color": "#f59e0b", "label": "偏低 <95%"},
    {"value": 90, "color": "#ef4444", "label": "危急 <90%"},
]
HR = [{"value": 100, "color": "#f59e0b"}, {"value": 120, "color": "#ef4444"}]


def test_live_examples():
    assert resolve_direction(LIVE_SPO2) == "below"
    assert resolve_direction(HR) == "above"


def test_explicit_direction_wins():
    assert resolve_direction([{"value": 1, "color": "red", "direction": "below"}]) == "below"
    # explicit beats severity ordering
    assert resolve_direction([
        {"value": 100, "color": "#f59e0b", "direction": "above"},
        {"value": 90, "color": "#ef4444"},
    ]) == "above"
    # first non-null in list order
    assert resolve_direction([
        {"value": 1, "color": "red"},
        {"value": 2, "color": "amber", "direction": "below"},
        {"value": 3, "color": "yellow", "direction": "above"},
    ]) == "below"


def test_severity_ordering_by_name_and_hue():
    assert resolve_direction([{"value": 0.95, "color": "amber"}, {"value": 0.99, "color": "green"}]) == "below"
    assert resolve_direction([{"value": 0.95, "color": "#f59e0b", "label": "warn <95%"},
                              {"value": 0.99, "color": "#34d399", "label": "healthy"}]) == "below"
    assert resolve_direction([{"value": 0.5, "color": "yellow"}, {"value": 1.0, "color": "red"}]) == "above"
    # equal severities fall through to labels / default
    assert resolve_direction([{"value": 1, "color": "red"}, {"value": 2, "color": "crimson"}]) == "above"
    assert resolve_direction([{"value": 1, "color": "red", "label": "低於下限"}, {"value": 2, "color": "red"}]) == "below"


def test_label_hints_and_default():
    assert resolve_direction([{"value": 88, "color": "red", "label": "危急 <88%"}]) == "below"
    assert resolve_direction([{"value": 88, "color": "red", "label": "≤ 88"}]) == "below"
    assert resolve_direction([{"value": 88, "color": "red", "label": "低於 88"}]) == "below"
    assert resolve_direction([{"value": 88, "color": "red", "label": "high"}]) == "above"
    assert resolve_direction([]) == "above"
    assert resolve_direction(None) == "above"


def test_resolver_accepts_models():
    models = [WidgetThreshold.model_validate(t) for t in LIVE_SPO2]
    assert resolve_direction(models) == "below"


def test_color_severity_ranks():
    assert color_severity("#ef4444") == 3 == color_severity("red") == color_severity("Critical")
    assert color_severity("#f59e0b") == 2 == color_severity("orange") == color_severity("amber")
    assert color_severity("yellow") == 1 == color_severity("gold")
    assert color_severity("#34d399") == 0 == color_severity("green") == color_severity("#9ca3af")
    assert color_severity(None) == 0 == color_severity("not-a-color!")
    assert color_severity("#f00") == 3  # short hex


def test_active_threshold_honours_direction():
    assert active_threshold(93, LIVE_SPO2)["label"] == "偏低 <95%"
    assert active_threshold(89, LIVE_SPO2)["label"] == "危急 <90%"
    assert active_threshold(97, LIVE_SPO2) is None
    assert active_threshold(110, HR)["value"] == 100
    assert active_threshold(125, HR)["value"] == 120
    assert active_threshold(95, HR) is None
    assert active_threshold(None, HR) is None
    assert active_threshold(5, []) is None


def test_threshold_model_direction_field():
    t = WidgetThreshold.model_validate({"value": 90, "color": "#ef4444", "direction": "below"})
    assert t.direction == "below"
    assert WidgetThreshold.model_validate({"value": 90, "color": "red"}).direction is None
    with pytest.raises(ValidationError):
        WidgetThreshold.model_validate({"value": 90, "color": "red", "direction": "sideways"})
    with pytest.raises(ValidationError):  # extra='forbid' still bites for op/condition
        WidgetThreshold.model_validate({"value": 90, "color": "red", "op": "<"})


def test_direction_round_trips_through_widget_spec():
    w = WidgetSpec.model_validate({
        "id": "spo2", "type": "stat_card", "title": "SpO2",
        "query": {"source": "prometheus", "promql": "max by (patient_id) (patient_spo2_percentage)",
                  "query_type": "instant"},
        "position": {"x": 0, "y": 0, "w": 3, "h": 3},
        "thresholds": [{**t, "direction": "below"} for t in LIVE_SPO2],
    })
    dumped = w.model_dump(mode="json")
    assert all(t["direction"] == "below" for t in dumped["thresholds"])
    # Values are untouched by direction resolution.
    assert [t["value"] for t in dumped["thresholds"]] == [94.0, 90.0]


def test_schema_doc_mentions_direction_and_queries():
    from app.specs.widget_schema_doc import WIDGET_SCHEMA_MARKDOWN

    assert "`direction`" in WIDGET_SCHEMA_MARKDOWN
    assert DIRECTION_RULES_DOC in WIDGET_SCHEMA_MARKDOWN
    assert "`queries`" in WIDGET_SCHEMA_MARKDOWN
    assert "NEVER embed a time range" in WIDGET_SCHEMA_MARKDOWN
    assert "stat_card` AND a `gauge" in WIDGET_SCHEMA_MARKDOWN
