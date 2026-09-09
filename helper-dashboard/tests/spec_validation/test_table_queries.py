"""WidgetSpec.queries (multi-metric table) + QuerySpec key/label/unit +
DashboardSpec.refresh_interval ms clamp."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.services.spec_validator import SpecValidationError, SpecValidator
from app.specs import DashboardSpec
from app.specs.dashboard_spec import MIN_REFRESH_SECONDS, clamp_refresh_interval
from app.specs.widget_spec import MAX_TABLE_QUERIES, WidgetSpec

HR = "max by (patient_id) (patient_heart_rate_bpm)"
SPO2 = "max by (patient_id) (patient_spo2_percentage)"
POS = {"x": 0, "y": 0, "w": 12, "h": 6}


def _table(**overrides):
    base = {"id": "tbl", "type": "table", "title": "各病患一覽", "position": POS}
    base.update(overrides)
    return base


def test_table_with_queries_may_omit_query_and_autofills_keys():
    w = WidgetSpec.model_validate(_table(queries=[
        {"promql": HR, "query_type": "instant", "label": "心率", "unit": "bpm"},
        {"promql": SPO2, "query_type": "instant", "key": "spo2"},
    ]))
    assert w.query is None
    assert [q.key for q in w.queries] == ["q1", "spo2"]
    assert w.queries[0].label == "心率" and w.queries[0].unit == "bpm"
    assert w.promqls() == [HR, SPO2]
    assert len(w.all_queries()) == 2
    dumped = w.model_dump(mode="json")
    assert dumped["query"] is None
    assert dumped["queries"][0]["key"] == "q1"


def test_table_may_carry_both_query_and_queries():
    w = WidgetSpec.model_validate(_table(
        query={"promql": HR, "query_type": "instant"},
        queries=[{"promql": SPO2, "query_type": "instant", "key": "spo2"}],
    ))
    assert w.promqls() == [HR, SPO2]


def test_queries_forbidden_on_non_table():
    with pytest.raises(ValidationError, match="only valid on a table"):
        WidgetSpec.model_validate({
            "id": "s", "type": "stat_card", "title": "x", "position": {"x": 0, "y": 0, "w": 3, "h": 3},
            "query": {"promql": HR, "query_type": "instant"},
            "queries": [{"promql": SPO2, "query_type": "instant"}],
        })


def test_query_required_without_queries():
    with pytest.raises(ValidationError, match="query is required"):
        WidgetSpec.model_validate(_table())
    with pytest.raises(ValidationError, match="query is required"):
        WidgetSpec.model_validate(_table(queries=[]))  # empty == not provided
    with pytest.raises(ValidationError):
        WidgetSpec.model_validate({"id": "s", "type": "stat_card", "title": "x",
                                   "position": {"x": 0, "y": 0, "w": 3, "h": 3}})


def test_duplicate_keys_and_limit():
    with pytest.raises(ValidationError, match="duplicated"):
        WidgetSpec.model_validate(_table(queries=[
            {"promql": HR, "key": "a"}, {"promql": SPO2, "key": "a"},
        ]))
    with pytest.raises(ValidationError):
        WidgetSpec.model_validate(_table(queries=[
            {"promql": HR, "key": f"k{i}"} for i in range(MAX_TABLE_QUERIES + 1)
        ]))
    assert MAX_TABLE_QUERIES == 6


def test_query_key_pattern_and_safe_text():
    with pytest.raises(ValidationError):
        WidgetSpec.model_validate(_table(queries=[{"promql": HR, "key": "1bad"}]))
    with pytest.raises(ValidationError):
        WidgetSpec.model_validate(_table(queries=[{"promql": HR, "key": "a-b"}]))
    with pytest.raises(ValidationError):
        WidgetSpec.model_validate(_table(queries=[{"promql": HR, "key": "ok", "label": "<script>x"}]))


def test_queries_promql_still_token_filtered():
    with pytest.raises(ValidationError, match="forbidden token"):
        WidgetSpec.model_validate(_table(queries=[{"promql": "up; drop", "key": "a"}]))


def test_dashboard_validator_accepts_queries_table():
    spec = SpecValidator().validate_dashboard({
        "dashboard_id": "d", "title": "T",
        "widgets": [_table(queries=[
            {"promql": HR, "query_type": "instant", "key": "hr"},
            {"promql": SPO2, "query_type": "instant", "key": "spo2"},
        ])],
    })
    assert spec.widgets[0].queries[1].key == "spo2"
    with pytest.raises(SpecValidationError):
        SpecValidator().validate_dashboard({
            "dashboard_id": "d", "title": "T",
            "widgets": [_table(queries=[{"promql": HR, "key": "hr", "render": "x"}])],
        })


def test_existing_specs_unchanged():
    """A classic widget round-trips byte-for-byte apart from the new
    null/None fields (query stays required and present)."""
    w = WidgetSpec.model_validate({
        "id": "w", "type": "line_chart", "title": "CPU",
        "query": {"source": "prometheus", "promql": "up", "query_type": "range", "range": "1h", "step": "30s"},
        "position": {"x": 0, "y": 0, "w": 6, "h": 6}, "encoding": {}, "thresholds": [], "options": {},
    })
    d = w.model_dump(mode="json")
    assert d["query"]["promql"] == "up" and d["queries"] is None
    assert d["query"]["key"] is None and d["query"]["label"] is None and d["query"]["unit"] is None


# ---------------------------------------------------------------------------
# refresh_interval: ms accepted, clamped >= 5s
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("500ms", "5s"), ("1s", "5s"), ("4s", "5s"), ("5s", "5s"), ("30s", "30s"),
    ("5m", "5m"), ("1h", "1h"), ("6000ms", "6s"), ("7500ms", "8s"), ("9000ms", "9s"),
])
def test_clamp_refresh_interval(raw, expected):
    assert clamp_refresh_interval(raw) == expected
    spec = DashboardSpec.model_validate({"dashboard_id": "d", "title": "T", "refresh_interval": raw})
    assert spec.refresh_interval == expected


def test_refresh_interval_min_is_five_seconds():
    assert MIN_REFRESH_SECONDS == 5
    with pytest.raises(ValidationError):
        DashboardSpec.model_validate({"dashboard_id": "d", "title": "T", "refresh_interval": "fast"})
