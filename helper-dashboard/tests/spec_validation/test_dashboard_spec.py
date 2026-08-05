"""Schema validation tests.

These assert the contract that every Helper agent must produce. If
Big guy changes a schema without updating these tests, CI should fail.
"""

import pytest

from app.services.spec_validator import SpecValidationError, SpecValidator


def _base_widget(**overrides):
    w = {
        "id": "w1",
        "type": "line_chart",
        "title": "CPU",
        "description": "",
        "query": {
            "source": "prometheus",
            "promql": "rate(http_requests_total[5m])",
            "query_type": "range",
            "range": "1h",
            "step": "30s",
        },
        "position": {"x": 0, "y": 0, "w": 6, "h": 6},
        "encoding": {},
        "thresholds": [],
        "options": {},
    }
    w.update(overrides)
    return w


def _base_dashboard(**overrides):
    d = {
        "dashboard_id": "demo",
        "title": "Demo",
        "description": "",
        "layout": {"columns": 12, "row_height": 40},
        "variables": [],
        "widgets": [_base_widget()],
        "refresh_interval": "30s",
    }
    d.update(overrides)
    return d


def test_valid_dashboard_round_trips():
    v = SpecValidator()
    spec = v.validate_dashboard(_base_dashboard())
    assert spec.title == "Demo"
    assert spec.widgets[0].type.value == "line_chart"


def test_rejects_unsupported_widget_type():
    # `sankey` is intentionally not in WidgetType. If the auto-extend
    # pipeline ever adds it, pick a fresh name that isn't cached yet.
    v = SpecValidator()
    d = _base_dashboard(widgets=[_base_widget(type="sankey")])
    with pytest.raises(SpecValidationError):
        v.validate_dashboard(d)


def test_rejects_script_in_title():
    v = SpecValidator()
    d = _base_dashboard(title="<script>alert(1)</script>")
    with pytest.raises(SpecValidationError):
        v.validate_dashboard(d)


def test_rejects_forbidden_widget_field():
    v = SpecValidator()
    bad_widget = _base_widget()
    bad_widget["raw_html"] = "<b>nope</b>"
    d = _base_dashboard(widgets=[bad_widget])
    with pytest.raises(SpecValidationError) as exc:
        v.validate_dashboard(d)
    assert any("raw_html" in e for e in exc.value.errors)


def test_rejects_duplicate_widget_ids():
    v = SpecValidator()
    d = _base_dashboard(widgets=[_base_widget(id="w1"), _base_widget(id="w1")])
    with pytest.raises(SpecValidationError):
        v.validate_dashboard(d)


def test_rejects_layout_overflow():
    v = SpecValidator()
    w = _base_widget()
    w["position"] = {"x": 10, "y": 0, "w": 6, "h": 6}  # 10+6 > 12 columns
    d = _base_dashboard(widgets=[w])
    with pytest.raises(SpecValidationError) as exc:
        v.validate_dashboard(d)
    assert any("overflows" in e for e in exc.value.errors)


def test_rejects_promql_with_shell_meta():
    v = SpecValidator()
    w = _base_widget()
    w["query"]["promql"] = "up; rm -rf /"
    d = _base_dashboard(widgets=[w])
    with pytest.raises(SpecValidationError):
        v.validate_dashboard(d)


def test_rejects_unknown_option_key():
    v = SpecValidator()
    w = _base_widget(options={"unsafe_exec": True})
    d = _base_dashboard(widgets=[w])
    with pytest.raises(SpecValidationError):
        v.validate_dashboard(d)


def test_rejects_bad_refresh_interval():
    v = SpecValidator()
    with pytest.raises(SpecValidationError):
        v.validate_dashboard(_base_dashboard(refresh_interval="every 30s"))
