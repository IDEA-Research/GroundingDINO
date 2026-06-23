"""Tests for the bar_chart widget type extension.

Validates that bar_chart is properly registered in the backend enum,
frontend schema, and can be used in dashboard/patch specs.
"""

from __future__ import annotations

import pytest

from app.services.patch_service import PatchService
from app.services.spec_validator import SpecValidationError, SpecValidator
from app.specs.widget_spec import WidgetType


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _widget(**overrides):
    w = {
        "id": "bar1",
        "type": "bar_chart",
        "title": "CPU Usage per Node",
        "description": "Bar chart showing CPU usage breakdown by node",
        "query": {
            "source": "prometheus",
            "promql": "node_cpu_seconds_total",
            "query_type": "instant",
        },
        "position": {"x": 0, "y": 0, "w": 6, "h": 6},
        "encoding": {},
        "thresholds": [],
        "options": {},
    }
    w.update(overrides)
    return w


def _dashboard(widgets=None):
    return SpecValidator().validate_dashboard(
        {
            "dashboard_id": "bar-test",
            "title": "Bar Test",
            "description": "",
            "layout": {"columns": 12, "row_height": 40},
            "variables": [],
            "widgets": widgets if widgets is not None else [_widget()],
            "refresh_interval": "30s",
        }
    )


def _patch(ops, *, target="bar-test"):
    return SpecValidator().validate_patch(
        {
            "patch_id": "p-bar",
            "reason": "test bar_chart",
            "target_dashboard_id": target,
            "created_by": "test",
            "operations": ops,
        }
    )


# ---------------------------------------------------------------------------
# 1. bar_chart is a valid WidgetType enum member
# ---------------------------------------------------------------------------

def test_bar_chart_in_widget_type_enum():
    assert "bar_chart" in [wt.value for wt in WidgetType]
    assert WidgetType("bar_chart") == WidgetType.bar_chart


# ---------------------------------------------------------------------------
# 2. A dashboard with a bar_chart widget validates successfully
# ---------------------------------------------------------------------------

def test_bar_chart_dashboard_validates():
    spec = _dashboard()
    assert len(spec.widgets) == 1
    assert spec.widgets[0].type == WidgetType.bar_chart
    assert spec.widgets[0].title == "CPU Usage per Node"


# ---------------------------------------------------------------------------
# 3. bar_chart-specific options (horizontal, show_values) are accepted
# ---------------------------------------------------------------------------

def test_bar_chart_options_accepted():
    spec = _dashboard(
        widgets=[
            _widget(options={
                "horizontal": True,
                "show_values": True,
                "show_grid": False,
                "show_legend": True,
                "stacked": False,
                "decimals": 1,
            }),
        ]
    )
    assert spec.widgets[0].options["horizontal"] is True
    assert spec.widgets[0].options["show_values"] is True
    assert spec.widgets[0].options["show_grid"] is False
    assert spec.widgets[0].options["decimals"] == 1


# ---------------------------------------------------------------------------
# 4. bar_chart can be added via PatchSpec
# ---------------------------------------------------------------------------

def test_bar_chart_add_via_patch():
    base = _dashboard(
        widgets=[
            {
                "id": "w1",
                "type": "line_chart",
                "title": "CPU",
                "query": {
                    "source": "prometheus",
                    "promql": "rate(node_cpu_seconds_total[5m])",
                    "query_type": "range",
                    "range": "1h",
                    "step": "30s",
                },
                "position": {"x": 0, "y": 0, "w": 6, "h": 6},
                "encoding": {},
                "thresholds": [],
                "options": {},
            }
        ]
    )
    patch = _patch(
        [{"op": "add_widget", "widget": _widget(id="bar2", position={"x": 6, "y": 0, "w": 6, "h": 6})}],
        target="bar-test",
    )
    new = PatchService().apply(base, patch)
    assert len(new.widgets) == 2
    assert new.widgets[1].type == WidgetType.bar_chart


# ---------------------------------------------------------------------------
# 5. bar_chart with decimals option validates
# ---------------------------------------------------------------------------

def test_bar_chart_with_decimals():
    spec = _dashboard(widgets=[_widget(options={"decimals": 2})])
    assert spec.widgets[0].options["decimals"] == 2


# ---------------------------------------------------------------------------
# 6. Invalid options on bar_chart are still rejected
# ---------------------------------------------------------------------------

def test_bar_chart_invalid_option_rejected():
    with pytest.raises(SpecValidationError):
        _dashboard(widgets=[_widget(options={"on_click": "alert(1)"})])


# ---------------------------------------------------------------------------
# 7. bar_chart with horizontal option validates
# ---------------------------------------------------------------------------

def test_bar_chart_horizontal_option():
    spec = _dashboard(widgets=[_widget(options={"horizontal": True})])
    assert spec.widgets[0].options["horizontal"] is True


# ---------------------------------------------------------------------------
# 8. bar_chart with show_values option validates
# ---------------------------------------------------------------------------

def test_bar_chart_show_values_option():
    spec = _dashboard(widgets=[_widget(options={"show_values": True})])
    assert spec.widgets[0].options["show_values"] is True
