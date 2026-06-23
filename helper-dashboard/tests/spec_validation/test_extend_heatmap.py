"""Tests for the heatmap widget type extension.

Validates that heatmap is properly registered in the backend enum,
frontend schema, and can be used in dashboard/patch specs.
"""

from __future__ import annotations

import pytest

from app.services.patch_service import PatchService
from app.services.spec_validator import SpecValidationError, SpecValidator
from app.specs.widget_spec import ALLOWED_OPTION_KEYS, WidgetType


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _widget(**overrides):
    w = {
        "id": "hm1",
        "type": "heatmap",
        "title": "Request Latency by Endpoint and Hour",
        "description": "Heatmap of p99 latency",
        "query": {
            "source": "prometheus",
            "promql": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
            "query_type": "instant",
        },
        "position": {"x": 0, "y": 0, "w": 12, "h": 8},
        "encoding": {},
        "thresholds": [],
        "options": {},
    }
    w.update(overrides)
    return w


def _dashboard(widgets=None):
    return SpecValidator().validate_dashboard(
        {
            "dashboard_id": "heatmap-test",
            "title": "Heatmap Test",
            "description": "",
            "layout": {"columns": 12, "row_height": 40},
            "variables": [],
            "widgets": widgets if widgets is not None else [_widget()],
            "refresh_interval": "30s",
        }
    )


def _patch(ops, *, target="heatmap-test"):
    return SpecValidator().validate_patch(
        {
            "patch_id": "p-heatmap",
            "reason": "test heatmap",
            "target_dashboard_id": target,
            "created_by": "test",
            "operations": ops,
        }
    )


# ---------------------------------------------------------------------------
# 1. heatmap is a valid WidgetType enum member
# ---------------------------------------------------------------------------

def test_heatmap_in_widget_type_enum():
    assert "heatmap" in [wt.value for wt in WidgetType]
    assert WidgetType("heatmap") == WidgetType.heatmap


# ---------------------------------------------------------------------------
# 2. A dashboard with a heatmap widget validates successfully
# ---------------------------------------------------------------------------

def test_heatmap_dashboard_validates():
    spec = _dashboard()
    assert len(spec.widgets) == 1
    assert spec.widgets[0].type == WidgetType.heatmap
    assert spec.widgets[0].title == "Request Latency by Endpoint and Hour"


# ---------------------------------------------------------------------------
# 3. heatmap-specific options (x_label, y_label, color_scale) are accepted
# ---------------------------------------------------------------------------

def test_heatmap_options_accepted():
    spec = _dashboard(
        widgets=[
            _widget(options={
                "x_label": "Hour of Day",
                "y_label": "Endpoint",
                "color_scale": "warm",
                "decimals": 2,
                "show_legend": True,
            }),
        ]
    )
    assert spec.widgets[0].options["x_label"] == "Hour of Day"
    assert spec.widgets[0].options["y_label"] == "Endpoint"
    assert spec.widgets[0].options["color_scale"] == "warm"
    assert spec.widgets[0].options["decimals"] == 2
    assert spec.widgets[0].options["show_legend"] is True


# ---------------------------------------------------------------------------
# 4. heatmap option keys are in ALLOWED_OPTION_KEYS
# ---------------------------------------------------------------------------

def test_heatmap_option_keys_in_allowed_set():
    assert "x_label" in ALLOWED_OPTION_KEYS
    assert "y_label" in ALLOWED_OPTION_KEYS
    assert "color_scale" in ALLOWED_OPTION_KEYS


# ---------------------------------------------------------------------------
# 5. heatmap can be added via PatchSpec
# ---------------------------------------------------------------------------

def test_heatmap_add_via_patch():
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
        [{"op": "add_widget", "widget": _widget(id="hm2", position={"x": 6, "y": 0, "w": 6, "h": 8})}],
        target="heatmap-test",
    )
    new = PatchService().apply(base, patch)
    assert len(new.widgets) == 2
    assert new.widgets[1].type == WidgetType.heatmap


# ---------------------------------------------------------------------------
# 6. heatmap with decimals option validates
# ---------------------------------------------------------------------------

def test_heatmap_with_decimals():
    spec = _dashboard(widgets=[_widget(options={"decimals": 3})])
    assert spec.widgets[0].options["decimals"] == 3


# ---------------------------------------------------------------------------
# 7. Invalid options on heatmap are still rejected
# ---------------------------------------------------------------------------

def test_heatmap_invalid_option_rejected():
    with pytest.raises(SpecValidationError):
        _dashboard(widgets=[_widget(options={"on_click": "alert(1)"})])


# ---------------------------------------------------------------------------
# 8. heatmap is in the schema doc
# ---------------------------------------------------------------------------

def test_heatmap_in_schema_doc():
    from app.specs.widget_schema_doc import WIDGET_SCHEMA_MARKDOWN
    assert "heatmap" in WIDGET_SCHEMA_MARKDOWN
    assert "Two-dimensional grid" in WIDGET_SCHEMA_MARKDOWN
