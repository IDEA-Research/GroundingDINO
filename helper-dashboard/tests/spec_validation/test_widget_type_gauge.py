"""Test that the WidgetType enum includes gauge."""

from app.specs.widget_spec import WidgetType


def test_widget_type_includes_gauge():
    assert "gauge" in [member.value for member in WidgetType]
