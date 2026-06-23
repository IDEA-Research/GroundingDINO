import pytest
from pydantic import ValidationError

from backend.app.specs.dashboard_spec import DashboardSpec
from backend.app.specs.widget_spec import QuerySpec, WidgetSpec


def test_dashboard_schema_accepts_supported_widget_types():
    dashboard = DashboardSpec(
        title="Patient Monitor",
        widgets=[
            WidgetSpec(
                title="Heart Rate",
                type="line_chart",
                query=QuerySpec(metric="patient_heart_rate_bpm"),
                unit="bpm",
            )
        ],
    )

    assert dashboard.widgets[0].type == "line_chart"
    assert dashboard.refresh_interval_ms == 5000


def test_widget_schema_rejects_unsupported_widget_type():
    with pytest.raises(ValidationError):
        WidgetSpec(
            title="Unsupported",
            type="pie_chart",
            query=QuerySpec(metric="patient_heart_rate_bpm"),
        )

