import pytest

from backend.app.services.query_service import QueryService
from backend.app.specs.widget_spec import QuerySpec, WidgetSpec


@pytest.mark.asyncio
async def test_query_service_returns_frontend_friendly_mock_points(monkeypatch):
    monkeypatch.setenv("PROMETHEUS_MODE", "mock")
    service = QueryService()
    widget = WidgetSpec(
        title="Heart Rate",
        type="line_chart",
        query=QuerySpec(metric="patient_heart_rate_bpm", time_range_seconds=60, step_seconds=5),
        unit="bpm",
    )

    result = await service.query_widget(widget)

    assert result["metadata"]["source"] == "mock"
    assert result["series"][0]["name"] == "Heart Rate"
    assert result["series"][0]["unit"] == "bpm"
    assert {"timestamp", "value"} <= set(result["series"][0]["points"][0])


@pytest.mark.asyncio
async def test_query_service_supports_blood_pressure_two_line_mock_widget(monkeypatch):
    monkeypatch.setenv("PROMETHEUS_MODE", "mock")
    service = QueryService()
    widget = WidgetSpec(
        title="Blood Pressure",
        type="line_chart",
        query=QuerySpec(metric="patient_systolic_bp_mmhg", time_range_seconds=60, step_seconds=5),
        unit="mmHg",
        metadata={"secondary_metric": "patient_diastolic_bp_mmhg"},
    )

    result = await service.query_widget(widget)

    assert [series["name"] for series in result["series"]] == [
        "Systolic Blood Pressure",
        "Diastolic Blood Pressure",
    ]
