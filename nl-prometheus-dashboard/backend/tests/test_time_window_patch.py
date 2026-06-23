from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from backend.app.agents.patch_agent import PatchAgent
from backend.app.agents.time_window_parser import parse_prompt_time_window
from backend.app.specs.dashboard_spec import DashboardSpec
from backend.app.specs.widget_spec import QuerySpec, WidgetSpec


def test_parse_absolute_clock_window():
    parsed = parse_prompt_time_window(
        "I want the data from 10:00 am to 10:30am",
        now=datetime(2026, 4, 28, 11, 5, tzinfo=ZoneInfo("Asia/Taipei")),
    )

    assert parsed is not None
    assert parsed.start.hour == 10
    assert parsed.start.minute == 0
    assert parsed.end.hour == 10
    assert parsed.end.minute == 30
    assert parsed.range_seconds == 1800


@pytest.mark.asyncio
async def test_patch_agent_updates_absolute_time_window_for_target_metric():
    dashboard = DashboardSpec(
        title="Vitals",
        widgets=[
            WidgetSpec(
                id="heart",
                title="Heart Rate Trend",
                type="line_chart",
                query=QuerySpec(metric="patient_heart_rate_bpm"),
            ),
            WidgetSpec(
                id="spo2",
                title="SpO2 Trend",
                type="line_chart",
                query=QuerySpec(metric="patient_spo2_percent"),
            ),
        ],
    )

    patch = await PatchAgent().generate_patch(
        dashboard_id=dashboard.id,
        prompt="Show the Heart Rate at 10:30 to 11",
        current_dashboard=dashboard,
    )

    assert len(patch.operations) == 1
    assert patch.operations[0].widget_id == "heart"
    query_updates = patch.operations[0].updates["query"]  # type: ignore[index]
    assert query_updates["time_range_seconds"] == 1800
    assert "T10:30:00" in query_updates["start_time"]
    assert "T11:00:00" in query_updates["end_time"]
