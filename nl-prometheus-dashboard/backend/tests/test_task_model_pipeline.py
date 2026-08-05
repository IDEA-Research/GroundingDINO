import pytest

from backend.app.agents.dashboard_spec_agent import DashboardSpecAgent
from backend.app.agents.task_model_agent import TaskModelAgent
from backend.app.specs.patch_spec import PatchOperation, PatchSpec


@pytest.mark.asyncio
async def test_prompt_maps_to_task_model_then_dashboard_spec():
    task_agent = TaskModelAgent()
    task_model = await task_agent.generate_task_model(
        "Build a patient monitoring dashboard with heart rate, SpO2, and warning thresholds."
    )

    dashboard = DashboardSpecAgent().generate_dashboard_from_task_model(task_model)
    metrics = {widget.query.metric for widget in dashboard.widgets}

    assert task_model.metadata["generation_mode"] == "mock_task_model"
    assert "heart_rate" in task_model.signals
    assert "spo2" in task_model.signals
    assert "threshold_detection" in task_model.analysis_intents
    assert "patient_heart_rate_bpm" in metrics
    assert "patient_spo2_percent" in metrics
    assert dashboard.metadata["generation_mode"] == "task_model_mapping"


@pytest.mark.asyncio
async def test_patch_updates_task_model_time_context():
    task_agent = TaskModelAgent()
    task_model = await task_agent.generate_task_model("Build a patient dashboard with heart rate.")
    patch = PatchSpec(
        dashboard_id="dash_test",
        operations=[
            PatchOperation(
                op="update_widget",
                widget_id="heart",
                updates={"query": {"time_range_seconds": 1800}},
            )
        ],
    )

    updated = task_agent.patch_task_model(
        current_task_model=task_model,
        prompt="Change time range to 30 minutes",
        patch=patch,
    )

    assert updated.time_context.time_range_seconds == 1800
