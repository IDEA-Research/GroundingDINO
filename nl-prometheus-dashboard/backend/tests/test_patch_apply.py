from backend.app.services.dashboard_service import DashboardService
from backend.app.services.version_service import VersionService
from backend.app.specs.dashboard_spec import DashboardSpec
from backend.app.specs.patch_spec import PatchOperation, PatchSpec
from backend.app.specs.widget_spec import QuerySpec, WidgetSpec


def test_patch_add_update_and_remove_widget(tmp_path):
    service = DashboardService(
        dashboards_dir=tmp_path / "dashboards",
        version_service=VersionService(tmp_path / "versions"),
    )
    dashboard = service.create_dashboard(
        DashboardSpec(
            title="Vitals",
            widgets=[
                WidgetSpec(
                    id="heart",
                    title="Heart Rate",
                    type="stat_card",
                    query=QuerySpec(metric="patient_heart_rate_bpm"),
                )
            ],
        )
    )

    patch = PatchSpec(
        dashboard_id=dashboard.id,
        operations=[
            PatchOperation(op="update_dashboard_title", title="ICU Vitals"),
            PatchOperation(
                op="update_widget",
                widget_id="heart",
                updates={"refresh_interval_ms": 10000},
            ),
            PatchOperation(
                op="add_widget",
                widget=WidgetSpec(
                    id="spo2",
                    title="SpO2",
                    type="threshold_card",
                    query=QuerySpec(metric="patient_spo2_percent"),
                ),
            ),
            PatchOperation(op="remove_widget", widget_id="spo2"),
        ],
    )

    updated = service.apply_patch(dashboard.id, patch)

    assert updated.title == "ICU Vitals"
    assert updated.widgets[0].refresh_interval_ms == 10000
    assert [widget.id for widget in updated.widgets] == ["heart"]
    assert updated.version == 2

    patch_logs = service.list_patch_logs(dashboard.id)
    assert len(patch_logs) == 1
    assert patch_logs[0]["diff"]["updated_widgets"] == ["heart"]
