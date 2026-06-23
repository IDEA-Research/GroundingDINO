from backend.app.services.dashboard_service import DashboardService
from backend.app.services.version_service import VersionService
from backend.app.specs.dashboard_spec import DashboardSpec
from backend.app.specs.widget_spec import QuerySpec, WidgetSpec


def test_dashboard_service_saves_versions_and_rolls_back(tmp_path):
    service = DashboardService(
        dashboards_dir=tmp_path / "dashboards",
        version_service=VersionService(tmp_path / "versions"),
    )
    created = service.create_dashboard(
        DashboardSpec(
            title="Original",
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
    updated = service.update_dashboard(created.id, created.model_copy(update={"title": "Updated"}))

    versions = service.list_versions(created.id)
    restored = service.rollback(created.id, versions[0]["version_id"])

    assert updated.version == 2
    assert len(versions) == 2
    assert restored.title == "Original"
    assert restored.version == 3
