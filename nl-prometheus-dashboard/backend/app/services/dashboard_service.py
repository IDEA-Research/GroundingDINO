from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .version_service import VersionService
from ..specs.dashboard_spec import DashboardSpec, DashboardVariableSpec, utc_now
from ..specs.patch_spec import PatchOperation, PatchOperationType, PatchSpec
from ..specs.task_model import MonitoringTaskModel
from ..specs.widget_spec import WidgetSpec


class DashboardNotFoundError(FileNotFoundError):
    pass


class DashboardService:
    def __init__(
        self,
        dashboards_dir: str | Path | None = None,
        version_service: VersionService | None = None,
    ) -> None:
        self.dashboards_dir = Path(dashboards_dir or Path(__file__).resolve().parents[1] / "storage" / "dashboards")
        self.dashboards_dir.mkdir(parents=True, exist_ok=True)
        self.version_service = version_service or VersionService()

    def create_dashboard(
        self,
        dashboard: DashboardSpec,
        *,
        task_model: MonitoringTaskModel | None = None,
    ) -> DashboardSpec:
        dashboard.version = 1
        dashboard.created_at = utc_now()
        dashboard.updated_at = utc_now()
        self._write_dashboard(dashboard)
        self.version_service.save_version(dashboard, "create_dashboard", task_model=task_model)
        return dashboard

    def get_dashboard(self, dashboard_id: str) -> DashboardSpec:
        path = self._dashboard_path(dashboard_id)
        if not path.exists():
            raise DashboardNotFoundError(f"Dashboard not found: {dashboard_id}")
        return DashboardSpec.model_validate_json(path.read_text(encoding="utf-8"))

    def update_dashboard(
        self,
        dashboard_id: str,
        dashboard: DashboardSpec,
        *,
        task_model: MonitoringTaskModel | None = None,
        reason: str = "update_dashboard",
    ) -> DashboardSpec:
        existing = self.get_dashboard(dashboard_id)
        dashboard.id = dashboard_id
        dashboard.created_at = existing.created_at
        dashboard.version = existing.version + 1
        dashboard.updated_at = utc_now()
        self._write_dashboard(dashboard)
        self.version_service.save_version(dashboard, reason, task_model=task_model)
        return dashboard

    def apply_patch(
        self,
        dashboard_id: str,
        patch: PatchSpec,
        *,
        prompt: str | None = None,
        task_model_before: MonitoringTaskModel | None = None,
        task_model_after: MonitoringTaskModel | None = None,
    ) -> DashboardSpec:
        if patch.dashboard_id != dashboard_id:
            raise ValueError("Patch dashboard_id does not match route dashboard_id")

        dashboard = self.get_dashboard(dashboard_id)
        updated = dashboard.model_copy(deep=True)
        for operation in patch.operations:
            updated = self._apply_operation(updated, operation)

        saved = self.update_dashboard(
            dashboard_id,
            updated,
            task_model=task_model_after,
            reason=f"patch:{patch.reason or 'dashboard_patch'}",
        )
        self.version_service.save_patch_log(
            dashboard_id=dashboard_id,
            prompt=prompt,
            patch=patch,
            before_dashboard=dashboard,
            after_dashboard=saved,
            task_model_before=task_model_before,
            task_model_after=task_model_after,
        )
        return saved

    def rollback(self, dashboard_id: str, version_id: str) -> DashboardSpec:
        record = self.version_service.get_version(dashboard_id, version_id)
        restored = DashboardSpec.model_validate(record["dashboard"])
        restored.version = self.get_dashboard(dashboard_id).version + 1
        restored.updated_at = utc_now()
        self._write_dashboard(restored)
        self.version_service.save_version(restored, f"rollback:{version_id}")
        return restored

    def list_versions(self, dashboard_id: str) -> list[dict[str, Any]]:
        return self.version_service.list_versions(dashboard_id)

    def list_patch_logs(self, dashboard_id: str) -> list[dict[str, Any]]:
        return self.version_service.list_patch_logs(dashboard_id)

    def _apply_operation(self, dashboard: DashboardSpec, operation: PatchOperation) -> DashboardSpec:
        if operation.op == PatchOperationType.ADD_WIDGET:
            dashboard.widgets.append(operation.widget)  # type: ignore[arg-type]
            return dashboard

        if operation.op == PatchOperationType.REMOVE_WIDGET:
            dashboard.widgets = [widget for widget in dashboard.widgets if widget.id != operation.widget_id]
            return dashboard

        if operation.op == PatchOperationType.UPDATE_WIDGET:
            dashboard.widgets = [
                self._merge_widget(widget, operation.updates or {}) if widget.id == operation.widget_id else widget
                for widget in dashboard.widgets
            ]
            return dashboard

        if operation.op == PatchOperationType.UPDATE_DASHBOARD_TITLE:
            dashboard.title = operation.title or dashboard.title
            return dashboard

        if operation.op == PatchOperationType.UPDATE_VARIABLE:
            dashboard.variables = self._update_variables(dashboard.variables, operation)
            return dashboard

        raise ValueError(f"Unsupported patch operation: {operation.op}")

    def _update_variables(
        self,
        variables: list[DashboardVariableSpec],
        operation: PatchOperation,
    ) -> list[DashboardVariableSpec]:
        if operation.variable is not None:
            variable_name = operation.variable_name or operation.variable.name
            next_variables = [variable for variable in variables if variable.name != variable_name]
            next_variables.append(operation.variable)
            return next_variables

        next_variables = []
        for variable in variables:
            if variable.name == operation.variable_name:
                merged = self._deep_merge(variable.model_dump(mode="json"), operation.updates or {})
                next_variables.append(DashboardVariableSpec.model_validate(merged))
            else:
                next_variables.append(variable)
        return next_variables

    def _merge_widget(self, widget: WidgetSpec, updates: dict[str, Any]) -> WidgetSpec:
        merged = self._deep_merge(widget.model_dump(mode="json"), updates)
        return WidgetSpec.model_validate(merged)

    def _deep_merge(self, original: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        merged = deepcopy(original)
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _dashboard_path(self, dashboard_id: str) -> Path:
        return self.dashboards_dir / f"{dashboard_id}.json"

    def _write_dashboard(self, dashboard: DashboardSpec) -> None:
        self._dashboard_path(dashboard.id).write_text(
            json.dumps(dashboard.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
