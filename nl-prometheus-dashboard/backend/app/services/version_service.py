from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..specs.dashboard_spec import DashboardSpec
from ..specs.patch_spec import PatchSpec
from ..specs.task_model import MonitoringTaskModel


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


class VersionService:
    def __init__(
        self,
        versions_dir: str | Path | None = None,
        patch_logs_dir: str | Path | None = None,
    ) -> None:
        self.versions_dir = Path(versions_dir or Path(__file__).resolve().parents[1] / "storage" / "versions")
        if patch_logs_dir is None and versions_dir is not None:
            self.patch_logs_dir = self.versions_dir.parent / "patch_logs"
        else:
            self.patch_logs_dir = Path(
                patch_logs_dir or Path(__file__).resolve().parents[1] / "storage" / "patch_logs"
            )
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.patch_logs_dir.mkdir(parents=True, exist_ok=True)

    def save_version(
        self,
        dashboard: DashboardSpec,
        reason: str,
        *,
        task_model: MonitoringTaskModel | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        version_id = f"v{dashboard.version}_{_utc_stamp()}"
        dashboard_dir = self.versions_dir / dashboard.id
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "version_id": version_id,
            "dashboard_id": dashboard.id,
            "dashboard_version": dashboard.version,
            "reason": reason,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dashboard": dashboard.model_dump(mode="json"),
            "task_model": task_model.model_dump(mode="json") if task_model is not None else None,
            "metadata": metadata or {},
        }
        (dashboard_dir / f"{version_id}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return record

    def save_patch_log(
        self,
        *,
        dashboard_id: str,
        prompt: str | None,
        patch: PatchSpec,
        before_dashboard: DashboardSpec,
        after_dashboard: DashboardSpec,
        task_model_before: MonitoringTaskModel | None = None,
        task_model_after: MonitoringTaskModel | None = None,
    ) -> dict[str, Any]:
        patch_id = f"patch_{_utc_stamp()}"
        dashboard_dir = self.patch_logs_dir / dashboard_id
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "patch_id": patch_id,
            "dashboard_id": dashboard_id,
            "prompt": prompt,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "patch": patch.model_dump(mode="json"),
            "diff": self._dashboard_diff(before_dashboard, after_dashboard),
            "before_dashboard": before_dashboard.model_dump(mode="json"),
            "after_dashboard": after_dashboard.model_dump(mode="json"),
            "task_model_before": task_model_before.model_dump(mode="json") if task_model_before is not None else None,
            "task_model_after": task_model_after.model_dump(mode="json") if task_model_after is not None else None,
        }
        (dashboard_dir / f"{patch_id}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return record

    def list_patch_logs(self, dashboard_id: str) -> list[dict[str, Any]]:
        dashboard_dir = self.patch_logs_dir / dashboard_id
        if not dashboard_dir.exists():
            return []
        records = []
        for path in sorted(dashboard_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                {
                    "patch_id": record["patch_id"],
                    "dashboard_id": record["dashboard_id"],
                    "created_at": record["created_at"],
                    "prompt": record.get("prompt"),
                    "diff": record["diff"],
                }
            )
        return records

    def list_versions(self, dashboard_id: str) -> list[dict[str, Any]]:
        dashboard_dir = self.versions_dir / dashboard_id
        if not dashboard_dir.exists():
            return []
        records = []
        for path in sorted(dashboard_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                {
                    "version_id": record["version_id"],
                    "dashboard_id": record["dashboard_id"],
                    "dashboard_version": record["dashboard_version"],
                    "reason": record["reason"],
                    "created_at": record["created_at"],
                }
            )
        return records

    def get_version(self, dashboard_id: str, version_id: str) -> dict[str, Any]:
        path = self.versions_dir / dashboard_id / f"{version_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Dashboard version not found: {dashboard_id}/{version_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _dashboard_diff(self, before: DashboardSpec, after: DashboardSpec) -> dict[str, Any]:
        before_widgets = {widget.id: widget.model_dump(mode="json") for widget in before.widgets}
        after_widgets = {widget.id: widget.model_dump(mode="json") for widget in after.widgets}
        before_ids = set(before_widgets)
        after_ids = set(after_widgets)
        shared_ids = before_ids & after_ids
        return {
            "title_changed": before.title != after.title,
            "version": {"before": before.version, "after": after.version},
            "added_widgets": sorted(after_ids - before_ids),
            "removed_widgets": sorted(before_ids - after_ids),
            "updated_widgets": sorted(
                widget_id for widget_id in shared_ids if before_widgets[widget_id] != after_widgets[widget_id]
            ),
            "time_range_changed": before.time_range_seconds != after.time_range_seconds,
            "refresh_interval_changed": before.refresh_interval_ms != after.refresh_interval_ms,
        }
