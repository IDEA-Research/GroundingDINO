from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..specs.task_model import MonitoringTaskModel


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


class TaskModelService:
    """Stores the evolving monitoring task model alongside a dashboard."""

    def __init__(self, task_models_dir: str | Path | None = None) -> None:
        self.task_models_dir = Path(
            task_models_dir or Path(__file__).resolve().parents[1] / "storage" / "task_models"
        )
        self.history_dir = self.task_models_dir / "history"
        self.task_models_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def save_task_model(
        self,
        dashboard_id: str,
        task_model: MonitoringTaskModel,
        *,
        reason: str,
        prompt: str | None = None,
        provider_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "dashboard_id": dashboard_id,
            "task_model_id": task_model.id,
            "reason": reason,
            "prompt": prompt,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider_metadata": provider_metadata or {},
            "task_model": task_model.model_dump(mode="json"),
        }
        latest_path = self._latest_path(dashboard_id)
        latest_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

        dashboard_history_dir = self.history_dir / dashboard_id
        dashboard_history_dir.mkdir(parents=True, exist_ok=True)
        (dashboard_history_dir / f"{_utc_stamp()}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return record

    def get_task_model(self, dashboard_id: str) -> MonitoringTaskModel:
        path = self._latest_path(dashboard_id)
        if not path.exists():
            raise FileNotFoundError(f"Task model not found: {dashboard_id}")
        record = json.loads(path.read_text(encoding="utf-8"))
        return MonitoringTaskModel.model_validate(record["task_model"])

    def list_task_model_history(self, dashboard_id: str) -> list[dict[str, Any]]:
        dashboard_history_dir = self.history_dir / dashboard_id
        if not dashboard_history_dir.exists():
            return []
        records = []
        for path in sorted(dashboard_history_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                {
                    "dashboard_id": record["dashboard_id"],
                    "task_model_id": record["task_model_id"],
                    "reason": record["reason"],
                    "created_at": record["created_at"],
                    "provider_metadata": record.get("provider_metadata", {}),
                }
            )
        return records

    def _latest_path(self, dashboard_id: str) -> Path:
        return self.task_models_dir / f"{dashboard_id}.json"
