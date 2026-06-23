"""Saved-dashboards store.

One JSON file per entry under `backend/app/storage/saved_dashboards/`.
Used by the orchestrator to preload library context at session start
and by `/api/saved_dashboards` for CRUD.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..specs import DashboardSpec, SavedDashboard, SavedDashboardSummary


_DIR = (
    Path(__file__).resolve().parent.parent / "storage" / "saved_dashboards"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:48] or f"saved-{uuid.uuid4().hex[:6]}"


def _summary(spec: DashboardSpec) -> dict:
    return {
        "widget_types": [w.type.value for w in spec.widgets],
        "promqls": [w.query.promql for w in spec.widgets],
    }


class SavedDashboardStore:
    def __init__(self, root: Path | None = None) -> None:
        self._dir = root or _DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        spec: DashboardSpec,
        *,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> SavedDashboard:
        saved_id = _slug(name)
        # If already exists, append a short uid.
        if (self._dir / f"{saved_id}.json").exists():
            saved_id = f"{saved_id}-{uuid.uuid4().hex[:4]}"
        entry = SavedDashboard(
            saved_id=saved_id,
            name=name[:96],
            description=description[:512],
            tags=list(dict.fromkeys((tags or [])[:16])),
            spec=spec,
            summary=_summary(spec),
            created_at=_now(),
            used_count=0,
        )
        (self._dir / f"{saved_id}.json").write_text(
            json.dumps(entry.model_dump(mode="json"), indent=2)
        )
        return entry

    def list(self) -> list[SavedDashboard]:
        out: list[SavedDashboard] = []
        for p in sorted(self._dir.glob("*.json")):
            try:
                out.append(SavedDashboard.model_validate_json(p.read_text()))
            except Exception:  # pragma: no cover
                continue
        return out

    def summaries(self) -> list[SavedDashboardSummary]:
        return [
            SavedDashboardSummary(
                saved_id=s.saved_id,
                name=s.name,
                description=s.description,
                tags=s.tags,
                widget_types=s.summary.get("widget_types") or [],
                promqls=s.summary.get("promqls") or [],
                used_count=s.used_count,
            )
            for s in self.list()
        ]

    def load(self, saved_id: str) -> SavedDashboard | None:
        p = self._dir / f"{saved_id}.json"
        if not p.exists():
            return None
        return SavedDashboard.model_validate_json(p.read_text())

    def delete(self, saved_id: str) -> bool:
        p = self._dir / f"{saved_id}.json"
        if not p.exists():
            return False
        p.unlink()
        return True

    def bump_used(self, saved_id: str) -> None:
        entry = self.load(saved_id)
        if entry is None:
            return
        entry = entry.model_copy(update={"used_count": entry.used_count + 1})
        (self._dir / f"{saved_id}.json").write_text(
            json.dumps(entry.model_dump(mode="json"), indent=2)
        )
