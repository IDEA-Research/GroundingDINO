"""Saved-dashboards store.

One JSON file per entry under `backend/app/storage/saved_dashboards/`.
Used by the orchestrator to preload library context at session start
and by `/api/saved_dashboards` for CRUD.

Deletion is SOFT: `delete` stamps `deleted_at` and keeps the file, so
the UI can offer 復原 (restore) and nothing is lost by a mis-tap.
Every listing / lookup path excludes deleted entries by default —
including `summaries()`, which is what the orchestrator injects into
Helper's prompt, so a trashed dashboard can never be "loaded" by chat.
`saved_id` is immutable: rename / duplicate / restore never change it.
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

# Suffix appended to a duplicated entry's name (user-facing, 繁體中文).
DUPLICATE_SUFFIX = " 副本"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:48] or f"saved-{uuid.uuid4().hex[:6]}"


def _summary(spec: DashboardSpec) -> dict:
    return {
        "widget_types": [w.type.value for w in spec.widgets],
        # `promqls()` covers both `query` and multi-metric `queries`
        # tables (whose `query` is None).
        "promqls": [p for w in spec.widgets for p in w.promqls()],
    }


class SavedDashboardStore:
    def __init__(self, root: Path | None = None) -> None:
        self._dir = root or _DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- helpers ------------------------------------------------------
    def _path(self, saved_id: str) -> Path:
        return self._dir / f"{saved_id}.json"

    def _write(self, entry: SavedDashboard) -> None:
        self._path(entry.saved_id).write_text(
            json.dumps(entry.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

    def _read(self, path: Path) -> SavedDashboard | None:
        try:
            return SavedDashboard.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:  # pragma: no cover - corrupt file
            return None

    # -- create -------------------------------------------------------
    def save(
        self,
        spec: DashboardSpec,
        *,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> SavedDashboard:
        saved_id = _slug(name)
        # If already exists (live OR trashed — the file is kept), append
        # a short uid so the new entry never overwrites it.
        if self._path(saved_id).exists():
            saved_id = f"{saved_id}-{uuid.uuid4().hex[:4]}"
        now = _now()
        entry = SavedDashboard(
            saved_id=saved_id,
            name=name[:96],
            description=description[:512],
            tags=list(dict.fromkeys((tags or [])[:16])),
            spec=spec,
            summary=_summary(spec),
            created_at=now,
            updated_at=now,
            used_count=0,
        )
        self._write(entry)
        return entry

    # -- read ---------------------------------------------------------
    def list(self, *, include_deleted: bool = False) -> list[SavedDashboard]:
        out: list[SavedDashboard] = []
        for p in sorted(self._dir.glob("*.json")):
            entry = self._read(p)
            if entry is None:
                continue
            if entry.deleted_at and not include_deleted:
                continue
            out.append(entry)
        return out

    def summaries(
        self, *, include_deleted: bool = False
    ) -> list[SavedDashboardSummary]:
        return [
            SavedDashboardSummary(
                saved_id=s.saved_id,
                name=s.name,
                description=s.description,
                tags=s.tags,
                widget_types=s.summary.get("widget_types") or [],
                promqls=s.summary.get("promqls") or [],
                used_count=s.used_count,
                deleted_at=s.deleted_at,
            )
            for s in self.list(include_deleted=include_deleted)
        ]

    def load(
        self, saved_id: str, *, include_deleted: bool = False
    ) -> SavedDashboard | None:
        """Live entry by id. Soft-deleted entries read as missing unless
        `include_deleted=True` (restore / developer listing)."""
        p = self._path(saved_id)
        if not p.exists():
            return None
        entry = self._read(p)
        if entry is None:
            return None
        if entry.deleted_at and not include_deleted:
            return None
        return entry

    # -- update -------------------------------------------------------
    def update(
        self,
        saved_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        spec: DashboardSpec | None = None,
    ) -> SavedDashboard | None:
        """Rename / re-describe / re-tag in place. `saved_id` never
        changes (the bar, LoadSavedIntent and used_count all key on it;
        Chinese names slug to `saved-<hex>` anyway).

        `spec` (Phase 2 layout engine): a pre-validated, already-merged
        DashboardSpec replacing the stored snapshot wholesale — the
        caller (api/saved_dashboards.py) owns the merge + validation,
        this just persists it and refreshes the derived `summary`.
        """
        entry = self.load(saved_id)
        if entry is None:
            return None
        changes: dict = {"updated_at": _now()}
        if name is not None and name.strip():
            changes["name"] = name.strip()[:96]
        if description is not None:
            changes["description"] = description[:512]
        if tags is not None:
            changes["tags"] = list(dict.fromkeys(tags[:16]))
        if spec is not None:
            changes["spec"] = spec
            changes["summary"] = _summary(spec)
        entry = entry.model_copy(update=changes)
        self._write(entry)
        return entry

    def duplicate(self, saved_id: str) -> SavedDashboard | None:
        """New library entry with the same spec/description/tags, named
        '{name} 副本'. The slug collision suffix keeps repeats unique."""
        entry = self.load(saved_id)
        if entry is None:
            return None
        base = entry.name
        room = 96 - len(DUPLICATE_SUFFIX)
        if len(base) > room:
            base = base[:room]
        return self.save(
            entry.spec,
            name=f"{base}{DUPLICATE_SUFFIX}",
            description=entry.description,
            tags=list(entry.tags),
        )

    # -- delete / restore --------------------------------------------
    def delete(self, saved_id: str) -> bool:
        """SOFT delete: stamp `deleted_at`, keep the file. Returns False
        when there is no live entry with this id."""
        entry = self.load(saved_id)
        if entry is None:
            return False
        entry = entry.model_copy(
            update={"deleted_at": _now(), "updated_at": _now()}
        )
        self._write(entry)
        return True

    def restore(self, saved_id: str) -> SavedDashboard | None:
        """Undo a soft delete. Idempotent on a live entry."""
        entry = self.load(saved_id, include_deleted=True)
        if entry is None:
            return None
        if entry.deleted_at:
            entry = entry.model_copy(
                update={"deleted_at": None, "updated_at": _now()}
            )
            self._write(entry)
        return entry

    def purge(self, saved_id: str) -> bool:
        """HARD delete (file removed). Not exposed over the API; used by
        housekeeping and tests."""
        p = self._path(saved_id)
        if not p.exists():
            return False
        p.unlink()
        return True

    def bump_used(self, saved_id: str) -> None:
        entry = self.load(saved_id)
        if entry is None:
            return
        entry = entry.model_copy(update={"used_count": entry.used_count + 1})
        self._write(entry)
