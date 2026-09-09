"""Saved-dashboards API.

Used by the frontend library drawer and by the orchestrator's
Save prompt flow. No developer-token gate — this is user-facing
library content. The one exception is `?include_deleted=1` on the
list route, which is honoured only with a valid developer header
(soft-deleted entries are a developer/support view; users get 復原
through the undo toast right after deleting).
"""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from ..services.dashboard_store import DashboardStore
from ..services.saved_dashboard_loader import load_saved_dashboard
from ..services.saved_dashboard_store import SavedDashboardStore
from ..services.spec_validator import SpecValidationError, SpecValidator
from ._auth import require_dev


router = APIRouter()
_store = SavedDashboardStore()
_dashboards = DashboardStore()
_validator = SpecValidator()


class SaveRequest(BaseModel):
    dashboard_id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=96)
    description: str = Field(default="", max_length=512)
    tags: list[str] = Field(default_factory=list, max_length=16)


class UpdateRequest(BaseModel):
    """PATCH body: every field optional; omitted fields are untouched."""

    name: str | None = Field(default=None, min_length=1, max_length=96)
    description: str | None = Field(default=None, max_length=512)
    tags: list[str] | None = Field(default=None, max_length=16)


def _dev_header_valid(token: str | None) -> bool:
    """True only when the header passes the SAME gate every developer
    route uses (never a separate, weaker check)."""
    try:
        require_dev(token)
    except HTTPException:
        return False
    return True


@router.post("/save")
def save(req: SaveRequest) -> dict:
    spec = _dashboards.load_dashboard(req.dashboard_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="dashboard not found")
    try:
        spec = _validator.validate_dashboard(spec.model_dump(mode="json", by_alias=True))
    except SpecValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors)
    entry = _store.save(
        spec,
        name=req.name,
        description=req.description,
        tags=req.tags,
    )
    return {"saved": entry.model_dump(mode="json")}


# Both forms registered explicitly: redirect_slashes is disabled on
# the FastAPI app so the browser never follows a cross-origin 307.
@router.get("")
@router.get("/")
def list_saved(
    include_deleted: int = Query(0, ge=0, le=1),
    x_developer_token: str | None = Header(default=None),
) -> dict:
    # Deleted entries are shown ONLY to a verified developer; a user
    # asking for them gets the normal list (no error, no leak).
    show_deleted = bool(include_deleted) and _dev_header_valid(x_developer_token)
    return {
        "saved": [
            s.model_dump(mode="json")
            for s in _store.summaries(include_deleted=show_deleted)
        ],
        "include_deleted": show_deleted,
    }


@router.get("/{saved_id}")
def get_saved(saved_id: str) -> dict:
    entry = _store.load(saved_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="not found")
    return {"saved": entry.model_dump(mode="json")}


@router.patch("/{saved_id}")
@router.patch("/{saved_id}/")
def update_saved(saved_id: str, req: UpdateRequest) -> dict:
    """Rename / re-describe / re-tag. `saved_id` never changes."""
    if req.name is None and req.description is None and req.tags is None:
        raise HTTPException(
            status_code=400, detail="nothing to update (name, description or tags)"
        )
    entry = _store.update(
        saved_id,
        name=req.name,
        description=req.description,
        tags=req.tags,
    )
    if entry is None:
        raise HTTPException(status_code=404, detail="not found")
    return {"saved": entry.model_dump(mode="json")}


@router.post("/{saved_id}/duplicate")
@router.post("/{saved_id}/duplicate/")
def duplicate_saved(saved_id: str) -> dict:
    entry = _store.duplicate(saved_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="not found")
    return {"saved": entry.model_dump(mode="json")}


# Both forms registered explicitly (same reason as the list route):
# redirect_slashes is disabled on the app, so a missing form would 404
# instead of redirecting.
@router.post("/{saved_id}/load")
@router.post("/{saved_id}/load/")
def load_saved(saved_id: str) -> dict:
    """Clone the saved entry into the working dashboard store.

    This is the SINGLE load surface — the frontend library button and
    the chat LoadSavedIntent both go through the same service function,
    so every load clones the snapshot under a FRESH dashboard_id (the
    stored id may be shared with newer specs; patching must target
    exactly what the user is looking at), persists the clone, and bumps
    the entry's used_count.
    """
    entry = _store.load(saved_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="not found")
    clone = load_saved_dashboard(entry, dashboards=_dashboards, saved=_store)
    # by_alias=True: decision_flow edges must serialize as `from` —
    # the frontend renders this dump directly.
    return {
        "saved_id": entry.saved_id,
        "name": entry.name,
        "dashboard": clone.model_dump(mode="json", by_alias=True),
    }


@router.delete("/{saved_id}")
@router.delete("/{saved_id}/")
def delete_saved(saved_id: str) -> dict:
    """SOFT delete — the entry leaves every list but stays restorable
    via POST /{saved_id}/restore (the UI's 復原 toast)."""
    ok = _store.delete(saved_id)
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    return {"ok": True, "restorable": True, "saved_id": saved_id}


@router.post("/{saved_id}/restore")
@router.post("/{saved_id}/restore/")
def restore_saved(saved_id: str) -> dict:
    entry = _store.restore(saved_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="not found")
    return {"ok": True, "saved": entry.model_dump(mode="json")}
