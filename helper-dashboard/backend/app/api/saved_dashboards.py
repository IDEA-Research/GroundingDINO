"""Saved-dashboards API.

Used by the frontend library drawer and by the orchestrator's
Save prompt flow. No developer-token gate — this is user-facing
library content.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.dashboard_store import DashboardStore
from ..services.saved_dashboard_store import SavedDashboardStore
from ..services.spec_validator import SpecValidationError, SpecValidator


router = APIRouter()
_store = SavedDashboardStore()
_dashboards = DashboardStore()
_validator = SpecValidator()


class SaveRequest(BaseModel):
    dashboard_id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=96)
    description: str = Field(default="", max_length=512)
    tags: list[str] = Field(default_factory=list, max_length=16)


@router.post("/save")
def save(req: SaveRequest) -> dict:
    spec = _dashboards.load_dashboard(req.dashboard_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="dashboard not found")
    try:
        spec = _validator.validate_dashboard(spec.model_dump(mode="json"))
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
def list_saved() -> dict:
    return {
        "saved": [s.model_dump(mode="json") for s in _store.summaries()]
    }


@router.get("/{saved_id}")
def get_saved(saved_id: str) -> dict:
    entry = _store.load(saved_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="not found")
    return {"saved": entry.model_dump(mode="json")}


@router.delete("/{saved_id}")
def delete_saved(saved_id: str) -> dict:
    ok = _store.delete(saved_id)
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    return {"ok": True}
