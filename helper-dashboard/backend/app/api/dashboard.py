"""Dashboard API — read/list stored DashboardSpec JSON files."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..services.dashboard_store import DashboardStore
from ..services.spec_validator import SpecValidator, SpecValidationError
from ..specs import DashboardSpec


router = APIRouter()
_store = DashboardStore()
_validator = SpecValidator()


# Both forms registered explicitly: redirect_slashes is disabled on
# the FastAPI app so the browser never follows a cross-origin 307.
@router.get("")
@router.get("/")
def list_dashboards() -> dict:
    return {"dashboards": _store.list_dashboard_ids()}


@router.get("/{dashboard_id}")
def get_dashboard(dashboard_id: str) -> dict:
    spec = _store.load_dashboard(dashboard_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="dashboard not found")
    # by_alias=True so decision_flow edges serialize their `from` key (aliased
    # from the `from_` field), which the frontend FlowEdge type expects.
    return {"spec": spec.model_dump(mode="json", by_alias=True)}


@router.post("/validate")
def validate_dashboard(payload: dict) -> dict:
    """Revalidate an in-memory spec without saving it.

    Used by the frontend inspector to check a spec before asking Helper
    to patch it further.
    """
    try:
        spec = _validator.validate_dashboard(payload)
    except SpecValidationError as exc:
        return {"ok": False, "errors": exc.errors}
    # by_alias=True: decision_flow edges must echo back as `from` (canonical),
    # matching get_dashboard above — a client re-rendering the echoed spec
    # would otherwise silently lose every branch edge.
    return {"ok": True, "spec": spec.model_dump(mode="json", by_alias=True)}
