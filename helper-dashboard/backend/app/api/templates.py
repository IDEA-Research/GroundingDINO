"""Templates API (`/api/templates`) — user-facing, no developer gate.

`GET /api/templates` lists the built-in neonatal templates;
`POST /api/templates/{id}/instantiate` validates the canned spec,
gives it a FRESH dashboard_id and persists it through the same
`DashboardStore.save_dashboard` every other creation path uses, so
`/api/dashboard/{id}`, chat patches and 儲存 all resolve against it.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..services.dashboard_store import DashboardStore
from ..services.saved_dashboard_loader import fresh_dashboard_id
from ..services.spec_validator import SpecValidationError, SpecValidator
from ..templates import (
    list_template_ids,
    load_template_raw,
    template_metrics,
)

router = APIRouter()
_dashboards = DashboardStore()
_validator = SpecValidator()


def _summary(template_id: str) -> dict:
    raw = load_template_raw(template_id)
    return {
        "id": template_id,
        "name": str(raw.get("title") or template_id),
        "description": str(raw.get("description") or ""),
        "widget_count": len(raw.get("widgets") or []),
        "widget_types": sorted({str(w.get("type")) for w in raw.get("widgets") or []}),
        "metrics": template_metrics(raw),
    }


# Both forms registered explicitly (redirect_slashes=False on the app).
@router.get("")
@router.get("/")
def list_templates() -> dict:
    return {"templates": [_summary(t) for t in list_template_ids()]}


@router.get("/{template_id}")
def get_template(template_id: str) -> dict:
    try:
        raw = load_template_raw(template_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="template not found")
    return {"template": _summary(template_id), "spec": raw}


@router.post("/{template_id}/instantiate")
@router.post("/{template_id}/instantiate/")
def instantiate_template(template_id: str) -> dict:
    try:
        raw = load_template_raw(template_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="template not found")
    try:
        spec = _validator.validate_dashboard(raw)
    except SpecValidationError as exc:  # pragma: no cover - templates are tested
        raise HTTPException(status_code=500, detail=exc.errors)
    clone = spec.model_copy(
        deep=True,
        update={"dashboard_id": fresh_dashboard_id(spec.dashboard_id)},
    )
    _dashboards.save_dashboard(clone)
    # by_alias=True: decision_flow edges (if a template ever has one)
    # must serialize as `from` — the frontend renders this dump directly.
    return {
        "template_id": template_id,
        "name": spec.title,
        "dashboard": clone.model_dump(mode="json", by_alias=True),
    }
