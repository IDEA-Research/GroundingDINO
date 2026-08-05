"""System-rule API — read surface + validated rule CRUD.

Non-clinical sibling of `/api/anomaly`. Every response carries
`non_diagnostic: true` (decision-support disclaimer) and rules are
structurally SHADOW (the spec cannot express anything else)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from ..services.system_rules import get_service
from ..specs.system_rule_spec import SYSTEM_METRIC_CATALOG, SystemAlertRuleSpec

router = APIRouter()


@router.get("")
@router.get("/")
def status() -> dict:
    return get_service().status()


@router.get("/catalog")
def catalog() -> dict:
    return {
        "catalog": [
            {
                "kind": e.kind,
                "unit": e.unit,
                "description": e.description,
                "min_threshold": e.min_threshold,
                "max_threshold": e.max_threshold,
            }
            for e in SYSTEM_METRIC_CATALOG.values()
        ],
        "non_diagnostic": True,
    }


@router.get("/alerts")
def alerts() -> dict:
    svc = get_service()
    return {"events": svc.alerts(), "non_diagnostic": True}


@router.post("/rules")
def create_rule(payload: dict) -> dict:
    try:
        rule = SystemAlertRuleSpec.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    get_service().add_rule(rule)
    return {
        "ok": True,
        "rule": rule.model_dump(by_alias=True),
        "summary": rule.human_summary(),
        "non_diagnostic": True,
    }


@router.delete("/rules/{rule_id}")
def delete_rule(rule_id: str) -> dict:
    removed = get_service().remove_rule(rule_id)
    if not removed:
        raise HTTPException(status_code=404, detail="rule not found")
    return {"ok": True, "deleted": rule_id}
