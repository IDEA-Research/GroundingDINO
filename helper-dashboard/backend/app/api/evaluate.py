"""Evaluate API — run browser evaluation against a stored dashboard.

Developer-gated (`X-Developer-Token`): a run launches Playwright and the
reports carry debugging traces, neither of which belongs in a user
deployment.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..services.browser_evaluator import BrowserEvaluator
from ..services.dashboard_store import DashboardStore
from ._auth import require_dev


router = APIRouter(dependencies=[Depends(require_dev)])
_store = DashboardStore()
_evaluator = BrowserEvaluator()


class EvaluateRequest(BaseModel):
    dashboard_id: str


@router.post("/run")
def run_evaluation(req: EvaluateRequest) -> dict:
    spec = _store.load_dashboard(req.dashboard_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="dashboard not found")
    report = _evaluator.evaluate(spec)
    _store.save_evaluation(report)
    return {"report": report.model_dump(mode="json")}


@router.get("/reports/{dashboard_id}")
def list_reports(dashboard_id: str) -> dict:
    return {"reports": _store.list_evaluation_reports(dashboard_id)}
