from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..services.dashboard_service import DashboardNotFoundError, DashboardService
from ..specs.dashboard_spec import DashboardSpec

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])
dashboard_service = DashboardService()


@router.get("/{dashboard_id}", response_model=DashboardSpec)
def get_dashboard(dashboard_id: str) -> DashboardSpec:
    try:
        return dashboard_service.get_dashboard(dashboard_id)
    except DashboardNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("", response_model=DashboardSpec)
def create_dashboard(dashboard: DashboardSpec) -> DashboardSpec:
    return dashboard_service.create_dashboard(dashboard)


@router.put("/{dashboard_id}", response_model=DashboardSpec)
def update_dashboard(dashboard_id: str, dashboard: DashboardSpec) -> DashboardSpec:
    try:
        return dashboard_service.update_dashboard(dashboard_id, dashboard)
    except DashboardNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{dashboard_id}/versions")
def list_dashboard_versions(dashboard_id: str) -> list[dict]:
    return dashboard_service.list_versions(dashboard_id)


@router.get("/{dashboard_id}/patch-logs")
def list_dashboard_patch_logs(dashboard_id: str) -> list[dict]:
    return dashboard_service.list_patch_logs(dashboard_id)


@router.post("/{dashboard_id}/rollback/{version_id}", response_model=DashboardSpec)
def rollback_dashboard(dashboard_id: str, version_id: str) -> DashboardSpec:
    try:
        return dashboard_service.rollback(dashboard_id, version_id)
    except (DashboardNotFoundError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
