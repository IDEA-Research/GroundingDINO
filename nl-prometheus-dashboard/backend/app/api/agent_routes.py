from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..agents.dashboard_spec_agent import DashboardSpecAgent
from ..agents.intent_agent import IntentAgent
from ..agents.patch_agent import PatchAgent
from ..agents.task_model_agent import TaskModelAgent
from ..llm.base import LLMConfigurationError, LLMJsonParseError, LLMOutputValidationError
from ..llm.local_llm_client import LocalLLMClient
from ..llm.openrouter_client import OpenRouterLLMClient
from ..services.dashboard_service import DashboardNotFoundError, DashboardService
from ..services.task_model_service import TaskModelService
from ..specs.dashboard_spec import DashboardSpec
from ..specs.metric_catalog import load_default_metric_catalog
from ..specs.patch_spec import PatchSpec
from ..specs.task_model import MonitoringTaskModel

router = APIRouter(prefix="/api/agent", tags=["agent"])

catalog = load_default_metric_catalog()
dashboard_service = DashboardService()
task_model_service = TaskModelService()
intent_agent = IntentAgent()


class CreateDashboardRequest(BaseModel):
    prompt: str
    context: dict[str, Any] = Field(default_factory=dict)


class CreateDashboardResponse(BaseModel):
    dashboard: DashboardSpec
    task_model: MonitoringTaskModel


class PatchDashboardRequest(BaseModel):
    dashboard_id: str
    prompt: str
    current_dashboard: DashboardSpec


class PatchDashboardResponse(BaseModel):
    patch: PatchSpec
    updated_dashboard: DashboardSpec
    task_model: MonitoringTaskModel


def _build_llm_client():
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    if provider == "openrouter":
        return OpenRouterLLMClient()
    if provider == "local":
        return LocalLLMClient()
    return None


def _provider_metadata() -> dict[str, str | None]:
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    model = None
    if provider == "openrouter":
        model = os.getenv("OPENROUTER_MODEL")
    if provider == "local":
        model = os.getenv("LOCAL_LLM_MODEL")
    return {"provider": provider, "model": model}


@router.post("/create-dashboard", response_model=CreateDashboardResponse)
async def create_dashboard(request: CreateDashboardRequest) -> CreateDashboardResponse:
    try:
        llm_client = _build_llm_client()
        intent_summary = intent_agent.extract(request.prompt, has_current_dashboard=False)
        task_agent = TaskModelAgent(catalog=catalog, llm_client=llm_client)
        task_model = await task_agent.generate_task_model(
            request.prompt,
            {**request.context, "intent": intent_summary.__dict__},
        )
        dashboard_agent = DashboardSpecAgent(catalog=catalog)
        dashboard = dashboard_agent.generate_dashboard_from_task_model(task_model)
        saved = dashboard_service.create_dashboard(dashboard, task_model=task_model)
        provider_metadata = {
            **_provider_metadata(),
            "intent_action": intent_summary.action,
            "intent_domain": intent_summary.domain,
        }
        task_model_service.save_task_model(
            saved.id,
            task_model,
            reason="create_dashboard",
            prompt=request.prompt,
            provider_metadata=provider_metadata,
        )
        return CreateDashboardResponse(dashboard=saved, task_model=task_model)
    except (LLMConfigurationError, LLMJsonParseError, LLMOutputValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/patch-dashboard", response_model=PatchDashboardResponse)
async def patch_dashboard(request: PatchDashboardRequest) -> PatchDashboardResponse:
    try:
        intent_summary = intent_agent.extract(request.prompt, has_current_dashboard=True)
        task_agent = TaskModelAgent(catalog=catalog, llm_client=None)
        try:
            persisted_dashboard = dashboard_service.get_dashboard(request.dashboard_id)
        except DashboardNotFoundError:
            request.current_dashboard.id = request.dashboard_id
            inferred_task_model = task_agent.infer_from_dashboard(request.current_dashboard)
            persisted_dashboard = dashboard_service.create_dashboard(
                request.current_dashboard,
                task_model=inferred_task_model,
            )
            task_model_service.save_task_model(
                request.dashboard_id,
                inferred_task_model,
                reason="infer_from_dashboard",
                prompt=None,
                provider_metadata={
                    **_provider_metadata(),
                    "intent_action": intent_summary.action,
                    "intent_domain": intent_summary.domain,
                },
            )

        try:
            current_task_model = task_model_service.get_task_model(request.dashboard_id)
        except FileNotFoundError:
            current_task_model = task_agent.infer_from_dashboard(persisted_dashboard)

        agent = PatchAgent(catalog=catalog, llm_client=_build_llm_client())
        patch = await agent.generate_patch(
            dashboard_id=request.dashboard_id,
            prompt=request.prompt,
            current_dashboard=persisted_dashboard,
            current_task_model=current_task_model,
        )
        updated_task_model = task_agent.patch_task_model(
            current_task_model=current_task_model,
            prompt=request.prompt,
            patch=patch,
            current_dashboard=persisted_dashboard,
        )
        updated = dashboard_service.apply_patch(
            request.dashboard_id,
            patch,
            prompt=request.prompt,
            task_model_before=current_task_model,
            task_model_after=updated_task_model,
        )
        task_model_service.save_task_model(
            request.dashboard_id,
            updated_task_model,
            reason="patch_dashboard",
            prompt=request.prompt,
            provider_metadata={
                **_provider_metadata(),
                "intent_action": intent_summary.action,
                "intent_domain": intent_summary.domain,
            },
        )
        return PatchDashboardResponse(patch=patch, updated_dashboard=updated, task_model=updated_task_model)
    except (LLMConfigurationError, LLMJsonParseError, LLMOutputValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
