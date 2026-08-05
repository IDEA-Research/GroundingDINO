from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..prometheus.client import PrometheusQueryError
from ..services.query_service import QueryService
from ..specs.widget_spec import WidgetSpec

router = APIRouter(prefix="/api/query", tags=["query"])
query_service = QueryService()


class WidgetQueryRequest(BaseModel):
    widget: WidgetSpec
    variables: dict[str, Any] = Field(default_factory=dict)


class WidgetQueryResponse(BaseModel):
    series: list[Any]
    metadata: dict[str, Any]


@router.post("/widget", response_model=WidgetQueryResponse)
async def query_widget(request: WidgetQueryRequest) -> WidgetQueryResponse:
    try:
        result = await query_service.query_widget(request.widget, request.variables)
        return WidgetQueryResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PrometheusQueryError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
