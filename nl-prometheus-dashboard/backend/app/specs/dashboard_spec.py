from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .widget_spec import WidgetSpec


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DashboardVariableSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    label: str
    type: Literal["text", "select"] = "text"
    default: str | None = None
    options: list[str] = Field(default_factory=list)
    required: bool = False
    description: str | None = None


class DashboardSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"dash_{uuid4().hex[:12]}")
    title: str
    description: str | None = None
    widgets: list[WidgetSpec] = Field(default_factory=list)
    variables: list[DashboardVariableSpec] = Field(default_factory=list)
    refresh_interval_ms: int = Field(5_000, ge=1_000, le=600_000)
    time_range_seconds: int = Field(300, ge=10, le=86_400)
    version: int = Field(1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
