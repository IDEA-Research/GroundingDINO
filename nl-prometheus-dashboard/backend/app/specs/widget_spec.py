from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WidgetType(str, Enum):
    LINE_CHART = "line_chart"
    STAT_CARD = "stat_card"
    THRESHOLD_CARD = "threshold_card"
    TABLE = "table"


class WidgetLayoutSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int = Field(0, ge=0)
    y: int = Field(0, ge=0)
    w: int = Field(4, ge=1, le=12)
    h: int = Field(3, ge=1, le=12)


class QuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(..., pattern=r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
    promql: str | None = Field(
        default=None,
        description="Optional restricted PromQL. If omitted, the metric selector is built from metric and label_matchers.",
    )
    query_type: Literal["instant", "range"] = "range"
    label_matchers: dict[str, str] = Field(default_factory=dict)
    time_range_seconds: int = Field(300, ge=10, le=86_400)
    step_seconds: int = Field(5, ge=1, le=3_600)
    start_time: datetime | None = None
    end_time: datetime | None = None

    @model_validator(mode="after")
    def validate_absolute_window(self) -> "QuerySpec":
        if (self.start_time is None) != (self.end_time is None):
            raise ValueError("start_time and end_time must be provided together")
        if self.start_time is not None and self.end_time is not None:
            if self.end_time <= self.start_time:
                raise ValueError("end_time must be after start_time")
            duration = (self.end_time - self.start_time).total_seconds()
            if duration > 86_400:
                raise ValueError("absolute time window exceeds 86400 seconds")
        return self

    def effective_promql(self) -> str:
        if self.promql:
            return self.promql
        if not self.label_matchers:
            return self.metric

        matchers = []
        for label, value in sorted(self.label_matchers.items()):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            matchers.append(f'{label}="{escaped}"')
        return f"{self.metric}" + "{" + ",".join(matchers) + "}"


class ThresholdSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    operator: Literal["gt", "gte", "lt", "lte", "eq"]
    value: float
    severity: Literal["info", "warning", "critical"] = "warning"
    message: str | None = None


class WidgetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"widget_{uuid4().hex[:12]}")
    title: str
    type: WidgetType
    query: QuerySpec
    unit: str | None = None
    thresholds: list[ThresholdSpec] = Field(default_factory=list)
    refresh_interval_ms: int | None = Field(default=None, ge=1_000, le=600_000)
    layout: WidgetLayoutSpec = Field(default_factory=WidgetLayoutSpec)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
