from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AnalysisIntent = Literal["trend_monitoring", "threshold_detection", "latest_value_summary", "tabular_review"]


class TimeContextSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_range_seconds: int = Field(300, ge=10, le=86_400)
    refresh_interval_ms: int = Field(5_000, ge=1_000, le=600_000)
    start_time: datetime | None = None
    end_time: datetime | None = None

    @model_validator(mode="after")
    def validate_absolute_window(self) -> "TimeContextSpec":
        if (self.start_time is None) != (self.end_time is None):
            raise ValueError("start_time and end_time must be provided together")
        if self.start_time is not None and self.end_time is not None:
            if self.end_time <= self.start_time:
                raise ValueError("end_time must be after start_time")
            duration = (self.end_time - self.start_time).total_seconds()
            if duration > 86_400:
                raise ValueError("absolute time window exceeds 86400 seconds")
        return self


class TaskConstraintsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_metrics: list[str] = Field(default_factory=list)
    max_time_range_seconds: int = Field(86_400, ge=10, le=604_800)
    min_step_interval_seconds: int = Field(5, ge=1, le=3_600)

    @field_validator("allowed_metrics")
    @classmethod
    def unique_allowed_metrics(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class MonitoringTaskModel(BaseModel):
    """A validated intermediate representation of the user's monitoring task.

    This model is intentionally not a UI model. It captures domain intent,
    entities, signals, relationships, analysis goals, time context, and metric
    constraints before a deterministic mapper produces DashboardSpec widgets.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"task_{uuid4().hex[:12]}")
    domain: str = "monitoring"
    monitoring_goal: str
    entities: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    analysis_intents: list[AnalysisIntent] = Field(default_factory=list)
    time_context: TimeContextSpec = Field(default_factory=TimeContextSpec)
    constraints: TaskConstraintsSpec = Field(default_factory=TaskConstraintsSpec)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("entities", "signals", "relationships", "analysis_intents")
    @classmethod
    def unique_text_values(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item for item in value if item))
