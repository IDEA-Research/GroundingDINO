"""DashboardSpec — the canonical JSON shape of a dashboard."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .widget_spec import WidgetSpec, _assert_safe_text


_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")


class DashboardLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: int = Field(default=12, ge=1, le=24)
    row_height: int = Field(default=40, ge=10, le=200)


class DashboardVariable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=32, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    label: str = Field(..., min_length=1, max_length=64)
    query: str | None = Field(default=None, max_length=256)
    default: str | None = Field(default=None, max_length=128)
    values: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("label", "query", "default")
    @classmethod
    def _safe_text(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _assert_safe_text(v, "variable text field")

    @field_validator("values")
    @classmethod
    def _safe_values(cls, v: list[str]) -> list[str]:
        for s in v:
            if len(s) > 128:
                raise ValueError("variable value too long")
            _assert_safe_text(s, "variable value")
        return v


# Fastest refresh the dashboard may run at. Anything faster (LLMs like
# to emit "1s" or "500ms") is clamped: every widget fires a Prometheus
# query per tick, and sub-5s ticks made the Jetson's Prometheus fall
# behind its own scrapes. Millisecond values are accepted and converted.
MIN_REFRESH_SECONDS = 5


def clamp_refresh_interval(value: str) -> str:
    """Normalise a refresh interval: `ms` -> whole seconds, and never
    below MIN_REFRESH_SECONDS. Values already >= 5s are returned as-is
    (so '30s', '5m', '1h' round-trip byte-for-byte)."""
    m = re.match(r"^(\d{1,4})(ms|s|m|h)$", value)
    if not m:
        return value
    num, unit = int(m.group(1)), m.group(2)
    if unit == "ms":
        seconds = num / 1000.0
    elif unit == "s":
        seconds = float(num)
    elif unit == "m":
        seconds = num * 60.0
    else:
        seconds = num * 3600.0
    if seconds >= MIN_REFRESH_SECONDS:
        if unit == "ms":
            return f"{int(round(seconds))}s"
        return value
    return f"{MIN_REFRESH_SECONDS}s"


class DashboardSpec(BaseModel):
    """Full dashboard definition."""

    model_config = ConfigDict(extra="forbid")

    dashboard_id: str = Field(..., min_length=1, max_length=64)
    title: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="", max_length=1024)
    layout: DashboardLayout = Field(default_factory=DashboardLayout)
    variables: list[DashboardVariable] = Field(default_factory=list, max_length=16)
    widgets: list[WidgetSpec] = Field(default_factory=list, max_length=64)
    refresh_interval: str = Field(default="30s", max_length=16)

    @field_validator("dashboard_id")
    @classmethod
    def _id_pattern(cls, v: str) -> str:
        if not _ID_PATTERN.match(v):
            raise ValueError("dashboard_id must match [a-zA-Z0-9_-]+")
        return v

    @field_validator("title", "description")
    @classmethod
    def _safe_text(cls, v: str) -> str:
        return _assert_safe_text(v, "dashboard text field")

    @field_validator("refresh_interval")
    @classmethod
    def _refresh_interval(cls, v: str) -> str:
        m = re.match(r"^(\d{1,4})(ms|s|m|h)$", v)
        if not m:
            raise ValueError("refresh_interval must look like '30s', '5m', '1h'")
        return clamp_refresh_interval(v)

    @model_validator(mode="after")
    def _unique_widget_ids_and_layout(self) -> "DashboardSpec":
        ids = [w.id for w in self.widgets]
        if len(ids) != len(set(ids)):
            raise ValueError("widget ids must be unique within a dashboard")
        cols = self.layout.columns
        for w in self.widgets:
            if w.position.x + w.position.w > cols:
                raise ValueError(
                    f"widget {w.id!r} overflows layout: "
                    f"x={w.position.x} w={w.position.w} columns={cols}"
                )
        return self
