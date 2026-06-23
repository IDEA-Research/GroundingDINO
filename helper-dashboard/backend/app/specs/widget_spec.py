"""WidgetSpec and related sub-schemas.

Widgets are the atomic unit of a dashboard. Their `type` is a strict
enum; adding a new type is a Big guy task that must also update the
frontend widget-toolkit and the validator.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Regex used to reject payloads that contain markup or script-like
# content. Applied to every free-text field.
_UNSAFE_PATTERN = re.compile(
    r"<\s*(script|iframe|object|embed|style)\b"
    r"|javascript:"
    r"|on[a-z]+\s*=",
    re.IGNORECASE,
)


def _assert_safe_text(value: str, field: str) -> str:
    if _UNSAFE_PATTERN.search(value):
        raise ValueError(f"{field} contains forbidden markup or script")
    return value


class WidgetType(str, Enum):
    line_chart = "line_chart"
    stat_card = "stat_card"
    gauge = "gauge"
    table = "table"
    alert_list = "alert_list"
    pie_chart = "pie_chart"
    bar_chart = "bar_chart"
    heatmap = "heatmap"


class QuerySource(str, Enum):
    prometheus = "prometheus"
    mock = "mock"


class QueryType(str, Enum):
    instant = "instant"
    range = "range"


class QuerySpec(BaseModel):
    """A single query attached to a widget."""

    model_config = ConfigDict(extra="forbid")

    source: QuerySource = QuerySource.prometheus
    promql: str = Field(..., min_length=1, max_length=512)
    query_type: QueryType = QueryType.range
    range: str | None = Field(default=None, max_length=16)
    step: str | None = Field(default=None, max_length=16)

    @field_validator("promql")
    @classmethod
    def _check_promql(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("promql must not be empty")
        # Block obvious shell / HTML / JS characters.
        forbidden = [";", "`", "<", ">", "$(" , "||"]
        for token in forbidden:
            if token in v:
                raise ValueError(f"promql contains forbidden token: {token!r}")
        _assert_safe_text(v, "promql")
        return v


class WidgetPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int = Field(..., ge=0, le=23)
    y: int = Field(..., ge=0, le=999)
    w: int = Field(..., ge=1, le=24)
    h: int = Field(..., ge=1, le=60)


class WidgetEncoding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit: str | None = Field(default=None, max_length=32)
    legend: str | None = Field(default=None, max_length=64)
    color: str | None = Field(
        default=None, pattern=r"^#[0-9a-fA-F]{3,8}$|^[a-zA-Z]{3,20}$"
    )

    @field_validator("unit", "legend")
    @classmethod
    def _safe_text(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _assert_safe_text(v, "encoding")


class WidgetThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    color: str = Field(..., pattern=r"^#[0-9a-fA-F]{3,8}$|^[a-zA-Z]{3,20}$")
    label: str | None = Field(default=None, max_length=32)

    @field_validator("label")
    @classmethod
    def _safe_label(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _assert_safe_text(v, "threshold.label")


# Allowed keys in the per-widget `options` dict. Anything else is
# dropped by the validator at `services/spec_validator.py`, but we keep
# this list here so both sides agree on what's legal.
ALLOWED_OPTION_KEYS: set[str] = {
    "decimals",
    "show_grid",
    "show_legend",
    "stacked",
    "fill",
    "min",
    "max",
    "columns",
    "severity_filter",
    "row_limit",
    "sort_by",
    "sort_dir",
    "show_labels",
    "donut",
    "horizontal",
    "show_values",
    "x_label",
    "y_label",
    "color_scale",
}

FORBIDDEN_WIDGET_FIELDS: set[str] = {
    "raw_html",
    "script",
    "component",
    "code",
    "iframe",
    "eval",
    "onclick",
    "onerror",
    "html",
    "jsx",
    "render",
}


class WidgetSpec(BaseModel):
    """A single widget inside a dashboard."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    type: WidgetType
    title: str = Field(..., min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=512)
    query: QuerySpec
    position: WidgetPosition
    encoding: WidgetEncoding = Field(default_factory=WidgetEncoding)
    thresholds: list[WidgetThreshold] = Field(default_factory=list, max_length=16)
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title", "description")
    @classmethod
    def _safe_text(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _assert_safe_text(v, "widget text field")

    @field_validator("options")
    @classmethod
    def _safe_options(cls, v: dict[str, Any]) -> dict[str, Any]:
        for key in v.keys():
            if key in FORBIDDEN_WIDGET_FIELDS:
                raise ValueError(f"option key {key!r} is forbidden")
            if key not in ALLOWED_OPTION_KEYS:
                raise ValueError(
                    f"option key {key!r} is not in the allowed option keys; "
                    f"extending the widget requires a DeveloperTicket"
                )
            val = v[key]
            if isinstance(val, str):
                _assert_safe_text(val, f"options.{key}")
        return v
