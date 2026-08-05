"""PatchSpec — typed, minimal edits to an existing DashboardSpec."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .widget_spec import WidgetSpec
from .dashboard_spec import DashboardLayout


class PatchOp(str, Enum):
    add_widget = "add_widget"
    remove_widget = "remove_widget"
    update_widget = "update_widget"
    update_dashboard = "update_dashboard"
    reorder_widgets = "reorder_widgets"


class AddWidgetOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal[PatchOp.add_widget] = PatchOp.add_widget
    widget: WidgetSpec


class RemoveWidgetOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal[PatchOp.remove_widget] = PatchOp.remove_widget
    widget_id: str = Field(..., min_length=1, max_length=64)


# Only these top-level WidgetSpec fields are patchable. `id` is
# intentionally excluded — renaming a widget identity is not a patch,
# it's a remove + add. `type` *is* patchable so Helper can switch a
# widget between supported toolkit types (e.g. line_chart -> table)
# in a single operation. The patched widget is re-validated as a full
# WidgetSpec before the dashboard is saved.
UPDATE_WIDGET_FIELDS: set[str] = {
    "type",
    "title",
    "description",
    "query",
    "position",
    "encoding",
    "thresholds",
    "options",
}


class UpdateWidgetOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal[PatchOp.update_widget] = PatchOp.update_widget
    widget_id: str = Field(..., min_length=1, max_length=64)
    fields: dict[str, Any]

    @field_validator("fields")
    @classmethod
    def _allowed_fields(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("update_widget.fields must not be empty")
        bad = set(v.keys()) - UPDATE_WIDGET_FIELDS
        if bad:
            raise ValueError(
                f"update_widget.fields contains unsupported keys: {sorted(bad)}"
            )
        return v


UPDATE_DASHBOARD_FIELDS: set[str] = {
    "title",
    "description",
    "refresh_interval",
    "layout",
}


class UpdateDashboardOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal[PatchOp.update_dashboard] = PatchOp.update_dashboard
    fields: dict[str, Any]

    @field_validator("fields")
    @classmethod
    def _allowed_fields(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("update_dashboard.fields must not be empty")
        bad = set(v.keys()) - UPDATE_DASHBOARD_FIELDS
        if bad:
            raise ValueError(
                f"update_dashboard.fields contains unsupported keys: {sorted(bad)}"
            )
        return v


class ReorderWidgetsOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal[PatchOp.reorder_widgets] = PatchOp.reorder_widgets
    order: list[str] = Field(..., min_length=1, max_length=64)

    @field_validator("order")
    @classmethod
    def _unique_order(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("reorder_widgets.order must be unique")
        for s in v:
            if not re.match(r"^[a-zA-Z0-9_\-]+$", s):
                raise ValueError(f"widget id {s!r} has invalid chars")
        return v


PatchOperation = (
    AddWidgetOp
    | RemoveWidgetOp
    | UpdateWidgetOp
    | UpdateDashboardOp
    | ReorderWidgetsOp
)


class PatchSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    reason: str = Field(..., min_length=1, max_length=256)
    target_dashboard_id: str = Field(..., min_length=1, max_length=64)
    created_by: str = Field(default="patch-agent", max_length=64)
    operations: list[PatchOperation] = Field(..., min_length=1, max_length=32)

    @model_validator(mode="after")
    def _non_empty(self) -> "PatchSpec":
        if not self.operations:
            raise ValueError("PatchSpec must have at least one operation")
        return self
