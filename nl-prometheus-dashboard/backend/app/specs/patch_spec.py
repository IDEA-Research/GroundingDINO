from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .dashboard_spec import DashboardVariableSpec
from .widget_spec import WidgetSpec


class PatchOperationType(str, Enum):
    ADD_WIDGET = "add_widget"
    REMOVE_WIDGET = "remove_widget"
    UPDATE_WIDGET = "update_widget"
    UPDATE_DASHBOARD_TITLE = "update_dashboard_title"
    UPDATE_VARIABLE = "update_variable"


class PatchOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: PatchOperationType
    widget_id: str | None = None
    widget: WidgetSpec | None = None
    title: str | None = None
    variable_name: str | None = None
    variable: DashboardVariableSpec | None = None
    updates: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_operation_payload(self) -> "PatchOperation":
        if self.op == PatchOperationType.ADD_WIDGET and self.widget is None:
            raise ValueError("add_widget requires widget")
        if self.op == PatchOperationType.REMOVE_WIDGET and not self.widget_id:
            raise ValueError("remove_widget requires widget_id")
        if self.op == PatchOperationType.UPDATE_WIDGET:
            if not self.widget_id:
                raise ValueError("update_widget requires widget_id")
            if not self.updates:
                raise ValueError("update_widget requires updates")
        if self.op == PatchOperationType.UPDATE_DASHBOARD_TITLE and not self.title:
            raise ValueError("update_dashboard_title requires title")
        if self.op == PatchOperationType.UPDATE_VARIABLE:
            if not self.variable_name and self.variable is None:
                raise ValueError("update_variable requires variable_name or variable")
            if self.variable is None and not self.updates:
                raise ValueError("update_variable requires variable or updates")
        return self


class PatchSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dashboard_id: str
    operations: list[PatchOperation] = Field(default_factory=list)
    reason: str | None = None
