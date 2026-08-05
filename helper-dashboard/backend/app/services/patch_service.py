"""Apply a PatchSpec to a DashboardSpec.

Every operation is typed. Patches are validated before application,
and the resulting DashboardSpec is validated after application.
"""

from __future__ import annotations

from ..specs import DashboardSpec, PatchSpec
from ..specs.patch_spec import (
    AddWidgetOp,
    PatchOp,
    RemoveWidgetOp,
    ReorderWidgetsOp,
    UpdateDashboardOp,
    UpdateWidgetOp,
)
from .spec_validator import SpecValidationError, SpecValidator


class PatchApplicationError(Exception):
    pass


class PatchService:
    def __init__(self, validator: SpecValidator | None = None):
        self._validator = validator or SpecValidator()

    def apply(self, dashboard: DashboardSpec, patch: PatchSpec) -> DashboardSpec:
        if patch.target_dashboard_id != dashboard.dashboard_id:
            raise PatchApplicationError(
                f"patch target {patch.target_dashboard_id!r} does not match "
                f"dashboard {dashboard.dashboard_id!r}"
            )

        data = dashboard.model_dump(mode="json")

        for op in patch.operations:
            if isinstance(op, AddWidgetOp):
                data["widgets"].append(op.widget.model_dump(mode="json"))
            elif isinstance(op, RemoveWidgetOp):
                before = len(data["widgets"])
                data["widgets"] = [w for w in data["widgets"] if w["id"] != op.widget_id]
                if len(data["widgets"]) == before:
                    raise PatchApplicationError(
                        f"remove_widget: no widget with id {op.widget_id!r}"
                    )
            elif isinstance(op, UpdateWidgetOp):
                found = False
                for w in data["widgets"]:
                    if w["id"] == op.widget_id:
                        for key, value in op.fields.items():
                            w[key] = value
                        found = True
                        break
                if not found:
                    raise PatchApplicationError(
                        f"update_widget: no widget with id {op.widget_id!r}"
                    )
            elif isinstance(op, UpdateDashboardOp):
                for key, value in op.fields.items():
                    data[key] = value
            elif isinstance(op, ReorderWidgetsOp):
                by_id = {w["id"]: w for w in data["widgets"]}
                missing = set(op.order) - set(by_id.keys())
                extra = set(by_id.keys()) - set(op.order)
                if missing or extra:
                    raise PatchApplicationError(
                        f"reorder_widgets: order mismatch (missing={sorted(missing)}, "
                        f"extra={sorted(extra)})"
                    )
                data["widgets"] = [by_id[wid] for wid in op.order]
            else:
                raise PatchApplicationError(f"unknown operation type: {op!r}")

        try:
            return self._validator.validate_dashboard(data)
        except SpecValidationError as exc:
            raise PatchApplicationError(
                "patched dashboard failed validation: " + "; ".join(exc.errors)
            ) from exc
