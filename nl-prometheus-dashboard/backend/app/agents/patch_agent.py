from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from .time_window_parser import parse_prompt_time_window
from ..llm.base import BaseLLMClient, LLMOutputValidationError
from ..specs.dashboard_spec import DashboardSpec
from ..specs.metric_catalog import MetricCatalog, load_default_metric_catalog
from ..specs.patch_spec import PatchOperation, PatchOperationType, PatchSpec
from ..specs.task_model import MonitoringTaskModel
from ..specs.widget_spec import QuerySpec, ThresholdSpec, WidgetLayoutSpec, WidgetSpec, WidgetType


class PatchAgent:
    SYSTEM_PROMPT = """You generate only JSON matching PatchSpec.
Do not regenerate the full dashboard.
PatchSpec is an auditable evolution step over a MonitoringTaskModel-backed dashboard.
Use only add_widget, remove_widget, update_widget, update_dashboard_title, or update_variable.
For update_widget, widget_id is required and updates MUST be a JSON object, never a scalar.
For refresh interval changes, use updates: {"refresh_interval_ms": milliseconds}.
For time range changes, use updates: {"query": {"time_range_seconds": seconds}}.
For absolute clock windows, use updates: {"query": {"start_time": ISO datetime, "end_time": ISO datetime, "time_range_seconds": seconds}}.
Only use metrics from the provided metric_catalog."""

    def __init__(
        self,
        *,
        catalog: MetricCatalog | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.catalog = catalog or load_default_metric_catalog()
        self.llm_client = llm_client

    async def generate_patch(
        self,
        *,
        dashboard_id: str,
        prompt: str,
        current_dashboard: DashboardSpec,
        current_task_model: MonitoringTaskModel | None = None,
    ) -> PatchSpec:
        if self.llm_client is not None:
            raw = await self.llm_client.generate_json(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                context={
                    "current_task_model": (
                        current_task_model.model_dump(mode="json") if current_task_model is not None else None
                    ),
                    "current_dashboard": current_dashboard.model_dump(mode="json"),
                    "metric_catalog": {
                        name: entry.model_dump(mode="json") for name, entry in self.catalog.entries.items()
                    },
                    "patch_schema": PatchSpec.model_json_schema(),
                },
            )
            try:
                return PatchSpec.model_validate(raw)
            except ValidationError as exc:
                try:
                    return self._repair_patch_dict(raw, dashboard_id, prompt, current_dashboard)
                except ValidationError as repair_exc:
                    raise LLMOutputValidationError(
                        f"PatchSpec validation failed: {exc}; repair failed: {repair_exc}"
                    ) from repair_exc

        return self._build_demo_patch(dashboard_id, prompt, current_dashboard)

    def _repair_patch_dict(
        self,
        raw: dict[str, Any],
        dashboard_id: str,
        prompt: str,
        dashboard: DashboardSpec,
    ) -> PatchSpec:
        raw_operations = raw.get("operations")
        if not isinstance(raw_operations, list):
            return self._build_demo_patch(dashboard_id, prompt, dashboard)

        operations: list[dict[str, Any]] = []
        for raw_operation in raw_operations:
            if not isinstance(raw_operation, dict):
                continue
            operation = self._repair_operation(raw_operation, prompt, dashboard)
            if isinstance(operation, list):
                operations.extend(operation)
            elif operation is not None:
                operations.append(operation)

        if not operations:
            return self._build_demo_patch(dashboard_id, prompt, dashboard)

        return PatchSpec.model_validate(
            {
                "dashboard_id": dashboard_id,
                "operations": operations,
                "reason": raw.get("reason") or "llm_schema_repair",
            }
        )

    def _repair_operation(
        self,
        raw_operation: dict[str, Any],
        prompt: str,
        dashboard: DashboardSpec,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        op = self._normalize_operation_type(str(raw_operation.get("op", "")), prompt)

        if op == PatchOperationType.UPDATE_DASHBOARD_TITLE:
            title = raw_operation.get("title")
            updates = raw_operation.get("updates")
            if not title and isinstance(updates, dict):
                title = updates.get("title")
            if not title and isinstance(updates, str):
                title = updates
            return {"op": op.value, "title": str(title or dashboard.title)}

        if op == PatchOperationType.UPDATE_WIDGET:
            updates = self._repair_updates(raw_operation.get("updates"), prompt)
            widget_id = raw_operation.get("widget_id")
            if widget_id:
                return {"op": op.value, "widget_id": str(widget_id), "updates": updates}
            return [
                {"op": op.value, "widget_id": widget.id, "updates": updates}
                for widget in dashboard.widgets
            ]

        if op == PatchOperationType.REMOVE_WIDGET:
            widget_id = raw_operation.get("widget_id")
            target = (
                next((widget for widget in dashboard.widgets if widget.id == widget_id), None)
                or self._find_widget_by_prompt(prompt, dashboard)
                or (dashboard.widgets[-1] if dashboard.widgets else None)
            )
            return {"op": op.value, "widget_id": target.id} if target is not None else None

        if op == PatchOperationType.ADD_WIDGET and isinstance(raw_operation.get("widget"), dict):
            return {"op": op.value, "widget": raw_operation["widget"]}

        if op == PatchOperationType.UPDATE_VARIABLE:
            operation = {"op": op.value}
            if raw_operation.get("variable_name"):
                operation["variable_name"] = raw_operation["variable_name"]
            if raw_operation.get("variable"):
                operation["variable"] = raw_operation["variable"]
            if isinstance(raw_operation.get("updates"), dict):
                operation["updates"] = raw_operation["updates"]
            return operation

        return None

    def _normalize_operation_type(self, raw_op: str, prompt: str) -> PatchOperationType:
        lowered_op = raw_op.lower()
        lowered_prompt = prompt.lower()
        for op in PatchOperationType:
            if lowered_op == op.value:
                return op
        if "title" in lowered_op or "title" in lowered_prompt or "標題" in lowered_prompt or "标题" in lowered_prompt:
            return PatchOperationType.UPDATE_DASHBOARD_TITLE
        if "remove" in lowered_op or "delete" in lowered_op:
            return PatchOperationType.REMOVE_WIDGET
        if "add" in lowered_op:
            return PatchOperationType.ADD_WIDGET
        if "variable" in lowered_op:
            return PatchOperationType.UPDATE_VARIABLE
        return PatchOperationType.UPDATE_WIDGET

    def _repair_updates(self, raw_updates: Any, prompt: str) -> dict[str, Any]:
        if isinstance(raw_updates, dict):
            return raw_updates
        if isinstance(raw_updates, (int, float)):
            lowered = prompt.lower()
            value = int(raw_updates)
            if "range" in lowered or "time range" in lowered:
                return {"query": {"time_range_seconds": value}}
            if value < 1_000 and ("second" in lowered or "秒" in lowered):
                value *= 1_000
            return {"refresh_interval_ms": value}
        return {"metadata": {"llm_repair_note": "empty update payload"}}

    def _build_demo_patch(self, dashboard_id: str, prompt: str, dashboard: DashboardSpec) -> PatchSpec:
        lowered = prompt.lower()
        operations: list[PatchOperation] = []

        title_match = re.search(r'(?:title|標題|标题)\s*[:：]?\s*"([^"]+)"', prompt, re.IGNORECASE)
        if title_match:
            operations.append(
                PatchOperation(op=PatchOperationType.UPDATE_DASHBOARD_TITLE, title=title_match.group(1))
            )

        refresh_seconds = self._extract_seconds(lowered)
        if refresh_seconds is not None and ("refresh" in lowered or "刷新" in lowered):
            for widget in dashboard.widgets:
                operations.append(
                    PatchOperation(
                        op=PatchOperationType.UPDATE_WIDGET,
                        widget_id=widget.id,
                        updates={"refresh_interval_ms": refresh_seconds * 1_000},
                    )
                )

        absolute_window = parse_prompt_time_window(prompt)
        if absolute_window is not None:
            target_widgets = self._find_widgets_by_prompt(prompt, dashboard) or dashboard.widgets
            for widget in target_widgets:
                operations.append(
                    PatchOperation(
                        op=PatchOperationType.UPDATE_WIDGET,
                        widget_id=widget.id,
                        updates={
                            "query": {
                                "start_time": absolute_window.start.isoformat(),
                                "end_time": absolute_window.end.isoformat(),
                                "time_range_seconds": absolute_window.range_seconds,
                            }
                        },
                    )
                )

        range_seconds = self._extract_seconds(lowered)
        if (
            absolute_window is None
            and range_seconds is not None
            and ("range" in lowered or "time" in lowered or "時間" in lowered)
        ):
            target_widgets = self._find_widgets_by_prompt(prompt, dashboard) or dashboard.widgets
            for widget in target_widgets:
                operations.append(
                    PatchOperation(
                        op=PatchOperationType.UPDATE_WIDGET,
                        widget_id=widget.id,
                        updates={"query": {"time_range_seconds": range_seconds, "start_time": None, "end_time": None}},
                    )
                )

        threshold_value = self._extract_number(lowered)
        threshold_keywords = ("threshold", "warning", "alert", "abnormal", "閾值", "阈值", "提示", "異常", "异常")
        if threshold_value is not None and any(keyword in lowered for keyword in threshold_keywords):
            target = self._find_widget_by_prompt(prompt, dashboard) or next(
                (widget for widget in dashboard.widgets if widget.thresholds),
                None,
            )
            if target is not None and target.thresholds:
                thresholds = [threshold.model_dump(mode="json") for threshold in target.thresholds]
                thresholds[-1]["value"] = threshold_value
                operations.append(
                    PatchOperation(
                        op=PatchOperationType.UPDATE_WIDGET,
                        widget_id=target.id,
                        updates={"thresholds": thresholds},
                    )
                )

        if any(keyword in lowered for keyword in ("remove", "delete", "刪除", "删除")) and dashboard.widgets:
            targets = self._find_widgets_by_prompt(prompt, dashboard) or [dashboard.widgets[-1]]
            for target in targets:
                operations.append(PatchOperation(op=PatchOperationType.REMOVE_WIDGET, widget_id=target.id))

        if any(keyword in lowered for keyword in ("add", "新增", "加入")):
            wants_threshold = any(keyword in lowered for keyword in threshold_keywords)
            metric_name = (
                self._select_metric_from_prompt(prompt)
                if wants_threshold
                else self._select_metric_not_already_used(prompt, dashboard)
            )
            if metric_name is not None:
                entry = self.catalog.require(metric_name)
                widget_type = WidgetType.THRESHOLD_CARD if wants_threshold and entry.normal_range is not None else WidgetType.LINE_CHART
                query_type = "instant" if widget_type == WidgetType.THRESHOLD_CARD else "range"
                title_suffix = "Alert" if widget_type == WidgetType.THRESHOLD_CARD else "Trend"
                operations.append(
                    PatchOperation(
                        op=PatchOperationType.ADD_WIDGET,
                        widget=WidgetSpec(
                            title=f"{entry.display_name} {title_suffix}",
                            type=widget_type,
                            unit=entry.unit,
                            query=QuerySpec(
                                metric=metric_name,
                                query_type=query_type,
                                time_range_seconds=300,
                                step_seconds=5,
                            ),
                            thresholds=self._thresholds_for(entry) if widget_type == WidgetType.THRESHOLD_CARD else [],
                            layout=WidgetLayoutSpec(x=0, y=len(dashboard.widgets) * 3, w=8, h=3),
                        ),
                    )
                )

        if not operations:
            operations.append(
                PatchOperation(
                    op=PatchOperationType.UPDATE_DASHBOARD_TITLE,
                    title=dashboard.title,
                )
            )

        return PatchSpec(dashboard_id=dashboard_id, operations=operations, reason="mock_catalog_patch")

    def _extract_seconds(self, text: str) -> int | None:
        minutes = re.search(r"(\d+)\s*(?:minute|minutes|min|m|分鐘|分)", text)
        if minutes:
            return int(minutes.group(1)) * 60
        seconds = re.search(r"(\d+)\s*(?:second|seconds|sec|s|秒)", text)
        if seconds:
            return int(seconds.group(1))
        return None

    def _extract_number(self, text: str) -> float | None:
        match = re.search(r"(-?\d+(?:\.\d+)?)", text)
        return float(match.group(1)) if match else None

    def _find_widget_by_prompt(self, prompt: str, dashboard: DashboardSpec) -> WidgetSpec | None:
        matches = self._find_widgets_by_prompt(prompt, dashboard)
        return matches[0] if matches else None

    def _find_widgets_by_prompt(self, prompt: str, dashboard: DashboardSpec) -> list[WidgetSpec]:
        lowered = prompt.lower()
        matches: list[WidgetSpec] = []
        for widget in dashboard.widgets:
            entry = self.catalog.get(widget.query.metric)
            display_name = entry.display_name.lower() if entry is not None else ""
            if (
                widget.query.metric.lower() in lowered
                or display_name in lowered
                or any(token and token in lowered for token in widget.title.lower().split())
            ):
                matches.append(widget)
                continue
            if widget.query.metric in {"patient_systolic_bp_mmhg", "patient_diastolic_bp_mmhg"} and any(
                keyword in lowered for keyword in ("blood pressure", "bp", "血壓", "血压")
            ):
                matches.append(widget)
        return matches

    def _select_metric_from_prompt(self, prompt: str) -> str | None:
        lowered = prompt.lower()
        for metric_name, entry in self.catalog.entries.items():
            if metric_name.lower() in lowered or entry.display_name.lower() in lowered:
                return metric_name
        keyword_map = {
            "patient_heart_rate_bpm": ("heart", "hr", "心率", "heart rate"),
            "patient_spo2_percent": ("spo2", "oxygen", "血氧"),
            "patient_systolic_bp_mmhg": ("systolic", "blood pressure", "bp", "血壓", "血压"),
            "patient_diastolic_bp_mmhg": ("diastolic", "blood pressure", "bp", "血壓", "血压"),
        }
        for metric_name, keywords in keyword_map.items():
            if any(keyword in lowered for keyword in keywords):
                return metric_name
        return None

    def _select_metric_not_already_used(self, prompt: str, dashboard: DashboardSpec) -> str | None:
        used = {widget.query.metric for widget in dashboard.widgets}
        lowered = prompt.lower()
        for metric_name, entry in self.catalog.entries.items():
            if metric_name in used:
                continue
            if entry.display_name.lower() in lowered or metric_name.lower() in lowered:
                return metric_name
        for metric_name in self.catalog.entries:
            if metric_name not in used:
                return metric_name
        return None

    def _thresholds_for(self, entry) -> list[ThresholdSpec]:
        thresholds: list[ThresholdSpec] = []
        if entry.normal_range is None:
            return thresholds
        if entry.normal_range.min is not None:
            thresholds.append(
                ThresholdSpec(
                    label=f"Low {entry.display_name}",
                    operator="lt",
                    value=entry.normal_range.min,
                    severity="critical",
                    message=f"{entry.display_name} is below normal range",
                )
            )
        if entry.normal_range.max is not None:
            thresholds.append(
                ThresholdSpec(
                    label=f"High {entry.display_name}",
                    operator="gt",
                    value=entry.normal_range.max,
                    severity="critical",
                    message=f"{entry.display_name} is above normal range",
                )
            )
        return thresholds
