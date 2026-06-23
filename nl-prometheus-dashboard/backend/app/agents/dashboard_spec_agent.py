from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import ValidationError

from ..llm.base import BaseLLMClient, LLMOutputValidationError
from ..specs.dashboard_spec import DashboardSpec, DashboardVariableSpec
from ..specs.metric_catalog import MetricCatalog, MetricCatalogEntry, load_default_metric_catalog
from ..specs.task_model import MonitoringTaskModel
from ..specs.widget_spec import QuerySpec, ThresholdSpec, WidgetLayoutSpec, WidgetSpec, WidgetType


class DashboardSpecAgent:
    """Maps a monitoring task model into a validated DashboardSpec.

    The preferred path is MonitoringTaskModel -> DashboardSpec. The legacy
    prompt method remains for compatibility, but it is intentionally catalog
    driven instead of free-form PromQL or UI-code driven.
    """

    SYSTEM_PROMPT = """You generate only JSON matching DashboardSpec.
Never generate React, HTML, JavaScript, Python, or executable code.
Only use metrics from the provided metric_catalog.
Metric catalog type values like gauge/counter are metric metadata, not widget.type.
Every widget.type MUST be one of: line_chart, stat_card, threshold_card, table.
Every widget.query.metric MUST be one catalog metric name.
Every widget.query.query_type MUST be instant or range.
PromQL must be omitted unless it uses the allowed metric exactly."""

    def __init__(
        self,
        *,
        catalog: MetricCatalog | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.catalog = catalog or load_default_metric_catalog()
        self.llm_client = llm_client

    async def generate_dashboard(self, prompt: str, context: dict[str, Any] | None = None) -> DashboardSpec:
        if self.llm_client is not None:
            raw = await self.llm_client.generate_json(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                context={
                    "context": context or {},
                    "metric_catalog": {
                        name: entry.model_dump(mode="json") for name, entry in self.catalog.entries.items()
                    },
                    "dashboard_schema": DashboardSpec.model_json_schema(),
                },
            )
            try:
                return DashboardSpec.model_validate(raw)
            except ValidationError as exc:
                try:
                    return self._repair_dashboard_dict(raw, prompt)
                except ValidationError as repair_exc:
                    raise LLMOutputValidationError(
                        f"DashboardSpec validation failed: {exc}; repair failed: {repair_exc}"
                    ) from repair_exc

        return self._build_demo_dashboard(prompt)

    def generate_dashboard_from_task_model(self, task_model: MonitoringTaskModel) -> DashboardSpec:
        selected_metrics = [
            metric_name for metric_name in task_model.constraints.allowed_metrics if metric_name in self.catalog.entries
        ]
        if not selected_metrics:
            selected_metrics = self._select_metrics(" ".join(task_model.signals))

        intents = set(task_model.analysis_intents or ["trend_monitoring", "latest_value_summary"])
        time_range_seconds = min(
            task_model.time_context.time_range_seconds,
            task_model.constraints.max_time_range_seconds,
        )
        step_seconds = self._step_for(
            time_range_seconds=time_range_seconds,
            min_step_seconds=task_model.constraints.min_step_interval_seconds,
        )

        widgets: list[WidgetSpec] = []
        row = 0
        for metric_name in selected_metrics:
            entry = self.catalog.require(metric_name)
            if "trend_monitoring" in intents:
                widgets.append(
                    self._line_chart(
                        metric_name,
                        entry,
                        row,
                        time_range_seconds=time_range_seconds,
                        step_seconds=step_seconds,
                        start_time=task_model.time_context.start_time,
                        end_time=task_model.time_context.end_time,
                    )
                )
            if "latest_value_summary" in intents and WidgetType.STAT_CARD in entry.recommended_widgets:
                widgets.append(
                    self._stat_card(
                        metric_name,
                        entry,
                        row,
                        time_range_seconds=time_range_seconds,
                        step_seconds=step_seconds,
                        start_time=task_model.time_context.start_time,
                        end_time=task_model.time_context.end_time,
                    )
                )
            if (
                "threshold_detection" in intents
                and WidgetType.THRESHOLD_CARD in entry.recommended_widgets
                and entry.normal_range is not None
            ):
                widgets.append(
                    self._threshold_card(
                        metric_name,
                        entry,
                        row,
                        time_range_seconds=time_range_seconds,
                        step_seconds=step_seconds,
                        start_time=task_model.time_context.start_time,
                        end_time=task_model.time_context.end_time,
                    )
                )
            if "tabular_review" in intents:
                widgets.append(
                    self._table_widget(
                        metric_name,
                        entry,
                        row,
                        time_range_seconds=time_range_seconds,
                        step_seconds=step_seconds,
                        start_time=task_model.time_context.start_time,
                        end_time=task_model.time_context.end_time,
                    )
                )
            row += 1

        return DashboardSpec(
            title=self._title_for_task_model(task_model),
            description=task_model.monitoring_goal,
            widgets=widgets,
            variables=self._variables_for_task_model(task_model),
            refresh_interval_ms=task_model.time_context.refresh_interval_ms,
            time_range_seconds=time_range_seconds,
            metadata={
                "generation_mode": "task_model_mapping",
                "task_model_id": task_model.id,
                "task_model_domain": task_model.domain,
                "catalog_metric_count": len(self.catalog.entries),
            },
        )

    def _build_demo_dashboard(self, prompt: str) -> DashboardSpec:
        selected_metrics = self._select_metrics(prompt)
        widgets: list[WidgetSpec] = []
        for index, metric_name in enumerate(selected_metrics):
            entry = self.catalog.require(metric_name)
            widgets.append(self._line_chart(metric_name, entry, index))
            if WidgetType.STAT_CARD in entry.recommended_widgets:
                widgets.append(self._stat_card(metric_name, entry, index))
            if WidgetType.THRESHOLD_CARD in entry.recommended_widgets and entry.normal_range is not None:
                widgets.append(self._threshold_card(metric_name, entry, index))

        return DashboardSpec(
            title="Patient Monitoring Dashboard",
            description="MVP dashboard generated from a catalog-aware natural language prompt.",
            widgets=widgets,
            variables=[
                DashboardVariableSpec(
                    name="patient_id",
                    label="Patient ID",
                    type="text",
                    default="demo-patient",
                    required=False,
                ),
                DashboardVariableSpec(
                    name="bed_id",
                    label="Bed ID",
                    type="text",
                    default="bed-01",
                    required=False,
                ),
            ],
            refresh_interval_ms=5_000,
            time_range_seconds=300,
            metadata={"generation_mode": "mock_catalog_agent"},
        )

    def _repair_dashboard_dict(self, raw: dict[str, Any], prompt: str) -> DashboardSpec:
        widgets = raw.get("widgets")
        if not isinstance(widgets, list) or not widgets:
            repaired = self._build_demo_dashboard(prompt)
            repaired.metadata["generation_mode"] = "llm_repair_fallback"
            return repaired

        repaired_widgets = []
        fallback_metrics = self._select_metrics(prompt)
        for index, widget in enumerate(widgets):
            if not isinstance(widget, dict):
                continue
            metric_name = self._metric_from_widget(widget, fallback_metrics, index)
            if metric_name is None:
                continue
            entry = self.catalog.require(metric_name)
            widget_type = self._repair_widget_type(widget, entry)
            query_type = "instant" if widget_type in {WidgetType.STAT_CARD, WidgetType.THRESHOLD_CARD} else "range"
            repaired_widgets.append(
                {
                    "id": widget.get("id") or f"widget_llm_{index}",
                    "title": str(widget.get("title") or entry.display_name),
                    "type": widget_type.value,
                    "query": self._repair_query(widget.get("query"), metric_name, query_type),
                    "unit": widget.get("unit") or entry.unit,
                    "thresholds": self._repair_thresholds(widget.get("thresholds"), entry),
                    "refresh_interval_ms": widget.get("refresh_interval_ms"),
                    "layout": self._repair_layout(widget.get("layout"), index, widget_type),
                    "metadata": self._metadata_dict(widget.get("metadata")),
                }
            )

        if not repaired_widgets:
            repaired = self._build_demo_dashboard(prompt)
            repaired.metadata["generation_mode"] = "llm_repair_fallback"
            return repaired

        repaired_dashboard = {
            "title": str(raw.get("title") or "Patient Monitoring Dashboard"),
            "description": raw.get("description") or "Dashboard generated from LLM JSON and schema-repaired.",
            "widgets": repaired_widgets,
            "variables": self._repair_variables(raw.get("variables")),
            "refresh_interval_ms": raw.get("refresh_interval_ms") or 5_000,
            "time_range_seconds": raw.get("time_range_seconds") or 300,
            "version": raw.get("version") or 1,
            "metadata": {**self._metadata_dict(raw.get("metadata")), "generation_mode": "llm_schema_repair"},
        }
        if raw.get("id"):
            repaired_dashboard["id"] = raw["id"]
        return DashboardSpec.model_validate(repaired_dashboard)

    def _metric_from_widget(self, widget: dict[str, Any], fallback_metrics: list[str], index: int) -> str | None:
        query = widget.get("query")
        candidates: list[Any] = [
            widget.get("metric"),
            query.get("metric") if isinstance(query, dict) else None,
            query if isinstance(query, str) else None,
        ]
        title = str(widget.get("title", "")).lower()
        for metric_name, entry in self.catalog.entries.items():
            if metric_name in candidates or metric_name.lower() in title or entry.display_name.lower() in title:
                return metric_name
        if index < len(fallback_metrics):
            return fallback_metrics[index]
        return next(iter(self.catalog.entries), None)

    def _repair_widget_type(self, widget: dict[str, Any], entry: MetricCatalogEntry) -> WidgetType:
        raw_type = str(widget.get("type", "")).lower()
        for widget_type in WidgetType:
            if raw_type == widget_type.value:
                return widget_type

        title = str(widget.get("title", "")).lower()
        if "alert" in title or "threshold" in title:
            return WidgetType.THRESHOLD_CARD
        if raw_type in {"number", "value", "card", "gauge", "metric"}:
            return WidgetType.STAT_CARD
        return entry.recommended_widgets[0]

    def _repair_query(self, raw_query: Any, metric_name: str, query_type: str) -> dict[str, Any]:
        query = raw_query if isinstance(raw_query, dict) else {}
        return {
            "metric": metric_name,
            "promql": query.get("promql") if isinstance(query.get("promql"), str) else None,
            "query_type": query.get("query_type") if query.get("query_type") in {"instant", "range"} else query_type,
            "label_matchers": query.get("label_matchers") if isinstance(query.get("label_matchers"), dict) else {},
            "time_range_seconds": query.get("time_range_seconds") or 300,
            "step_seconds": query.get("step_seconds") or 5,
        }

    def _repair_thresholds(self, raw_thresholds: Any, entry: MetricCatalogEntry) -> list[dict[str, Any]]:
        if isinstance(raw_thresholds, list):
            repaired = []
            for threshold in raw_thresholds:
                if isinstance(threshold, dict) and threshold.get("operator") in {"gt", "gte", "lt", "lte", "eq"}:
                    repaired.append(
                        {
                            "label": str(threshold.get("label") or entry.display_name),
                            "operator": threshold["operator"],
                            "value": threshold.get("value", 0),
                            "severity": (
                                threshold.get("severity")
                                if threshold.get("severity") in {"info", "warning", "critical"}
                                else "warning"
                            ),
                            "message": threshold.get("message"),
                        }
                    )
            if repaired:
                return repaired
        return [threshold.model_dump(mode="json") for threshold in self._thresholds_for(entry)]

    def _repair_layout(self, raw_layout: Any, index: int, widget_type: WidgetType) -> dict[str, int]:
        layout = raw_layout if isinstance(raw_layout, dict) else {}
        default_width = 8 if widget_type == WidgetType.LINE_CHART else 4
        return {
            "x": int(layout.get("x", 0)),
            "y": int(layout.get("y", index * 3)),
            "w": int(layout.get("w", default_width)),
            "h": int(layout.get("h", 3)),
        }

    def _repair_variables(self, raw_variables: Any) -> list[dict[str, Any]]:
        if isinstance(raw_variables, list):
            repaired = []
            for variable in raw_variables:
                if isinstance(variable, dict) and variable.get("name"):
                    repaired.append(
                        {
                            "name": variable["name"],
                            "label": variable.get("label") or variable["name"],
                            "type": variable.get("type") if variable.get("type") in {"text", "select"} else "text",
                            "default": variable.get("default"),
                            "options": variable.get("options") if isinstance(variable.get("options"), list) else [],
                            "required": bool(variable.get("required", False)),
                            "description": variable.get("description"),
                        }
                    )
            if repaired:
                return repaired
        return [
            {
                "name": "patient_id",
                "label": "Patient ID",
                "type": "text",
                "default": "demo-patient",
                "options": [],
                "required": False,
            },
            {
                "name": "bed_id",
                "label": "Bed ID",
                "type": "text",
                "default": "bed-01",
                "options": [],
                "required": False,
            },
        ]

    def _metadata_dict(self, value: Any) -> dict[str, str | int | float | bool]:
        if not isinstance(value, dict):
            return {}
        return {
            str(key): item
            for key, item in value.items()
            if isinstance(item, (str, int, float, bool))
        }

    def _select_metrics(self, prompt: str) -> list[str]:
        lowered = prompt.lower()
        selected: list[str] = []
        keyword_map = {
            "patient_heart_rate_bpm": ("heart", "hr", "心率", "heart rate"),
            "patient_spo2_percent": ("spo2", "oxygen", "血氧"),
            "patient_systolic_bp_mmhg": ("blood pressure", "bp", "systolic", "血壓", "血压"),
            "patient_diastolic_bp_mmhg": ("blood pressure", "bp", "diastolic", "血壓", "血压"),
        }
        for metric_name, keywords in keyword_map.items():
            if any(keyword in lowered for keyword in keywords):
                selected.append(metric_name)

        if not selected:
            selected = list(self.catalog.entries.keys())[:3]
        return selected

    def _thresholds_for(self, entry: MetricCatalogEntry) -> list[ThresholdSpec]:
        if entry.normal_range is None:
            return []
        thresholds: list[ThresholdSpec] = []
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

    def _line_chart(
        self,
        metric_name: str,
        entry: MetricCatalogEntry,
        index: int,
        *,
        time_range_seconds: int = 300,
        step_seconds: int = 5,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> WidgetSpec:
        return WidgetSpec(
            title=f"{entry.display_name} Trend",
            type=WidgetType.LINE_CHART,
            unit=entry.unit,
            query=QuerySpec(
                metric=metric_name,
                time_range_seconds=time_range_seconds,
                step_seconds=step_seconds,
                start_time=start_time,
                end_time=end_time,
            ),
            thresholds=self._thresholds_for(entry),
            layout=WidgetLayoutSpec(x=0, y=index * 6, w=8, h=3),
        )

    def _stat_card(
        self,
        metric_name: str,
        entry: MetricCatalogEntry,
        index: int,
        *,
        time_range_seconds: int = 300,
        step_seconds: int = 5,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> WidgetSpec:
        return WidgetSpec(
            title=entry.display_name,
            type=WidgetType.STAT_CARD,
            unit=entry.unit,
            query=QuerySpec(
                metric=metric_name,
                query_type="instant",
                time_range_seconds=time_range_seconds,
                step_seconds=step_seconds,
                start_time=start_time,
                end_time=end_time,
            ),
            thresholds=self._thresholds_for(entry),
            layout=WidgetLayoutSpec(x=8, y=index * 6, w=4, h=2),
        )

    def _threshold_card(
        self,
        metric_name: str,
        entry: MetricCatalogEntry,
        index: int,
        *,
        time_range_seconds: int = 300,
        step_seconds: int = 5,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> WidgetSpec:
        return WidgetSpec(
            title=f"{entry.display_name} Alert",
            type=WidgetType.THRESHOLD_CARD,
            unit=entry.unit,
            query=QuerySpec(
                metric=metric_name,
                query_type="instant",
                time_range_seconds=time_range_seconds,
                step_seconds=step_seconds,
                start_time=start_time,
                end_time=end_time,
            ),
            thresholds=self._thresholds_for(entry),
            layout=WidgetLayoutSpec(x=8, y=index * 6 + 2, w=4, h=2),
        )

    def _table_widget(
        self,
        metric_name: str,
        entry: MetricCatalogEntry,
        index: int,
        *,
        time_range_seconds: int = 300,
        step_seconds: int = 5,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> WidgetSpec:
        return WidgetSpec(
            title=f"{entry.display_name} Samples",
            type=WidgetType.TABLE,
            unit=entry.unit,
            query=QuerySpec(
                metric=metric_name,
                time_range_seconds=time_range_seconds,
                step_seconds=step_seconds,
                start_time=start_time,
                end_time=end_time,
            ),
            thresholds=self._thresholds_for(entry),
            layout=WidgetLayoutSpec(x=0, y=index * 6 + 3, w=12, h=3),
        )

    def _step_for(self, *, time_range_seconds: int, min_step_seconds: int) -> int:
        return max(min_step_seconds, min(3_600, max(1, time_range_seconds // 120)))

    def _title_for_task_model(self, task_model: MonitoringTaskModel) -> str:
        if task_model.domain == "medical_monitoring":
            return "Patient Monitoring Dashboard"
        return "Live Monitoring Dashboard"

    def _variables_for_task_model(self, task_model: MonitoringTaskModel) -> list[DashboardVariableSpec]:
        variables: list[DashboardVariableSpec] = []
        if "patient" in task_model.entities:
            variables.append(
                DashboardVariableSpec(
                    name="patient_id",
                    label="Patient ID",
                    type="text",
                    default="demo-patient",
                    required=False,
                )
            )
        if "bed" in task_model.entities:
            variables.append(
                DashboardVariableSpec(
                    name="bed_id",
                    label="Bed ID",
                    type="text",
                    default="bed-01",
                    required=False,
                )
            )
        return variables
