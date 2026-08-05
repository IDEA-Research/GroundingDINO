from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from .time_window_parser import parse_prompt_time_window
from ..llm.base import BaseLLMClient, LLMOutputValidationError
from ..specs.dashboard_spec import DashboardSpec
from ..specs.metric_catalog import MetricCatalog, load_default_metric_catalog
from ..specs.patch_spec import PatchOperationType, PatchSpec
from ..specs.task_model import MonitoringTaskModel, TaskConstraintsSpec, TimeContextSpec
from ..specs.widget_spec import WidgetType


class MetricResolver:
    """Maps natural-language monitoring signals to catalog-backed metrics."""

    SIGNAL_TO_METRICS = {
        "heart_rate": ("patient_heart_rate_bpm",),
        "spo2": ("patient_spo2_percent",),
        "systolic_bp": ("patient_systolic_bp_mmhg",),
        "diastolic_bp": ("patient_diastolic_bp_mmhg",),
    }

    KEYWORDS = {
        "heart_rate": ("heart rate", "heart", "hr", "心率", "心跳"),
        "spo2": ("spo2", "oxygen", "oxygen saturation", "血氧"),
        "systolic_bp": ("systolic", "收縮壓", "收缩压"),
        "diastolic_bp": ("diastolic", "舒張壓", "舒张压"),
    }

    BLOOD_PRESSURE_KEYWORDS = ("blood pressure", "bp", "血壓", "血压")

    def __init__(self, catalog: MetricCatalog) -> None:
        self.catalog = catalog

    def resolve_prompt(self, prompt: str) -> tuple[list[str], list[str]]:
        lowered = prompt.lower()
        signals: list[str] = []
        for signal, keywords in self.KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                signals.append(signal)

        if any(keyword in lowered for keyword in self.BLOOD_PRESSURE_KEYWORDS):
            signals.extend(["systolic_bp", "diastolic_bp"])

        if not signals:
            signals = ["heart_rate", "spo2", "systolic_bp"]

        return self._dedupe(signals), self.metrics_for_signals(signals)

    def metrics_for_signals(self, signals: list[str]) -> list[str]:
        metrics: list[str] = []
        for signal in signals:
            for metric_name in self.SIGNAL_TO_METRICS.get(signal, ()):
                if metric_name in self.catalog.entries:
                    metrics.append(metric_name)
        return self._dedupe(metrics)

    def signals_for_metrics(self, metrics: list[str]) -> list[str]:
        signals: list[str] = []
        for metric_name in metrics:
            for signal, signal_metrics in self.SIGNAL_TO_METRICS.items():
                if metric_name in signal_metrics:
                    signals.append(signal)
        return self._dedupe(signals)

    def _dedupe(self, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


class TaskModelAgent:
    """Produces and evolves MonitoringTaskModel, not UI code or widgets."""

    SYSTEM_PROMPT = """You generate only JSON matching MonitoringTaskModel.
Do not generate React, HTML, JavaScript, Python, PromQL, or executable code.
Represent the user's monitoring task before UI generation.
Use only metrics from constraints.allowed_metrics and the provided metric_catalog.
Use analysis_intents only from: trend_monitoring, threshold_detection, latest_value_summary, tabular_review.
Unknown JSON fields are not allowed."""

    def __init__(
        self,
        *,
        catalog: MetricCatalog | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.catalog = catalog or load_default_metric_catalog()
        self.llm_client = llm_client
        self.metric_resolver = MetricResolver(self.catalog)

    async def generate_task_model(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> MonitoringTaskModel:
        if self.llm_client is not None:
            signals, metrics = self.metric_resolver.resolve_prompt(prompt)
            raw = await self.llm_client.generate_json(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                context={
                    "context": context or {},
                    "metric_catalog": {
                        name: entry.model_dump(mode="json") for name, entry in self.catalog.entries.items()
                    },
                    "resolved_signal_candidates": signals,
                    "allowed_metric_candidates": metrics,
                    "task_model_schema": MonitoringTaskModel.model_json_schema(),
                },
            )
            try:
                task_model = MonitoringTaskModel.model_validate(raw)
            except ValidationError as exc:
                raise LLMOutputValidationError(f"MonitoringTaskModel validation failed: {exc}") from exc
            return self._ground_task_model(task_model, prompt, generation_mode="llm_task_model")

        return self._build_demo_task_model(prompt)

    def patch_task_model(
        self,
        *,
        current_task_model: MonitoringTaskModel,
        prompt: str,
        patch: PatchSpec,
        current_dashboard: DashboardSpec | None = None,
    ) -> MonitoringTaskModel:
        updated = current_task_model.model_copy(deep=True)
        lowered = prompt.lower()
        prompt_signals, prompt_metrics = self.metric_resolver.resolve_prompt(prompt)

        if any(keyword in lowered for keyword in ("remove", "delete", "刪除", "删除")):
            metrics_to_remove = set(prompt_metrics)
            for operation in patch.operations:
                if operation.op == PatchOperationType.REMOVE_WIDGET and current_dashboard is not None:
                    widget = next(
                        (candidate for candidate in current_dashboard.widgets if candidate.id == operation.widget_id),
                        None,
                    )
                    if widget is not None:
                        metrics_to_remove.add(widget.query.metric)

            updated.constraints.allowed_metrics = [
                metric_name for metric_name in updated.constraints.allowed_metrics if metric_name not in metrics_to_remove
            ]
            removed_signals = set(self.metric_resolver.signals_for_metrics(list(metrics_to_remove)))
            updated.signals = [signal for signal in updated.signals if signal not in removed_signals]

        if any(keyword in lowered for keyword in ("add", "新增", "加入", "warning", "alert", "提示", "異常", "异常")):
            for signal in prompt_signals:
                if signal not in updated.signals:
                    updated.signals.append(signal)
            for metric_name in prompt_metrics:
                if metric_name not in updated.constraints.allowed_metrics:
                    updated.constraints.allowed_metrics.append(metric_name)

        for operation in patch.operations:
            if operation.op == PatchOperationType.ADD_WIDGET and operation.widget is not None:
                metric_name = operation.widget.query.metric
                if metric_name not in updated.constraints.allowed_metrics:
                    updated.constraints.allowed_metrics.append(metric_name)
                for signal in self.metric_resolver.signals_for_metrics([metric_name]):
                    if signal not in updated.signals:
                        updated.signals.append(signal)

        if any(keyword in lowered for keyword in ("threshold", "warning", "alert", "abnormal", "閾值", "阈值", "提示", "異常", "异常")):
            if "threshold_detection" not in updated.analysis_intents:
                updated.analysis_intents.append("threshold_detection")
        if any(keyword in lowered for keyword in ("line", "trend", "chart", "graph", "折線", "折线", "趨勢", "趋势")):
            if "trend_monitoring" not in updated.analysis_intents:
                updated.analysis_intents.append("trend_monitoring")
        if any(keyword in lowered for keyword in ("stat", "card", "latest", "summary", "目前", "最新")):
            if "latest_value_summary" not in updated.analysis_intents:
                updated.analysis_intents.append("latest_value_summary")

        seconds = self._extract_seconds(lowered)
        absolute_window = parse_prompt_time_window(prompt)
        if absolute_window is not None:
            updated.time_context.start_time = absolute_window.start
            updated.time_context.end_time = absolute_window.end
            updated.time_context.time_range_seconds = min(
                absolute_window.range_seconds,
                updated.constraints.max_time_range_seconds,
            )
        if seconds is not None and any(keyword in lowered for keyword in ("range", "time", "時間", "范围", "範圍")):
            updated.time_context.time_range_seconds = min(seconds, updated.constraints.max_time_range_seconds)
            updated.time_context.start_time = None
            updated.time_context.end_time = None
        if seconds is not None and any(keyword in lowered for keyword in ("refresh", "poll", "interval", "刷新", "更新")):
            updated.time_context.refresh_interval_ms = max(seconds * 1_000, 1_000)

        updated.metadata["last_patch_mode"] = "patch_task_model"
        return MonitoringTaskModel.model_validate(updated.model_dump(mode="json"))

    def infer_from_dashboard(self, dashboard: DashboardSpec) -> MonitoringTaskModel:
        metrics = list(dict.fromkeys(widget.query.metric for widget in dashboard.widgets))
        intents: list[str] = []
        widget_types = {widget.type for widget in dashboard.widgets}
        if WidgetType.LINE_CHART in widget_types:
            intents.append("trend_monitoring")
        if WidgetType.THRESHOLD_CARD in widget_types:
            intents.append("threshold_detection")
        if WidgetType.STAT_CARD in widget_types:
            intents.append("latest_value_summary")
        if WidgetType.TABLE in widget_types:
            intents.append("tabular_review")

        return MonitoringTaskModel(
            domain="medical_monitoring",
            monitoring_goal=dashboard.title,
            entities=["patient", "bed", "metric"],
            signals=self.metric_resolver.signals_for_metrics(metrics),
            relationships=[
                "patient has metrics",
                "metric belongs to patient_id",
                "metric belongs to bed_id",
            ],
            analysis_intents=intents or ["trend_monitoring", "latest_value_summary"],
            time_context=TimeContextSpec(
                time_range_seconds=dashboard.time_range_seconds,
                refresh_interval_ms=dashboard.refresh_interval_ms,
                start_time=dashboard.widgets[0].query.start_time if dashboard.widgets else None,
                end_time=dashboard.widgets[0].query.end_time if dashboard.widgets else None,
            ),
            constraints=TaskConstraintsSpec(allowed_metrics=metrics),
            metadata={"generation_mode": "inferred_from_dashboard"},
        )

    def _build_demo_task_model(self, prompt: str) -> MonitoringTaskModel:
        signals, metrics = self.metric_resolver.resolve_prompt(prompt)
        absolute_window = parse_prompt_time_window(prompt)
        time_range_seconds = (
            absolute_window.range_seconds
            if absolute_window is not None
            else self._extract_seconds(prompt.lower()) or 300
        )
        return MonitoringTaskModel(
            domain=self._domain_for_prompt(prompt),
            monitoring_goal=self._goal_for_prompt(prompt),
            entities=["patient", "bed", "metric"],
            signals=signals,
            relationships=[
                "patient has metrics",
                "metric belongs to patient_id",
                "metric belongs to bed_id",
            ],
            analysis_intents=self._analysis_intents(prompt),
            time_context=TimeContextSpec(
                time_range_seconds=min(time_range_seconds, 86_400),
                refresh_interval_ms=5_000,
                start_time=absolute_window.start if absolute_window is not None else None,
                end_time=absolute_window.end if absolute_window is not None else None,
            ),
            constraints=TaskConstraintsSpec(
                allowed_metrics=metrics,
                max_time_range_seconds=86_400,
                min_step_interval_seconds=5,
            ),
            metadata={"generation_mode": "mock_task_model"},
        )

    def _ground_task_model(
        self,
        task_model: MonitoringTaskModel,
        prompt: str,
        *,
        generation_mode: str,
    ) -> MonitoringTaskModel:
        prompt_signals, prompt_metrics = self.metric_resolver.resolve_prompt(prompt)
        signals = task_model.signals or prompt_signals
        allowed_metrics = [
            metric_name for metric_name in task_model.constraints.allowed_metrics if metric_name in self.catalog.entries
        ]
        if not allowed_metrics:
            allowed_metrics = self.metric_resolver.metrics_for_signals(signals) or prompt_metrics

        task_model.signals = list(dict.fromkeys(signals))
        task_model.constraints.allowed_metrics = list(dict.fromkeys(allowed_metrics))
        if not task_model.analysis_intents:
            task_model.analysis_intents = self._analysis_intents(prompt)
        absolute_window = parse_prompt_time_window(prompt)
        if absolute_window is not None:
            task_model.time_context.start_time = absolute_window.start
            task_model.time_context.end_time = absolute_window.end
            task_model.time_context.time_range_seconds = min(
                absolute_window.range_seconds,
                task_model.constraints.max_time_range_seconds,
            )
        task_model.metadata["generation_mode"] = generation_mode
        return MonitoringTaskModel.model_validate(task_model.model_dump(mode="json"))

    def _analysis_intents(self, prompt: str) -> list[str]:
        lowered = prompt.lower()
        intents: list[str] = []
        if any(keyword in lowered for keyword in ("line", "trend", "chart", "graph", "折線", "折线", "趨勢", "趋势")):
            intents.append("trend_monitoring")
        if any(keyword in lowered for keyword in ("threshold", "warning", "alert", "abnormal", "閾值", "阈值", "提示", "異常", "异常")):
            intents.append("threshold_detection")
        if any(keyword in lowered for keyword in ("stat", "card", "latest", "summary", "目前", "最新")):
            intents.append("latest_value_summary")
        if "table" in lowered or "表格" in lowered:
            intents.append("tabular_review")
        return intents or ["trend_monitoring", "latest_value_summary"]

    def _domain_for_prompt(self, prompt: str) -> str:
        lowered = prompt.lower()
        if any(keyword in lowered for keyword in ("patient", "medical", "icu", "hospital", "病人", "患者", "醫療", "医疗")):
            return "medical_monitoring"
        return "monitoring"

    def _goal_for_prompt(self, prompt: str) -> str:
        cleaned = " ".join(prompt.split())
        return cleaned[:160] if cleaned else "Live monitoring dashboard"

    def _extract_seconds(self, text: str) -> int | None:
        hours = re.search(r"(\d+)\s*(?:hour|hours|hr|h|小時|小时)", text)
        if hours:
            return int(hours.group(1)) * 3_600
        minutes = re.search(r"(\d+)\s*(?:minute|minutes|min|m|分鐘|分钟|分)", text)
        if minutes:
            return int(minutes.group(1)) * 60
        seconds = re.search(r"(\d+)\s*(?:second|seconds|sec|s|秒)", text)
        if seconds:
            return int(seconds.group(1))
        return None
