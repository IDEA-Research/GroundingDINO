from __future__ import annotations

import math
import os
import time
from typing import Any

from ..prometheus.client import PrometheusClient
from ..prometheus.promql_validator import PromQLValidator
from ..specs.metric_catalog import MetricCatalog, load_default_metric_catalog
from ..specs.widget_spec import WidgetSpec


class QueryService:
    MOCK_METRICS = {
        "patient_heart_rate_bpm": {"base": 82.0, "amplitude": 9.0, "period": 7.0, "phase": 0.0},
        "patient_spo2_percent": {"base": 97.0, "amplitude": 1.8, "period": 9.0, "phase": 1.2},
        "patient_systolic_bp_mmhg": {"base": 116.0, "amplitude": 8.0, "period": 11.0, "phase": 2.0},
        "patient_diastolic_bp_mmhg": {"base": 74.0, "amplitude": 5.0, "period": 13.0, "phase": 2.8},
    }

    def __init__(
        self,
        *,
        catalog: MetricCatalog | None = None,
        prometheus_client: PrometheusClient | None = None,
        validator: PromQLValidator | None = None,
    ) -> None:
        self.catalog = catalog or load_default_metric_catalog()
        self.prometheus_client = prometheus_client or PrometheusClient()
        self.validator = validator or PromQLValidator(self.catalog)
        self.prometheus_mode = os.getenv("PROMETHEUS_MODE", "mock").lower()

    async def query_widget(self, widget: WidgetSpec, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        validation = self.validator.validate_query_spec(widget.query)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))

        secondary_metric = self._secondary_metric(widget)
        if secondary_metric is not None and secondary_metric not in self.catalog.metric_names():
            raise ValueError(f"Secondary metric is not allowed by catalog: {secondary_metric}")

        if self.prometheus_mode == "mock":
            return self._mock_widget_response(widget, secondary_metric)

        query = self._apply_variables(widget.query.effective_promql(), variables or {})
        now = time.time()
        series = await self._query_prometheus_series(
            metric=widget.query.metric,
            query=query,
            widget=widget,
            now=now,
        )
        friendly_series = self._prometheus_to_friendly_series(widget.query.metric, series)

        if secondary_metric is not None:
            secondary_series = await self._query_prometheus_series(
                metric=secondary_metric,
                query=secondary_metric,
                widget=widget,
                now=now,
            )
            friendly_series.extend(self._prometheus_to_friendly_series(secondary_metric, secondary_series))

        return {
            "series": friendly_series,
            "metadata": {
                "source": "prometheus",
                "query": query,
                "query_type": widget.query.query_type,
                "metric": widget.query.metric,
                "secondary_metric": secondary_metric,
                "unit": widget.unit,
                "start_time": widget.query.start_time.isoformat() if widget.query.start_time else None,
                "end_time": widget.query.end_time.isoformat() if widget.query.end_time else None,
            },
        }

    async def _query_prometheus_series(
        self,
        *,
        metric: str,
        query: str,
        widget: WidgetSpec,
        now: float,
    ) -> dict[str, Any]:
        if widget.query.query_type == "instant":
            query_time = widget.query.end_time.timestamp() if widget.query.end_time is not None else now
            return await self.prometheus_client.instant_query(query=query, time=query_time)

        start, end = self._query_window(widget, now)
        return await self.prometheus_client.range_query(
            query=query if metric == widget.query.metric else metric,
            start=start,
            end=end,
            step=widget.query.step_seconds,
        )

    def _mock_widget_response(self, widget: WidgetSpec, secondary_metric: str | None) -> dict[str, Any]:
        metrics = [widget.query.metric]
        if secondary_metric is not None:
            metrics.append(secondary_metric)

        now = time.time()
        timestamps = self._mock_timestamps(widget, now)
        series = []
        for metric_name in metrics:
            entry = self.catalog.require(metric_name)
            series.append(
                {
                    "name": entry.display_name,
                    "unit": entry.unit,
                    "points": [
                        {
                            "timestamp": timestamp_ms,
                            "value": self._mock_value(metric_name, timestamp_ms / 1000),
                        }
                        for timestamp_ms in timestamps
                    ],
                }
            )

        return {
            "series": series,
            "metadata": {
                "source": "mock",
                "query_type": widget.query.query_type,
                "metric": widget.query.metric,
                "secondary_metric": secondary_metric,
                "start_time": widget.query.start_time.isoformat() if widget.query.start_time else None,
                "end_time": widget.query.end_time.isoformat() if widget.query.end_time else None,
            },
        }

    def _mock_timestamps(self, widget: WidgetSpec, now: float) -> list[int]:
        if widget.query.query_type == "instant":
            query_time = widget.query.end_time.timestamp() if widget.query.end_time is not None else now
            return [int(query_time * 1000)]

        start, end = self._query_window(widget, now)
        step = max(widget.query.step_seconds, 1)
        point_count = max(2, min(240, int((end - start) / step) + 1))
        if point_count >= 240:
            step = max(step, int((end - start) / 239) or 1)
            start = end - step * 239
            point_count = 240
        return [int((start + index * step) * 1000) for index in range(point_count)]

    def _mock_value(self, metric_name: str, timestamp_seconds: float) -> float:
        spec = self.MOCK_METRICS.get(metric_name, {"base": 50.0, "amplitude": 10.0, "period": 30.0, "phase": 0.0})
        base = spec["base"]
        amplitude = spec["amplitude"]
        period = spec["period"]
        phase = spec["phase"]
        value = (
            base
            + amplitude * math.sin(timestamp_seconds / period + phase)
            + amplitude * 0.18 * math.sin(timestamp_seconds / 7.0 + phase)
        )
        return round(value, 1)

    def _prometheus_to_friendly_series(self, metric_name: str, data: dict[str, Any]) -> list[dict[str, Any]]:
        entry = self.catalog.require(metric_name)
        result = data.get("result", [])
        friendly = []
        for index, prometheus_series in enumerate(result):
            labels = prometheus_series.get("metric", {})
            label_suffix = self._label_suffix(labels)
            values = prometheus_series.get("values")
            if values is None and prometheus_series.get("value") is not None:
                values = [prometheus_series["value"]]
            points = [
                {"timestamp": int(float(timestamp) * 1000), "value": float(raw_value)}
                for timestamp, raw_value in (values or [])
            ]
            friendly.append(
                {
                    "name": f"{entry.display_name}{label_suffix}" if label_suffix else entry.display_name,
                    "unit": entry.unit,
                    "points": points,
                }
            )
        if not friendly:
            friendly.append({"name": entry.display_name, "unit": entry.unit, "points": []})
        return friendly

    def _secondary_metric(self, widget: WidgetSpec) -> str | None:
        value = widget.metadata.get("secondary_metric")
        return value if isinstance(value, str) and value else None

    def _label_suffix(self, labels: dict[str, Any]) -> str:
        patient_id = labels.get("patient_id")
        bed_id = labels.get("bed_id")
        parts = [part for part in (patient_id, bed_id) if part]
        return f" ({', '.join(parts)})" if parts else ""

    def _query_window(self, widget: WidgetSpec, now: float) -> tuple[float, float]:
        if widget.query.start_time is not None and widget.query.end_time is not None:
            return widget.query.start_time.timestamp(), widget.query.end_time.timestamp()
        return now - widget.query.time_range_seconds, now

    def _apply_variables(self, query: str, variables: dict[str, Any]) -> str:
        rendered = query
        for name, value in variables.items():
            safe_value = str(value).replace("\\", "\\\\").replace('"', '\\"')
            rendered = rendered.replace(f"${name}", safe_value)
        return rendered
