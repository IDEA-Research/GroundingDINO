from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .widget_spec import WidgetType


class NormalRangeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None


class MetricCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: str
    unit: str | None = None
    type: Literal["gauge", "counter", "histogram", "summary", "unknown"] = "unknown"
    labels: list[str] = Field(default_factory=list)
    normal_range: NormalRangeSpec | None = None
    recommended_widgets: list[WidgetType] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_recommended_widgets(self) -> "MetricCatalogEntry":
        if not self.recommended_widgets:
            raise ValueError("recommended_widgets must contain at least one widget type")
        return self


class MetricCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: dict[str, MetricCatalogEntry] = Field(default_factory=dict)

    @classmethod
    def load_from_file(cls, path: str | Path) -> "MetricCatalog":
        catalog_path = Path(path)
        raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
        entries = {
            metric_name: MetricCatalogEntry.model_validate({"name": metric_name, **config})
            for metric_name, config in raw.items()
        }
        return cls(entries=entries)

    def get(self, metric_name: str) -> MetricCatalogEntry | None:
        return self.entries.get(metric_name)

    def require(self, metric_name: str) -> MetricCatalogEntry:
        entry = self.get(metric_name)
        if entry is None:
            raise KeyError(f"Metric is not in catalog: {metric_name}")
        return entry

    def metric_names(self) -> set[str]:
        return set(self.entries.keys())

    def label_names(self) -> set[str]:
        labels: set[str] = set()
        for entry in self.entries.values():
            labels.update(entry.labels)
        return labels


def default_metric_catalog_path() -> Path:
    env_path = os.getenv("METRIC_CATALOG_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.is_absolute():
            return candidate
        project_candidate = Path(__file__).resolve().parents[3] / candidate
        if project_candidate.exists():
            return project_candidate
        return candidate
    return Path(__file__).resolve().parents[3] / "configs" / "metric_catalog.yaml"


def load_default_metric_catalog() -> MetricCatalog:
    return MetricCatalog.load_from_file(default_metric_catalog_path())
