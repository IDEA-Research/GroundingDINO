from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..specs.metric_catalog import MetricCatalog
from ..specs.widget_spec import QuerySpec


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


class PromQLValidator:
    GLOBAL_SCAN_PATTERNS = [
        re.compile(r'\{[^}]*__name__\s*=~\s*"\.\*"[^}]*\}'),
        re.compile(r"\{[^}]*__name__\s*=~\s*'\.\*'[^}]*\}"),
    ]
    IDENTIFIER_PATTERN = re.compile(r"(?<![a-zA-Z0-9_:])([a-zA-Z_:][a-zA-Z0-9_:]*)(?![a-zA-Z0-9_:])")
    QUOTED_STRING_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')

    RESERVED_WORDS = {
        "and",
        "or",
        "unless",
        "by",
        "without",
        "on",
        "ignoring",
        "group_left",
        "group_right",
        "bool",
        "offset",
        "sum",
        "avg",
        "min",
        "max",
        "count",
        "stddev",
        "stdvar",
        "topk",
        "bottomk",
        "quantile",
        "rate",
        "irate",
        "increase",
        "delta",
        "idelta",
        "avg_over_time",
        "min_over_time",
        "max_over_time",
        "sum_over_time",
        "count_over_time",
        "last_over_time",
        "histogram_quantile",
        "scalar",
        "vector",
        "time",
        "round",
        "clamp_min",
        "clamp_max",
    }

    def __init__(
        self,
        catalog: MetricCatalog,
        *,
        max_range_seconds: int = 86_400,
        min_step_seconds: int = 1,
        max_points: int = 5_000,
        max_query_chars: int = 512,
    ) -> None:
        self.catalog = catalog
        self.max_range_seconds = max_range_seconds
        self.min_step_seconds = min_step_seconds
        self.max_points = max_points
        self.max_query_chars = max_query_chars

    def validate_query_spec(self, query_spec: QuerySpec) -> ValidationResult:
        errors: list[str] = []
        if query_spec.metric not in self.catalog.metric_names():
            errors.append(f"Metric is not allowed by catalog: {query_spec.metric}")

        promql = query_spec.effective_promql()
        errors.extend(self.validate_promql(promql).errors)
        time_range_seconds = query_spec.time_range_seconds
        if query_spec.start_time is not None and query_spec.end_time is not None:
            time_range_seconds = int((query_spec.end_time - query_spec.start_time).total_seconds())
        errors.extend(self.validate_range(time_range_seconds, query_spec.step_seconds).errors)

        return ValidationResult(valid=not errors, errors=errors)

    def validate_promql(self, promql: str) -> ValidationResult:
        errors: list[str] = []
        if len(promql) > self.max_query_chars:
            errors.append(f"PromQL query is too long; max {self.max_query_chars} characters")

        for pattern in self.GLOBAL_SCAN_PATTERNS:
            if pattern.search(promql):
                errors.append('Global __name__ regex scans are not allowed')
                break

        identifiers = self._extract_identifiers(promql)
        allowed_names = self.catalog.metric_names()
        allowed_labels = self.catalog.label_names() | {"__name__"}
        unknown = sorted(
            token
            for token in identifiers
            if token not in allowed_names
            and token not in allowed_labels
            and token not in self.RESERVED_WORDS
        )
        if unknown:
            errors.append(f"PromQL contains identifiers outside the metric catalog: {', '.join(unknown)}")

        return ValidationResult(valid=not errors, errors=errors)

    def validate_range(self, time_range_seconds: int, step_seconds: int) -> ValidationResult:
        errors: list[str] = []
        if time_range_seconds > self.max_range_seconds:
            errors.append(f"Time range exceeds max {self.max_range_seconds} seconds")
        if step_seconds < self.min_step_seconds:
            errors.append(f"Query step must be at least {self.min_step_seconds} second(s)")
        if step_seconds > 0 and (time_range_seconds / step_seconds) > self.max_points:
            errors.append(f"Query would return too many points; max {self.max_points}")
        return ValidationResult(valid=not errors, errors=errors)

    def _extract_identifiers(self, promql: str) -> set[str]:
        without_strings = self.QUOTED_STRING_PATTERN.sub("", promql)
        without_durations = re.sub(r"\[[^\]]+\]", "", without_strings)
        return set(self.IDENTIFIER_PATTERN.findall(without_durations))
