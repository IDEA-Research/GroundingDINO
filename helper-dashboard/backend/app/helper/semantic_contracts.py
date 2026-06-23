"""Semantic contracts for Helper outputs.

Used by the AgentValidationRetryLoop to decide whether a Helper's
DashboardSpec / PatchSpec output actually satisfies the user's
request — not just whether it parses.

Contracts are pure, deterministic, and side-effect-free. They never
call an LLM. They produce structured error lists that the retry loop
turns into feedback messages for the next attempt.

Naming convention: one class per contract; one `contract_for_*`
builder per operation kind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Widget-id / title matching helpers (mirrors runtime._first_id_in_text
# but deterministic and side-effect-free)
# ---------------------------------------------------------------------------


_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on",
    "usage", "rate", "chart", "widget", "panel", "list",
})


def _widget_matches_keyword(widget: dict, keyword: str) -> bool:
    kw = keyword.lower().strip()
    if not kw:
        return False
    wid = (widget.get("id") or "").lower()
    title = (widget.get("title") or "").lower()
    if kw in wid or kw in title:
        return True
    # Token match on title
    tokens = [
        t for t in re.split(r"[\s/\-]+", title)
        if t and t not in _STOPWORDS and len(t) > 2
    ]
    return any(kw == t for t in tokens)


def _find_widget_id(
    widgets: list[dict], keyword: str
) -> str | None:
    for w in widgets:
        if _widget_matches_keyword(w, keyword):
            return w.get("id")
    return None


# ---------------------------------------------------------------------------
# Contract protocol + result type
# ---------------------------------------------------------------------------


@dataclass
class ContractCheck:
    """Return type from SemanticContract.check."""
    ok: bool
    errors: list[str] = field(default_factory=list)
    op_summary: list[str] = field(default_factory=list)


class SemanticContract(Protocol):
    name: str

    def check(self, output: dict, args: dict) -> ContractCheck: ...

    def feedback_message(
        self, errors: list[str], args: dict
    ) -> str: ...


# ---------------------------------------------------------------------------
# Shared patch utilities
# ---------------------------------------------------------------------------


# Derived dynamically from the Pydantic WidgetType enum so any
# widget type added by the rescue_extend operation (which mutates
# widget_spec.py at runtime) is automatically reflected in semantic
# contract checks. Returns the live set at call time so it survives
# `importlib.reload(widget_spec)`.
def _allowed_widget_types() -> frozenset[str]:
    from ..specs.widget_spec import WidgetType
    return frozenset(wt.value for wt in WidgetType)


# Backwards-compatible name for direct attribute access. Callers
# should prefer `_allowed_widget_types()` for the live view.
_ALLOWED_WIDGET_TYPES = _allowed_widget_types()

_ALLOWED_OPS = frozenset({
    "add_widget", "remove_widget", "update_widget",
    "update_dashboard", "reorder_widgets",
})

_CRITICAL_KEYWORDS = ("error", "latency", "p95", "p99",
                       "availability", "up")


def _output_is_patch(output: dict) -> bool:
    return output.get("type") == "PatchSpec"


def _output_is_dashboard(output: dict) -> bool:
    return output.get("type") == "DashboardSpec"


def _patch_ops(output: dict) -> list[dict]:
    return (output.get("spec") or {}).get("operations") or []


def _dashboard_widgets(output: dict) -> list[dict]:
    return (output.get("spec") or {}).get("widgets") or []


def _op_summary(ops: list[dict]) -> list[str]:
    out: list[str] = []
    for op in ops:
        kind = op.get("op") or "?"
        if kind == "update_widget":
            wid = op.get("widget_id") or "?"
            fields = list((op.get("fields") or {}).keys())
            out.append(f"update_widget:{wid}:{'/'.join(fields)}")
        elif kind == "add_widget":
            w = op.get("widget") or {}
            out.append(f"add_widget:{w.get('id', '?')}:{w.get('type', '?')}")
        elif kind == "remove_widget":
            out.append(f"remove_widget:{op.get('widget_id', '?')}")
        elif kind == "reorder_widgets":
            out.append(
                f"reorder_widgets:[{','.join((op.get('order') or [])[:4])}...]"
            )
        elif kind == "update_dashboard":
            out.append(
                f"update_dashboard:{'/'.join((op.get('fields') or {}).keys())}"
            )
        else:
            out.append(str(kind))
    return out


def _widget_type_allowed(w: dict) -> bool:
    return w.get("type") in _allowed_widget_types()


def _basic_patch_errors(output: dict) -> list[str]:
    """Envelope + operation allow-list check. Shared by all patch
    contracts."""
    errors: list[str] = []
    if not _output_is_patch(output):
        errors.append(
            f"Expected output type 'PatchSpec', got {output.get('type')!r}."
        )
        return errors
    ops = _patch_ops(output)
    if not ops:
        errors.append("PatchSpec.operations is empty; at least one operation required.")
    for i, op in enumerate(ops):
        kind = op.get("op")
        if kind not in _ALLOWED_OPS:
            errors.append(
                f"operations[{i}].op={kind!r} is not in the allowed set "
                f"{sorted(_ALLOWED_OPS)}."
            )
        if kind == "add_widget":
            w = op.get("widget") or {}
            if not _widget_type_allowed(w):
                errors.append(
                    f"operations[{i}].widget.type={w.get('type')!r} not in "
                    f"allowed widget types {sorted(_allowed_widget_types())}."
                )
    return errors


# ---------------------------------------------------------------------------
# Patch contracts
# ---------------------------------------------------------------------------


@dataclass
class AnyValidPatchContract:
    name: str = "any_valid_patch"

    def check(self, output: dict, args: dict) -> ContractCheck:
        errors = _basic_patch_errors(output)
        ops = _patch_ops(output) if _output_is_patch(output) else []
        return ContractCheck(
            ok=not errors,
            errors=errors,
            op_summary=_op_summary(ops),
        )

    def feedback_message(self, errors: list[str], args: dict) -> str:
        bullets = "\n".join(f"  - {e}" for e in errors[:6])
        change = _requested_change(args)
        return (
            f"Your previous PatchSpec was rejected for the user request "
            f"{change!r}. Issues:\n{bullets}\n"
            f"Return a corrected PatchSpec JSON only. Use only operations "
            f"in {sorted(_ALLOWED_OPS)} and only widget types in "
            f"{sorted(_allowed_widget_types())}."
        )


@dataclass
class MakeProminentContract:
    target_keyword: str
    name: str = "make_prominent"

    def check(self, output: dict, args: dict) -> ContractCheck:
        errors = _basic_patch_errors(output)
        ops = _patch_ops(output)
        if errors:
            return ContractCheck(False, errors, _op_summary(ops))

        hits = [
            op for op in ops
            if op.get("op") == "update_widget"
            and self.target_keyword.lower() in (op.get("widget_id") or "").lower()
        ]
        if not hits:
            errors.append(
                f"Expected at least one update_widget targeting a widget "
                f"whose id contains {self.target_keyword!r}."
            )
            return ContractCheck(False, errors, _op_summary(ops))

        def is_prominent(op: dict) -> bool:
            pos = (op.get("fields") or {}).get("position") or {}
            w = pos.get("w")
            y = pos.get("y")
            if isinstance(w, int) and w >= 8:
                return True
            if isinstance(y, int) and y == 0:
                return True
            return False

        if not any(is_prominent(op) for op in hits):
            errors.append(
                f"update_widget on {self.target_keyword!r} found, but it "
                f"does not make the widget more prominent. Set "
                f"position.w >= 8 or position.y == 0."
            )

        return ContractCheck(
            ok=not errors, errors=errors, op_summary=_op_summary(ops),
        )

    def feedback_message(self, errors: list[str], args: dict) -> str:
        change = _requested_change(args)
        bullets = "\n".join(f"  - {e}" for e in errors[:6])
        return (
            f"Your previous PatchSpec was rejected because the user asked "
            f"to make {self.target_keyword} more prominent (request: "
            f"{change!r}).\nIssues:\n{bullets}\n"
            f"Return a corrected PatchSpec JSON only. It must include at "
            f"least one update_widget operation targeting a widget whose "
            f"id or title contains {self.target_keyword!r}. The update "
            f"must make the widget more prominent by setting "
            f"position.w >= 8 OR position.y == 0."
        )


@dataclass
class MoveCriticalToTopContract:
    name: str = "move_critical_to_top"

    def check(self, output: dict, args: dict) -> ContractCheck:
        errors = _basic_patch_errors(output)
        ops = _patch_ops(output)
        if errors:
            return ContractCheck(False, errors, _op_summary(ops))

        reorder = next(
            (op for op in ops if op.get("op") == "reorder_widgets"), None,
        )
        if reorder is None:
            errors.append(
                "Expected at least one reorder_widgets operation."
            )
            return ContractCheck(False, errors, _op_summary(ops))

        dashboard = args.get("dashboard") or {}
        widgets_by_id = {
            w.get("id"): w for w in dashboard.get("widgets") or []
        }
        order = reorder.get("order") or []
        if len(order) < 2:
            errors.append("reorder_widgets.order must have at least 2 ids.")
            return ContractCheck(False, errors, _op_summary(ops))

        def is_critical(wid: str) -> bool:
            w = widgets_by_id.get(wid) or {}
            blob = (w.get("id") or "").lower() + " " + (w.get("title") or "").lower()
            return any(kw in blob for kw in _CRITICAL_KEYWORDS) \
                or (w.get("type") in {"gauge", "alert_list"})

        first_two = order[:2]
        if not any(is_critical(wid) for wid in first_two):
            errors.append(
                "reorder_widgets.order does not place any critical "
                f"widget in the first two positions (got {first_two!r}). "
                f"Critical keywords: {list(_CRITICAL_KEYWORDS)}."
            )

        return ContractCheck(
            ok=not errors, errors=errors, op_summary=_op_summary(ops),
        )

    def feedback_message(self, errors: list[str], args: dict) -> str:
        change = _requested_change(args)
        bullets = "\n".join(f"  - {e}" for e in errors[:6])
        return (
            f"Your previous PatchSpec was rejected because the user asked "
            f"to move critical widgets to the top (request: {change!r}).\n"
            f"Issues:\n{bullets}\n"
            f"Return a corrected PatchSpec JSON only. Use exactly one "
            f"reorder_widgets operation. Critical widgets are those "
            f"whose id or title contains any of "
            f"{list(_CRITICAL_KEYWORDS)}, or whose type is 'gauge' or "
            f"'alert_list'. Place at least one such widget in the first "
            f"two positions of `order`."
        )


@dataclass
class AddThresholdContract:
    target_keyword: str
    expected_value: float
    raw_value_label: str
    name: str = "add_threshold"

    def check(self, output: dict, args: dict) -> ContractCheck:
        errors = _basic_patch_errors(output)
        ops = _patch_ops(output)
        if errors:
            return ContractCheck(False, errors, _op_summary(ops))

        upd = next(
            (op for op in ops if op.get("op") == "update_widget"
             and self.target_keyword.lower() in
                 (op.get("widget_id") or "").lower()),
            None,
        )
        if upd is None:
            errors.append(
                f"Expected at least one update_widget targeting a widget "
                f"whose id contains {self.target_keyword!r}."
            )
            return ContractCheck(False, errors, _op_summary(ops))

        thresholds = (upd.get("fields") or {}).get("thresholds") or []
        has_value = any(
            isinstance(t, dict)
            and abs(float(t.get("value", 0)) - self.expected_value) < 1e-6
            for t in thresholds
        )
        if not has_value:
            errors.append(
                f"update_widget on {self.target_keyword!r} exists, but "
                f"fields.thresholds does not contain value "
                f"{self.expected_value} (for request "
                f"{self.raw_value_label!r}). Current thresholds: "
                f"{[t.get('value') if isinstance(t, dict) else t for t in thresholds]}"
            )

        return ContractCheck(
            ok=not errors, errors=errors, op_summary=_op_summary(ops),
        )

    def feedback_message(self, errors: list[str], args: dict) -> str:
        change = _requested_change(args)
        bullets = "\n".join(f"  - {e}" for e in errors[:6])
        return (
            f"Your previous PatchSpec was rejected because the user asked "
            f"to add a threshold {self.raw_value_label!r} to the "
            f"{self.target_keyword} widget (request: {change!r}).\n"
            f"Issues:\n{bullets}\n"
            f"Return a corrected PatchSpec JSON only. Use one "
            f"update_widget operation targeting the widget whose id "
            f"contains {self.target_keyword!r}, and add a threshold with "
            f"value {self.expected_value} to "
            f"fields.thresholds (preserve any existing ones)."
        )


@dataclass
class ChangeChartContract:
    from_keyword: str
    to_keyword: str
    expected_promql_substrings: list[str]
    name: str = "change_chart"

    def check(self, output: dict, args: dict) -> ContractCheck:
        errors = _basic_patch_errors(output)
        ops = _patch_ops(output)
        if errors:
            return ContractCheck(False, errors, _op_summary(ops))

        upd = next(
            (op for op in ops if op.get("op") == "update_widget"
             and self.from_keyword.lower() in (op.get("widget_id") or "").lower()),
            None,
        )
        if upd is None:
            errors.append(
                f"Expected at least one update_widget targeting a widget "
                f"whose id contains {self.from_keyword!r} (the 'from' "
                f"widget to swap)."
            )
            return ContractCheck(False, errors, _op_summary(ops))

        query = (upd.get("fields") or {}).get("query") or {}
        promql = (query.get("promql") or "").lower()
        hit = any(sub.lower() in promql for sub in self.expected_promql_substrings)
        if not hit:
            errors.append(
                f"update_widget on {self.from_keyword!r} found but "
                f"fields.query.promql={promql[:80]!r} does not contain "
                f"any of the expected substrings for "
                f"{self.to_keyword!r}: "
                f"{self.expected_promql_substrings}."
            )

        return ContractCheck(
            ok=not errors, errors=errors, op_summary=_op_summary(ops),
        )

    def feedback_message(self, errors: list[str], args: dict) -> str:
        change = _requested_change(args)
        bullets = "\n".join(f"  - {e}" for e in errors[:6])
        return (
            f"Your previous PatchSpec was rejected because the user asked "
            f"to change the {self.from_keyword} chart to a "
            f"{self.to_keyword} chart (request: {change!r}).\n"
            f"Issues:\n{bullets}\n"
            f"Return a corrected PatchSpec JSON only. Use one "
            f"update_widget operation targeting the {self.from_keyword} "
            f"widget, with fields.query.promql containing one of "
            f"{self.expected_promql_substrings}."
        )


@dataclass
class RemoveWidgetContract:
    target_keyword: str
    name: str = "remove_widget"

    def check(self, output: dict, args: dict) -> ContractCheck:
        errors = _basic_patch_errors(output)
        ops = _patch_ops(output)
        if errors:
            return ContractCheck(False, errors, _op_summary(ops))
        hit = next(
            (op for op in ops if op.get("op") == "remove_widget"
             and self.target_keyword.lower() in (op.get("widget_id") or "").lower()),
            None,
        )
        if hit is None:
            errors.append(
                f"Expected remove_widget with widget_id containing "
                f"{self.target_keyword!r}."
            )
        return ContractCheck(
            ok=not errors, errors=errors, op_summary=_op_summary(ops),
        )

    def feedback_message(self, errors: list[str], args: dict) -> str:
        change = _requested_change(args)
        return (
            f"Your previous PatchSpec was rejected for the remove request "
            f"{change!r}. Return one remove_widget with widget_id "
            f"containing {self.target_keyword!r}."
        )


# ---------------------------------------------------------------------------
# Generate contracts
# ---------------------------------------------------------------------------


@dataclass
class AnyValidGenerateContract:
    name: str = "any_valid_generate"

    def check(self, output: dict, args: dict) -> ContractCheck:
        errors: list[str] = []
        if not _output_is_dashboard(output):
            errors.append(
                f"Expected output type 'DashboardSpec', got "
                f"{output.get('type')!r}."
            )
            return ContractCheck(False, errors, [])
        widgets = _dashboard_widgets(output)
        if not widgets:
            errors.append("DashboardSpec.widgets is empty; expected at least 1.")
        for i, w in enumerate(widgets):
            if not _widget_type_allowed(w):
                errors.append(
                    f"widgets[{i}].type={w.get('type')!r} not in allowed "
                    f"widget types {sorted(_allowed_widget_types())}."
                )
        return ContractCheck(
            ok=not errors, errors=errors,
            op_summary=[f"widgets={len(widgets)}"],
        )

    def feedback_message(self, errors: list[str], args: dict) -> str:
        bullets = "\n".join(f"  - {e}" for e in errors[:6])
        return (
            f"Your previous DashboardSpec was rejected.\n"
            f"Issues:\n{bullets}\n"
            f"Return a corrected DashboardSpec JSON only. Every widget "
            f"type must be in {sorted(_allowed_widget_types())}."
        )


@dataclass
class ApiObservabilityGenerateContract:
    min_widgets: int = 6
    name: str = "api_observability_generate"

    def check(self, output: dict, args: dict) -> ContractCheck:
        base = AnyValidGenerateContract().check(output, args)
        if not base.ok:
            return base
        widgets = _dashboard_widgets(output)
        errors: list[str] = []
        if len(widgets) < self.min_widgets:
            errors.append(
                f"API observability dashboard should have >= {self.min_widgets} "
                f"widgets; got {len(widgets)}."
            )
        has_request_rate = any(
            "rate(http_requests_total" in (w.get("query") or {}).get("promql", "")
            for w in widgets
        )
        if not has_request_rate:
            errors.append(
                "Missing a widget with rate(http_requests_total[...]) PromQL."
            )
        has_percentile = any(
            "histogram_quantile" in (w.get("query") or {}).get("promql", "")
            and "by (le)" in (w.get("query") or {}).get("promql", "")
            for w in widgets
        )
        if not has_percentile:
            errors.append(
                "Missing a latency percentile widget with "
                "histogram_quantile(..., sum(rate(..._bucket[...])) by (le))."
            )
        has_threshold = any(w.get("thresholds") for w in widgets)
        if not has_threshold:
            errors.append(
                "No widget has a threshold. Critical widgets (error rate, "
                "latency, availability) should carry thresholds."
            )
        return ContractCheck(
            ok=not errors, errors=errors,
            op_summary=[f"widgets={len(widgets)}"],
        )

    def feedback_message(self, errors: list[str], args: dict) -> str:
        bullets = "\n".join(f"  - {e}" for e in errors[:6])
        return (
            f"Your previous API observability DashboardSpec was rejected.\n"
            f"Issues:\n{bullets}\n"
            f"Return a corrected DashboardSpec JSON only with at least "
            f"{self.min_widgets} widgets, a request-rate widget, a "
            f"latency-percentile widget, and at least one threshold."
        )


# ---------------------------------------------------------------------------
# Builders (keyword-driven)
# ---------------------------------------------------------------------------


_PROMINENT_KWS = ("more prominent", "bigger", "highlight", "promote")
_REORDER_KWS = (
    "move critical", "critical widgets", "critical to the top",
    "reorder", "move to the top",
)
_REMOVE_KWS = ("remove ", "delete ")


def _requested_change(args: dict) -> str:
    intent = args.get("intent") or {}
    changes = intent.get("requested_changes") or []
    return changes[0] if changes else ""


def _normalize_threshold_value(raw: float, unit: str) -> float:
    unit = (unit or "").lower()
    if unit in ("%", "percent"):
        return raw / 100.0
    if unit == "ms":
        return raw / 1000.0
    return float(raw)


def _extract_target_keyword(text: str) -> str | None:
    """Very light heuristic: if the user mentions 'latency', 'cpu',
    'error rate', etc., return that token as the target keyword for
    a widget match."""
    low = text.lower()
    for kw in ("error rate", "error-rate", "latency", "cpu", "memory",
                "ram", "p95", "p99", "availability", "alerts", "request"):
        if kw in low:
            # normalize: error-rate -> error, p95 -> latency context
            if kw in ("error rate", "error-rate"):
                return "error"
            if kw in ("p95", "p99"):
                return "latency"
            if kw == "ram":
                return "memory"
            return kw
    return None


_TO_METRIC_SUBSTRS = {
    "cpu": ["process_cpu_seconds_total", "node_cpu_seconds_total"],
    "memory": ["process_resident_memory_bytes", "node_memory_"],
    "ram": ["process_resident_memory_bytes", "node_memory_"],
    "requests": ["http_requests_total"],
    "latency": ["http_request_duration_seconds_bucket", "histogram_quantile"],
    "errors": ["status=~\"5..\"", "error"],
}


def contract_for_patch(
    requested_changes: list[str], dashboard: dict,
) -> SemanticContract:
    text = " ".join(requested_changes).lower().strip()
    if not text:
        return AnyValidPatchContract()

    # 1. reorder / move-critical-to-top
    if any(k in text for k in _REORDER_KWS):
        return MoveCriticalToTopContract()

    # 2. change X chart to Y chart
    m = re.search(
        r"change\s+(?:the\s+)?(\w+)\s+chart\s+to\s+(?:a\s+)?(\w+)",
        text,
    )
    if m:
        from_kw = m.group(1)
        to_kw = m.group(2)
        substrs = _TO_METRIC_SUBSTRS.get(to_kw, [to_kw])
        return ChangeChartContract(
            from_keyword=from_kw,
            to_keyword=to_kw,
            expected_promql_substrings=substrs,
        )

    # 3. add X threshold at N
    if "threshold" in text:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|s|ms)?", text)
        if m:
            raw = float(m.group(1))
            unit = m.group(2) or ""
            target = _extract_target_keyword(text) or "error"
            return AddThresholdContract(
                target_keyword=target,
                expected_value=_normalize_threshold_value(raw, unit),
                raw_value_label=f"{raw}{unit}".strip(),
            )

    # 4. make X more prominent
    if any(k in text for k in _PROMINENT_KWS):
        target = _extract_target_keyword(text)
        if target:
            return MakeProminentContract(target_keyword=target)

    # 5. remove X
    if any(k in text for k in _REMOVE_KWS):
        target = _extract_target_keyword(text)
        if target:
            return RemoveWidgetContract(target_keyword=target)

    # default
    return AnyValidPatchContract()


def contract_for_generate(requirements: dict) -> SemanticContract:
    title = (requirements.get("title") or "").lower()
    goal = (requirements.get("goal") or "").lower()
    hints = " ".join(requirements.get("metrics_hints") or []).lower()
    blob = f"{title} {goal} {hints}"

    api_obs_keywords = (
        "api", "http_requests", "http request", "request rate",
        "latency", "p95", "p99", "error rate", "availability",
        "observability",
    )
    hits = sum(1 for kw in api_obs_keywords if kw in blob)
    if hits >= 2:
        return ApiObservabilityGenerateContract()
    return AnyValidGenerateContract()
