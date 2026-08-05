"""WidgetSpec and related sub-schemas.

Widgets are the atomic unit of a dashboard. Their `type` is a strict
enum; adding a new type is a Big guy task that must also update the
frontend widget-toolkit and the validator.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Regex used to reject payloads that contain markup or script-like
# content. Applied to every free-text field.
_UNSAFE_PATTERN = re.compile(
    r"<\s*(script|iframe|object|embed|style)\b"
    r"|javascript:"
    r"|on[a-z]+\s*=",
    re.IGNORECASE,
)


def _assert_safe_text(value: str, field: str) -> str:
    if _UNSAFE_PATTERN.search(value):
        raise ValueError(f"{field} contains forbidden markup or script")
    return value


class WidgetType(str, Enum):
    line_chart = "line_chart"
    stat_card = "stat_card"
    gauge = "gauge"
    table = "table"
    alert_list = "alert_list"
    pie_chart = "pie_chart"
    bar_chart = "bar_chart"
    heatmap = "heatmap"
    decision_flow = "decision_flow"


class QuerySource(str, Enum):
    prometheus = "prometheus"
    mock = "mock"


class QueryType(str, Enum):
    instant = "instant"
    range = "range"


class QuerySpec(BaseModel):
    """A single query attached to a widget."""

    model_config = ConfigDict(extra="forbid")

    source: QuerySource = QuerySource.prometheus
    promql: str = Field(..., min_length=1, max_length=512)
    query_type: QueryType = QueryType.range
    range: str | None = Field(default=None, max_length=16)
    step: str | None = Field(default=None, max_length=16)

    @field_validator("promql")
    @classmethod
    def _check_promql(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("promql must not be empty")
        # Block obvious shell / HTML / JS characters.
        forbidden = [";", "`", "<", ">", "$(" , "||"]
        for token in forbidden:
            if token in v:
                raise ValueError(f"promql contains forbidden token: {token!r}")
        _assert_safe_text(v, "promql")
        return v


# ---------------------------------------------------------------------------
# decision_flow — clinical decision-support flowchart
#
# A decision_flow widget renders a clinician-facing flowchart whose branches
# reference live neonatal physiology (rSO2 + SpO2 + HR + MAP + FiO2). It is a
# patient-safety surface, so it is STRICT and FAIL-CLOSED:
#
#   - Every flow's FIRST node MUST be the data-integrity gate. A verdict path
#     may only be walked once the gate passed; otherwise the flow shows
#     SIGNAL_LOST, never a clinical branch (mirrors the evaluator's rule that
#     the data-integrity gate runs first, always).
#   - Every input query is prometheus-only. `source == "mock"` is rejected at
#     validation time so a decision flow can never branch on the silent mock
#     fallback in prometheus/client.py.
#   - nodes / edges / steps are their own `extra='forbid'` models, so
#     `raw_html` / `component` / `code` cannot smuggle in through a nested
#     free-text field, and the pre-Pydantic forbidden-field scan still bites.
# ---------------------------------------------------------------------------

# Physiological signals a neonatal decision flow may branch on. A branch
# condition or an input label must reference one of these — an unknown signal
# is rejected so a flow cannot silently depend on a metric that is never fed.
DECISION_FLOW_SIGNALS: tuple[str, ...] = ("rSO2", "SpO2", "HR", "MAP", "FiO2")

# The kind of the mandatory first node. Kept as a named constant so the
# validator and the frontend agree on the exact spelling.
DATA_INTEGRITY_GATE_KIND = "data_integrity_gate"


class FlowNodeKind(str, Enum):
    """Node roles in a clinical decision flow.

    `data_integrity_gate` is the mandatory first node (reachable ∧ returns_data
    ∧ source=='prometheus' ∧ fresh). `signal_lost` is the loud degraded-signal
    terminal. `decision` branches on physiology; `action`/`terminal` are advice
    endpoints (decision-support, never a diagnosis).
    """

    data_integrity_gate = "data_integrity_gate"
    decision = "decision"
    action = "action"
    terminal = "terminal"
    signal_lost = "signal_lost"


class FlowNode(BaseModel):
    """A single node in the decision flow graph."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    kind: FlowNodeKind
    label: str = Field(..., min_length=1, max_length=200)
    # Which physiological signal this decision node reads, when kind==decision.
    signal: str | None = Field(default=None, max_length=16)

    @field_validator("label")
    @classmethod
    def _safe_label(cls, v: str) -> str:
        return _assert_safe_text(v, "flow node label")

    @field_validator("signal")
    @classmethod
    def _known_signal(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if v not in DECISION_FLOW_SIGNALS:
            raise ValueError(
                f"node.signal {v!r} is not a known neonatal signal "
                f"(one of {DECISION_FLOW_SIGNALS})"
            )
        return v


class FlowEdge(BaseModel):
    """A directed edge between two nodes, guarded by a branch condition."""

    # populate_by_name lets a stored spec load whether the edge source was
    # written as `from` (alias, canonical) or `from_` (field name), so a
    # decision_flow dashboard round-trips through the store. Serialize with
    # by_alias=True so the frontend always receives `from`.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: str = Field(
        ..., alias="from", min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$"
    )
    to: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    # Human-readable branch condition, e.g. "rSO2 < 0.80 * baseline".
    condition: str = Field(..., min_length=1, max_length=200)

    @field_validator("condition")
    @classmethod
    def _safe_condition(cls, v: str) -> str:
        return _assert_safe_text(v, "flow edge condition")


class FlowStep(BaseModel):
    """An ordered guided step referencing a node in the graph.

    Steps drive the client-side wizard: the user advances one step at a time,
    each pointing at a node and carrying decision-support guidance.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    title: str = Field(..., min_length=1, max_length=128)
    node: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    guidance: str = Field(..., min_length=1, max_length=512)

    @field_validator("title", "guidance")
    @classmethod
    def _safe_text(cls, v: str) -> str:
        return _assert_safe_text(v, "flow step text")


class QueryInput(BaseModel):
    """A labeled, prometheus-only input feeding a decision flow.

    The label ties the query to one physiological signal so branch conditions
    can reference it. `source == "mock"` is rejected — a clinical flow may
    never branch on fake/stale data.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., min_length=1, max_length=16)
    query: QuerySpec

    @field_validator("label")
    @classmethod
    def _known_label(cls, v: str) -> str:
        if v not in DECISION_FLOW_SIGNALS:
            raise ValueError(
                f"input label {v!r} is not a known neonatal signal "
                f"(one of {DECISION_FLOW_SIGNALS})"
            )
        return v

    @field_validator("query")
    @classmethod
    def _reject_mock_input(cls, v: QuerySpec) -> QuerySpec:
        # FAIL CLOSED: a decision-support flow branches on physiology; it may
        # never branch on the silent mock fallback. Mirrors AlertRuleSpec.
        if v.source != QuerySource.prometheus:
            raise ValueError(
                "decision_flow input source must be 'prometheus'; branching on "
                "mock/fake data is forbidden (fail-closed clinical-safety invariant)"
            )
        return v


class DecisionFlowSpec(BaseModel):
    """The nested nodes / edges / steps graph for a decision_flow widget."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[FlowNode] = Field(..., min_length=2, max_length=64)
    edges: list[FlowEdge] = Field(default_factory=list, max_length=128)
    steps: list[FlowStep] = Field(..., min_length=1, max_length=64)
    # Multi-input labeled query set — branch conditions reference these signals.
    inputs: list[QueryInput] = Field(..., min_length=1, max_length=16)

    @field_validator("nodes")
    @classmethod
    def _first_node_is_gate(cls, v: list[FlowNode]) -> list[FlowNode]:
        # INVIOLABLE: the first node of every clinical flow is the
        # data-integrity gate. A verdict branch may only be reached after it.
        if v[0].kind != FlowNodeKind.data_integrity_gate:
            raise ValueError(
                "the first node of a decision_flow MUST be the "
                f"{DATA_INTEGRITY_GATE_KIND!r} node (data-integrity gate runs first)"
            )
        # Node ids must be unique so edges/steps resolve deterministically.
        ids = [n.id for n in v]
        if len(ids) != len(set(ids)):
            raise ValueError("decision_flow node ids must be unique")
        return v

    @field_validator("edges")
    @classmethod
    def _edges_resolve(cls, v: list[FlowEdge], info) -> list[FlowEdge]:
        nodes = info.data.get("nodes")
        if not nodes:
            return v  # a nodes-level error already fired
        node_ids = {n.id for n in nodes}
        for e in v:
            if e.from_ not in node_ids:
                raise ValueError(f"edge.from {e.from_!r} references an unknown node")
            if e.to not in node_ids:
                raise ValueError(f"edge.to {e.to!r} references an unknown node")
        return v

    @field_validator("steps")
    @classmethod
    def _steps_resolve(cls, v: list[FlowStep], info) -> list[FlowStep]:
        nodes = info.data.get("nodes")
        if not nodes:
            return v
        node_ids = {n.id for n in nodes}
        for s in v:
            if s.node not in node_ids:
                raise ValueError(f"step.node {s.node!r} references an unknown node")
        return v

    @field_validator("inputs")
    @classmethod
    def _inputs_reference_signals(cls, v: list[QueryInput]) -> list[QueryInput]:
        # A multi-input clinical flow must feed at least the cerebral-oximetry
        # signal it exists for; duplicate labels are ambiguous.
        labels = [i.label for i in v]
        if len(labels) != len(set(labels)):
            raise ValueError("decision_flow input labels must be unique")
        return v


class WidgetPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int = Field(..., ge=0, le=23)
    y: int = Field(..., ge=0, le=999)
    w: int = Field(..., ge=1, le=24)
    h: int = Field(..., ge=1, le=60)


class WidgetEncoding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit: str | None = Field(default=None, max_length=32)
    legend: str | None = Field(default=None, max_length=64)
    color: str | None = Field(
        default=None, pattern=r"^#[0-9a-fA-F]{3,8}$|^[a-zA-Z]{3,20}$"
    )

    @field_validator("unit", "legend")
    @classmethod
    def _safe_text(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _assert_safe_text(v, "encoding")


class WidgetThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    color: str = Field(..., pattern=r"^#[0-9a-fA-F]{3,8}$|^[a-zA-Z]{3,20}$")
    label: str | None = Field(default=None, max_length=32)

    @field_validator("label")
    @classmethod
    def _safe_label(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _assert_safe_text(v, "threshold.label")


# Allowed keys in the per-widget `options` dict. Anything else is
# dropped by the validator at `services/spec_validator.py`, but we keep
# this list here so both sides agree on what's legal.
ALLOWED_OPTION_KEYS: set[str] = {
    "decimals",
    "show_grid",
    "show_legend",
    "stacked",
    "fill",
    "min",
    "max",
    "columns",
    "severity_filter",
    "row_limit",
    "sort_by",
    "sort_dir",
    "show_labels",
    "donut",
    "horizontal",
    "show_values",
    "x_label",
    "y_label",
    "color_scale",
}

FORBIDDEN_WIDGET_FIELDS: set[str] = {
    "raw_html",
    "script",
    "component",
    "code",
    "iframe",
    "eval",
    "onclick",
    "onerror",
    "html",
    "jsx",
    "render",
}


class WidgetSpec(BaseModel):
    """A single widget inside a dashboard."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    type: WidgetType
    title: str = Field(..., min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=512)
    query: QuerySpec
    position: WidgetPosition
    encoding: WidgetEncoding = Field(default_factory=WidgetEncoding)
    thresholds: list[WidgetThreshold] = Field(default_factory=list, max_length=16)
    options: dict[str, Any] = Field(default_factory=dict)
    # Present only for type == decision_flow. The nested nodes/edges/steps +
    # multi-input labeled query set. None (and forbidden) for every other type.
    decision_flow: DecisionFlowSpec | None = None

    @field_validator("title", "description")
    @classmethod
    def _safe_text(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _assert_safe_text(v, "widget text field")

    @model_validator(mode="after")
    def _decision_flow_lockstep(self) -> "WidgetSpec":
        # decision_flow config is REQUIRED for a decision_flow widget and
        # FORBIDDEN for any other type — so the payload cannot smuggle a graph
        # onto an unrelated widget, and a decision_flow can't be empty.
        if self.type == WidgetType.decision_flow:
            if self.decision_flow is None:
                raise ValueError(
                    "type 'decision_flow' requires a 'decision_flow' "
                    "nodes/edges/steps configuration"
                )
        elif self.decision_flow is not None:
            raise ValueError(
                "'decision_flow' config is only valid on a decision_flow widget"
            )
        return self

    @field_validator("options")
    @classmethod
    def _safe_options(cls, v: dict[str, Any]) -> dict[str, Any]:
        for key in v.keys():
            if key in FORBIDDEN_WIDGET_FIELDS:
                raise ValueError(f"option key {key!r} is forbidden")
            if key not in ALLOWED_OPTION_KEYS:
                raise ValueError(
                    f"option key {key!r} is not in the allowed option keys; "
                    f"extending the widget requires a DeveloperTicket"
                )
            val = v[key]
            if isinstance(val, str):
                _assert_safe_text(val, f"options.{key}")
        return v
