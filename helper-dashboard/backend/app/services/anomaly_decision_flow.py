"""Evaluator -> decision_flow widget projection (INC5 end-to-end wiring).

The evaluator produces an :class:`AnomalyEvaluationReport`; a clinician-facing
``decision_flow`` widget (INC4, see ``widget_spec.DecisionFlowSpec``) renders a
guided flowchart. This module is the bridge:

  - :func:`build_rso2_decision_flow_widget` emits a STRICT, spec-validated
    ``decision_flow`` WidgetSpec for the neonatal rSO2 rule. It is the static
    graph: the mandatory data-integrity-gate first node, a signal-lost
    terminal, the rSO2 (+ SpO2 / HR / MAP / FiO2) inputs, all prometheus-only.
    Building it through :class:`WidgetSpec` means it inherits every existing
    security invariant (extra='forbid', forbidden-field scan, mock rejection).

  - :func:`project_report` maps a live report to the runtime state the widget
    overlays on that static graph: which node is "active", whether the flow is
    in SIGNAL_LOST (gate failed) vs a clinical branch, and the persistent
    non-diagnostic banner. It NEVER re-derives a clinical verdict — it reflects
    the core's verdict. A degraded signal is surfaced LOUDLY (the flow parks on
    the signal_lost terminal), never reassured-and-hidden.

The projection is a plain dataclass so the FastAPI surface and the e2e test can
serialise it without a browser.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..specs.alert_rule_spec import AlertRuleSpec
from ..specs.anomaly_evaluation_report import AlertState, AnomalyEvaluationReport
from ..specs.widget_spec import (
    DATA_INTEGRITY_GATE_KIND,
    WidgetSpec,
    WidgetType,
)

# Persistent, non-suppressible disclaimer for the alerting surface.
NON_DIAGNOSTIC_LABEL = (
    "Decision-support, not a diagnosis. Neonatal rSO2 monitor; verify the "
    "patient, not this screen."
)

# The active node the flow parks on for each evaluator state. Every state maps
# to a node in the static graph so the widget always has a defined position —
# a degraded state resolves to the loud signal_lost terminal, never to "ok".
_STATE_TO_NODE: dict[AlertState, str] = {
    AlertState.signal_lost: "lost",
    AlertState.insufficient_baseline: "gate",
    AlertState.pending: "d_rso2",
    AlertState.firing: "act",
    AlertState.resolved: "ok",
}


def _q(promql: str) -> dict[str, Any]:
    return {"source": "prometheus", "promql": promql, "query_type": "instant"}


def build_rso2_decision_flow_widget(
    *,
    widget_id: str = "neo-rso2-decision-flow",
    rso2_metric: str = "rso2_left",
) -> WidgetSpec:
    """Return a validated ``decision_flow`` widget for the neonatal rSO2 rule.

    Validation runs through :class:`WidgetSpec`, so this inherits the INC4
    invariants: the first node MUST be the data-integrity gate, inputs are
    prometheus-only (mock rejected), and no forbidden free-text field can be
    smuggled in. If any invariant is violated this RAISES at build time — the
    widget is never shipped half-formed.
    """
    flow = {
        "nodes": [
            {
                "id": "gate",
                "kind": DATA_INTEGRITY_GATE_KIND,
                "label": "Data-integrity gate: reachable, returns data, "
                "source=prometheus, fresh",
            },
            {
                "id": "d_rso2",
                "kind": "decision",
                "label": "rSO2 < 0.80 x 24h baseline, sustained 5m?",
                "signal": "rSO2",
            },
            {
                "id": "d_spo2",
                "kind": "decision",
                "label": "SpO2 also low (co-desaturation)?",
                "signal": "SpO2",
            },
            {
                "id": "act",
                "kind": "action",
                "label": "Escalate: check probe placement, reposition, "
                "assess perfusion, call clinician",
            },
            {"id": "ok", "kind": "terminal", "label": "Within baseline - continue monitoring"},
            {
                "id": "lost",
                "kind": "signal_lost",
                "label": "SIGNAL LOST - data-integrity gate failed; do not "
                "interpret a clinical branch",
            },
        ],
        "edges": [
            {"from": "gate", "to": "d_rso2", "condition": "gate passed"},
            {"from": "gate", "to": "lost", "condition": "gate failed"},
            {"from": "d_rso2", "to": "d_spo2", "condition": "rSO2 breach sustained"},
            {"from": "d_rso2", "to": "ok", "condition": "rSO2 within baseline"},
            {"from": "d_spo2", "to": "act", "condition": "co-desaturation"},
            {"from": "d_spo2", "to": "act", "condition": "isolated cerebral desat"},
        ],
        "steps": [
            {
                "id": "s1",
                "title": "Verify signal integrity",
                "node": "gate",
                "guidance": "Confirm the probe is reachable, data is fresh, and "
                "the source is prometheus before reading any branch.",
            },
            {
                "id": "s2",
                "title": "Assess cerebral rSO2",
                "node": "d_rso2",
                "guidance": "Compare live rSO2 to 0.80x its 24h baseline; a "
                "sustained 5m breach is a decision-support flag, not a diagnosis.",
            },
            {
                "id": "s3",
                "title": "Escalation guidance",
                "node": "act",
                "guidance": "Decision-support only: verify the patient and probe; "
                "escalate per unit protocol.",
            },
        ],
        "inputs": [
            {"label": "rSO2", "query": _q(rso2_metric)},
            {"label": "SpO2", "query": _q("spo2")},
            {"label": "HR", "query": _q("hr")},
            {"label": "MAP", "query": _q("map")},
            {"label": "FiO2", "query": _q("fio2")},
        ],
    }
    widget = {
        "id": widget_id,
        "type": "decision_flow",
        "title": "Neonatal rSO2 desaturation decision support",
        "description": "Guided, decision-support response to a sustained "
        "cerebral desaturation. Not a diagnosis; not a clinical alarm.",
        "query": _q(rso2_metric),
        "position": {"x": 0, "y": 0, "w": 12, "h": 12},
        "encoding": {},
        "thresholds": [],
        "options": {},
        "decision_flow": flow,
    }
    return WidgetSpec.model_validate(widget)


@dataclass
class DecisionFlowRuntime:
    """The live overlay the widget renders on top of the static graph.

    This is what the evaluator feeds the widget each tick. It carries the
    active node, the loud degradation banner (when signal is lost / baseline
    insufficient), and the paging posture — but it is NEVER a diagnosis.
    """

    rule_id: str
    widget_id: str
    ts: str
    state: str
    active_node: str
    signal_lost: bool
    degraded: bool
    would_page: bool
    paged: bool
    mode: str
    value: float | None
    baseline: float | None
    threshold: float | None
    banner: str
    non_diagnostic: bool = True
    signal_lost_reason: str | None = None
    messages: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def project_report(
    report: AnomalyEvaluationReport,
    *,
    widget_id: str = "neo-rso2-decision-flow",
) -> DecisionFlowRuntime:
    """Map an evaluator report to the decision_flow widget's runtime overlay.

    Reflects — never re-derives — the clinical verdict. A degraded signal
    (signal_lost / insufficient_baseline) parks the flow on a loud node and
    sets a conspicuous banner; it is surfaced, not hidden.
    """
    state = report.state
    active_node = _STATE_TO_NODE.get(state, "gate")
    signal_lost = state == AlertState.signal_lost
    degraded = state in (AlertState.signal_lost, AlertState.insufficient_baseline)

    # Paging posture. Prefer the current tick's events (the firing/signal_lost
    # EDGE carries would_page/paged); but a STANDING firing/signal_lost with no
    # new event this tick is still a would-page state, so derive it from the
    # state when there is no fresh event — the widget must not "forget" it is
    # in a firing posture between edges.
    would_page = any(ev.would_page for ev in report.events)
    paged = any(ev.paged for ev in report.events)
    if not report.events and state in (AlertState.firing, AlertState.signal_lost):
        would_page = True
        paged = report.mode == "active"
    slr: str | None = None
    for ev in report.events:
        if ev.signal_lost_reason is not None:
            slr = ev.signal_lost_reason.value
            break

    if signal_lost:
        banner = (
            f"SIGNAL LOST ({slr or 'unknown'}) - the data-integrity gate "
            f"failed. No clinical verdict. {NON_DIAGNOSTIC_LABEL}"
        )
    elif state == AlertState.insufficient_baseline:
        banner = (
            f"INSUFFICIENT BASELINE - <24h history; the rule is not firing. "
            f"{NON_DIAGNOSTIC_LABEL}"
        )
    elif state == AlertState.firing:
        banner = (
            f"FIRING ({report.mode.upper()}) - sustained rSO2 desaturation flag. "
            f"{NON_DIAGNOSTIC_LABEL}"
        )
    else:
        banner = NON_DIAGNOSTIC_LABEL

    return DecisionFlowRuntime(
        rule_id=report.rule_id,
        widget_id=widget_id,
        ts=report.ts,
        state=state.value,
        active_node=active_node,
        signal_lost=signal_lost,
        degraded=degraded,
        would_page=would_page,
        paged=paged,
        mode=report.mode,
        value=report.value,
        baseline=report.baseline,
        threshold=report.threshold,
        banner=banner,
        signal_lost_reason=slr,
        messages=[ev.message for ev in report.events if ev.message],
    )


def widget_for_rule(rule: AlertRuleSpec, *, widget_id: str | None = None) -> WidgetSpec:
    """Convenience: build the decision_flow widget for a given rule's metric."""
    return build_rso2_decision_flow_widget(
        widget_id=widget_id or f"{rule.id}-flow",
        rso2_metric=rule.metric,
    )
