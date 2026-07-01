"""Tier0 tests for the `decision_flow` clinical decision-support widget (INC4).

A decision_flow widget carries a nested nodes/edges/steps graph plus a
multi-input labeled query set over the neonatal signals (rSO2 + SpO2 + HR +
MAP + FiO2). It is patient-safety-critical, so validation is FAIL-CLOSED:

    - the FIRST node MUST be the data-integrity gate,
    - `raw_html` / `component` / `code` are rejected anywhere in the tree,
    - a `source == "mock"` input is rejected (never branch on fake data),
    - the decision_flow config is required for a decision_flow widget and
      forbidden on any other type.
"""

from __future__ import annotations

import pytest

from app.services.spec_validator import SpecValidationError, SpecValidator
from app.specs.widget_spec import (
    DATA_INTEGRITY_GATE_KIND,
    DECISION_FLOW_SIGNALS,
    DecisionFlowSpec,
    WidgetSpec,
    WidgetType,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _q(promql: str, source: str = "prometheus") -> dict:
    return {"source": source, "promql": promql, "query_type": "instant"}


def _flow(**overrides) -> dict:
    flow = {
        "nodes": [
            {
                "id": "gate",
                "kind": "data_integrity_gate",
                "label": "Data-integrity gate: reachable, fresh, prometheus",
            },
            {
                "id": "d_rso2",
                "kind": "decision",
                "label": "rSO2 < 0.80 x 24h baseline?",
                "signal": "rSO2",
            },
            {
                "id": "d_spo2",
                "kind": "decision",
                "label": "SpO2 low?",
                "signal": "SpO2",
            },
            {
                "id": "act",
                "kind": "action",
                "label": "Escalate: check probe, reposition, call clinician",
            },
            {"id": "ok", "kind": "terminal", "label": "Continue monitoring"},
            {
                "id": "lost",
                "kind": "signal_lost",
                "label": "SIGNAL LOST - do not interpret",
            },
        ],
        "edges": [
            {"from": "gate", "to": "d_rso2", "condition": "gate passed"},
            {"from": "gate", "to": "lost", "condition": "gate failed"},
            {"from": "d_rso2", "to": "d_spo2", "condition": "rSO2 breach"},
            {"from": "d_rso2", "to": "ok", "condition": "rSO2 normal"},
            {"from": "d_spo2", "to": "act", "condition": "SpO2 low"},
        ],
        "steps": [
            {
                "id": "s1",
                "title": "Verify signal",
                "node": "gate",
                "guidance": "Confirm probe reachable and data fresh first.",
            },
            {
                "id": "s2",
                "title": "Assess rSO2",
                "node": "d_rso2",
                "guidance": "Compare live rSO2 to 0.80x its 24h baseline.",
            },
        ],
        "inputs": [
            {"label": "rSO2", "query": _q("neonatal_rso2")},
            {"label": "SpO2", "query": _q("neonatal_spo2")},
            {"label": "HR", "query": _q("neonatal_hr")},
            {"label": "MAP", "query": _q("neonatal_map")},
            {"label": "FiO2", "query": _q("neonatal_fio2")},
        ],
    }
    flow.update(overrides)
    return flow


def _widget(**overrides) -> dict:
    w = {
        "id": "df1",
        "type": "decision_flow",
        "title": "rSO2 desaturation decision support",
        "description": "Guided response to a sustained cerebral desaturation.",
        "query": {"source": "prometheus", "promql": "neonatal_rso2", "query_type": "instant"},
        "position": {"x": 0, "y": 0, "w": 12, "h": 10},
        "encoding": {},
        "thresholds": [],
        "options": {},
        "decision_flow": _flow(),
    }
    w.update(overrides)
    return w


def _dashboard(widgets=None):
    return SpecValidator().validate_dashboard(
        {
            "dashboard_id": "df-test",
            "title": "Decision Flow Test",
            "description": "",
            "layout": {"columns": 12, "row_height": 40},
            "variables": [],
            "widgets": widgets if widgets is not None else [_widget()],
            "refresh_interval": "30s",
        }
    )


# ---------------------------------------------------------------------------
# 1. decision_flow is a valid WidgetType enum member
# ---------------------------------------------------------------------------

def test_decision_flow_in_widget_type_enum():
    assert "decision_flow" in [wt.value for wt in WidgetType]
    assert WidgetType("decision_flow") == WidgetType.decision_flow


# ---------------------------------------------------------------------------
# 2. A decision_flow spec validates with nodes / edges / steps + multi-input
# ---------------------------------------------------------------------------

def test_decision_flow_validates_with_nodes_edges_steps_inputs():
    spec = WidgetSpec.model_validate(_widget())
    assert spec.type == WidgetType.decision_flow
    df = spec.decision_flow
    assert isinstance(df, DecisionFlowSpec)
    assert len(df.nodes) == 6
    assert len(df.edges) == 5
    assert len(df.steps) == 2
    # multi-input labeled query set references all five neonatal signals
    labels = {i.label for i in df.inputs}
    assert labels == set(DECISION_FLOW_SIGNALS)


def test_decision_flow_validates_inside_a_dashboard():
    spec = _dashboard()
    assert spec.widgets[0].type == WidgetType.decision_flow
    assert spec.widgets[0].decision_flow is not None


# ---------------------------------------------------------------------------
# 3. First node MUST be the data-integrity gate
# ---------------------------------------------------------------------------

def test_first_node_is_data_integrity_gate():
    df = WidgetSpec.model_validate(_widget()).decision_flow
    assert df.nodes[0].kind.value == DATA_INTEGRITY_GATE_KIND


def test_first_node_not_gate_rejected():
    flow = _flow()
    flow["nodes"][0]["kind"] = "decision"
    flow["nodes"][0]["signal"] = "rSO2"
    with pytest.raises(Exception):
        WidgetSpec.model_validate(_widget(decision_flow=flow))


# ---------------------------------------------------------------------------
# 4. Smuggling raw_html / component / code is rejected (anywhere in the tree)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["raw_html", "component", "code"])
def test_smuggled_forbidden_field_on_node_rejected(field):
    flow = _flow()
    flow["nodes"][0][field] = "<b>x</b>" if field == "raw_html" else "Evil"
    with pytest.raises(SpecValidationError):
        SpecValidator().validate_widget(_widget(decision_flow=flow))


@pytest.mark.parametrize("field", ["raw_html", "component", "code"])
def test_smuggled_forbidden_field_on_flow_root_rejected(field):
    flow = _flow()
    flow[field] = "Evil"
    with pytest.raises(SpecValidationError):
        SpecValidator().validate_widget(_widget(decision_flow=flow))


@pytest.mark.parametrize("field", ["raw_html", "component", "code"])
def test_smuggled_forbidden_field_on_step_rejected(field):
    flow = _flow()
    flow["steps"][0][field] = "Evil"
    with pytest.raises(SpecValidationError):
        SpecValidator().validate_widget(_widget(decision_flow=flow))


# ---------------------------------------------------------------------------
# 5. A source=mock input is rejected (never branch on fake/stale data)
# ---------------------------------------------------------------------------

def test_mock_input_rejected():
    flow = _flow()
    flow["inputs"][0]["query"]["source"] = "mock"
    with pytest.raises(Exception):
        WidgetSpec.model_validate(_widget(decision_flow=flow))


def test_mock_input_rejected_via_validator():
    flow = _flow()
    flow["inputs"][1]["query"]["source"] = "mock"
    with pytest.raises(SpecValidationError):
        SpecValidator().validate_widget(_widget(decision_flow=flow))


# ---------------------------------------------------------------------------
# 6. decision_flow config lockstep with type
# ---------------------------------------------------------------------------

def test_decision_flow_config_required_for_decision_flow_type():
    w = _widget()
    del w["decision_flow"]
    with pytest.raises(Exception):
        WidgetSpec.model_validate(w)


def test_decision_flow_config_forbidden_on_other_type():
    w = _widget(type="line_chart")
    with pytest.raises(Exception):
        WidgetSpec.model_validate(w)


# ---------------------------------------------------------------------------
# 7. Graph integrity: edges/steps must resolve; unknown signal rejected
# ---------------------------------------------------------------------------

def test_edge_to_unknown_node_rejected():
    flow = _flow()
    flow["edges"].append({"from": "gate", "to": "nope", "condition": "x"})
    with pytest.raises(Exception):
        WidgetSpec.model_validate(_widget(decision_flow=flow))


def test_step_referencing_unknown_node_rejected():
    flow = _flow()
    flow["steps"][0]["node"] = "nope"
    with pytest.raises(Exception):
        WidgetSpec.model_validate(_widget(decision_flow=flow))


def test_unknown_signal_on_decision_node_rejected():
    flow = _flow()
    flow["nodes"][1]["signal"] = "BLOOD_GLUCOSE"
    with pytest.raises(Exception):
        WidgetSpec.model_validate(_widget(decision_flow=flow))


def test_unknown_input_label_rejected():
    flow = _flow()
    flow["inputs"][0]["label"] = "TEMP"
    with pytest.raises(Exception):
        WidgetSpec.model_validate(_widget(decision_flow=flow))


# ---------------------------------------------------------------------------
# 8. decision_flow is taught in the schema doc
# ---------------------------------------------------------------------------

def test_decision_flow_in_schema_doc():
    from app.specs.widget_schema_doc import WIDGET_SCHEMA_MARKDOWN

    assert "decision_flow" in WIDGET_SCHEMA_MARKDOWN
    assert DATA_INTEGRITY_GATE_KIND in WIDGET_SCHEMA_MARKDOWN
    # all five neonatal signals are documented
    for sig in DECISION_FLOW_SIGNALS:
        assert sig in WIDGET_SCHEMA_MARKDOWN
