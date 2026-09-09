"""Orchestrator title normaliser (五.3) + bin/helpercode prompt rules.

The normaliser strips ONLY a trailing time-range parenthetical; the
prompt rules are appended to the generate/patch system prompts and the
heuristic provider never emits a stat_card+gauge twin.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import signal
import sys
from pathlib import Path

import pytest

from app.helper.orchestrator import (
    Orchestrator,
    _PENDING_SAVE,
    normalize_widget_title,
    normalize_widget_titles,
)
from app.specs import DashboardSpec

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("title,expected", [
    ("心率趨勢（過去 1 小時）", "心率趨勢"),
    ("心率趨勢(過去 1 小時)", "心率趨勢"),
    ("心率趨勢（過去1小時）", "心率趨勢"),
    ("血氧趨勢（最近 5 分鐘）", "血氧趨勢"),
    ("血氧趨勢（近 15 分）", "血氧趨勢"),
    ("SpO2 trend (last 1h)", "SpO2 trend"),
    ("SpO2 trend (last 24 hours)", "SpO2 trend"),
    ("HR (past 5m)", "HR"),
    ("心率趨勢（過去一小時）", "心率趨勢"),
    ("rSO2 左右腦 (Last 6 Hrs)", "rSO2 左右腦"),
])
def test_strips_trailing_time_range(title, expected):
    assert normalize_widget_title(title) == expected
    assert normalize_widget_title(expected) == expected  # idempotent


@pytest.mark.parametrize("title", [
    "心率", "SpO2 (%)", "心率（左側）", "過去 1 小時心率", "(過去 1 小時) 心率",
    "Request latency p95", "5xx error rate", "心率 (bpm)", "各病患一覽",
])
def test_leaves_other_titles_alone(title):
    assert normalize_widget_title(title) == title


def test_never_empties_a_title():
    assert normalize_widget_title("（過去 1 小時）") == "（過去 1 小時）"
    assert normalize_widget_title("") == ""


def test_normalize_widget_titles_rebuilds_spec_and_reports():
    spec = DashboardSpec.model_validate({
        "dashboard_id": "d", "title": "T（過去 1 小時）",  # dashboard title untouched
        "widgets": [
            {"id": "a", "type": "line_chart", "title": "心率趨勢（過去 1 小時）",
             "query": {"promql": "up", "query_type": "range", "range": "1h", "step": "30s"},
             "position": {"x": 0, "y": 0, "w": 6, "h": 6}},
            {"id": "b", "type": "stat_card", "title": "心率",
             "query": {"promql": "up", "query_type": "instant"},
             "position": {"x": 6, "y": 0, "w": 3, "h": 3}},
        ],
    })
    out, notes = normalize_widget_titles(spec)
    assert [w.title for w in out.widgets] == ["心率趨勢", "心率"]
    assert out.title == "T（過去 1 小時）"
    assert notes == ["title normalized: '心率趨勢（過去 1 小時）' -> '心率趨勢'"]
    same, none = normalize_widget_titles(out)
    assert same is out and none == []


def test_orchestrator_applies_normalizer_to_generated_spec(monkeypatch):
    """Scripted runtime emits a range-bearing title; the delivered
    dashboard has it stripped and a dev-facing warning explains."""
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_HELPERCODE", raising=False)
    _PENDING_SAVE.clear()
    orch = Orchestrator()
    real_invoke = orch._runtime.invoke_operation

    def _invoke(operation, args, **kw):
        out = real_invoke(operation, args, **kw)
        if operation == "generate_dashboard" and isinstance(out, dict) and out.get("spec"):
            for w in out["spec"].get("widgets", []):
                w["title"] = w["title"] + "（過去 1 小時）"
        return out

    monkeypatch.setattr(orch._runtime, "invoke_operation", _invoke)
    monkeypatch.setattr(orch._retry._runtime, "invoke_operation", _invoke, raising=False)
    res = orch.handle_user_message(session_id="norm-1", message="Show me a CPU dashboard")
    assert res["intent_type"] == "DashboardSpec"
    titles = [w["title"] for w in res["dashboard"]["widgets"]]
    assert titles and all("過去" not in t for t in titles)
    assert any(w.startswith("title normalized:") for w in res["warnings"])
    # The save prompt appended by the orchestrator is 繁中 (the mock
    # classifier's own message_to_user, in runtime.py, is out of scope).
    assert "存到你的資料庫" in res["user_reply"]


# ---------------------------------------------------------------------------
# bin/helpercode prompt rules (loaded as a module; the script installs
# POSIX-only signal handlers at import, which we stub on Windows)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def helpercode():
    # bin/helpercode reads signal.SIGHUP/SIGQUIT at import (POSIX only);
    # stub the attributes for the import and remove them again afterwards
    # so no other test module sees a fake signal set.
    added = []
    for name in ("SIGHUP", "SIGQUIT"):
        if not hasattr(signal, name):
            setattr(signal, name, 99)  # only the attribute lookup runs at import
            added.append(name)
    path = _REPO_ROOT / "bin" / "helpercode"
    # No `.py` suffix -> spec_from_file_location() has no loader for it;
    # name the source loader explicitly.
    loader = importlib.machinery.SourceFileLoader("helpercode_under_test", str(path))
    spec = importlib.util.spec_from_loader("helpercode_under_test", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["helpercode_under_test"] = module
    try:
        loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("helpercode_under_test", None)
        for name in added:
            delattr(signal, name)


def test_generate_and_patch_prompts_carry_authoring_rules(helpercode):
    for op, agent in (("generate_dashboard", "dashboard-spec-agent"), ("patch_dashboard", "patch-agent")):
        prompt = helpercode._build_system_prompt(op, agent, "m")
        assert "Dashboard authoring rules" in prompt
        assert "NEVER put a time range in a title" in prompt
        assert "stat_card` AND a `gauge" in prompt
        assert "MUST use `queries`" in prompt
        assert "MUST set `direction`" in prompt
        # The rules block comes AFTER the agent body so it overrides it.
        assert prompt.index("Dashboard authoring rules") > prompt.index("Output contract")
    # Non-authoring operations don't get the block.
    assert "Dashboard authoring rules" not in helpercode._build_system_prompt(
        "user_message", "helper-chat-agent", "m"
    )


def test_schema_hints_in_prompt_document_new_fields(helpercode):
    prompt = helpercode._build_system_prompt("generate_dashboard", "dashboard-spec-agent", "m")
    assert "`direction`" in prompt and "`queries`" in prompt and "`key`" in prompt


def test_heuristic_never_emits_stat_and_gauge_twins(helpercode):
    widgets = helpercode._build_widgets(["stat_card", "gauge"], ["node_cpu_seconds_total"])
    instant = [w for w in widgets if w["type"] in ("stat_card", "gauge")]
    assert len(instant) == 1 and instant[0]["type"] == "stat_card"
    # Different metrics still get their own instant widget.
    widgets = helpercode._build_widgets(["stat_card", "gauge"], ["node_cpu_seconds_total", "node_load1"])
    assert len({w["query"]["promql"] for w in widgets}) == len(widgets)


def test_heuristic_sample_thresholds_carry_direction(helpercode):
    widgets = helpercode._api_observability_widgets()
    for w in widgets:
        for t in w["thresholds"]:
            assert t.get("direction") in ("above", "below"), (w["id"], t)
    gauge = next(w for w in widgets if w["type"] == "gauge")
    assert all(t["direction"] == "below" for t in gauge["thresholds"])
    # And the sample still validates against the backend schema.
    from app.services.spec_validator import SpecValidator

    SpecValidator().validate_dashboard({
        "dashboard_id": "api-gold", "title": "API", "widgets": widgets,
    })
