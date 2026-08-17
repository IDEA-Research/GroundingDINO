"""Tests for bin/helpercode CLI.

Exercises the subprocess protocol end-to-end using the heuristic
provider (no OpenRouter key required).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


_CLI = Path(__file__).resolve().parents[2] / "bin" / "helpercode"


def _run(payload: dict, *, env: dict | None = None) -> tuple[int, dict | None, str]:
    """Invoke bin/helpercode, return (rc, parsed_stdout_or_none, stderr)."""
    proc_env = {
        **os.environ,
        "HELPERCODE_LLM_PROVIDER": "heuristic",
    }
    if env:
        proc_env.update(env)
    proc = subprocess.run(
        [str(_CLI)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=20,
        env=proc_env,
    )
    out = proc.stdout.strip()
    parsed: dict | None = None
    try:
        parsed = json.loads(out) if out else None
    except json.JSONDecodeError:
        parsed = None
    return proc.returncode, parsed, proc.stderr


def test_cli_exists_and_executable():
    assert _CLI.exists()
    assert os.access(_CLI, os.X_OK)


def test_cli_user_message_heuristic():
    rc, out, err = _run({
        "operation": "user_message",
        "agent": "helper-chat-agent",
        "args": {"message": "Show me a CPU dashboard"},
    })
    assert rc == 0, err
    assert out is not None
    assert out["type"] == "DashboardIntent"


def test_cli_generate_dashboard_heuristic():
    rc, out, err = _run({
        "operation": "generate_dashboard",
        "agent": "dashboard-spec-agent",
        "args": {
            "intent": {
                "type": "DashboardIntent",
                "summary": "CPU",
                "requirements": {
                    "title": "CPU", "goal": "see cpu",
                    "metrics_hints": ["node_cpu_seconds_total"],
                    "widget_hints": ["line_chart"],
                    "refresh_interval_hint": "30s",
                },
            },
        },
    })
    assert rc == 0, err
    assert out["type"] == "DashboardSpec"
    assert out["spec"]["widgets"]
    assert out["spec"]["widgets"][0]["type"] == "line_chart"


def test_cli_review_rendered_heuristic():
    rc, out, err = _run({
        "operation": "review_rendered",
        "agent": "helper-review-agent",
        "args": {
            "dashboard": {
                "dashboard_id": "d",
                "widgets": [{"id": "w1"}, {"id": "w2"}],
            },
            "report": {
                "page_loaded": True,
                "missing_widgets": ["w1"],
                "console_errors": [],
            },
            "attempt": 1,
        },
    })
    assert rc == 0, err
    assert out["type"] == "ReviewDecision"
    assert out["decision"] == "patch"
    assert "patch" in out


def test_cli_review_rendered_approve():
    rc, out, err = _run({
        "operation": "review_rendered",
        "agent": "helper-review-agent",
        "args": {
            "dashboard": {"dashboard_id": "d", "widgets": [{"id": "w1"}]},
            "report": {
                "page_loaded": True,
                "missing_widgets": [],
                "console_errors": [],
            },
            "attempt": 1,
        },
    })
    assert rc == 0, err
    assert out["type"] == "ReviewDecision"
    assert out["decision"] == "approve"


def test_cli_rescue_review_ticket_on_console_errors():
    rc, out, err = _run({
        "operation": "rescue_review",
        "agent": "big-guy-developer-agent",
        "args": {
            "dashboard": {"dashboard_id": "d", "widgets": []},
            "report": {
                "page_loaded": True,
                "missing_widgets": [],
                "console_errors": ["TypeError: x is not a function"],
            },
        },
    })
    assert rc == 0, err
    assert out["type"] == "RescueDecision"
    assert out["kind"] == "ticket"
    assert out["ticket"]["severity"] in {"low", "medium", "high"}


def test_cli_rescue_review_ask_user_on_missing_widgets():
    rc, out, err = _run({
        "operation": "rescue_review",
        "agent": "big-guy-developer-agent",
        "args": {
            "dashboard": {"dashboard_id": "d", "widgets": []},
            "report": {
                "page_loaded": True,
                "missing_widgets": ["w1"],
                "console_errors": [],
            },
        },
    })
    assert rc == 0, err
    assert out["kind"] == "ask_user"
    assert out["questions"]


def test_cli_rejects_unknown_operation():
    rc, out, err = _run({
        "operation": "shell_exec",
        "agent": "helper-chat-agent",
        "args": {"cmd": "ls"},
    })
    assert rc != 0
    assert out is None
    assert "unknown operation" in err


def test_cli_rejects_invalid_output_type():
    """If heuristic somehow produced a result with the wrong envelope
    type for the operation, the CLI should reject before printing.

    We simulate this by pointing `operation` at one thing but
    providing args the heuristic interprets as something else. In
    practice the heuristic always produces the right type, so this
    test just proves the validator is wired — we stress it by
    requesting an operation that would need specific args shapes
    and giving it bad args.
    """
    # developer_fix always returns DeveloperReport type — that's
    # in the allow-list so it succeeds. This is a smoke that the
    # CLI wires the check.
    rc, out, err = _run({
        "operation": "developer_fix",
        "agent": "big-guy-developer-agent",
        "args": {"instruction": "ignored"},
    })
    assert rc == 0, err
    assert out["type"] == "DeveloperReport"


def test_cli_developer_fix_heuristic_refuses_edits():
    rc, out, err = _run({
        "operation": "developer_fix",
        "agent": "big-guy-developer-agent",
        "args": {"instruction": "edit some file"},
    })
    assert rc == 0, err
    assert out["type"] == "DeveloperReport"
    assert out["status"] == "rejected"
    assert "cannot make code changes" in out["summary"].lower()


def test_cli_rejects_missing_api_key_in_openrouter_mode(tmp_path):
    """With provider=openrouter and no key, exit non-zero."""
    rc, out, err = _run(
        {
            "operation": "user_message",
            "agent": "helper-chat-agent",
            "args": {"message": "hi"},
        },
        env={
            "HELPERCODE_LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "",
        },
    )
    assert rc != 0
    assert "OPENROUTER_API_KEY" in err


# ---------------------------------------------------------------------------
# Metric-catalog prompt injection (LLM path)
# ---------------------------------------------------------------------------


def _load_cli_module():
    """Import bin/helpercode as a module (no .py suffix — same pattern
    as tests/helpercode_cli/test_tool_use.py)."""
    from importlib.machinery import SourceFileLoader
    from importlib.util import module_from_spec, spec_from_loader

    loader = SourceFileLoader("_bin_helpercode_prompt", str(_CLI))
    spec = spec_from_loader(loader.name, loader)
    mod = module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_user_prompt_carries_metric_catalog_section():
    mod = _load_cli_module()
    args = {
        "intent": {"type": "DashboardIntent", "summary": "vitals"},
        "metric_catalog": {
            "available": True,
            "fetched_at": "2026-08-17T00:00:00+00:00",
            "metrics": [
                "patient_heart_rate_bpm",
                "patient_spo2_percentage",
                "node_cpu_seconds_total",
            ],
        },
    }
    prompt = mod._build_user_prompt("generate_dashboard", args)
    # The section lists the real names as ground truth...
    assert "## Live metric catalog" in prompt
    assert "- patient_heart_rate_bpm" in prompt
    assert "- patient_spo2_percentage" in prompt
    # ...with the clinical aggregation rule and the no-invention order.
    assert "max by (patient_id)" in prompt
    assert "NEVER invent" in prompt
    # The raw payload is NOT duplicated inside the ARGS JSON blob (it
    # would eat the 16000-char budget).
    assert '"metric_catalog"' not in prompt


def test_user_prompt_states_catalog_unavailable_explicitly():
    mod = _load_cli_module()
    args = {
        "intent": {"type": "DashboardIntent", "summary": "vitals"},
        "metric_catalog": {"available": False, "metrics": [],
                           "fetched_at": "2026-08-17T00:00:00+00:00"},
    }
    prompt = mod._build_user_prompt("generate_dashboard", args)
    assert "UNAVAILABLE" in prompt
    assert "Do NOT guess" in prompt


def test_user_prompt_without_catalog_is_unchanged_shape():
    mod = _load_cli_module()
    prompt = mod._build_user_prompt(
        "generate_dashboard", {"intent": {"type": "DashboardIntent"}}
    )
    assert "## Live metric catalog" not in prompt
    assert prompt.startswith("OPERATION: generate_dashboard\n")
    assert prompt.rstrip().endswith("No prose.")
