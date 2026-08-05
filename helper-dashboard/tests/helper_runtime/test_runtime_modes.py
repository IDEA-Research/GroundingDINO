"""OpenCodeRuntime adapter tests.

Covers the three runtime modes (mock / opencode / auto), the
operation allow-list, the user-vs-developer gate, the subprocess
contract, and the output-contract validation.

Uses small on-disk stub executables to stand in for the real
`opencode` CLI so we can verify the end-to-end protocol without a
real install.
"""

from __future__ import annotations

import json
import os
import stat
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from app.helper.runtime import (
    BIG_GUY,
    DEVELOPER_OPERATIONS,
    EXPECTED_OUTPUT_TYPES,
    HELPER_AGENTS,
    OPERATIONS,
    OpenCodeRuntime,
    RuntimeError_,
    USER_OPERATIONS,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_stub(path: Path, script: str) -> Path:
    path.write_text(f"#!/usr/bin/env python3\n{script}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _clear_runtime_env(monkeypatch):
    for k in (
        "HELPER_DASHBOARD_OPENCODE",
        "HELPER_DASHBOARD_OPENCODE_BIN",
        "HELPER_DASHBOARD_OPENCODE_CMD",
        "HELPER_DASHBOARD_OPENCODE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(k, raising=False)


def _configure_stub(monkeypatch, stub_path: Path, *, mode: str = "opencode"):
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE", mode)
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE_BIN", str(stub_path))
    # Simplest possible argv: just the binary, nothing else.
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE_CMD", json.dumps(["{bin}"]))


# ---------------------------------------------------------------------------
# Mode: mock is the default
# ---------------------------------------------------------------------------


def test_mock_is_default(monkeypatch):
    _clear_runtime_env(monkeypatch)
    rt = OpenCodeRuntime()
    out = rt.invoke_operation(
        "user_message", {"message": "Show me a CPU dashboard"}
    )
    assert out["type"] == "DashboardIntent"
    assert out["runtime_used"] == "mock"
    assert out.get("fallback_reason") is None


def test_mock_mode_explicit(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE", "mock")
    rt = OpenCodeRuntime()
    out = rt.invoke_operation("user_message", {"message": "hi"})
    assert out["runtime_used"] == "mock"


def test_unknown_mode_is_refused(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE", "magic")
    rt = OpenCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("user_message", {"message": "hi"})
    assert "unknown HELPER_DASHBOARD_OPENCODE mode" in str(exc.value)


# ---------------------------------------------------------------------------
# Mode: opencode must NOT silently fall back
# ---------------------------------------------------------------------------


def test_opencode_mode_missing_binary_raises(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE", "opencode")
    monkeypatch.setenv(
        "HELPER_DASHBOARD_OPENCODE_BIN", "/nonexistent/opencode-xyz-1234"
    )
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE_CMD", json.dumps(["{bin}"]))
    rt = OpenCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("user_message", {"message": "hi"})
    assert "opencode runtime failure" in str(exc.value)
    assert "binary_missing" in str(exc.value)


def test_opencode_mode_non_json_stdout_raises(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(tmp_path / "opencode", 'print("not json at all")')
    _configure_stub(monkeypatch, stub, mode="opencode")
    rt = OpenCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("user_message", {"message": "hi"})
    assert "non_json" in str(exc.value)


def test_opencode_mode_nonzero_exit_raises(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "opencode",
        textwrap.dedent(
            """
            import sys
            print('boom', file=sys.stderr)
            sys.exit(7)
            """
        ),
    )
    _configure_stub(monkeypatch, stub, mode="opencode")
    rt = OpenCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("user_message", {"message": "hi"})
    assert "nonzero_exit" in str(exc.value)


def test_opencode_mode_missing_type_field_raises(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "opencode",
        'import json; print(json.dumps({"not_typed": True}))',
    )
    _configure_stub(monkeypatch, stub, mode="opencode")
    rt = OpenCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("user_message", {"message": "hi"})
    assert "missing_type" in str(exc.value)


def test_opencode_mode_invalid_output_type_raises(monkeypatch, tmp_path):
    """An otherwise well-formed response whose `type` is not in the
    allow-list for this operation must be rejected."""
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "opencode",
        textwrap.dedent(
            """
            import json
            # generate_dashboard expects DashboardSpec or DeveloperTicket.
            # Returning a PatchSpec here is wrong for this operation.
            print(json.dumps({"type": "PatchSpec", "spec": {}}))
            """
        ),
    )
    _configure_stub(monkeypatch, stub, mode="opencode")
    rt = OpenCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("generate_dashboard", {"intent": {}})
    assert "invalid output for 'generate_dashboard'" in str(exc.value)


def test_opencode_mode_success_marks_runtime_used(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "opencode",
        textwrap.dedent(
            """
            import json
            print(json.dumps({"type": "UserResponse", "message": "hi back"}))
            """
        ),
    )
    _configure_stub(monkeypatch, stub, mode="opencode")
    rt = OpenCodeRuntime()
    out = rt.invoke_operation("user_message", {"message": "hi"})
    assert out["type"] == "UserResponse"
    assert out["runtime_used"] == "opencode"
    assert out.get("fallback_reason") is None


# ---------------------------------------------------------------------------
# Mode: auto — may fall back, but must mark it explicitly
# ---------------------------------------------------------------------------


def test_auto_mode_uses_real_on_success(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "opencode",
        textwrap.dedent(
            """
            import json
            print(json.dumps({"type": "UserResponse", "message": "real reply"}))
            """
        ),
    )
    _configure_stub(monkeypatch, stub, mode="auto")
    rt = OpenCodeRuntime()
    out = rt.invoke_operation("user_message", {"message": "hi"})
    assert out["runtime_used"] == "opencode"
    assert out["message"] == "real reply"


def test_auto_mode_falls_back_with_explicit_marker_on_missing_binary(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE", "auto")
    monkeypatch.setenv(
        "HELPER_DASHBOARD_OPENCODE_BIN", "/nonexistent/opencode-xyz-1234"
    )
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE_CMD", json.dumps(["{bin}"]))
    rt = OpenCodeRuntime()
    out = rt.invoke_operation("user_message", {"message": "Show me a CPU dashboard"})
    assert out["runtime_used"] == "mock_fallback"
    assert out.get("fallback_reason")
    assert "binary_missing" in out["fallback_reason"]
    assert out["type"] == "DashboardIntent"  # mock output kept


def test_auto_mode_falls_back_on_non_json(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(tmp_path / "opencode", 'print("garbage")')
    _configure_stub(monkeypatch, stub, mode="auto")
    rt = OpenCodeRuntime()
    out = rt.invoke_operation(
        "user_message", {"message": "Show me a CPU dashboard"}
    )
    assert out["runtime_used"] == "mock_fallback"
    assert "non_json" in out["fallback_reason"]


# ---------------------------------------------------------------------------
# Operation / agent allow-list
# ---------------------------------------------------------------------------


def test_unknown_operation_is_refused(monkeypatch):
    _clear_runtime_env(monkeypatch)
    rt = OpenCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("shell_exec", {"cmd": "ls"})
    assert "unknown operation" in str(exc.value)


def test_developer_fix_is_blocked_from_user_flow(monkeypatch):
    """Normal /api/chat must never call developer_fix. The runtime
    wrapper rejects it unless the caller explicitly passes
    `developer=True`."""
    _clear_runtime_env(monkeypatch)
    rt = OpenCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("developer_fix", {"ticket_id": "tkt-1"})
    assert "developer-only" in str(exc.value)


def test_developer_fix_requires_developer_flag(monkeypatch, tmp_path):
    """Even when a stub would happily respond, the runtime refuses
    developer_fix unless `developer=True` is passed. This is where
    the user-vs-developer gate lives."""
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "opencode",
        textwrap.dedent(
            """
            import json
            print(json.dumps({
                "type": "DeveloperReport",
                "report_id": "r1",
                "summary": "done",
                "status": "resolved"
            }))
            """
        ),
    )
    _configure_stub(monkeypatch, stub, mode="opencode")
    rt = OpenCodeRuntime()

    with pytest.raises(RuntimeError_):
        rt.invoke_operation("developer_fix", {"ticket_id": "tkt-1"})

    out = rt.invoke_operation(
        "developer_fix", {"ticket_id": "tkt-1"}, developer=True
    )
    assert out["type"] == "DeveloperReport"


def test_operation_agent_mapping_is_complete_and_disjoint():
    # Every operation maps to exactly one agent.
    for op, meta in OPERATIONS.items():
        assert meta["agent"], f"operation {op} has no agent"
    # User and developer operations are disjoint.
    assert USER_OPERATIONS.isdisjoint(DEVELOPER_OPERATIONS)
    # All defined operations are classified.
    assert USER_OPERATIONS | DEVELOPER_OPERATIONS == set(OPERATIONS.keys())
    # Every operation has an expected output type set.
    for op in OPERATIONS:
        assert EXPECTED_OUTPUT_TYPES[op]


def test_big_guy_not_in_helper_agents_set():
    assert BIG_GUY not in HELPER_AGENTS


def test_compat_invoke_helper_rejects_big_guy(monkeypatch):
    _clear_runtime_env(monkeypatch)
    rt = OpenCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_helper(BIG_GUY, {"ticket_id": "tkt-1"})
    assert "not a Helper agent" in str(exc.value)


def test_compat_invoke_helper_rejects_unknown_agent(monkeypatch):
    _clear_runtime_env(monkeypatch)
    rt = OpenCodeRuntime()
    with pytest.raises(RuntimeError_):
        rt.invoke_helper("mystery-agent", {"message": "hi"})


# ---------------------------------------------------------------------------
# Subprocess safety
# ---------------------------------------------------------------------------


def test_subprocess_uses_shell_false_and_fixed_cwd(monkeypatch, tmp_path):
    """Confirms the adapter never invokes subprocess.run with
    shell=True, pins cwd to the project root, and never injects
    user-supplied text into argv."""
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "opencode",
        'import json; print(json.dumps({"type":"UserResponse","message":"ok"}))',
    )
    _configure_stub(monkeypatch, stub, mode="opencode")

    captured: dict = {}
    import app.helper.runtime as runtime_mod
    orig = runtime_mod.subprocess.run

    def spy(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return orig(*args, **kwargs)

    monkeypatch.setattr(runtime_mod.subprocess, "run", spy)

    rt = OpenCodeRuntime()
    # user args contain shell metacharacters — they must never reach argv.
    rt.invoke_operation(
        "user_message", {"message": "rm -rf / && echo ;`whoami`"}
    )

    assert "args" in captured
    cmd = captured["args"][0]
    # `shell` must be either absent (defaults to False) or explicitly False.
    assert captured["kwargs"].get("shell", False) is False
    # cwd is pinned to the project root, which is the dir containing README.md.
    cwd = captured["kwargs"].get("cwd")
    assert cwd and (Path(cwd) / "README.md").exists()
    # User text does not appear anywhere in the argv.
    flat = " ".join(cmd)
    assert "rm -rf" not in flat
    assert "whoami" not in flat


def test_user_args_are_sent_on_stdin_not_argv(monkeypatch, tmp_path):
    """The stub echoes the argv and the stdin payload so we can
    assert where user data actually goes."""
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "opencode",
        textwrap.dedent(
            """
            import json, sys
            argv = sys.argv[1:]
            payload = json.loads(sys.stdin.read())
            print(json.dumps({
                "type": "UserResponse",
                "message": "ok",
                "argv": argv,
                "stdin_args_message": payload["args"].get("message", ""),
            }))
            """
        ),
    )
    _configure_stub(monkeypatch, stub, mode="opencode")
    rt = OpenCodeRuntime()
    out = rt.invoke_operation(
        "user_message", {"message": "rm -rf / ; echo pwned"}
    )
    assert "rm -rf" not in " ".join(out["argv"])
    assert out["stdin_args_message"] == "rm -rf / ; echo pwned"


def test_agent_substitution_uses_allow_listed_value(monkeypatch, tmp_path):
    """`{agent}` in the command template is substituted from the
    operation mapping, not from any caller-controlled field."""
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "opencode",
        textwrap.dedent(
            """
            import json, sys
            argv = sys.argv[1:]
            print(json.dumps({
                "type": "DashboardSpec",
                "spec": {
                    "dashboard_id": "d",
                    "title": "t",
                    "description": "",
                    "layout": {"columns": 12, "row_height": 40},
                    "variables": [],
                    "widgets": [],
                    "refresh_interval": "30s"
                },
                "_argv": argv,
            }))
            """
        ),
    )
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE", "opencode")
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE_BIN", str(stub))
    monkeypatch.setenv(
        "HELPER_DASHBOARD_OPENCODE_CMD",
        json.dumps(["{bin}", "--agent", "{agent}"]),
    )
    rt = OpenCodeRuntime()
    # Caller supplies args that try to contaminate the agent field —
    # the wrapper ignores them; `{agent}` is resolved from OPERATIONS.
    out = rt.invoke_operation(
        "generate_dashboard",
        {"intent": {}, "agent": "evil-agent-name"},  # ignored
    )
    assert "--agent" in out["_argv"]
    assert "dashboard-spec-agent" in out["_argv"]
    assert "evil-agent-name" not in out["_argv"]


def test_malformed_cmd_env_is_ignored(monkeypatch, tmp_path):
    """If the operator sets HELPER_DASHBOARD_OPENCODE_CMD to something
    that isn't a JSON array of strings, we log and fall back to the
    default template rather than crashing or shell-evaluating."""
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "opencode",
        'import json; print(json.dumps({"type":"UserResponse","message":"ok"}))',
    )
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE", "opencode")
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE_BIN", str(stub))
    # A shell-injection attempt.
    monkeypatch.setenv(
        "HELPER_DASHBOARD_OPENCODE_CMD", "rm -rf / ; echo pwned"
    )
    rt = OpenCodeRuntime()
    out = rt.invoke_operation("user_message", {"message": "hi"})
    assert out["type"] == "UserResponse"


# ---------------------------------------------------------------------------
# Output contract: invalid specs produced by the runtime are rejected
# downstream even when the envelope type is correct
# ---------------------------------------------------------------------------


def test_invalid_dashboard_spec_payload_rejected_by_orchestrator(monkeypatch, tmp_path):
    """When the envelope is `DashboardSpec` but the inner spec fails
    schema validation, the orchestrator must refuse to save it and
    must not surface internal details to the user."""
    _clear_runtime_env(monkeypatch)
    # Stub routes: the classifier returns DashboardIntent, the generator
    # returns a DashboardSpec with a forbidden field.
    stub = _write_stub(
        tmp_path / "opencode",
        textwrap.dedent(
            """
            import json, sys
            payload = json.loads(sys.stdin.read())
            op = payload["operation"]
            if op == "user_message":
                print(json.dumps({
                    "type": "DashboardIntent",
                    "summary": "demo",
                    "requirements": {
                        "title": "demo",
                        "goal": "",
                        "metrics_hints": [],
                        "widget_hints": ["line_chart"],
                        "refresh_interval_hint": "30s"
                    },
                    "clarification_needed": False,
                    "message_to_user": "building"
                }))
            elif op == "generate_dashboard":
                print(json.dumps({
                    "type": "DashboardSpec",
                    "spec": {
                        "dashboard_id": "demo",
                        "title": "<script>bad</script>",
                        "description": "",
                        "layout": {"columns": 12, "row_height": 40},
                        "variables": [],
                        "widgets": [],
                        "refresh_interval": "30s"
                    }
                }))
            else:
                print(json.dumps({"type": "UserResponse", "message": "noop"}))
            """
        ),
    )
    _configure_stub(monkeypatch, stub, mode="opencode")

    from app.helper.orchestrator import Orchestrator
    orch = Orchestrator()
    res = orch.handle_user_message(
        session_id="s-invalid", message="build me a dashboard"
    )
    # The reply is the safe generic error; no dashboard was saved.
    assert res["dashboard"] is None
    assert res["warnings"]
    assert "<script>" not in res["user_reply"]


def test_invalid_patch_spec_payload_rejected(monkeypatch, tmp_path):
    """A well-typed PatchSpec envelope whose operations contain an
    unknown `op` must be refused by validation."""
    _clear_runtime_env(monkeypatch)
    # First create a dashboard via mock so there is something to patch.
    from app.helper.orchestrator import Orchestrator
    orch = Orchestrator()
    first = orch.handle_user_message(
        session_id="s-patch", message="Show me a CPU dashboard"
    )
    did = first["dashboard"]["dashboard_id"]

    # Now switch to opencode with a stub that returns a bogus patch.
    stub = _write_stub(
        tmp_path / "opencode",
        textwrap.dedent(
            """
            import json, sys
            payload = json.loads(sys.stdin.read())
            op = payload["operation"]
            if op == "user_message":
                print(json.dumps({
                    "type": "PatchIntent",
                    "target_dashboard_id": payload["args"].get("current_dashboard_id"),
                    "requested_changes": ["change something"],
                    "message_to_user": "ok"
                }))
            elif op == "patch_dashboard":
                print(json.dumps({
                    "type": "PatchSpec",
                    "spec": {
                        "patch_id": "p1",
                        "reason": "x",
                        "target_dashboard_id": payload["args"]["dashboard"]["dashboard_id"],
                        "created_by": "patch-agent",
                        "operations": [{"op": "exec", "cmd": "rm -rf /"}]
                    }
                }))
            else:
                print(json.dumps({"type": "UserResponse", "message": "noop"}))
            """
        ),
    )
    _configure_stub(monkeypatch, stub, mode="opencode")
    orch2 = Orchestrator()
    res = orch2.handle_user_message(
        session_id="s-patch",
        message="change something",
        current_dashboard_id=did,
    )
    assert res["patch"] is None
    assert res["warnings"]


# ---------------------------------------------------------------------------
# Agent config: Helper permissions stay locked at the config layer
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_helper_agent_md_configs_lock_permissions():
    """Primary enforcement of Helper permissions is the agent .md
    frontmatter (and opencode.json). This test reads each Helper
    agent file and asserts edit/shell/webfetch are all false."""
    for agent_name in HELPER_AGENTS:
        path = _REPO_ROOT / ".opencode" / "agent" / f"{agent_name}.md"
        text = path.read_text()
        # The frontmatter lives between the first two `---` fences.
        assert text.startswith("---"), f"{agent_name}.md missing frontmatter"
        _, frontmatter, _ = text.split("---", 2)
        for key in ("edit", "shell", "webfetch"):
            # Accept either `edit: false` or `  edit: false` indentation.
            assert (
                f"{key}: false" in frontmatter
            ), f"{agent_name}: expected `{key}: false` in frontmatter"


def test_big_guy_md_marks_internal_only():
    path = _REPO_ROOT / ".opencode" / "agent" / f"{BIG_GUY}.md"
    text = path.read_text()
    assert "internal: true" in text
    # Big guy is allowed to have edit/shell — but he's not a Helper.
    assert "edit: true" in text
    assert "shell: true" in text


def test_opencode_json_locks_helper_permissions_and_gates_developer_fix():
    cfg = json.loads((_REPO_ROOT / "opencode.json").read_text())
    for agent_name in HELPER_AGENTS:
        perms = cfg["agents"][agent_name]["permissions"]
        assert perms["edit"] is False
        assert perms["shell"] is False
    assert cfg["agents"][BIG_GUY].get("internal") is True
    assert "developer-fix" in cfg["boundaries"]["developer_only"]
    assert "developer-fix" not in cfg["boundaries"]["user_can_invoke"]
