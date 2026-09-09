"""HelperCodeRuntime adapter tests.

Covers the three runtime modes (mock / helpercode / auto), the
operation allow-list, the user-vs-developer gate, the subprocess
contract, and the output-contract validation.

Uses small on-disk stub executables to stand in for the real
`helpercode` CLI so we can verify the end-to-end protocol without a
real install.
"""

from __future__ import annotations

import json
import os
import re
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
    HelperCodeRuntime,
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
        "HELPER_DASHBOARD_HELPERCODE",
        "HELPER_DASHBOARD_HELPERCODE_BIN",
        "HELPER_DASHBOARD_HELPERCODE_CMD",
        "HELPER_DASHBOARD_HELPERCODE_TIMEOUT_SECONDS",
        "HELPER_DASHBOARD_DEVFIX_RUNTIME",
        "HELPER_DASHBOARD_DEVFIX_BIN",
        "HELPER_DASHBOARD_DEVFIX_MODEL",
        "HELPER_DASHBOARD_DEVFIX_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(k, raising=False)


def _configure_stub(monkeypatch, stub_path: Path, *, mode: str = "helpercode"):
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE", mode)
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE_BIN", str(stub_path))
    # Simplest possible argv: just the binary, nothing else.
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE_CMD", json.dumps(["{bin}"]))


# ---------------------------------------------------------------------------
# Mode: mock is the default
# ---------------------------------------------------------------------------


def test_mock_is_default(monkeypatch):
    _clear_runtime_env(monkeypatch)
    rt = HelperCodeRuntime()
    out = rt.invoke_operation(
        "user_message", {"message": "Show me a CPU dashboard"}
    )
    assert out["type"] == "DashboardIntent"
    assert out["runtime_used"] == "mock"
    assert out.get("fallback_reason") is None


def test_mock_mode_explicit(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE", "mock")
    rt = HelperCodeRuntime()
    out = rt.invoke_operation("user_message", {"message": "hi"})
    assert out["runtime_used"] == "mock"


def test_unknown_mode_is_refused(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE", "magic")
    rt = HelperCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("user_message", {"message": "hi"})
    assert "unknown HELPER_DASHBOARD_HELPERCODE mode" in str(exc.value)


# ---------------------------------------------------------------------------
# Mode: helpercode must NOT silently fall back
# ---------------------------------------------------------------------------


def test_helpercode_mode_missing_binary_raises(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE", "helpercode")
    monkeypatch.setenv(
        "HELPER_DASHBOARD_HELPERCODE_BIN", "/nonexistent/helpercode-xyz-1234"
    )
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE_CMD", json.dumps(["{bin}"]))
    rt = HelperCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("user_message", {"message": "hi"})
    assert "helpercode runtime failure" in str(exc.value)
    assert "binary_missing" in str(exc.value)


def test_helpercode_mode_non_json_stdout_raises(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(tmp_path / "helpercode", 'print("not json at all")')
    _configure_stub(monkeypatch, stub, mode="helpercode")
    rt = HelperCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("user_message", {"message": "hi"})
    assert "non_json" in str(exc.value)


def test_helpercode_mode_nonzero_exit_raises(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "helpercode",
        textwrap.dedent(
            """
            import sys
            print('boom', file=sys.stderr)
            sys.exit(7)
            """
        ),
    )
    _configure_stub(monkeypatch, stub, mode="helpercode")
    rt = HelperCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("user_message", {"message": "hi"})
    assert "nonzero_exit" in str(exc.value)


def test_helpercode_mode_missing_type_field_raises(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "helpercode",
        'import json; print(json.dumps({"not_typed": True}))',
    )
    _configure_stub(monkeypatch, stub, mode="helpercode")
    rt = HelperCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("user_message", {"message": "hi"})
    assert "missing_type" in str(exc.value)


def test_helpercode_mode_invalid_output_type_raises(monkeypatch, tmp_path):
    """An otherwise well-formed response whose `type` is not in the
    allow-list for this operation must be rejected."""
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "helpercode",
        textwrap.dedent(
            """
            import json
            # generate_dashboard expects DashboardSpec or DeveloperTicket.
            # Returning a PatchSpec here is wrong for this operation.
            print(json.dumps({"type": "PatchSpec", "spec": {}}))
            """
        ),
    )
    _configure_stub(monkeypatch, stub, mode="helpercode")
    rt = HelperCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("generate_dashboard", {"intent": {}})
    assert "invalid output for 'generate_dashboard'" in str(exc.value)


def test_helpercode_mode_success_marks_runtime_used(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "helpercode",
        textwrap.dedent(
            """
            import json
            print(json.dumps({"type": "UserResponse", "message": "hi back"}))
            """
        ),
    )
    _configure_stub(monkeypatch, stub, mode="helpercode")
    rt = HelperCodeRuntime()
    out = rt.invoke_operation("user_message", {"message": "hi"})
    assert out["type"] == "UserResponse"
    assert out["runtime_used"] == "helpercode"
    assert out.get("fallback_reason") is None


# ---------------------------------------------------------------------------
# Mode: auto — may fall back, but must mark it explicitly
# ---------------------------------------------------------------------------


def test_auto_mode_uses_real_on_success(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "helpercode",
        textwrap.dedent(
            """
            import json
            print(json.dumps({"type": "UserResponse", "message": "real reply"}))
            """
        ),
    )
    _configure_stub(monkeypatch, stub, mode="auto")
    rt = HelperCodeRuntime()
    out = rt.invoke_operation("user_message", {"message": "hi"})
    assert out["runtime_used"] == "helpercode"
    assert out["message"] == "real reply"


def test_auto_mode_falls_back_with_explicit_marker_on_missing_binary(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE", "auto")
    monkeypatch.setenv(
        "HELPER_DASHBOARD_HELPERCODE_BIN", "/nonexistent/helpercode-xyz-1234"
    )
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE_CMD", json.dumps(["{bin}"]))
    rt = HelperCodeRuntime()
    out = rt.invoke_operation("user_message", {"message": "Show me a CPU dashboard"})
    assert out["runtime_used"] == "mock_fallback"
    assert out.get("fallback_reason")
    assert "binary_missing" in out["fallback_reason"]
    assert out["type"] == "DashboardIntent"  # mock output kept


def test_auto_mode_falls_back_on_non_json(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(tmp_path / "helpercode", 'print("garbage")')
    _configure_stub(monkeypatch, stub, mode="auto")
    rt = HelperCodeRuntime()
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
    rt = HelperCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("shell_exec", {"cmd": "ls"})
    assert "unknown operation" in str(exc.value)


def test_developer_fix_is_blocked_from_user_flow(monkeypatch):
    """Normal /api/chat must never call developer_fix. The runtime
    wrapper rejects it unless the caller explicitly passes
    `developer=True`."""
    _clear_runtime_env(monkeypatch)
    rt = HelperCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("developer_fix", {"ticket_id": "tkt-1"})
    assert "developer-only" in str(exc.value)


def test_developer_fix_requires_developer_flag(monkeypatch, tmp_path):
    """Even when a stub would happily respond, the runtime refuses
    developer_fix unless `developer=True` is passed. This is where
    the user-vs-developer gate lives."""
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "helpercode",
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
    _configure_stub(monkeypatch, stub, mode="helpercode")
    rt = HelperCodeRuntime()

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
    rt = HelperCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_helper(BIG_GUY, {"ticket_id": "tkt-1"})
    assert "not a Helper agent" in str(exc.value)


def test_compat_invoke_helper_rejects_unknown_agent(monkeypatch):
    _clear_runtime_env(monkeypatch)
    rt = HelperCodeRuntime()
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
        tmp_path / "helpercode",
        'import json; print(json.dumps({"type":"UserResponse","message":"ok"}))',
    )
    _configure_stub(monkeypatch, stub, mode="helpercode")

    captured: dict = {}
    import app.helper.runtime as runtime_mod
    orig = runtime_mod.subprocess.run

    def spy(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return orig(*args, **kwargs)

    monkeypatch.setattr(runtime_mod.subprocess, "run", spy)

    rt = HelperCodeRuntime()
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
        tmp_path / "helpercode",
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
    _configure_stub(monkeypatch, stub, mode="helpercode")
    rt = HelperCodeRuntime()
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
        tmp_path / "helpercode",
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
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE", "helpercode")
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE_BIN", str(stub))
    monkeypatch.setenv(
        "HELPER_DASHBOARD_HELPERCODE_CMD",
        json.dumps(["{bin}", "--agent", "{agent}"]),
    )
    rt = HelperCodeRuntime()
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
    """If the operator sets HELPER_DASHBOARD_HELPERCODE_CMD to something
    that isn't a JSON array of strings, we log and fall back to the
    default template rather than crashing or shell-evaluating."""
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "helpercode",
        'import json; print(json.dumps({"type":"UserResponse","message":"ok"}))',
    )
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE", "helpercode")
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE_BIN", str(stub))
    # A shell-injection attempt.
    monkeypatch.setenv(
        "HELPER_DASHBOARD_HELPERCODE_CMD", "rm -rf / ; echo pwned"
    )
    rt = HelperCodeRuntime()
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
        tmp_path / "helpercode",
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
    _configure_stub(monkeypatch, stub, mode="helpercode")

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

    # Now switch to helpercode with a stub that returns a bogus patch.
    stub = _write_stub(
        tmp_path / "helpercode",
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
    _configure_stub(monkeypatch, stub, mode="helpercode")
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


def _frontmatter(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---"), f"{path.name} missing frontmatter"
    _, frontmatter, _ = text.split("---", 2)
    return frontmatter


def test_helper_agent_md_configs_lock_permissions():
    """Primary enforcement of Helper permissions is the agent .md
    frontmatter, in the official OpenCode schema (since 517ec41):
    `permission.edit: deny` and `permission.bash: deny`. The INTENT is
    unchanged — user-facing agents can never edit code or run shell."""
    for agent_name in HELPER_AGENTS:
        path = _REPO_ROOT / ".opencode" / "agent" / f"{agent_name}.md"
        frontmatter = _frontmatter(path)
        assert "permission:" in frontmatter, (
            f"{agent_name}: expected a `permission:` block"
        )
        for key in ("edit", "bash"):
            assert re.search(rf"^\s+{key}:\s*deny\s*$", frontmatter, re.M), (
                f"{agent_name}: expected `{key}: deny` in frontmatter"
            )
            assert f"{key}: allow" not in frontmatter, (
                f"{agent_name}: `{key}: allow` must never appear"
            )


def test_big_guy_md_marks_internal_only():
    """Big guy (hand-rolled CLI flavor) is a hidden subagent with edit
    access — official schema keys since 517ec41 (`hidden: true`
    replaced the non-standard `internal: true`)."""
    frontmatter = _frontmatter(_REPO_ROOT / ".opencode" / "agent" / f"{BIG_GUY}.md")
    assert "hidden: true" in frontmatter
    assert "mode: subagent" in frontmatter
    # Big guy is allowed edit (+ gated bash) — but he's not a Helper.
    assert "edit: allow" in frontmatter
    assert re.search(r"^\s+bash:\s*(ask|allow)\s*$", frontmatter, re.M)


def test_big_guy_devfix_md_is_primary_with_default_deny():
    """The real-OpenCode devfix agent MUST be `mode: primary`
    (`run --agent` on a subagent silently falls back to the default
    build agent) and its edit/bash pattern records MUST lead with the
    catch-all deny (matching is last-match-wins, so `"*": deny` first
    means everything not explicitly allowed is denied)."""
    frontmatter = _frontmatter(
        _REPO_ROOT / ".opencode" / "agent" / "big-guy-developer-fix.md"
    )
    assert re.search(r"^mode:\s*primary\s*$", frontmatter, re.M)
    assert "mode: subagent" not in frontmatter
    assert "hidden: true" in frontmatter
    lines = [ln.rstrip() for ln in frontmatter.splitlines()]
    for record in ("edit", "bash"):
        idx = lines.index(f"  {record}:")
        assert lines[idx + 1].strip() == '"*": deny', (
            f"permission.{record} must have the catch-all deny FIRST, "
            f"got {lines[idx + 1]!r}"
        )
    # Headless safety net: webfetch/websearch/external_directory/task
    # stay hard-denied for this agent.
    for key in ("webfetch", "websearch", "external_directory", "task"):
        assert re.search(rf"^\s+{key}:\s*deny\s*$", frontmatter, re.M), (
            f"expected `{key}: deny`"
        )

    # Bash allow-list is EXACTLY the test/check commands. `node *` was an
    # arbitrary-code hole (node -e '<js>' inside the worktree) and must
    # never come back.
    assert '"node *"' not in frontmatter
    assert "node *" not in frontmatter
    bash_block = frontmatter.split("  bash:\n", 1)[1].split("\n  read:", 1)[0]
    bash_allow = re.findall(r'^\s+"([^"]+)":\s*allow\s*$', bash_block, re.M)
    assert bash_allow == [
        "python3 -m pytest*", "pytest*", "npm test*", "npx tsc*", "npm run build*",
    ], bash_allow
    # Edit allow globs match the ABSOLUTE worktree path (`**/` prefix) and
    # cover exactly the render layer + tests.
    edit_block = frontmatter.split("  edit:\n", 1)[1].split("\n  bash:", 1)[0]
    edit_allow = re.findall(r'^\s+"([^"]+)":\s*allow\s*$', edit_block, re.M)
    assert edit_allow and all(g.startswith("**/") for g in edit_allow), edit_allow
    assert set(edit_allow) == {
        "**/backend/app/services/browser_evaluator.py",
        "**/frontend/lib/renderer.tsx",
        "**/frontend/lib/spec-schema.ts",
        "**/frontend/widget-toolkit/**",
        "**/tests/**",
    }, edit_allow


def test_opencode_json_locks_helper_permissions_and_gates_developer_fix():
    """Since 517ec41, opencode.json is minimal ($schema + instructions);
    per-agent permissions live in .opencode/agent/*.md and the command
    boundary is prose in .opencode/PROJECT.md + the runtime gate. This
    locks the layout so the old non-standard keys never come back."""
    cfg = json.loads((_REPO_ROOT / "opencode.json").read_text())
    assert cfg.get("$schema") == "https://opencode.ai/config.json"
    assert ".opencode/PROJECT.md" in cfg.get("instructions", [])
    assert "agents" not in cfg
    assert "boundaries" not in cfg

    project = (_REPO_ROOT / ".opencode" / "PROJECT.md").read_text()
    assert "Developer-only: developer-fix" in project

    # The developer gate itself is enforced in the runtime.
    assert "developer_fix" in DEVELOPER_OPERATIONS
    rt = HelperCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("developer_fix", {"instruction": "x"})
    assert "developer-only" in str(exc.value)


# ---------------------------------------------------------------------------
# DevFix routing: HELPER_DASHBOARD_DEVFIX_RUNTIME switches ONLY
# developer_fix between bin/helpercode and the real OpenCode CLI
# ---------------------------------------------------------------------------


_DEVFIX_SENTINEL = {
    "type": "DeveloperReport",
    "report_id": "sentinel",
    "status": "rejected",
    "summary": "sentinel report from the devfix runtime stub",
    "runtime_used": "opencode",
}


def test_devfix_routing_opencode_cli_switch(monkeypatch, tmp_path):
    """With HELPER_DASHBOARD_DEVFIX_RUNTIME=opencode-cli, developer_fix
    goes to the DevFix runtime while chat ops still hit _real."""
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELPER_DASHBOARD_DEVFIX_RUNTIME", "opencode-cli")
    stub = _write_stub(
        tmp_path / "helpercode",
        'import json; print(json.dumps({"type": "UserResponse", "message": "hi"}))',
    )
    _configure_stub(monkeypatch, stub, mode="helpercode")

    rt = HelperCodeRuntime()
    devfix_calls: list[tuple[str, str]] = []

    def fake_devfix_invoke(operation, agent, args):
        devfix_calls.append((operation, agent))
        return dict(_DEVFIX_SENTINEL)

    monkeypatch.setattr(rt._devfix, "invoke", fake_devfix_invoke)

    out = rt.invoke_operation(
        "developer_fix", {"instruction": "fix it"}, developer=True,
    )
    assert out["runtime_used"] == "opencode"
    assert out["report_id"] == "sentinel"
    assert devfix_calls == [("developer_fix", "big-guy-developer-agent")]

    # Chat op still goes through the hand-rolled CLI stub.
    chat = rt.invoke_operation("user_message", {"message": "hi"})
    assert chat["runtime_used"] == "helpercode"
    assert devfix_calls == [("developer_fix", "big-guy-developer-agent")]


def test_devfix_routing_default_stays_on_helpercode(monkeypatch, tmp_path):
    """With default env (no HELPER_DASHBOARD_DEVFIX_RUNTIME), the old
    behavior is byte-identical: developer_fix hits _real."""
    _clear_runtime_env(monkeypatch)
    stub = _write_stub(
        tmp_path / "helpercode",
        textwrap.dedent(
            """
            import json
            print(json.dumps({
                "type": "DeveloperReport",
                "report_id": "from-real",
                "status": "rejected",
                "summary": "hand-rolled CLI answered",
            }))
            """
        ),
    )
    _configure_stub(monkeypatch, stub, mode="helpercode")

    rt = HelperCodeRuntime()

    def _boom(operation, agent, args):  # pragma: no cover - must not run
        raise AssertionError("devfix runtime must not be used by default")

    monkeypatch.setattr(rt._devfix, "invoke", _boom)

    out = rt.invoke_operation(
        "developer_fix", {"instruction": "fix it"}, developer=True,
    )
    assert out["report_id"] == "from-real"
    assert out["runtime_used"] == "helpercode"


def test_devfix_routing_unknown_value_falls_back_to_helpercode(monkeypatch, tmp_path):
    """A typo'd switch value must fail safe (helpercode), never route
    to the paid CLI."""
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELPER_DASHBOARD_DEVFIX_RUNTIME", "opencode")  # wrong value
    stub = _write_stub(
        tmp_path / "helpercode",
        textwrap.dedent(
            """
            import json
            print(json.dumps({
                "type": "DeveloperReport",
                "report_id": "from-real",
                "status": "rejected",
                "summary": "hand-rolled CLI answered",
            }))
            """
        ),
    )
    _configure_stub(monkeypatch, stub, mode="helpercode")
    rt = HelperCodeRuntime()

    def _boom(operation, agent, args):  # pragma: no cover - must not run
        raise AssertionError("unknown switch value must not reach devfix")

    monkeypatch.setattr(rt._devfix, "invoke", _boom)
    out = rt.invoke_operation(
        "developer_fix", {"instruction": "fix it"}, developer=True,
    )
    assert out["runtime_used"] == "helpercode"


def test_devfix_routing_mock_mode_ignores_switch(monkeypatch):
    """mock mode stays fully offline even with the switch flipped."""
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE", "mock")
    monkeypatch.setenv("HELPER_DASHBOARD_DEVFIX_RUNTIME", "opencode-cli")
    rt = HelperCodeRuntime()

    def _boom(operation, agent, args):  # pragma: no cover - must not run
        raise AssertionError("mock mode must never reach devfix")

    monkeypatch.setattr(rt._devfix, "invoke", _boom)
    out = rt.invoke_operation(
        "developer_fix", {"instruction": "fix it"}, developer=True,
    )
    assert out["runtime_used"] == "mock"
