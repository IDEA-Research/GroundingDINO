"""DevFixOpenCodeCLIRuntime tests with a FAKE opencode CLI.

The fake CLI is a tmp_path python script that prints canned JSONL
(mixed with garbage non-JSON lines) and exits 0/1, so the full
subprocess + stream-parse + report-extraction path runs without any
network or LLM. devfix_worktree is stubbed out for speed — its own
behavior is covered in test_worktree.py.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest

from app.helper import devfix_worktree as dw
from app.helper.runtime import (
    DevFixOpenCodeCLIRuntime,
    RuntimeError_,
    _HelperCodeInvalidOutput,
    _HelperCodeUnavailable,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_worktree(tmp_path, monkeypatch):
    """Replace devfix_worktree's lifecycle with cheap stubs and record
    every call."""
    calls: dict[str, list] = {
        "create": [], "remove": [], "copy_back": [], "sweep": [],
    }
    project_dir = tmp_path / "wt" / "helper-dashboard"
    project_dir.mkdir(parents=True)
    handle = dw.WorktreeHandle(
        ticket_id="t-stub",
        git_root=tmp_path,
        worktree_root=tmp_path / "wt",
        project_dir=project_dir,
    )

    def fake_create(ticket_id, *, payload=None, git_root=None):
        calls["create"].append((ticket_id, payload))
        return handle

    monkeypatch.setattr(dw, "create", fake_create)
    monkeypatch.setattr(
        dw, "remove", lambda h: calls["remove"].append(h.worktree_root)
    )
    monkeypatch.setattr(
        dw, "sweep_stale",
        lambda *a, **k: calls["sweep"].append(True) or [],
    )
    monkeypatch.setattr(dw, "collect_changes", lambda h: [])
    monkeypatch.setattr(
        dw, "copy_back",
        lambda h, allowlist_check=None: calls["copy_back"].append(True) or [],
    )
    return calls


def _make_fake_cli(
    tmp_path: Path,
    stdout_text: str,
    *,
    stderr_text: str = "",
    rc: int = 0,
    sleep_s: float = 0.0,
) -> Path:
    """A python script that dumps its argv, checks env hygiene, prints
    the canned streams, and exits with `rc`."""
    (tmp_path / "cli-out.txt").write_text(stdout_text, encoding="utf-8")
    (tmp_path / "cli-err.txt").write_text(stderr_text, encoding="utf-8")
    script = tmp_path / "fake-opencode"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys, time\n"
        f"base = {str(tmp_path)!r}\n"
        "with open(os.path.join(base, 'cli-argv.json'), 'w') as f:\n"
        "    json.dump({'argv': sys.argv, 'env_has_openrouter_key':\n"
        "               'OPENROUTER_API_KEY' in os.environ,\n"
        "               'cwd': os.getcwd()}, f)\n"
        f"time.sleep({sleep_s!r})\n"
        "sys.stdout.write(open(os.path.join(base, 'cli-out.txt')).read())\n"
        "sys.stderr.write(open(os.path.join(base, 'cli-err.txt')).read())\n"
        f"sys.exit({rc!r})\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


@pytest.fixture()
def devfix_env(tmp_path, monkeypatch):
    def configure(script: Path, *, timeout: str | None = None):
        monkeypatch.setenv("HELPER_DASHBOARD_DEVFIX_BIN", str(script))
        monkeypatch.setenv("HELPER_DASHBOARD_DEVFIX_HOME", str(tmp_path))
        monkeypatch.delenv("HELPER_DASHBOARD_DEVFIX_PATH", raising=False)
        if timeout is not None:
            monkeypatch.setenv(
                "HELPER_DASHBOARD_DEVFIX_TIMEOUT_SECONDS", timeout
            )
        else:
            monkeypatch.delenv(
                "HELPER_DASHBOARD_DEVFIX_TIMEOUT_SECONDS", raising=False
            )
        # Prove the child env is built explicitly, not inherited.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-must-not-leak")
    return configure


def _ev(type_: str, **part) -> str:
    return json.dumps({
        "type": type_, "timestamp": 1, "sessionID": "s", "part": part,
    })


_REPORT = {
    "type": "DeveloperReport",
    "report_id": "r1",
    "ticket_id": "t-9",
    "instruction": None,
    "actions_taken": ["edited frontend/lib/renderer.tsx"],
    "tests_run": ["python3 -m pytest tests/spec_validation -q"],
    "summary": "fixed it",
    "status": "resolved",
}

_ARGS = {"ticket_id": "t-9", "ticket": {}, "instruction": "", "auto": True}


def _invoke(args=None):
    return DevFixOpenCodeCLIRuntime().invoke(
        "developer_fix", "big-guy-developer-agent", args or dict(_ARGS),
    )


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_happy_fence_returns_report(tmp_path, monkeypatch, stub_worktree, devfix_env):
    final_text = (
        "All done. Here is the report:\n\n```json\n"
        + json.dumps(_REPORT, indent=2)
        + "\n```"
    )
    stdout = "\n".join([
        "some non-JSON notice line from the CLI",
        _ev("step_start"),
        _ev("tool_use", tool="read"),
        "another garbage line {not json",
        _ev("text", text="intermediate thoughts"),
        _ev("step_finish"),
        _ev("text", text=final_text),
        "",
    ])
    script = _make_fake_cli(tmp_path, stdout)
    devfix_env(script)

    out = _invoke()

    assert out["type"] == "DeveloperReport"
    assert out["report_id"] == "r1"
    assert out["runtime_used"] == "opencode"
    # resolved -> copy_back ran; worktree always removed; sweep ran.
    assert stub_worktree["copy_back"] == [True]
    assert len(stub_worktree["remove"]) == 1
    assert stub_worktree["sweep"] == [True]
    # Payload passed to create() has the RealHelperCodeRuntime stdin shape.
    (ticket_id, payload) = stub_worktree["create"][0]
    assert ticket_id == "t-9"
    assert payload == {
        "operation": "developer_fix",
        "agent": "big-guy-developer-agent",
        "command": "developer-fix",
        "args": dict(_ARGS),
    }

    meta = json.loads((tmp_path / "cli-argv.json").read_text())
    argv = meta["argv"]
    assert "--auto" not in argv, "NEVER pass --auto (headless ask must reject)"
    assert argv[argv.index("--agent") + 1] == "big-guy-developer-fix"
    assert argv[argv.index("--format") + 1] == "json"
    assert argv[-1].startswith("Operation developer_fix.")  # message positional
    assert meta["env_has_openrouter_key"] is False, "env must not be inherited"


def test_non_resolved_report_skips_copy_back(
    tmp_path, monkeypatch, stub_worktree, devfix_env,
):
    report = dict(_REPORT, status="rejected")
    stdout = _ev(
        "text", text="```json\n" + json.dumps(report) + "\n```",
    )
    script = _make_fake_cli(tmp_path, stdout)
    devfix_env(script)

    out = _invoke()
    assert out["status"] == "rejected"
    assert stub_worktree["copy_back"] == []
    assert len(stub_worktree["remove"]) == 1


def test_whole_text_as_json_fallback(tmp_path, monkeypatch, stub_worktree, devfix_env):
    report = dict(_REPORT, status="needs_context")
    script = _make_fake_cli(tmp_path, _ev("text", text=json.dumps(report)))
    devfix_env(script)
    out = _invoke()
    assert out["type"] == "DeveloperReport"
    assert out["status"] == "needs_context"
    assert stub_worktree["copy_back"] == []


# ---------------------------------------------------------------------------
# failure classification
# ---------------------------------------------------------------------------


def test_no_fence_no_json_is_non_json(tmp_path, monkeypatch, stub_worktree, devfix_env):
    script = _make_fake_cli(
        tmp_path, _ev("text", text="I did things but forgot the report."),
    )
    devfix_env(script)
    with pytest.raises(_HelperCodeInvalidOutput) as exc:
        _invoke()
    assert exc.value.kind == "non_json"
    assert len(stub_worktree["remove"]) == 1  # worktree cleaned up anyway


def test_fence_without_type_is_missing_type(
    tmp_path, monkeypatch, stub_worktree, devfix_env,
):
    script = _make_fake_cli(
        tmp_path, _ev("text", text='```json\n{"status": "resolved"}\n```'),
    )
    devfix_env(script)
    with pytest.raises(_HelperCodeInvalidOutput) as exc:
        _invoke()
    assert exc.value.kind == "missing_type"


def test_no_text_event_is_empty_output(tmp_path, monkeypatch, stub_worktree, devfix_env):
    stdout = "\n".join([_ev("step_start"), _ev("step_finish"), "notice"])
    script = _make_fake_cli(tmp_path, stdout)
    devfix_env(script)
    with pytest.raises(_HelperCodeInvalidOutput) as exc:
        _invoke()
    assert exc.value.kind == "empty_output"


def test_rc1_with_402_stderr_is_credits_exhausted(
    tmp_path, monkeypatch, stub_worktree, devfix_env,
):
    script = _make_fake_cli(
        tmp_path, "",
        stderr_text="Error: HTTP 402 Payment Required — credits exhausted",
        rc=1,
    )
    devfix_env(script)
    with pytest.raises(_HelperCodeInvalidOutput) as exc:
        _invoke()
    assert exc.value.kind == "credits_exhausted"
    assert len(stub_worktree["remove"]) == 1


def test_rc1_with_error_event_classified(
    tmp_path, monkeypatch, stub_worktree, devfix_env,
):
    stdout = json.dumps({
        "type": "error", "timestamp": 1, "sessionID": "s",
        "part": {"message": "provider rate limit exceeded (HTTP 429)"},
    })
    script = _make_fake_cli(tmp_path, stdout, rc=1)
    devfix_env(script)
    with pytest.raises(_HelperCodeInvalidOutput) as exc:
        _invoke()
    assert exc.value.kind == "rate_limited"


def test_timeout_kills_process_group(tmp_path, monkeypatch, stub_worktree, devfix_env):
    script = _make_fake_cli(tmp_path, _ev("text", text="late"), sleep_s=30.0)
    devfix_env(script, timeout="1")
    t0 = time.monotonic()
    with pytest.raises(_HelperCodeUnavailable) as exc:
        _invoke()
    elapsed = time.monotonic() - t0
    assert exc.value.kind == "timeout"
    assert elapsed < 15, "SIGKILL to the process group must end the run fast"
    assert len(stub_worktree["remove"]) == 1


def test_missing_binary_is_binary_missing(tmp_path, monkeypatch, stub_worktree):
    monkeypatch.setenv(
        "HELPER_DASHBOARD_DEVFIX_BIN", str(tmp_path / "no-such-opencode")
    )
    with pytest.raises(_HelperCodeUnavailable) as exc:
        _invoke()
    assert exc.value.kind == "binary_missing"
    assert stub_worktree["create"] == []  # refused before any worktree work


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def test_wrong_operation_refused(tmp_path, stub_worktree):
    with pytest.raises(RuntimeError_):
        DevFixOpenCodeCLIRuntime().invoke(
            "user_message", "helper-chat-agent", {},
        )
    assert stub_worktree["create"] == []


def test_wrong_agent_event_flags_subagent_fallback(
    tmp_path, monkeypatch, stub_worktree, devfix_env,
):
    stdout = "\n".join([
        json.dumps({
            "type": "step_start", "timestamp": 1, "sessionID": "s",
            "part": {"agent": "build"},
        }),
        _ev("text", text="```json\n" + json.dumps(_REPORT) + "\n```"),
    ])
    script = _make_fake_cli(tmp_path, stdout)
    devfix_env(script)
    with pytest.raises(_HelperCodeInvalidOutput) as exc:
        _invoke()
    assert exc.value.kind == "missing_type"
    assert "fallback" in str(exc.value)
    assert stub_worktree["copy_back"] == []


def test_resolved_with_offlist_changes_raises_runtime_error(
    tmp_path, monkeypatch, stub_worktree, devfix_env,
):
    def bad_copy_back(h, allowlist_check=None):
        raise dw.CopyBackViolation(["backend/app/main.py"])

    monkeypatch.setattr(dw, "copy_back", bad_copy_back)
    script = _make_fake_cli(
        tmp_path, _ev("text", text="```json\n" + json.dumps(_REPORT) + "\n```"),
    )
    devfix_env(script)
    with pytest.raises(RuntimeError_) as exc:
        _invoke()
    assert "devfix changes outside allow-list" in str(exc.value)
    assert "backend/app/main.py" in str(exc.value)
    assert len(stub_worktree["remove"]) == 1  # cleanup still happens


# ---------------------------------------------------------------------------
# live run log (devfix_log integration — the conftest autouse fixture
# points HELPER_DASHBOARD_DEVFIX_RUNS_DIR at tmp_path / "devfix_runs")
# ---------------------------------------------------------------------------


def _only_run_dir(tmp_path: Path) -> Path:
    root = tmp_path / "devfix_runs"
    dirs = [d for d in root.iterdir() if d.is_dir()]
    assert len(dirs) == 1, f"expected exactly one run dir, got {dirs}"
    return dirs[0]


def test_happy_run_streams_live_log(tmp_path, monkeypatch, stub_worktree, devfix_env):
    final_text = "Done.\n```json\n" + json.dumps(_REPORT) + "\n```"
    stdout = "\n".join([
        "garbage notice line",
        _ev("step_start"),
        _ev("reasoning", text="let me look at the widget toolkit first"),
        _ev("tool_use", tool="read",
            state={"status": "completed", "title": "frontend/lib/api.ts"}),
        _ev("text", text=final_text),
        "",
    ])
    script = _make_fake_cli(tmp_path, stdout)
    devfix_env(script)

    out = _invoke()
    assert out["status"] == "resolved"

    run_dir = _only_run_dir(tmp_path)
    assert run_dir.name.startswith("t-9-")

    # --thinking must be on the argv so reasoning events exist at all.
    meta_cli = json.loads((tmp_path / "cli-argv.json").read_text())
    assert "--thinking" in meta_cli["argv"]

    # events.jsonl: the raw JSON events, garbage lines skipped.
    raw_events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [e["type"] for e in raw_events] == [
        "step_start", "reasoning", "tool_use", "text",
    ]

    # meta.json reached the terminal status.
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "resolved"
    assert meta["ticket_id"] == "t-9"
    assert meta["finished_at"] is not None

    # log.txt: rendered lifecycle + event lines, in order.
    log = (run_dir / "log.txt").read_text(encoding="utf-8")
    lines = log.splitlines()
    assert any("worktree created:" in line for line in lines)
    assert any("💭 let me look at the widget toolkit first" in line
               for line in lines)
    assert any("📖 read frontend/lib/api.ts ✓" in line for line in lines)
    assert any(line.split(" ", 1)[1].startswith("💬") for line in lines
               if " " in line)
    assert any("copy-back: 0 files applied" in line for line in lines)
    assert lines[-1].endswith("🏁 run finished: resolved")


def test_rejected_run_logs_rejected_no_copyback_note(
    tmp_path, monkeypatch, stub_worktree, devfix_env,
):
    report = dict(_REPORT, status="rejected")
    stdout = _ev("text", text="```json\n" + json.dumps(report) + "\n```")
    script = _make_fake_cli(tmp_path, stdout)
    devfix_env(script)

    out = _invoke()
    assert out["status"] == "rejected"

    run_dir = _only_run_dir(tmp_path)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "rejected"
    log = (run_dir / "log.txt").read_text(encoding="utf-8")
    assert "copy-back" not in log
    assert "🏁 run finished: rejected" in log


def test_timeout_marks_run_failed_timeout(
    tmp_path, monkeypatch, stub_worktree, devfix_env,
):
    script = _make_fake_cli(tmp_path, _ev("text", text="late"), sleep_s=20.0)
    devfix_env(script, timeout="1")
    with pytest.raises(_HelperCodeUnavailable):
        _invoke()

    run_dir = _only_run_dir(tmp_path)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed"
    assert meta["error_kind"] == "timeout"
    log = (run_dir / "log.txt").read_text(encoding="utf-8")
    assert "🏁 run finished: failed (timeout)" in log


def test_non_json_final_message_marks_run_failed(
    tmp_path, monkeypatch, stub_worktree, devfix_env,
):
    script = _make_fake_cli(
        tmp_path, _ev("text", text="I did things but forgot the report."),
    )
    devfix_env(script)
    with pytest.raises(_HelperCodeInvalidOutput):
        _invoke()

    run_dir = _only_run_dir(tmp_path)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed"
    assert meta["error_kind"] == "non_json"
