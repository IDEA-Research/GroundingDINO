"""Cancel registry, pid/pgid recording and cancel classification for
DevFixOpenCodeCLIRuntime — driven through a FAKE Popen so the full
`_run_in_worktree` path runs without a real subprocess (portable to the
Windows dev box; the Jetson target uses the real killpg path).

`os.getpgid` / `os.killpg` are monkeypatched: the fake pgid must never
reach a real process group.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from app.helper import devfix_worktree as dw
from app.helper import runtime as runtime_mod
from app.helper.runtime import DevFixOpenCodeCLIRuntime, _HelperCodeUnavailable


# ---------------------------------------------------------------------------
# fakes / fixtures
# ---------------------------------------------------------------------------


class _FakeProc:
    """Popen stand-in. `block` (an Event) makes wait() hang until the
    process is terminated/killed or the timeout elapses."""

    def __init__(self, stdout_text: str, rc: int = 0, block: threading.Event | None = None):
        self.pid = 4242
        self.stdout = io.StringIO(stdout_text)
        self.stderr = io.StringIO("")
        self.returncode: int | None = None
        self._rc = rc
        self._block = block
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        if self._block is not None and not self._block.wait(timeout=timeout):
            raise subprocess.TimeoutExpired(cmd="fake-opencode", timeout=timeout)
        if self.returncode is None:
            self.returncode = -15 if (self.terminated or self.killed) else self._rc
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if self._block is not None:
            self._block.set()

    def kill(self):
        self.killed = True
        if self._block is not None:
            self._block.set()


@pytest.fixture()
def stub_worktree(tmp_path, monkeypatch):
    calls: dict[str, list] = {"create": [], "remove": [], "copy_back": []}
    project_dir = tmp_path / "wt" / "helper-dashboard"
    project_dir.mkdir(parents=True)
    handle = dw.WorktreeHandle(
        ticket_id="t-stub", git_root=tmp_path, worktree_root=tmp_path / "wt",
        project_dir=project_dir,
    )
    monkeypatch.setattr(dw, "create", lambda ticket_id, *, payload=None, git_root=None: (
        calls["create"].append(ticket_id) or handle))
    monkeypatch.setattr(dw, "remove", lambda h: calls["remove"].append(h.worktree_root))
    monkeypatch.setattr(dw, "sweep_stale", lambda *a, **k: [])
    monkeypatch.setattr(dw, "collect_changes", lambda h: [])
    monkeypatch.setattr(dw, "copy_back", lambda h, allowlist_check=None: (
        calls["copy_back"].append(True) or []))
    return calls


@pytest.fixture()
def devfix_env(tmp_path, monkeypatch):
    # An existing executable file stands in for the OpenCode binary; it is
    # never launched because Popen is faked.
    monkeypatch.setenv("HELPER_DASHBOARD_DEVFIX_BIN", sys.executable)
    monkeypatch.setenv("HELPER_DASHBOARD_DEVFIX_HOME", str(tmp_path))
    monkeypatch.delenv("HELPER_DASHBOARD_DEVFIX_PATH", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_DEVFIX_TIMEOUT_SECONDS", raising=False)


@pytest.fixture()
def fake_pg(monkeypatch):
    """Fake process-group primitives: getpgid -> 777; killpg records the
    signal and, per `release_on`, terminates/kills the fake proc."""
    state = {"proc": None, "killpg": [], "release_on": {"term": True, "kill": True}}
    import signal

    def _getpgid(pid):
        return 777

    def _killpg(pgid, sig):
        state["killpg"].append((pgid, sig))
        proc = state["proc"]
        if proc is None:
            return
        is_kill = sig == getattr(signal, "SIGKILL", -9)
        if is_kill and state["release_on"]["kill"]:
            proc.kill()
        elif not is_kill and state["release_on"]["term"]:
            proc.terminate()

    monkeypatch.setattr(os, "getpgid", _getpgid, raising=False)
    monkeypatch.setattr(os, "killpg", _killpg, raising=False)
    return state


@pytest.fixture()
def spawn(monkeypatch, fake_pg):
    """Install a fake subprocess.Popen returning the given proc."""
    holder: dict = {"argv": None, "kwargs": None}

    def install(proc: _FakeProc):
        fake_pg["proc"] = proc

        def _popen(argv, **kwargs):
            holder["argv"] = list(argv)
            holder["kwargs"] = kwargs
            return proc

        monkeypatch.setattr(runtime_mod.subprocess, "Popen", _popen)
        return holder

    return install


def _ev(type_: str, **part) -> str:
    return json.dumps({"type": type_, "timestamp": 1, "sessionID": "s", "part": part})


_REPORT = {
    "type": "DeveloperReport", "report_id": "r1", "ticket_id": "t-9",
    "instruction": None, "actions_taken": ["edited renderer"],
    "tests_run": ["python3 -m pytest tests/spec_validation -q"],
    "summary": "fixed it", "status": "resolved",
}
_ARGS = {"ticket_id": "t-9", "ticket": {}, "instruction": "", "auto": True}


def _invoke(args=None):
    return DevFixOpenCodeCLIRuntime().invoke(
        "developer_fix", "big-guy-developer-agent", args or dict(_ARGS),
    )


def _only_run_dir(tmp_path: Path) -> Path:
    root = tmp_path / "devfix_runs"
    dirs = [d for d in root.iterdir() if d.is_dir()]
    assert len(dirs) == 1, dirs
    return dirs[0]


def _meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))


def _wait_for(pred, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_spawn_records_pid_pgid_and_success_meta(tmp_path, stub_worktree, devfix_env, spawn):
    stdout = "\n".join([
        _ev("tool_use", tool="read", state={"status": "completed",
            "input": {"filePath": str(tmp_path / "wt" / "helper-dashboard" / "frontend" / "lib" / "renderer.tsx")}}),
        _ev("tool_use", tool="bash", state={"status": "completed",
            "input": {"command": "python3 -m pytest tests/spec_validation -q"},
            "output": "7 passed in 1.0s", "metadata": {"exit": 0}}),
        _ev("text", text="```json\n" + json.dumps(_REPORT) + "\n```"),
        "",
    ])
    holder = spawn(_FakeProc(stdout))

    out = _invoke()

    assert out["status"] == "resolved"
    run_dir = _only_run_dir(tmp_path)
    assert out["run_id"] == run_dir.name
    meta = _meta(run_dir)
    assert meta["pid"] == 4242
    assert meta["pgid"] == 777
    assert meta["status"] == "success"
    assert meta["report_status"] == "resolved"
    assert meta["applied"] == {"status": "applied", "files": []}
    assert meta["report_summary"] == "fixed it"
    assert meta["actions_taken"] == ["edited renderer"]
    assert meta["tests_reported"] == ["python3 -m pytest tests/spec_validation -q"]
    # Tool-path relativized against the worktree project dir.
    assert meta["files_read"] == ["frontend/lib/renderer.tsx"]
    assert meta["tests"]["ran"] is True and meta["tests"]["passed"] == 7
    assert meta["verification"] == {"status": "passed"}
    # Diff captured on the terminal state (empty here: stub collect_changes).
    assert (run_dir / "diff.patch").exists()
    assert meta["diff_files"] == []
    # Registry is clean afterwards; Popen got a fresh session + no --auto.
    assert runtime_mod.active_devfix_run_ids() == []
    assert holder["kwargs"]["start_new_session"] is True
    assert "--auto" not in holder["argv"]
    assert len(stub_worktree["remove"]) == 1


def test_request_cancel_unknown_run_is_false():
    assert runtime_mod.request_devfix_cancel("no-such-run") is False
    assert runtime_mod.is_devfix_cancel_requested("no-such-run") is False


def test_cancel_running_run_sigterm_then_cancelled(tmp_path, stub_worktree, devfix_env, spawn, fake_pg):
    block = threading.Event()
    proc = _FakeProc("", rc=0, block=block)
    spawn(proc)
    result: dict = {}

    def _go():
        try:
            _invoke()
        except Exception as exc:  # noqa: BLE001 - captured for the assertion
            result["exc"] = exc

    t = threading.Thread(target=_go, daemon=True)
    t.start()
    assert _wait_for(lambda: bool(runtime_mod.active_devfix_run_ids())), "run never registered"
    run_id = runtime_mod.active_devfix_run_ids()[0]
    run_dir = _only_run_dir(tmp_path)
    assert run_dir.name == run_id
    assert _meta(run_dir)["status"] == "running"
    assert _meta(run_dir)["pid"] == 4242 and _meta(run_dir)["pgid"] == 777

    assert runtime_mod.request_devfix_cancel(run_id, grace_s=5.0) is True
    t.join(timeout=10)
    assert not t.is_alive()

    import signal
    # SIGTERM went to the process GROUP (fake killpg), which ended the run.
    assert fake_pg["killpg"] and fake_pg["killpg"][0][0] == 777
    assert fake_pg["killpg"][0][1] == getattr(signal, "SIGTERM")
    assert proc.terminated is True and proc.killed is False
    exc = result["exc"]
    assert isinstance(exc, _HelperCodeUnavailable) and exc.kind == "user_cancel"
    meta = _meta(run_dir)
    assert meta["status"] == "cancelled"
    assert meta["error_kind"] == "user_cancel"
    assert meta["report_status"] == "cancelled"
    assert meta["applied"]["status"] == "n/a"
    assert (run_dir / "diff.patch").exists()  # captured before removal
    assert len(stub_worktree["remove"]) == 1
    assert runtime_mod.active_devfix_run_ids() == []
    log = (run_dir / "log.txt").read_text(encoding="utf-8")
    assert "🏁 run finished: cancelled (user_cancel)" in log


def test_cancel_escalates_to_sigkill_after_grace(tmp_path, stub_worktree, devfix_env, spawn, fake_pg):
    fake_pg["release_on"]["term"] = False  # process ignores SIGTERM
    block = threading.Event()
    proc = _FakeProc("", rc=0, block=block)
    spawn(proc)
    result: dict = {}

    def _go():
        try:
            _invoke()
        except Exception as exc:  # noqa: BLE001
            result["exc"] = exc

    t = threading.Thread(target=_go, daemon=True)
    t.start()
    assert _wait_for(lambda: bool(runtime_mod.active_devfix_run_ids()))
    run_id = runtime_mod.active_devfix_run_ids()[0]
    t0 = time.monotonic()
    assert runtime_mod.request_devfix_cancel(run_id, grace_s=0.2) is True
    # Still registered (the process ignores SIGTERM) -> flag is visible.
    assert runtime_mod.is_devfix_cancel_requested(run_id) is True
    t.join(timeout=10)
    assert not t.is_alive()
    assert time.monotonic() - t0 >= 0.2
    assert proc.killed is True, "hard kill must follow the grace period"
    assert isinstance(result["exc"], _HelperCodeUnavailable)
    assert result["exc"].kind == "user_cancel"
    assert _meta(_only_run_dir(tmp_path))["status"] == "cancelled"


def test_timeout_uses_group_kill_and_marks_timeout(tmp_path, monkeypatch, stub_worktree, devfix_env, spawn, fake_pg):
    monkeypatch.setenv("HELPER_DASHBOARD_DEVFIX_TIMEOUT_SECONDS", "0.3")
    block = threading.Event()  # never released voluntarily
    proc = _FakeProc("", rc=0, block=block)
    spawn(proc)
    with pytest.raises(_HelperCodeUnavailable) as exc:
        _invoke()
    assert exc.value.kind == "timeout"
    import signal
    assert any(sig == getattr(signal, "SIGKILL", None) for _, sig in fake_pg["killpg"]) or proc.killed
    meta = _meta(_only_run_dir(tmp_path))
    assert meta["status"] == "timeout"
    assert meta["error_kind"] == "timeout"
    assert runtime_mod.active_devfix_run_ids() == []


def test_resumes_precreated_run_record(tmp_path, stub_worktree, devfix_env, spawn):
    from app.helper import devfix_log

    pre = devfix_log.start_run(
        "t-9", instruction="manual instruction", model="m", status="queued",
        source="manual", title="Manual title",
    )
    pre.close()
    spawn(_FakeProc(_ev("text", text="```json\n" + json.dumps(_REPORT) + "\n```") + "\n"))

    out = _invoke(dict(_ARGS, run_id=pre.run_id, auto=False))

    assert out["run_id"] == pre.run_id
    run_dir = _only_run_dir(tmp_path)  # no second record was created
    assert run_dir.name == pre.run_id
    meta = _meta(run_dir)
    assert meta["status"] == "success"
    assert meta["source"] == "manual"
    assert meta["title"] == "Manual title"
    assert meta["instruction"] == "manual instruction"
    assert meta["pid"] == 4242
