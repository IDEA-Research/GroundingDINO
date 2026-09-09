"""/api/devfix + /api/capabilities endpoints against a tmp storage dir.

The suite-wide conftest fixture already points
HELPER_DASHBOARD_DEVFIX_RUNS_DIR at tmp_path/devfix_runs, so runs
written through devfix_log here are exactly what the API reads back.

Every /api/devfix route is developer-gated: tests send
X-Developer-Token and assert 403 without it.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.helper import auto_fix, devfix_log, devfix_queue
from app.main import app

client = TestClient(app)

TOKEN = "test-dev-token-0123"
HDR = {"X-Developer-Token": TOKEN}


@pytest.fixture(autouse=True)
def _dev_token(monkeypatch):
    monkeypatch.setenv("HELPER_DASHBOARD_DEV_TOKEN", TOKEN)


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method,path", [
    ("get", "/api/devfix/runs"),
    ("get", "/api/devfix/runs/some-run"),
    ("get", "/api/devfix/runs/some-run/events"),
    ("get", "/api/devfix/runs/some-run/diff"),
    ("get", "/api/devfix/config"),
    ("post", "/api/devfix/runs"),
    ("post", "/api/devfix/runs/some-run/cancel"),
    ("post", "/api/devfix/runs/some-run/retry"),
    ("post", "/api/evaluate/run"),
    ("get", "/api/evaluate/reports/d1"),
    ("get", "/api/developer/tickets"),
])
def test_dev_routes_403_without_or_with_wrong_token(method, path, monkeypatch):
    body = {"instruction": "x", "dashboard_id": "d"} if method == "post" else None
    res = client.request(method.upper(), path, json=body)
    assert res.status_code == 403, (path, res.text)
    res = client.request(
        method.upper(), path, json=body, headers={"X-Developer-Token": "wrong"},
    )
    assert res.status_code == 403, (path, res.text)
    # Fail closed: token unset server-side -> even the right header is refused.
    monkeypatch.delenv("HELPER_DASHBOARD_DEV_TOKEN", raising=False)
    res = client.request(method.upper(), path, json=body, headers=HDR)
    assert res.status_code == 403, (path, res.text)


# ---------------------------------------------------------------------------
# capabilities (public)
# ---------------------------------------------------------------------------


def test_capabilities_public_booleans(monkeypatch):
    monkeypatch.delenv("HELPER_DASHBOARD_DEVFIX_LOCAL_RUNS", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_ALLOW_SIMULATION", raising=False)
    res = client.get("/api/capabilities")  # no header
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {
        "dev_token_configured", "devfix_local_runs", "auto_fix_enabled",
        "simulation_available", "evaluate_enabled",
    }
    assert all(isinstance(v, bool) for v in body.values())
    assert body["dev_token_configured"] is True
    assert body["devfix_local_runs"] is False  # kill switch off
    assert body["auto_fix_enabled"] is False   # conftest kill switch
    assert body["simulation_available"] is False
    assert body["evaluate_enabled"] is True
    assert client.get("/api/capabilities/").status_code == 200

    monkeypatch.delenv("HELPER_DASHBOARD_DEV_TOKEN", raising=False)
    monkeypatch.setenv("HELPER_DASHBOARD_ALLOW_SIMULATION", "1")
    body = client.get("/api/capabilities").json()
    assert body["dev_token_configured"] is False
    assert body["evaluate_enabled"] is False
    assert body["simulation_available"] is True


def test_capabilities_local_runs_true_when_fully_enabled(local_runs_env):
    assert client.get("/api/capabilities").json()["devfix_local_runs"] is True


# ---------------------------------------------------------------------------
# read side
# ---------------------------------------------------------------------------


def test_list_runs_empty():
    res = client.get("/api/devfix/runs", headers=HDR)
    assert res.status_code == 200
    assert res.json() == {"runs": []}


def test_config_read_only(monkeypatch):
    monkeypatch.delenv("HELPER_DASHBOARD_DEVFIX_LOCAL_RUNS", raising=False)
    res = client.get("/api/devfix/config", headers=HDR)
    assert res.status_code == 200
    cfg = res.json()
    assert cfg["local_runs_enabled"] is False
    assert cfg["local_runs_reason"] == "local_runs_disabled"
    assert "project_root" in cfg and "worktree_parent" in cfg
    assert isinstance(cfg["bin_exists"], bool)
    assert set(cfg["daily_quota"]) == {"limit", "used"}


def test_list_and_incremental_read(tmp_path):
    logger = devfix_log.start_run(
        "t-api", instruction="fix the header", model="m/x",
    )
    logger.note("worktree created: /tmp/wt")
    logger.note("step one")

    # list shows the run, newest first, with the additive meta fields.
    res = client.get("/api/devfix/runs", headers=HDR)
    assert res.status_code == 200
    runs = res.json()["runs"]
    assert [r["run_id"] for r in runs] == [logger.run_id]
    assert runs[0]["status"] == "running"
    assert runs[0]["instruction_preview"] == "fix the header"
    assert runs[0]["instruction"] == "fix the header"
    assert runs[0]["source"] == "auto"
    assert runs[0]["applied"] == {"status": "n/a", "files": []}

    # full read from cursor 0.
    res = client.get(f"/api/devfix/runs/{logger.run_id}?after=0", headers=HDR)
    assert res.status_code == 200
    body = res.json()
    assert body["meta"]["run_id"] == logger.run_id
    assert len(body["lines"]) == 2
    assert body["next_cursor"] == 2

    # nothing new yet — incremental read is empty.
    res = client.get(f"/api/devfix/runs/{logger.run_id}?after=2", headers=HDR)
    assert res.json()["lines"] == []

    # append + finish, then tail from the old cursor: only the new lines.
    logger.note("step two")
    logger.finish("resolved")
    res = client.get(f"/api/devfix/runs/{logger.run_id}?after=2", headers=HDR)
    body = res.json()
    assert len(body["lines"]) == 2  # "step two" + finish line
    assert "step two" in body["lines"][0]
    assert body["next_cursor"] == 4
    assert body["meta"]["status"] == "success"
    assert body["meta"]["report_status"] == "resolved"


def test_events_and_diff_endpoints(monkeypatch):
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "planted-secret-value-77")
    logger = devfix_log.start_run("t-ev", instruction="x", model="m")
    logger.event({"type": "step_start", "timestamp": 1, "sessionID": "s", "part": {}})
    logger.event({"type": "text", "timestamp": 2, "sessionID": "s",
                  "part": {"text": "token planted-secret-value-77 here"}})
    res = client.get(f"/api/devfix/runs/{logger.run_id}/events?after=0", headers=HDR)
    assert res.status_code == 200
    body = res.json()
    assert [e["type"] for e in body["events"]] == ["step_start", "text"]
    assert "planted-secret-value-77" not in res.text
    assert body["next_cursor"] == 2
    res = client.get(f"/api/devfix/runs/{logger.run_id}/events?after=2", headers=HDR)
    assert res.json() == {"events": [], "next_cursor": 2}

    logger.finish("resolved")
    logger.save_diff(
        "diff --git a/frontend/widget-toolkit/New.tsx b/frontend/widget-toolkit/New.tsx\n"
        "new file mode 100644\n--- /dev/null\n+++ b/frontend/widget-toolkit/New.tsx\n"
        "@@ -0,0 +1 @@\n+export const k = 'planted-secret-value-77';\n",
        [{"path": "frontend/widget-toolkit/New.tsx", "additions": 1, "deletions": 0, "status": "added"}],
    )
    res = client.get(f"/api/devfix/runs/{logger.run_id}/diff", headers=HDR)
    assert res.status_code == 200
    body = res.json()
    assert body["files"] == [{"path": "frontend/widget-toolkit/New.tsx", "additions": 1, "deletions": 0, "status": "added"}]
    assert body["patch"].startswith("diff --git")
    assert "planted-secret-value-77" not in body["patch"]


def test_read_unknown_run_is_404():
    for path in ("", "/events", "/diff"):
        res = client.get(f"/api/devfix/runs/no-such-run-20990101T000000{path}", headers=HDR)
        assert res.status_code == 404, path


def test_read_malformed_run_id_is_404():
    # Traversal-shaped and otherwise malformed ids must 404, never
    # touch the filesystem outside the store.
    for bad in ("..%2F..%2Fetc%2Fpasswd", "a.b", "..", "run%3Brm"):
        res = client.get(f"/api/devfix/runs/{bad}", headers=HDR)
        assert res.status_code == 404, bad
        res = client.post(f"/api/devfix/runs/{bad}/cancel", headers=HDR)
        assert res.status_code in (404, 409), bad


# ---------------------------------------------------------------------------
# execution side
# ---------------------------------------------------------------------------


class _Store:
    def __init__(self):
        self.tickets = {}

    def save_ticket(self, t):
        self.tickets[t.ticket_id] = t

    def load_ticket(self, ticket_id):
        return self.tickets.get(ticket_id)


class _Runtime:
    def __init__(self, status="resolved", gate: threading.Event | None = None):
        self.status = status
        self.gate = gate
        self.calls = []
        self.started = threading.Event()

    def invoke_operation(self, operation, args, *, developer=False):
        self.calls.append(dict(args))
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(timeout=20)
        return {"type": "DeveloperReport", "report_id": "r", "status": self.status,
                "summary": "stub", "actions_taken": [], "tests_run": []}


def _wait_terminal(run_id, timeout_s=15.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        meta = devfix_log.read_meta(run_id)
        if meta and meta["status"] in devfix_log.TERMINAL_STATUSES:
            return meta
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish")


def _wait_idle(timeout_s=15.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not devfix_queue.is_run_active() and not devfix_queue.snapshot()["pending_run_ids"]:
            return
        time.sleep(0.02)
    raise AssertionError("queue did not drain")


@pytest.fixture()
def local_runs_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HELPER_DASHBOARD_DEVFIX_LOCAL_RUNS", "1")
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE", "helpercode")
    monkeypatch.setenv("HELPER_DASHBOARD_DEVFIX_RUNTIME", "opencode-cli")
    monkeypatch.setenv("HELPER_DASHBOARD_DEVFIX_BIN", sys.executable)
    monkeypatch.delenv("HELPER_DASHBOARD_AUTO_FIX_DAILY_QUOTA", raising=False)
    writable = tmp_path / "w" / "f.py"
    writable.parent.mkdir(parents=True)
    writable.write_text("o", encoding="utf-8")
    protected = tmp_path / "p" / "anomaly_core.py"
    protected.parent.mkdir(parents=True)
    protected.write_text("c", encoding="utf-8")
    monkeypatch.setattr(auto_fix, "_writable_file_paths", lambda: [writable])
    monkeypatch.setattr(auto_fix, "_writable_dir_paths", lambda: [tmp_path / "toolkit"])
    monkeypatch.setattr(auto_fix, "_protected_paths", lambda: [protected])
    monkeypatch.setattr(auto_fix, "_AUDIT_DIR", tmp_path / "audit")
    auto_fix.reset_quota_for_tests()
    devfix_queue.reset_for_tests()
    gates: list[threading.Event] = []
    yield {"gates": gates}
    # Release every gate a test opened so no worker thread outlives the
    # test (a blocked worker would hold the single-flight lock and
    # contaminate the next test's queue).
    for g in gates:
        g.set()
    _wait_idle()
    devfix_queue.reset_for_tests()
    auto_fix.reset_quota_for_tests()


def test_start_refused_409_when_kill_switch_off(monkeypatch):
    monkeypatch.delenv("HELPER_DASHBOARD_DEVFIX_LOCAL_RUNS", raising=False)
    res = client.post("/api/devfix/runs", json={"instruction": "fix it"}, headers=HDR)
    assert res.status_code == 409
    assert res.json() == {"detail": "本機執行層未啟用", "reason": "local_runs_disabled"}
    assert client.get("/api/devfix/runs", headers=HDR).json()["runs"] == []
    # retry + cancel are behind the same switch.
    res = client.post("/api/devfix/runs/any-run/retry", headers=HDR)
    assert res.status_code == 409 and res.json()["reason"] == "local_runs_disabled"
    res = client.post("/api/devfix/runs/any-run/cancel", headers=HDR)
    assert res.status_code == 409 and res.json()["reason"] == "local_runs_disabled"


def test_start_refused_409_in_mock_mode(monkeypatch):
    monkeypatch.setenv("HELPER_DASHBOARD_DEVFIX_LOCAL_RUNS", "1")
    monkeypatch.setenv("HELPER_DASHBOARD_HELPERCODE", "mock")
    res = client.post("/api/devfix/runs", json={"instruction": "fix it"}, headers=HDR)
    assert res.status_code == 409
    assert res.json()["detail"] == "本機執行層未啟用"
    assert res.json()["reason"] == "mock_mode"


def test_start_body_rejects_commands_paths_and_model(local_runs_env):
    devfix_queue.set_backends_for_tests(runtime=_Runtime(), store=_Store())
    for extra in ({"command": "rm -rf /"}, {"path": "/etc"}, {"model": "x"}, {"argv": ["a"]}):
        res = client.post("/api/devfix/runs", json={"instruction": "ok", **extra}, headers=HDR)
        assert res.status_code == 422, extra
    assert client.post("/api/devfix/runs", json={"title": "no instruction"}, headers=HDR).status_code == 422
    assert client.post("/api/devfix/runs", json={"instruction": ""}, headers=HDR).status_code == 422
    assert client.post("/api/devfix/runs", json={"instruction": "   "}, headers=HDR).status_code == 422
    assert client.post("/api/devfix/runs", json={"instruction": "x" * 2001}, headers=HDR).status_code == 422
    assert client.get("/api/devfix/runs", headers=HDR).json()["runs"] == []


def test_start_queue_cancel_retry_flow(local_runs_env):
    store = _Store()
    gate = threading.Event()
    local_runs_env["gates"].append(gate)
    rt = _Runtime(status="rejected", gate=gate)
    devfix_queue.set_backends_for_tests(runtime=rt, store=store)

    res = client.post("/api/devfix/runs", json={"instruction": "first task", "title": "T1"}, headers=HDR)
    assert res.status_code == 202, res.text
    first = res.json()
    assert set(first) == {"run_id", "ticket_id", "status"}
    assert first["status"] == "queued"
    assert rt.started.wait(timeout=10)

    # Second one queues behind the active run; history shows both.
    res = client.post("/api/devfix/runs", json={"instruction": "second task"}, headers=HDR)
    assert res.status_code == 202
    second = res.json()
    runs = {r["run_id"]: r for r in client.get("/api/devfix/runs", headers=HDR).json()["runs"]}
    assert runs[first["run_id"]]["status"] == "running"
    assert runs[second["run_id"]]["status"] == "queued"
    assert runs[first["run_id"]]["source"] == "manual"
    assert runs[first["run_id"]]["title"] == "T1"
    cfg = client.get("/api/devfix/config", headers=HDR).json()
    assert cfg["queue"]["active_run_id"] == first["run_id"]
    assert cfg["queue"]["pending_run_ids"] == [second["run_id"]]

    # Retry refused while a run is active.
    res = client.post(f"/api/devfix/runs/{first['run_id']}/retry", headers=HDR)
    assert res.status_code == 409 and res.json()["reason"] == "run_active"

    # Cancel the queued one -> removed, cancelled/user_cancel.
    res = client.post(f"/api/devfix/runs/{second['run_id']}/cancel", headers=HDR)
    assert res.status_code == 202, res.text
    assert res.json()["result"] == "removed"
    meta = client.get(f"/api/devfix/runs/{second['run_id']}", headers=HDR).json()["meta"]
    assert meta["status"] == "cancelled" and meta["error_kind"] == "user_cancel"

    # Cancel of a run that is "running" on disk but has no live CLI
    # process here (the stub never spawned one) is refused honestly.
    res = client.post(f"/api/devfix/runs/{first['run_id']}/cancel", headers=HDR)
    assert res.status_code == 409 and res.json()["reason"] == "no_live_process"

    gate.set()
    meta = _wait_terminal(first["run_id"])
    assert meta["status"] == "waiting"
    _wait_idle()

    # Cancel of a finished run -> 409 not_active; unknown -> 404.
    res = client.post(f"/api/devfix/runs/{first['run_id']}/cancel", headers=HDR)
    assert res.status_code == 409 and res.json()["reason"] == "not_active"
    assert client.post("/api/devfix/runs/no-such-run-1/cancel", headers=HDR).status_code == 404

    # Retry now succeeds and links to the parent.
    res = client.post(f"/api/devfix/runs/{first['run_id']}/retry", headers=HDR)
    assert res.status_code == 202, res.text
    child = res.json()
    cmeta = _wait_terminal(child["run_id"])
    assert cmeta["parent_run_id"] == first["run_id"]
    assert cmeta["attempt"] == 2
    assert cmeta["instruction"] == "first task"
    assert client.post("/api/devfix/runs/no-such-run-2/retry", headers=HDR).status_code == 404


def test_queue_full_is_409(local_runs_env):
    gate = threading.Event()
    local_runs_env["gates"].append(gate)
    rt = _Runtime(gate=gate)
    devfix_queue.set_backends_for_tests(runtime=rt, store=_Store())
    assert client.post("/api/devfix/runs", json={"instruction": "active"}, headers=HDR).status_code == 202
    assert rt.started.wait(timeout=10)
    for i in range(5):
        assert client.post("/api/devfix/runs", json={"instruction": f"q{i}"}, headers=HDR).status_code == 202
    res = client.post("/api/devfix/runs", json={"instruction": "overflow"}, headers=HDR)
    assert res.status_code == 409 and res.json()["reason"] == "queue_full"
    gate.set()


def test_quota_exhausted_is_409(local_runs_env, monkeypatch):
    monkeypatch.setenv("HELPER_DASHBOARD_AUTO_FIX_DAILY_QUOTA", "0")
    devfix_queue.set_backends_for_tests(runtime=_Runtime(), store=_Store())
    res = client.post("/api/devfix/runs", json={"instruction": "x"}, headers=HDR)
    assert res.status_code == 409 and res.json()["reason"] == "quota_exhausted"
