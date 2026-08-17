"""/api/devfix endpoints against a tmp storage dir.

The suite-wide conftest fixture already points
HELPER_DASHBOARD_DEVFIX_RUNS_DIR at tmp_path/devfix_runs, so runs
written through devfix_log here are exactly what the API reads back.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.helper import devfix_log
from app.main import app

client = TestClient(app)


def test_list_runs_empty():
    res = client.get("/api/devfix/runs")
    assert res.status_code == 200
    assert res.json() == {"runs": []}


def test_list_and_incremental_read(tmp_path):
    logger = devfix_log.start_run(
        "t-api", instruction="fix the header", model="m/x",
    )
    logger.note("worktree created: /tmp/wt")
    logger.note("step one")

    # list shows the run, newest first.
    res = client.get("/api/devfix/runs")
    assert res.status_code == 200
    runs = res.json()["runs"]
    assert [r["run_id"] for r in runs] == [logger.run_id]
    assert runs[0]["status"] == "running"
    assert runs[0]["instruction_preview"] == "fix the header"

    # full read from cursor 0.
    res = client.get(f"/api/devfix/runs/{logger.run_id}?after=0")
    assert res.status_code == 200
    body = res.json()
    assert body["meta"]["run_id"] == logger.run_id
    assert len(body["lines"]) == 2
    assert body["next_cursor"] == 2

    # nothing new yet — incremental read is empty.
    res = client.get(f"/api/devfix/runs/{logger.run_id}?after=2")
    assert res.json()["lines"] == []

    # append + finish, then tail from the old cursor: only the new lines.
    logger.note("step two")
    logger.finish("resolved")
    res = client.get(f"/api/devfix/runs/{logger.run_id}?after=2")
    body = res.json()
    assert len(body["lines"]) == 2  # "step two" + finish line
    assert "step two" in body["lines"][0]
    assert body["next_cursor"] == 4
    assert body["meta"]["status"] == "resolved"


def test_read_unknown_run_is_404():
    res = client.get("/api/devfix/runs/no-such-run-20990101T000000")
    assert res.status_code == 404


def test_read_malformed_run_id_is_404():
    # Traversal-shaped and otherwise malformed ids must 404, never
    # touch the filesystem outside the store.
    for bad in ("..%2F..%2Fetc%2Fpasswd", "a.b", "..", "run%3Brm"):
        res = client.get(f"/api/devfix/runs/{bad}")
        assert res.status_code == 404, bad
