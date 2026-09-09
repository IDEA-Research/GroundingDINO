"""Saved-dashboards tests.

Covers the store, the API, and the orchestrator's save-prompt flow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.helper.orchestrator import Orchestrator, _PENDING_SAVE
from app.main import app
from app.services.saved_dashboard_store import SavedDashboardStore
from app.specs import DashboardSpec


_STORAGE_SUBDIRS = ("dashboards", "tickets", "evaluation_reports", "saved_dashboards")


@pytest.fixture(autouse=True)
def _isolated_storage():
    """Give every test in this module an empty view of the real storage
    dirs WITHOUT ever deleting real data: back up existing *.json files
    in-memory, clear the dirs for the test, then restore the originals
    and remove anything the test created. `backend/app/storage/` is not
    test-isolated at the module level (unlike devfix_runs/system_rules,
    see tests/conftest.py) because DashboardStore/SavedDashboardStore
    resolve an absolute, non-redirectable `_STORAGE_ROOT` by design —
    see the "immutable" comment in dashboard_store.py. Running this
    suite against the real app (e.g. on a live deployment box) must
    never wipe or leak into real saved dashboards/tickets/reports."""
    root = Path("backend/app/storage")
    backups: dict[Path, bytes] = {}
    for sub in _STORAGE_SUBDIRS:
        d = root / sub
        if d.exists():
            for p in d.glob("*.json"):
                backups[p] = p.read_bytes()
                p.unlink()
    _PENDING_SAVE.clear()
    try:
        yield
    finally:
        for sub in _STORAGE_SUBDIRS:
            d = root / sub
            if d.exists():
                for p in d.glob("*.json"):
                    if p not in backups:
                        p.unlink()
        for p, content in backups.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
        _PENDING_SAVE.clear()


def test_store_save_list_load_delete(tmp_path):
    store = SavedDashboardStore(root=tmp_path)
    spec = DashboardSpec.model_validate({
        "dashboard_id": "demo", "title": "Demo", "description": "",
        "layout": {"columns": 12, "row_height": 40}, "variables": [],
        "widgets": [], "refresh_interval": "30s",
    })
    entry = store.save(spec, name="prod CPU", description="d", tags=["prod", "cpu"])
    assert entry.saved_id
    assert entry.name == "prod CPU"

    items = store.list()
    assert len(items) == 1
    assert items[0].saved_id == entry.saved_id

    sums = store.summaries()
    assert sums[0].name == "prod CPU"
    assert sums[0].tags == ["prod", "cpu"]

    loaded = store.load(entry.saved_id)
    assert loaded is not None
    assert loaded.spec.title == "Demo"

    assert store.delete(entry.saved_id) is True
    assert store.load(entry.saved_id) is None


def test_store_unique_slug_on_name_collision(tmp_path):
    store = SavedDashboardStore(root=tmp_path)
    spec = DashboardSpec.model_validate({
        "dashboard_id": "demo", "title": "Demo", "description": "",
        "layout": {"columns": 12, "row_height": 40}, "variables": [],
        "widgets": [], "refresh_interval": "30s",
    })
    a = store.save(spec, name="prod CPU")
    b = store.save(spec, name="prod CPU")
    assert a.saved_id != b.saved_id
    assert store.load(a.saved_id).name == "prod CPU"
    assert store.load(b.saved_id).name == "prod CPU"


def _chat_post_and_wait(client, payload: dict, timeout_s: float = 10.0) -> dict:
    """Drive the async chat job pattern end-to-end.

    POST /api/chat/message returns {job_id, status:"queued"}; the
    actual ChatResponse is delivered through GET /api/chat/message/
    <job_id>.result once status flips to "done". This helper hides
    the polling so callers can treat chat like a sync RPC.
    """
    import time as _time

    r = client.post("/api/chat/message", json=payload)
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    deadline = _time.time() + timeout_s
    while _time.time() < deadline:
        s = client.get(f"/api/chat/message/{job_id}")
        assert s.status_code == 200, s.text
        body = s.json()
        if body["status"] == "done":
            return body["result"]
        if body["status"] == "error":
            raise AssertionError(
                f"chat job failed: {body.get('error') or '(no error message)'}"
            )
        _time.sleep(0.05)
    raise AssertionError(f"chat job {job_id} timed out after {timeout_s}s")


def test_api_save_list_delete(monkeypatch):
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_HELPERCODE", raising=False)
    client = TestClient(app)

    # Build a dashboard via chat (async job pattern).
    result = _chat_post_and_wait(client, {
        "session_id": "sd-api",
        "message": "Show me a CPU dashboard",
    })
    did = result["dashboard"]["dashboard_id"]

    # Save it.
    r = client.post("/api/saved_dashboards/save", json={
        "dashboard_id": did,
        "name": "my-library-cpu",
        "description": "cpu template",
        "tags": ["cpu"],
    })
    assert r.status_code == 200
    saved = r.json()["saved"]
    assert saved["name"] == "my-library-cpu"

    # List.
    r = client.get("/api/saved_dashboards/")
    assert r.status_code == 200
    names = [s["name"] for s in r.json()["saved"]]
    assert "my-library-cpu" in names

    # Get one.
    r = client.get(f"/api/saved_dashboards/{saved['saved_id']}")
    assert r.status_code == 200

    # Delete.
    r = client.delete(f"/api/saved_dashboards/{saved['saved_id']}")
    assert r.status_code == 200

    # Re-list: gone.
    r = client.get("/api/saved_dashboards/")
    assert saved["saved_id"] not in [s["saved_id"] for s in r.json()["saved"]]


def test_api_save_dashboard_not_found():
    client = TestClient(app)
    r = client.post("/api/saved_dashboards/save", json={
        "dashboard_id": "no-such-dashboard",
        "name": "x",
    })
    assert r.status_code == 404


def test_orchestrator_save_prompt_yes(monkeypatch):
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_HELPERCODE", raising=False)
    o = Orchestrator()

    r1 = o.handle_user_message(
        session_id="sp-1", message="Show me a CPU dashboard",
    )
    assert r1["intent_type"] == "DashboardSpec"
    assert r1["save_prompt_for"] is not None
    # Reply contains the (繁中) save prompt text.
    assert "存到你的資料庫" in r1["user_reply"]

    r2 = o.handle_user_message(session_id="sp-1", message="prod cpu")
    assert "已儲存為「prod cpu」" in r2["user_reply"]
    sums = SavedDashboardStore().summaries()
    assert any(s.name == "prod cpu" for s in sums)


def test_orchestrator_save_prompt_no(monkeypatch):
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_HELPERCODE", raising=False)
    o = Orchestrator()

    r1 = o.handle_user_message(
        session_id="sp-2", message="Show me a CPU dashboard",
    )
    assert r1["save_prompt_for"] is not None
    r2 = o.handle_user_message(session_id="sp-2", message="no")
    assert "不儲存" in r2["user_reply"]
    sums = SavedDashboardStore().summaries()
    assert all(s.name != "no" for s in sums)


def test_orchestrator_save_prompt_falls_through_on_next_request(monkeypatch):
    """If user replies to a save prompt with a brand-new dashboard
    request, the save prompt is dropped and the new dashboard is
    built."""
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_HELPERCODE", raising=False)
    o = Orchestrator()
    o.handle_user_message(session_id="sp-3", message="Show me a CPU dashboard")
    # Second message is a full new dashboard request — should NOT be
    # interpreted as a library name.
    r = o.handle_user_message(
        session_id="sp-3",
        message="Show me a memory dashboard with a line chart",
    )
    assert r["intent_type"] == "DashboardSpec"
    assert r["save_prompt_for"] is not None  # a new save prompt was issued
