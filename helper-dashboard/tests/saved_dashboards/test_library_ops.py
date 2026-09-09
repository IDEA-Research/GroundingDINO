"""Saved-library management: rename (PATCH), duplicate, SOFT delete +
restore, developer-only trash listing, stable saved_id, both '' and
'/' route forms, and the chat save-answer guard.

Store-level tests use tmp_path; API tests point the router's module
store at tmp_path too, so nothing touches the real storage dir.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import saved_dashboards as saved_api
from app.helper.orchestrator import (
    Orchestrator,
    _PENDING_SAVE,
    looks_like_library_name,
)
from app.main import app
from app.services.saved_dashboard_store import DUPLICATE_SUFFIX, SavedDashboardStore
from app.specs import DashboardSpec


def _spec(dashboard_id: str = "demo", title: str = "Demo") -> DashboardSpec:
    return DashboardSpec.model_validate({
        "dashboard_id": dashboard_id, "title": title, "description": "",
        "layout": {"columns": 12, "row_height": 40}, "variables": [],
        "widgets": [{
            "id": "w1", "type": "stat_card", "title": "心率",
            "query": {"source": "prometheus", "promql": "max by (patient_id) (patient_heart_rate_bpm)",
                      "query_type": "instant"},
            "position": {"x": 0, "y": 0, "w": 3, "h": 3},
            "encoding": {"unit": "bpm"}, "thresholds": [], "options": {},
        }],
        "refresh_interval": "30s",
    })


@pytest.fixture
def store(tmp_path):
    return SavedDashboardStore(root=tmp_path / "saved")


@pytest.fixture
def api(monkeypatch, store):
    monkeypatch.setattr(saved_api, "_store", store)
    return TestClient(app)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


def test_update_keeps_saved_id_stable(store):
    entry = store.save(_spec(), name="新生兒心率", description="d", tags=["nicu"])
    assert entry.saved_id.startswith("saved-")  # Chinese name -> slug fallback
    updated = store.update(entry.saved_id, name="心率與血氧", tags=["nicu", "hr"])
    assert updated is not None
    assert updated.saved_id == entry.saved_id
    assert updated.name == "心率與血氧"
    assert updated.description == "d"  # untouched
    assert updated.tags == ["nicu", "hr"]
    assert updated.updated_at and updated.updated_at >= entry.created_at
    assert store.load(entry.saved_id).name == "心率與血氧"
    assert store.update("nope", name="x") is None


def test_duplicate_appends_suffix_and_copies_spec(store):
    entry = store.save(_spec("orig", "Prod"), name="prod cpu", description="dd", tags=["a"])
    dup = store.duplicate(entry.saved_id)
    assert dup is not None
    assert dup.name == f"prod cpu{DUPLICATE_SUFFIX}" == "prod cpu 副本"
    assert dup.saved_id != entry.saved_id
    assert dup.spec == entry.spec
    assert dup.description == "dd" and dup.tags == ["a"]
    assert dup.used_count == 0
    # Repeats stay unique.
    dup2 = store.duplicate(entry.saved_id)
    assert dup2.saved_id not in {entry.saved_id, dup.saved_id}
    assert len(store.list()) == 3


def test_soft_delete_excludes_and_restore_brings_back(store):
    a = store.save(_spec("a"), name="alpha")
    b = store.save(_spec("b"), name="beta")
    assert store.delete(a.saved_id) is True
    # Gone from every default view...
    assert [s.saved_id for s in store.list()] == [b.saved_id]
    assert [s.name for s in store.summaries()] == ["beta"]
    assert store.load(a.saved_id) is None
    assert store.delete(a.saved_id) is False  # already deleted
    # ...but still on disk with a deleted_at stamp.
    trashed = store.load(a.saved_id, include_deleted=True)
    assert trashed is not None and trashed.deleted_at
    assert {s.saved_id for s in store.list(include_deleted=True)} == {a.saved_id, b.saved_id}
    dev_view = store.summaries(include_deleted=True)
    assert any(s.deleted_at for s in dev_view if s.saved_id == a.saved_id)
    # restore
    restored = store.restore(a.saved_id)
    assert restored is not None and restored.deleted_at is None
    assert store.load(a.saved_id) is not None
    assert store.restore(a.saved_id).saved_id == a.saved_id  # idempotent
    assert store.restore("nope") is None


def test_bump_used_ignores_deleted(store):
    a = store.save(_spec(), name="alpha")
    store.delete(a.saved_id)
    store.bump_used(a.saved_id)
    assert store.load(a.saved_id, include_deleted=True).used_count == 0


def test_save_never_overwrites_a_trashed_file(store):
    a = store.save(_spec(), name="prod cpu")
    store.delete(a.saved_id)
    b = store.save(_spec(), name="prod cpu")
    assert b.saved_id != a.saved_id
    assert store.load(a.saved_id, include_deleted=True).deleted_at


def test_summary_promqls_cover_queries_tables(store):
    spec = DashboardSpec.model_validate({
        "dashboard_id": "t", "title": "T", "widgets": [{
            "id": "tbl", "type": "table", "title": "各病患一覽",
            "position": {"x": 0, "y": 0, "w": 12, "h": 6},
            "queries": [
                {"promql": "max by (patient_id) (patient_heart_rate_bpm)", "query_type": "instant", "key": "hr"},
                {"promql": "max by (patient_id) (patient_spo2_percentage)", "query_type": "instant", "key": "spo2"},
            ],
        }],
    })
    entry = store.save(spec, name="tbl")
    assert len(entry.summary["promqls"]) == 2


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_api_patch_duplicate_delete_restore_flow(api, store, monkeypatch):
    entry = store.save(_spec(), name="alpha", tags=["x"])
    sid = entry.saved_id

    # PATCH (both forms)
    r = api.patch(f"/api/saved_dashboards/{sid}", json={"name": "alpha 2"})
    assert r.status_code == 200 and r.json()["saved"]["name"] == "alpha 2"
    assert r.json()["saved"]["saved_id"] == sid
    r = api.patch(f"/api/saved_dashboards/{sid}/", json={"tags": ["y", "y"]})
    assert r.status_code == 200 and r.json()["saved"]["tags"] == ["y"]
    assert api.patch(f"/api/saved_dashboards/{sid}", json={}).status_code == 400
    assert api.patch("/api/saved_dashboards/nope", json={"name": "z"}).status_code == 404

    # duplicate
    r = api.post(f"/api/saved_dashboards/{sid}/duplicate")
    assert r.status_code == 200
    dup = r.json()["saved"]
    assert dup["name"] == "alpha 2 副本" and dup["saved_id"] != sid
    assert api.post(f"/api/saved_dashboards/{sid}/duplicate/").status_code == 200
    assert api.post("/api/saved_dashboards/nope/duplicate").status_code == 404

    # soft delete -> list excludes, get 404, load 404
    r = api.delete(f"/api/saved_dashboards/{sid}")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "restorable": True, "saved_id": sid}
    ids = [s["saved_id"] for s in api.get("/api/saved_dashboards/").json()["saved"]]
    assert sid not in ids and dup["saved_id"] in ids
    assert api.get(f"/api/saved_dashboards/{sid}").status_code == 404
    assert api.post(f"/api/saved_dashboards/{sid}/load").status_code == 404
    assert api.delete(f"/api/saved_dashboards/{sid}").status_code == 404

    # include_deleted without a dev header -> silently the normal list
    monkeypatch.setenv("HELPER_DASHBOARD_DEV_TOKEN", "tok")
    r = api.get("/api/saved_dashboards?include_deleted=1")
    assert r.json()["include_deleted"] is False
    assert sid not in [s["saved_id"] for s in r.json()["saved"]]
    r = api.get("/api/saved_dashboards?include_deleted=1", headers={"X-Developer-Token": "wrong"})
    assert r.json()["include_deleted"] is False
    # ...with a valid dev header the trash shows, stamped
    r = api.get("/api/saved_dashboards/?include_deleted=1", headers={"X-Developer-Token": "tok"})
    assert r.json()["include_deleted"] is True
    row = next(s for s in r.json()["saved"] if s["saved_id"] == sid)
    assert row["deleted_at"]

    # restore (both forms), then back in the list with the same id
    r = api.post(f"/api/saved_dashboards/{sid}/restore")
    assert r.status_code == 200 and r.json()["saved"]["deleted_at"] is None
    assert api.post(f"/api/saved_dashboards/{sid}/restore/").status_code == 200
    assert sid in [s["saved_id"] for s in api.get("/api/saved_dashboards").json()["saved"]]
    assert api.post("/api/saved_dashboards/nope/restore").status_code == 404


def test_api_delete_trailing_slash_form(api, store):
    entry = store.save(_spec(), name="beta")
    assert api.delete(f"/api/saved_dashboards/{entry.saved_id}/").status_code == 200


def test_summary_deleted_at_absent_from_default_listing(api, store):
    store.save(_spec(), name="gamma")
    rows = api.get("/api/saved_dashboards").json()["saved"]
    assert rows and all(r["deleted_at"] is None for r in rows)


# ---------------------------------------------------------------------------
# chat save-answer guard (繁中 follow-ups are never swallowed as names)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["prod cpu", "新生兒心率", "NICU 床位 3", "rSO2 左右腦"])
def test_looks_like_library_name_accepts_short_names(name):
    assert looks_like_library_name(name) is True


@pytest.mark.parametrize("msg", [
    "改成5分", "把血氧換成心率", "幫我加一個表格", "請顯示過去一小時", "看一下心率",
    "刪掉第二張", "調整範圍", "設定門檻", "新增病患", "移除圖表", "小時", "分",
    "這是什麼?", "存起來嗎？", "x" * 25, "two\nlines", "show me memory", "add a chart",
])
def test_looks_like_library_name_rejects_edits_and_questions(msg):
    assert looks_like_library_name(msg) is False


def _fresh_orchestrator(monkeypatch, tmp_path):
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_HELPERCODE", raising=False)
    _PENDING_SAVE.clear()
    o = Orchestrator()
    o._saved = SavedDashboardStore(root=tmp_path / "lib")
    return o


def test_chinese_follow_up_after_save_prompt_is_not_saved_as_name(monkeypatch, tmp_path):
    o = _fresh_orchestrator(monkeypatch, tmp_path)
    r1 = o.handle_user_message(session_id="g-1", message="Show me a CPU dashboard")
    assert r1["save_prompt_for"] is not None
    assert "存到你的資料庫" in r1["user_reply"]
    r2 = o.handle_user_message(session_id="g-1", message="改成5分")
    assert all(s.name != "改成5分" for s in o._saved.summaries())
    assert r2["intent_type"] != "UserResponse" or "已儲存" not in r2["user_reply"]


def test_chinese_decline_and_chinese_name(monkeypatch, tmp_path):
    o = _fresh_orchestrator(monkeypatch, tmp_path)
    o.handle_user_message(session_id="g-2", message="Show me a CPU dashboard")
    r = o.handle_user_message(session_id="g-2", message="不用")
    assert r["user_reply"] == "好的，不儲存。"
    assert o._saved.summaries() == []

    o.handle_user_message(session_id="g-3", message="Show me a CPU dashboard")
    r = o.handle_user_message(session_id="g-3", message="產線 CPU")
    assert "已儲存為「產線 CPU」" in r["user_reply"]
    assert [s.name for s in o._saved.summaries()] == ["產線 CPU"]
