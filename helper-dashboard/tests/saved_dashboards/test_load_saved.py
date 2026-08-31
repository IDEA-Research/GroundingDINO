"""Saved-dashboard LOAD surface tests.

Covers the single load surface (POST /api/saved_dashboards/{id}/load),
the orchestrator's LoadSavedIntent dispatch (which must go through the
SAME clone logic), and the chat request's current_dashboard_id updating
session memory before classification.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.helper.orchestrator import Orchestrator, _PENDING_SAVE
from app.main import app
from app.services.dashboard_store import DashboardStore
from app.services.saved_dashboard_store import SavedDashboardStore
from app.specs import DashboardSpec


def _reset_storage():
    root = Path("backend/app/storage")
    for sub in ("dashboards", "tickets", "evaluation_reports", "saved_dashboards"):
        d = root / sub
        if d.exists():
            for p in d.glob("*.json"):
                p.unlink()
    _PENDING_SAVE.clear()


def _spec(dashboard_id: str = "demo", title: str = "Demo") -> DashboardSpec:
    return DashboardSpec.model_validate({
        "dashboard_id": dashboard_id, "title": title, "description": "",
        "layout": {"columns": 12, "row_height": 40}, "variables": [],
        "widgets": [], "refresh_interval": "30s",
    })


# ---------------------------------------------------------------------------
# POST /api/saved_dashboards/{saved_id}/load
# ---------------------------------------------------------------------------


def test_load_endpoint_clones_with_fresh_id_and_bumps_used():
    _reset_storage()
    client = TestClient(app)
    store = SavedDashboardStore()
    entry = store.save(_spec("orig-dash", "Prod CPU"), name="prod CPU")
    assert store.load(entry.saved_id).used_count == 0

    r = client.post(f"/api/saved_dashboards/{entry.saved_id}/load")
    assert r.status_code == 200
    body = r.json()
    assert body["saved_id"] == entry.saved_id
    assert body["name"] == "prod CPU"
    clone = body["dashboard"]
    # The clone is a full DashboardSpec with a FRESH dashboard_id: a
    # later chat patch must target what the user is looking at, never
    # a different spec that happens to share the stored id.
    assert clone["title"] == "Prod CPU"
    assert clone["dashboard_id"] != "orig-dash"
    assert clone["dashboard_id"].startswith("orig-dash-")
    # The clone is persisted in the working dashboard store.
    assert DashboardStore().load_dashboard(clone["dashboard_id"]) is not None
    # used_count bumped.
    assert store.load(entry.saved_id).used_count == 1

    # A second load produces ANOTHER fresh clone and bumps again.
    r2 = client.post(f"/api/saved_dashboards/{entry.saved_id}/load")
    assert r2.status_code == 200
    clone2 = r2.json()["dashboard"]
    assert clone2["dashboard_id"] != clone["dashboard_id"]
    assert store.load(entry.saved_id).used_count == 2


def test_load_endpoint_404_when_missing():
    _reset_storage()
    client = TestClient(app)
    r = client.post("/api/saved_dashboards/no-such-entry/load")
    assert r.status_code == 404
    assert r.json()["detail"]


def test_load_endpoint_trailing_slash_form_registered():
    # redirect_slashes=False on the app: both forms must be explicit
    # routes, or the proxy-escaping 307 bug comes back.
    _reset_storage()
    client = TestClient(app)
    store = SavedDashboardStore()
    entry = store.save(_spec(), name="slash form")
    r = client.post(f"/api/saved_dashboards/{entry.saved_id}/load/")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Orchestrator LoadSavedIntent dispatch
# ---------------------------------------------------------------------------


def test_orchestrator_loads_saved_dashboard_via_chat(monkeypatch):
    """End-to-end (mock runtime): 'load my saved X' resolves the entry,
    clones it with a fresh id, updates session memory, and returns a
    dashboard-bearing response — no LLM regeneration."""
    _reset_storage()
    monkeypatch.delenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW", raising=False)
    monkeypatch.delenv("HELPER_DASHBOARD_HELPERCODE", raising=False)
    store = SavedDashboardStore()
    entry = store.save(_spec("cpu-dash", "Prod CPU"), name="prod cpu")

    o = Orchestrator()
    r = o.handle_user_message(
        session_id="ls-1", message="load my saved prod cpu",
    )
    assert r["intent_type"] == "DashboardSpec"
    assert r["dashboard"] is not None
    assert r["dashboard"]["title"] == "Prod CPU"
    assert r["dashboard"]["dashboard_id"] != "cpu-dash"  # fresh clone
    # Loading does not re-ask to save — it's already in the library.
    assert r["save_prompt_for"] is None
    # Session memory now points at the clone so follow-up patches
    # target exactly what the user is looking at.
    assert (
        o._memory.get_current_dashboard_id("ls-1")
        == r["dashboard"]["dashboard_id"]
    )
    # Shared clone logic bumped used_count (same path as the endpoint).
    assert store.load(entry.saved_id).used_count == 1
    # The clone is persisted for /api/dashboard/{id} + patching.
    assert (
        DashboardStore().load_dashboard(r["dashboard"]["dashboard_id"])
        is not None
    )


def test_orchestrator_load_saved_ambiguous_returns_clarification():
    """A LoadSavedIntent whose name_query matches several entries must
    come back as a ClarificationRequest listing the candidates."""
    _reset_storage()
    store = SavedDashboardStore()
    store.save(_spec("a", "A"), name="prod cpu")
    store.save(_spec("b", "B"), name="prod memory")

    o = Orchestrator()
    r = o._handle_load_saved_intent(
        session_id="ls-2",
        message="open my saved prod dashboard",
        intent={
            "type": "LoadSavedIntent",
            "saved_id": None,
            "name_query": "prod",
            "message_to_user": None,
        },
        runtime_meta={"runtime_used": "mock", "fallback_reason": None},
    )
    assert r["intent_type"] == "ClarificationRequest"
    assert r["clarification_questions"]
    q = r["clarification_questions"][0]
    assert "prod cpu" in q and "prod memory" in q
    assert r["dashboard"] is None
    # Nothing was cloned or bumped.
    assert all(s.used_count == 0 for s in store.summaries())


def test_orchestrator_load_saved_no_match_lists_library():
    _reset_storage()
    store = SavedDashboardStore()
    store.save(_spec(), name="prod cpu")

    o = Orchestrator()
    r = o._handle_load_saved_intent(
        session_id="ls-3",
        message="load my saved thing",
        intent={
            "type": "LoadSavedIntent",
            "saved_id": None,
            "name_query": "zzz-not-a-name",
            "message_to_user": None,
        },
        runtime_meta={"runtime_used": "mock", "fallback_reason": None},
    )
    assert r["intent_type"] == "UserResponse"
    assert "prod cpu" in r["user_reply"]  # friendly library listing
    assert r["dashboard"] is None


def test_mock_classifier_emits_load_saved_intent(monkeypatch):
    from app.helper.runtime import HelperCodeRuntime

    monkeypatch.delenv("HELPER_DASHBOARD_HELPERCODE", raising=False)
    rt = HelperCodeRuntime()
    out = rt.invoke_operation("user_message", {
        "message": "open my saved prod cpu",
        "saved_dashboards": [{
            "saved_id": "prod-cpu", "name": "prod cpu", "tags": [],
            "widget_types": [], "promqls": [], "used_count": 0,
            "description": "",
        }],
    })
    assert out["type"] == "LoadSavedIntent"
    assert out["saved_id"] == "prod-cpu"


def test_mock_classifier_clarifies_when_library_ambiguous(monkeypatch):
    from app.helper.runtime import HelperCodeRuntime

    monkeypatch.delenv("HELPER_DASHBOARD_HELPERCODE", raising=False)
    rt = HelperCodeRuntime()
    out = rt.invoke_operation("user_message", {
        "message": "open my saved prod dashboard",
        "saved_dashboards": [
            {"saved_id": "prod-cpu", "name": "prod cpu", "tags": ["prod"],
             "widget_types": [], "promqls": [], "used_count": 0,
             "description": ""},
            {"saved_id": "prod-mem", "name": "prod memory", "tags": ["prod"],
             "widget_types": [], "promqls": [], "used_count": 0,
             "description": ""},
        ],
    })
    assert out["type"] == "ClarificationRequest"
    assert "prod cpu" in out["questions"][0]
    assert "prod memory" in out["questions"][0]


# ---------------------------------------------------------------------------
# Chat request current_dashboard_id → session memory (before classification)
# ---------------------------------------------------------------------------


def test_chat_current_dashboard_id_updates_session_memory():
    _reset_storage()
    o = Orchestrator()
    r = o.handle_user_message(
        session_id="cm-1",
        message="hello there",
        current_dashboard_id="dash-the-user-is-looking-at",
    )
    # A plain greeting stays a UserResponse...
    assert r["intent_type"] == "UserResponse"
    # ...but memory was updated BEFORE classification, so the next
    # patch-shaped message targets what the user is looking at.
    assert (
        o._memory.get_current_dashboard_id("cm-1")
        == "dash-the-user-is-looking-at"
    )
