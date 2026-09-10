"""PATCH /api/saved_dashboards/{id} widget CONTENT updates.

Phase 2 (react-grid-layout, commit b7664085) widened this route's
`widgets` field to carry a per-widget `position` so drag/resize could be
persisted for an already-saved dashboard. This module covers the follow
-up that widens the SAME field to widget CONTENT (title / description /
thresholds / options) — the fields the panel-edit drawer
(frontend WidgetEditDrawer.tsx) edits — so those edits survive a reload
of a saved dashboard (`api.loadSaved`) instead of only living in
`hd.lastOpened` localStorage.

Store-level tests use tmp_path; API tests point the router's module
store at tmp_path too (same fixture pattern as test_library_ops.py), so
nothing touches the real storage dir.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import saved_dashboards as saved_api
from app.main import app
from app.services.saved_dashboard_store import SavedDashboardStore
from app.specs import DashboardSpec


def _spec(dashboard_id: str = "demo", title: str = "Demo") -> DashboardSpec:
    return DashboardSpec.model_validate({
        "dashboard_id": dashboard_id, "title": title, "description": "",
        "layout": {"columns": 12, "row_height": 40}, "variables": [],
        "widgets": [{
            "id": "w1", "type": "stat_card", "title": "心率",
            "description": "原始描述",
            "query": {"source": "prometheus", "promql": "max by (patient_id) (patient_heart_rate_bpm)",
                      "query_type": "instant"},
            "position": {"x": 0, "y": 0, "w": 3, "h": 3},
            "encoding": {"unit": "bpm"},
            "thresholds": [
                {"value": 90, "color": "#ff0000", "label": "低", "direction": "below"},
            ],
            "options": {"decimals": 0},
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


def _reread(tmp_path, saved_id: str):
    """A BRAND NEW store instance over the same directory — proves the
    write actually reached disk rather than only living on the fixture's
    in-memory object."""
    fresh = SavedDashboardStore(root=tmp_path / "saved")
    entry = fresh.load(saved_id)
    assert entry is not None
    return entry


def _w1(entry):
    return next(w for w in entry.spec.widgets if w.id == "w1")


# ---------------------------------------------------------------------------
# (a) content update succeeds and really lands on disk
# ---------------------------------------------------------------------------


def test_content_update_persists_title_thresholds_options(api, store, tmp_path):
    entry = store.save(_spec(), name="content-a")
    sid = entry.saved_id

    r = api.patch(f"/api/saved_dashboards/{sid}", json={
        "widgets": [{
            "id": "w1",
            "title": "新標題",
            "description": "新描述",
            "thresholds": [
                {"value": 80, "color": "#00ff00", "label": "正常", "direction": "above"},
            ],
            "options": {"decimals": 2, "show_legend": True},
        }],
    })
    assert r.status_code == 200, r.text
    w = next(w for w in r.json()["saved"]["spec"]["widgets"] if w["id"] == "w1")
    assert w["title"] == "新標題"
    assert w["description"] == "新描述"
    assert w["thresholds"] == [
        {"value": 80, "color": "#00ff00", "label": "正常", "direction": "above"},
    ]
    assert w["options"] == {"decimals": 2, "show_legend": True}
    # type/query/position untouched by a content-only update.
    assert w["type"] == "stat_card"
    assert w["position"] == {"x": 0, "y": 0, "w": 3, "h": 3}

    # Reread from a FRESH store instance -> really persisted, not just
    # returned in the response.
    reloaded = _w1(_reread(tmp_path, sid))
    assert reloaded.title == "新標題"
    assert reloaded.description == "新描述"
    assert len(reloaded.thresholds) == 1
    assert reloaded.thresholds[0].value == 80
    assert reloaded.thresholds[0].color == "#00ff00"
    assert reloaded.options == {"decimals": 2, "show_legend": True}


# ---------------------------------------------------------------------------
# (b) invalid threshold data -> 422, no partial write
# ---------------------------------------------------------------------------


def test_invalid_threshold_rejected_with_no_partial_write(api, store, tmp_path):
    entry = store.save(_spec(), name="content-b")
    sid = entry.saved_id

    r = api.patch(f"/api/saved_dashboards/{sid}", json={
        "widgets": [{
            "id": "w1",
            "title": "壞掉的更新",  # would succeed on its own
            "thresholds": [
                {"value": 80, "color": "not-a-color", "label": "壞掉"},
            ],
        }],
    })
    assert r.status_code == 422
    assert "color" in str(r.json()["detail"]).lower() or "pattern" in str(r.json()["detail"]).lower()

    # Nothing was written: title AND thresholds are exactly as before —
    # the bad threshold must not let the good title field through.
    reloaded = _w1(_reread(tmp_path, sid))
    assert reloaded.title == "心率"
    assert len(reloaded.thresholds) == 1
    assert reloaded.thresholds[0].color == "#ff0000"


def test_invalid_layout_overflow_still_rejected_after_content_merge(api, store, tmp_path):
    """A content edit combined with a position that overflows the grid
    still 422s via the SAME full-dashboard validation path (no separate,
    looser rule set for the new content fields)."""
    entry = store.save(_spec(), name="content-b2")
    sid = entry.saved_id

    r = api.patch(f"/api/saved_dashboards/{sid}", json={
        "widgets": [{
            "id": "w1",
            "title": "OK 標題",
            "position": {"x": 10, "y": 0, "w": 6, "h": 3},  # 10+6 > 12 columns
        }],
    })
    assert r.status_code == 422
    reloaded = _w1(_reread(tmp_path, sid))
    assert reloaded.title == "心率"
    assert reloaded.position.x == 0


# ---------------------------------------------------------------------------
# (c) Phase 2 old behavior: position-only update still works untouched
# ---------------------------------------------------------------------------


def test_position_only_update_unchanged_behavior(api, store, tmp_path):
    entry = store.save(_spec(), name="content-c")
    sid = entry.saved_id

    r = api.patch(f"/api/saved_dashboards/{sid}", json={
        "widgets": [{"id": "w1", "position": {"x": 3, "y": 3, "w": 4, "h": 4}}],
    })
    assert r.status_code == 200, r.text
    w = next(w for w in r.json()["saved"]["spec"]["widgets"] if w["id"] == "w1")
    assert w["position"] == {"x": 3, "y": 3, "w": 4, "h": 4}
    # Content fields untouched by a position-only update.
    assert w["title"] == "心率"
    assert w["description"] == "原始描述"
    assert w["thresholds"] == [
        {"value": 90, "color": "#ff0000", "label": "低", "direction": "below"},
    ]
    assert w["options"] == {"decimals": 0}

    reloaded = _w1(_reread(tmp_path, sid))
    assert reloaded.position.x == 3 and reloaded.position.y == 3
    assert reloaded.title == "心率"
    assert reloaded.options == {"decimals": 0}


def test_layout_only_update_still_works(api, store, tmp_path):
    """The other Phase 2 field (`layout`) is untouched by this change."""
    entry = store.save(_spec(), name="content-c2")
    sid = entry.saved_id
    r = api.patch(f"/api/saved_dashboards/{sid}", json={
        "layout": {"columns": 24, "row_height": 30},
    })
    assert r.status_code == 200, r.text
    assert r.json()["saved"]["spec"]["layout"] == {"columns": 24, "row_height": 30}


# ---------------------------------------------------------------------------
# (d) unknown fields rejected at the VALIDATION-LOGIC layer, not by a
#     schema-level extra=forbid catch-all.
# ---------------------------------------------------------------------------


def test_unknown_option_key_rejected_by_spec_validator(api, store, tmp_path):
    entry = store.save(_spec(), name="content-d")
    sid = entry.saved_id

    r = api.patch(f"/api/saved_dashboards/{sid}", json={
        "widgets": [{"id": "w1", "options": {"totally_unknown_key": True}}],
    })
    assert r.status_code == 422
    detail = str(r.json()["detail"])
    # This exact wording comes from WidgetSpec._safe_options in
    # widget_spec.py (the SAME allowlist check every dashboard build
    # goes through) — proves the rejection is semantic, not a generic
    # "extra fields not permitted" schema error.
    assert "not in the allowed option keys" in detail
    assert "totally_unknown_key" in detail

    reloaded = _w1(_reread(tmp_path, sid))
    assert reloaded.options == {"decimals": 0}  # unchanged


def test_forbidden_option_key_rejected(api, store, tmp_path):
    """A FORBIDDEN key (not just unlisted) is also caught by the same
    semantic layer, pre-empting the allowlist message with its own."""
    entry = store.save(_spec(), name="content-d2")
    sid = entry.saved_id
    r = api.patch(f"/api/saved_dashboards/{sid}", json={
        "widgets": [{"id": "w1", "options": {"raw_html": "<b>x</b>"}}],
    })
    assert r.status_code == 422
    assert "forbidden" in str(r.json()["detail"]).lower()


def test_unknown_top_level_widget_field_rejected_structurally(api, store):
    """A field that isn't part of the update schema at all (typo /
    made-up key) is rejected at the FastAPI/Pydantic request-parsing
    layer — WidgetContentUpdate is `extra="forbid"`, not `extra="allow"`."""
    entry = store.save(_spec(), name="content-d3")
    sid = entry.saved_id
    r = api.patch(f"/api/saved_dashboards/{sid}", json={
        "widgets": [{"id": "w1", "not_a_real_field": 123}],
    })
    assert r.status_code == 422
