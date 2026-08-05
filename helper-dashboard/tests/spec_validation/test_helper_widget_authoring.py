"""Helper-authors-widget-instances tests.

These tests assert the architectural correction:

    Widget INSTANCE creation/modification  -> Helper's job.
    Widget TOOLKIT source-code extension   -> Big guy's job.

Each numbered test corresponds to an item in the developer directive.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.patch_service import PatchApplicationError, PatchService
from app.services.spec_validator import SpecValidationError, SpecValidator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _widget(**overrides):
    w = {
        "id": "w1",
        "type": "line_chart",
        "title": "CPU",
        "description": "",
        "query": {
            "source": "prometheus",
            "promql": "rate(http_requests_total[5m])",
            "query_type": "range",
            "range": "1h",
            "step": "30s",
        },
        "position": {"x": 0, "y": 0, "w": 6, "h": 6},
        "encoding": {},
        "thresholds": [],
        "options": {},
    }
    w.update(overrides)
    return w


def _dashboard(widgets=None):
    return SpecValidator().validate_dashboard(
        {
            "dashboard_id": "demo",
            "title": "Demo",
            "description": "",
            "layout": {"columns": 12, "row_height": 40},
            "variables": [],
            "widgets": widgets if widgets is not None else [_widget()],
            "refresh_interval": "30s",
        }
    )


def _patch(ops, *, target="demo"):
    return SpecValidator().validate_patch(
        {
            "patch_id": "p1",
            "reason": "test",
            "target_dashboard_id": target,
            "created_by": "patch-agent",
            "operations": ops,
        }
    )


# ---------------------------------------------------------------------------
# 1. Helper can add a widget through PatchSpec
# ---------------------------------------------------------------------------

def test_1_helper_can_add_widget_via_patch():
    spec = _dashboard()
    new_widget = _widget(id="w2", type="stat_card", title="Up",
                         query={
                             "source": "prometheus",
                             "promql": "sum(up)",
                             "query_type": "instant",
                         },
                         position={"x": 6, "y": 0, "w": 3, "h": 4})
    patch = _patch([{"op": "add_widget", "widget": new_widget}])
    new = PatchService().apply(spec, patch)
    assert [w.id for w in new.widgets] == ["w1", "w2"]
    assert new.widgets[1].type.value == "stat_card"


# ---------------------------------------------------------------------------
# 2. Helper can modify a widget through PatchSpec
# ---------------------------------------------------------------------------

def test_2_helper_can_modify_widget_via_patch():
    spec = _dashboard()
    patch = _patch(
        [
            {
                "op": "update_widget",
                "widget_id": "w1",
                "fields": {"title": "Renamed", "description": "edited"},
            }
        ]
    )
    new = PatchService().apply(spec, patch)
    assert new.widgets[0].title == "Renamed"
    assert new.widgets[0].description == "edited"


# Bonus: change widget type among supported types
def test_2b_helper_can_change_widget_type_among_supported_types():
    spec = _dashboard()
    patch = _patch(
        [
            {
                "op": "update_widget",
                "widget_id": "w1",
                "fields": {
                    "type": "table",
                    "query": {
                        "source": "prometheus",
                        "promql": "topk(10, http_requests_total)",
                        "query_type": "instant",
                    },
                },
            }
        ]
    )
    new = PatchService().apply(spec, patch)
    assert new.widgets[0].type.value == "table"


# ---------------------------------------------------------------------------
# 3. Helper can change a widget PromQL query through PatchSpec
# ---------------------------------------------------------------------------

def test_3_helper_can_change_promql_via_patch():
    spec = _dashboard()
    patch = _patch(
        [
            {
                "op": "update_widget",
                "widget_id": "w1",
                "fields": {
                    "query": {
                        "source": "prometheus",
                        "promql": "rate(node_cpu_seconds_total[1m])",
                        "query_type": "range",
                        "range": "30m",
                        "step": "15s",
                    }
                },
            }
        ]
    )
    new = PatchService().apply(spec, patch)
    assert new.widgets[0].query.promql == "rate(node_cpu_seconds_total[1m])"
    assert new.widgets[0].query.range == "30m"


# ---------------------------------------------------------------------------
# 4. Helper can change layout safely
# ---------------------------------------------------------------------------

def test_4_helper_can_resize_and_move_widget_safely():
    spec = _dashboard()
    patch = _patch(
        [
            {
                "op": "update_widget",
                "widget_id": "w1",
                "fields": {"position": {"x": 2, "y": 3, "w": 8, "h": 5}},
            }
        ]
    )
    new = PatchService().apply(spec, patch)
    assert new.widgets[0].position.x == 2
    assert new.widgets[0].position.w == 8


def test_4b_layout_changes_that_overflow_are_rejected():
    spec = _dashboard()
    patch = _patch(
        [
            {
                "op": "update_widget",
                "widget_id": "w1",
                "fields": {"position": {"x": 10, "y": 0, "w": 6, "h": 6}},
            }
        ]
    )
    with pytest.raises(PatchApplicationError):
        PatchService().apply(spec, patch)


def test_4c_helper_can_reorder_widgets():
    spec = _dashboard(
        widgets=[
            _widget(id="w1"),
            _widget(id="w2", position={"x": 6, "y": 0, "w": 6, "h": 6}),
        ]
    )
    patch = _patch([{"op": "reorder_widgets", "order": ["w2", "w1"]}])
    new = PatchService().apply(spec, patch)
    assert [w.id for w in new.widgets] == ["w2", "w1"]


# ---------------------------------------------------------------------------
# 5. Invalid widget types are rejected
# ---------------------------------------------------------------------------

def test_5_invalid_widget_type_in_add_is_rejected():
    # `sankey` is intentionally not in WidgetType — if the auto-extend
    # pipeline later adds it, pick a fresh name that isn't yet cached.
    with pytest.raises(SpecValidationError):
        _patch(
            [
                {
                    "op": "add_widget",
                    "widget": _widget(id="w2", type="sankey"),
                }
            ]
        )


def test_5b_invalid_widget_type_in_update_is_rejected():
    spec = _dashboard()
    # The patch itself validates (the fields dict accepts arbitrary
    # types syntactically) — the rejection happens at post-patch
    # dashboard re-validation. Either path must refuse the patch.
    try:
        patch = _patch(
            [
                {
                    "op": "update_widget",
                    "widget_id": "w1",
                    "fields": {"type": "sankey"},
                }
            ]
        )
    except SpecValidationError:
        return  # rejected at patch validation, good

    with pytest.raises((SpecValidationError, PatchApplicationError)):
        PatchService().apply(spec, patch)


# ---------------------------------------------------------------------------
# 6. Unsafe widget fields are rejected
# ---------------------------------------------------------------------------

def test_6_unsafe_widget_fields_rejected_in_add_widget():
    bad = _widget()
    bad["raw_html"] = "<script>alert(1)</script>"
    with pytest.raises(SpecValidationError) as exc:
        _patch([{"op": "add_widget", "widget": bad}])
    assert any("raw_html" in e or "forbidden" in e for e in exc.value.errors)


def test_6b_script_content_in_title_rejected():
    bad = _widget(title="<script>alert(1)</script>")
    with pytest.raises(SpecValidationError):
        _patch([{"op": "add_widget", "widget": bad}])


def test_6c_unknown_option_key_rejected():
    bad = _widget(options={"on_click": "alert(1)"})
    with pytest.raises(SpecValidationError):
        _patch([{"op": "add_widget", "widget": bad}])


# ---------------------------------------------------------------------------
# 7. Widget instance creation does NOT require a developer token
# ---------------------------------------------------------------------------

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


def test_7_widget_instance_creation_has_no_developer_token_requirement(monkeypatch, tmp_path):
    """The user-facing /api/chat endpoint must create widgets without
    any X-Developer-Token header. A successful dashboard-creation
    response is direct proof that Helper authored widget instances
    through the normal user flow.
    """
    # Isolate storage so the test doesn't litter the repo.
    monkeypatch.setenv("HELPER_DASHBOARD_DEV_TOKEN", "")  # effectively unset
    # The app is module-level; ensure a clean client.
    client = TestClient(app)

    # Async job pattern: POST returns job_id, poll GET for the result.
    # IMPORTANT: no X-Developer-Token header is sent.
    body = _chat_post_and_wait(client, {
        "session_id": "s-widget-test",
        "message": "Show me a CPU dashboard",
    })
    assert body["intent_type"] == "DashboardSpec"
    assert body["dashboard"] is not None
    widgets = body["dashboard"]["widgets"]
    assert len(widgets) >= 1
    # Every widget must use a supported type.
    assert all(
        w["type"] in {"line_chart", "stat_card", "gauge", "table", "alert_list"}
        for w in widgets
    )

    # And a follow-up patch (also widget-instance work) is equally
    # allowed with no dev token.
    did = body["dashboard"]["dashboard_id"]
    body2 = _chat_post_and_wait(client, {
        "session_id": "s-widget-test",
        "message": "add a stat card",
        "current_dashboard_id": did,
    })
    assert body2["intent_type"] == "PatchSpec"
    assert body2["patch"] is not None
    assert any(op["op"] == "add_widget" for op in body2["patch"]["operations"])


# ---------------------------------------------------------------------------
# 8. Widget TOOLKIT extension still requires DeveloperTicket / dev token
# ---------------------------------------------------------------------------

def test_8_widget_toolkit_extension_still_requires_developer_access(monkeypatch):
    """Two invariants at once:

    (a) A widget type that is NOT in the toolkit cannot be smuggled in
        via any Helper-authored spec — validation rejects it.
    (b) The /api/developer endpoints (the channel Big guy uses to
        extend the toolkit) are locked without a developer token.
    """
    # (a) spec validator refuses unknown widget types
    v = SpecValidator()
    with pytest.raises(SpecValidationError):
        v.validate_widget(_widget(type="flame_graph"))

    # (b) developer endpoints are gated
    monkeypatch.delenv("HELPER_DASHBOARD_DEV_TOKEN", raising=False)
    client = TestClient(app)

    r = client.get("/api/developer/tickets")
    assert r.status_code == 403

    # Even with a token env set, a client without the header is denied.
    monkeypatch.setenv("HELPER_DASHBOARD_DEV_TOKEN", "secret-xyz")
    r = client.get("/api/developer/tickets")
    assert r.status_code == 403
    r = client.get("/api/developer/tickets", headers={"X-Developer-Token": "wrong"})
    assert r.status_code == 403
    # Correct token succeeds.
    r = client.get("/api/developer/tickets", headers={"X-Developer-Token": "secret-xyz"})
    assert r.status_code == 200
