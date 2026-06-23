"""Draft-shadow fix tests.

Verifies that during the review loop, the frontend can fetch the
in-progress draft via /api/dashboard/<id> even though it hasn't
been saved to disk yet. Before this fix, every review loop failed
because the Playwright evaluator got 404s.
"""

from __future__ import annotations

from app.services.dashboard_store import DashboardStore
from app.specs import DashboardSpec


def _spec(did="demo-draft"):
    return DashboardSpec.model_validate({
        "dashboard_id": did,
        "title": "Draft",
        "description": "",
        "layout": {"columns": 12, "row_height": 40},
        "variables": [],
        "widgets": [],
        "refresh_interval": "30s",
    })


def test_draft_visible_via_load(tmp_path):
    store = DashboardStore(root=tmp_path)
    spec = _spec()
    # Not saved yet — no file on disk.
    assert store.load_dashboard(spec.dashboard_id) is None

    # Set draft shadow.
    store.set_draft(spec)
    loaded = store.load_dashboard(spec.dashboard_id)
    assert loaded is not None
    assert loaded.dashboard_id == "demo-draft"

    # Clear.
    store.clear_draft(spec.dashboard_id)
    assert store.load_dashboard(spec.dashboard_id) is None


def test_draft_cleared_on_promote(tmp_path):
    store = DashboardStore(root=tmp_path)
    spec = _spec()
    store.set_draft(spec)
    store.save_dashboard(spec)
    # After save, draft entry is dropped but file serves the load.
    assert spec.dashboard_id not in DashboardStore._DRAFTS
    assert store.load_dashboard(spec.dashboard_id) is not None


def test_draft_shadow_is_process_wide(tmp_path):
    """Multiple Store instances share the in-memory shadow because
    it's class-level. This matches how review_loop/orchestrator/api
    share the state via their own instances."""
    a = DashboardStore(root=tmp_path)
    b = DashboardStore(root=tmp_path)
    spec = _spec("shared")
    a.set_draft(spec)
    assert b.load_dashboard("shared") is not None
    b.clear_draft("shared")
    assert a.load_dashboard("shared") is None


def test_draft_takes_precedence_over_saved_disk(tmp_path):
    """When the review loop is mid-flight and a draft is active for an
    id that also has a saved disk version, the draft must win — not
    the stale saved disk. Without this, LLMs that reuse common ids
    like `k8s-node-cpu-memory` see the previous save shadow the new
    draft and review_loop reports widgets missing forever.

    This test pins the M5 fix to the C-1 ordering bug found in the
    live walk-through demo on 2026-05-26."""
    store = DashboardStore(root=tmp_path)
    on_disk = _spec()
    on_disk = on_disk.model_copy(update={"title": "OLD_SAVED"})
    store.save_dashboard(on_disk)

    draft = on_disk.model_copy(update={"title": "NEW_DRAFT"})
    store.set_draft(draft)

    loaded = store.load_dashboard(on_disk.dashboard_id)
    assert loaded.title == "NEW_DRAFT", (
        "draft must shadow saved disk during review loop"
    )

    # After review ends and clear_draft fires, the saved disk version
    # is what callers see again.
    store.clear_draft(on_disk.dashboard_id)
    after = store.load_dashboard(on_disk.dashboard_id)
    assert after.title == "OLD_SAVED"


def test_draft_survives_in_memory_loss(tmp_path):
    """The disk shadow (`_drafts/<id>.json`) must let a draft survive
    even if the in-memory `_DRAFTS` dict is wiped — this simulates a
    second uvicorn worker handling the GET, which has no shared
    memory with the worker that set the draft."""
    store = DashboardStore(root=tmp_path)
    spec = _spec("cross-worker")
    store.set_draft(spec)

    # Simulate the other worker: clear the in-memory dict but leave
    # the on-disk shadow intact.
    DashboardStore._DRAFTS.pop("cross-worker", None)

    loaded = store.load_dashboard("cross-worker")
    assert loaded is not None
    assert loaded.dashboard_id == "cross-worker"

    # clear_draft must remove BOTH memory and disk so the next
    # session starts clean.
    store.clear_draft("cross-worker")
    assert store.load_dashboard("cross-worker") is None
    assert not DashboardStore._draft_path("cross-worker").exists()
