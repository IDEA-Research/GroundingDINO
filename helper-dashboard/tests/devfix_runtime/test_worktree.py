"""devfix_worktree tests — against a scratch git repo in tmp_path.

No network, no LLM. The scratch repo mirrors the real layout: the
project lives in a `helper-dashboard/` subdirectory of the git root,
with one file on the copy-back allow-list and one off it.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from app.helper import devfix_worktree as dw


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        text=True, capture_output=True, check=True,
    )


@pytest.fixture()
def scratch_repo(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    proj = root / "helper-dashboard"
    (proj / "frontend" / "lib").mkdir(parents=True)
    (proj / "frontend" / "widget-toolkit").mkdir(parents=True)
    (proj / "backend" / "app").mkdir(parents=True)
    (proj / "frontend" / "lib" / "renderer.tsx").write_text(
        "export const A = 1;\n"
    )
    (proj / "frontend" / "widget-toolkit" / "Existing.tsx").write_text(
        "existing widget\n"
    )
    (proj / "backend" / "app" / "main.py").write_text("print('hi')\n")

    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")

    # Keep every worktree inside tmp_path, never near the real repo.
    monkeypatch.setenv(
        "HELPER_DASHBOARD_DEVFIX_WORKTREE_DIR", str(tmp_path / "wts")
    )
    return root


# ---------------------------------------------------------------------------
# Allow-list predicate
# ---------------------------------------------------------------------------


def test_is_copyback_allowed():
    assert dw.is_copyback_allowed("frontend/lib/renderer.tsx")
    assert dw.is_copyback_allowed("frontend/lib/spec-schema.ts")
    assert dw.is_copyback_allowed("backend/app/services/browser_evaluator.py")
    assert dw.is_copyback_allowed("frontend/widget-toolkit/NewWidget.tsx")
    assert dw.is_copyback_allowed("tests/spec_validation/test_new.py")
    assert dw.is_copyback_allowed("tests/review_loop/sub/dir/test_x.py")

    assert not dw.is_copyback_allowed("backend/app/main.py")
    assert not dw.is_copyback_allowed("frontend/lib/api.ts")
    assert not dw.is_copyback_allowed("bin/helpercode")
    assert not dw.is_copyback_allowed(".env")
    assert not dw.is_copyback_allowed("frontend/widget-toolkit")  # dir itself
    assert not dw.is_copyback_allowed("frontend/widget-toolkit/../lib/api.ts")
    assert not dw.is_copyback_allowed("")


# ---------------------------------------------------------------------------
# create / overlay
# ---------------------------------------------------------------------------


def test_create_overlays_dirty_tracked_changes(scratch_repo):
    root = scratch_repo
    # Uncommitted tracked modification in the main tree must be visible
    # inside the worktree (the live backend runs this code).
    (root / "helper-dashboard" / "frontend" / "lib" / "renderer.tsx").write_text(
        "export const A = 2;  // dirty\n"
    )

    h = dw.create(
        "ticket-1", payload={"operation": "developer_fix"}, git_root=root,
    )
    try:
        assert h.project_dir.is_dir()
        assert (
            h.project_dir / "frontend" / "lib" / "renderer.tsx"
        ).read_text() == "export const A = 2;  // dirty\n"

        payload = json.loads(
            (h.project_dir / ".devfix" / "payload.json").read_text()
        )
        assert payload == {"operation": "developer_fix"}
        assert not (h.project_dir / ".env").exists()

        # The overlay itself is baseline, not a run change.
        assert dw.collect_changes(h) == []
    finally:
        dw.remove(h)


def test_create_with_clean_tree_and_no_payload(scratch_repo):
    root = scratch_repo
    h = dw.create("t-clean", git_root=root)
    try:
        assert (
            h.project_dir / "frontend" / "lib" / "renderer.tsx"
        ).read_text() == "export const A = 1;\n"
        assert not (h.project_dir / ".devfix").exists()
        assert dw.collect_changes(h) == []
    finally:
        dw.remove(h)


# ---------------------------------------------------------------------------
# collect_changes
# ---------------------------------------------------------------------------


def test_collect_changes_modified_untracked_and_devfix_excluded(scratch_repo):
    root = scratch_repo
    h = dw.create("t-collect", payload={"x": 1}, git_root=root)
    try:
        (h.project_dir / "frontend" / "widget-toolkit" / "Existing.tsx").write_text(
            "changed by agent\n"
        )
        (h.project_dir / "frontend" / "widget-toolkit" / "New.tsx").write_text(
            "brand new widget\n"
        )
        (h.project_dir / ".devfix" / "scratch.json").write_text("{}")

        assert dw.collect_changes(h) == [
            "frontend/widget-toolkit/Existing.tsx",
            "frontend/widget-toolkit/New.tsx",
        ]
    finally:
        dw.remove(h)


# ---------------------------------------------------------------------------
# copy_back — all-or-nothing allow-list
# ---------------------------------------------------------------------------


def test_copy_back_refuses_all_when_any_path_off_list(scratch_repo):
    root = scratch_repo
    h = dw.create("t-violation", git_root=root)
    try:
        (h.project_dir / "frontend" / "lib" / "renderer.tsx").write_text(
            "export const A = 99;  // allowed change\n"
        )
        (h.project_dir / "backend" / "app" / "main.py").write_text(
            "print('EVIL')\n"
        )

        with pytest.raises(dw.CopyBackViolation) as exc:
            dw.copy_back(h)
        assert "backend/app/main.py" in str(exc.value)

        # All-or-nothing: the ALLOWED change must not have been copied.
        assert (
            root / "helper-dashboard" / "frontend" / "lib" / "renderer.tsx"
        ).read_text() == "export const A = 1;\n"
        assert (
            root / "helper-dashboard" / "backend" / "app" / "main.py"
        ).read_text() == "print('hi')\n"
    finally:
        dw.remove(h)


def test_copy_back_applies_allowed_changes_byte_for_byte(scratch_repo):
    root = scratch_repo
    h = dw.create("t-apply", git_root=root)
    try:
        new_renderer = "export const A = 42;  // fixed\n"
        new_widget = b"new widget bytes \xf0\x9f\x9b\xa0\n"
        (h.project_dir / "frontend" / "lib" / "renderer.tsx").write_text(
            new_renderer
        )
        (h.project_dir / "frontend" / "widget-toolkit" / "New.tsx").write_bytes(
            new_widget
        )

        applied = dw.copy_back(h)
        assert applied == [
            "frontend/lib/renderer.tsx",
            "frontend/widget-toolkit/New.tsx",
        ]
        assert (
            root / "helper-dashboard" / "frontend" / "lib" / "renderer.tsx"
        ).read_text() == new_renderer
        assert (
            root / "helper-dashboard" / "frontend" / "widget-toolkit" / "New.tsx"
        ).read_bytes() == new_widget
    finally:
        dw.remove(h)


def test_copy_back_custom_predicate(scratch_repo):
    root = scratch_repo
    h = dw.create("t-pred", git_root=root)
    try:
        (h.project_dir / "frontend" / "lib" / "renderer.tsx").write_text("x\n")
        with pytest.raises(dw.CopyBackViolation):
            dw.copy_back(h, allowlist_check=lambda p: False)
        assert (
            root / "helper-dashboard" / "frontend" / "lib" / "renderer.tsx"
        ).read_text() == "export const A = 1;\n"
    finally:
        dw.remove(h)


# ---------------------------------------------------------------------------
# remove / sweep
# ---------------------------------------------------------------------------


def test_remove_is_idempotent_and_leaves_repo_clean(scratch_repo):
    root = scratch_repo
    h = dw.create("t-remove", git_root=root)
    wt = h.worktree_root
    assert wt.is_dir()

    dw.remove(h)
    assert not wt.exists()
    listed = _git(root, "worktree", "list", "--porcelain").stdout
    assert str(wt) not in listed

    # Already-gone must be tolerated.
    dw.remove(h)


def test_sweep_stale_removes_only_old_worktrees(scratch_repo, tmp_path):
    root = scratch_repo
    h_old = dw.create("t-old", git_root=root)
    h_new = dw.create("t-new", git_root=root)
    try:
        old_ts = time.time() - 48 * 3600
        os.utime(h_old.worktree_root, (old_ts, old_ts))

        removed = dw.sweep_stale(max_age_h=24, git_root=root)

        assert h_old.worktree_root.name in removed
        assert not h_old.worktree_root.exists()
        assert h_new.worktree_root.is_dir()
        listed = _git(root, "worktree", "list", "--porcelain").stdout
        assert str(h_old.worktree_root) not in listed
    finally:
        dw.remove(h_new)
        dw.remove(h_old)


def test_sweep_stale_missing_parent_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "HELPER_DASHBOARD_DEVFIX_WORKTREE_DIR", str(tmp_path / "does-not-exist")
    )
    assert dw.sweep_stale(git_root=tmp_path) == []
