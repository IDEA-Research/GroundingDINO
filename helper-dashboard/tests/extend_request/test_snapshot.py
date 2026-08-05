"""M7: snapshot + auto-rollback tests for the rescue_extend write set."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.helper import extend_snapshot
from app.helper.extend_snapshot import (
    _REPO_ROOT,
    _allowed_paths_for,
    cleanup_old_snapshots,
    restore_from,
    take_snapshot,
)


@pytest.fixture
def isolated_snapshots_dir(tmp_path, monkeypatch):
    """Redirect snapshot directory into tmp_path so tests don't
    pollute the real audit storage."""
    monkeypatch.setattr(extend_snapshot, "_SNAPSHOTS_DIR", tmp_path / "snaps")
    return tmp_path / "snaps"


# ---------------------------------------------------------------------------
# Snapshot integrity
# ---------------------------------------------------------------------------


def test_snapshot_records_existing_files(isolated_snapshots_dir):
    # Use a widget_type that definitely never has a component file on
    # disk — pie_chart is now real after the live extend on 2026-05-26.
    handle = take_snapshot("nonexistent_chart")
    spec = _REPO_ROOT / "backend" / "app" / "specs" / "widget_spec.py"
    component = (
        _REPO_ROOT / "frontend" / "widget-toolkit" / "NonexistentChartWidget.tsx"
    )
    assert handle.files[spec][0] is True   # existed before
    assert b"WidgetType" in handle.files[spec][1]
    assert handle.files[component][0] is False  # did not exist
    assert handle.files[component][1] == b""


def test_snapshot_writes_disk_manifest(isolated_snapshots_dir):
    handle = take_snapshot("heatmap")
    assert handle.disk_dir is not None
    manifest = handle.disk_dir / "manifest.json"
    assert manifest.exists()
    text = manifest.read_text(encoding="utf-8")
    assert "heatmap" in text
    assert "widget_spec.py" in text


def test_snapshot_rejects_invalid_widget_type(isolated_snapshots_dir):
    with pytest.raises(ValueError):
        take_snapshot("PieChart")   # uppercase
    with pytest.raises(ValueError):
        take_snapshot("../etc")     # path traversal-shaped
    with pytest.raises(ValueError):
        take_snapshot("")           # empty


# ---------------------------------------------------------------------------
# Restore integrity
# ---------------------------------------------------------------------------


def test_restore_overwrites_modified_existing_file(isolated_snapshots_dir, tmp_path):
    # Use a temp file mocked into the allowlist.
    fake_target = tmp_path / "fake_widget_spec.py"
    fake_target.write_text("original\n", encoding="utf-8")

    handle = extend_snapshot.SnapshotHandle(widget_type="pie_chart")
    handle.files[fake_target.resolve()] = (True, b"original\n")

    # Big guy modifies the file.
    fake_target.write_text("EVIL EDIT\n", encoding="utf-8")

    result = restore_from(handle)
    assert fake_target.read_text(encoding="utf-8") == "original\n"
    # The path is outside _REPO_ROOT, so _safe_relative returns the
    # absolute path — check by suffix.
    assert any(p.endswith("fake_widget_spec.py") for p in result["restored"])
    assert result["errors"] == []


def test_restore_deletes_files_created_by_extend(isolated_snapshots_dir, tmp_path):
    fake_new = tmp_path / "fake_new_widget.tsx"

    handle = extend_snapshot.SnapshotHandle(widget_type="pie_chart")
    handle.files[fake_new.resolve()] = (False, b"")  # didn't exist before

    # Big guy creates the file.
    fake_new.write_text("created by big guy", encoding="utf-8")
    assert fake_new.exists()

    result = restore_from(handle)
    assert not fake_new.exists()
    assert any(p.endswith("fake_new_widget.tsx") for p in result["deleted"])


def test_restore_does_not_raise_on_io_error(isolated_snapshots_dir):
    """A snapshot referring to an unwritable path must not crash the
    rollback flow — errors are collected and returned, not raised."""
    handle = extend_snapshot.SnapshotHandle(widget_type="pie_chart")
    handle.files[Path("/proc/this/cannot/exist")] = (True, b"x")
    result = restore_from(handle)
    # Either an error gets recorded (if write fails) or it gets restored
    # somewhere (unlikely under /proc). Either way: no raise.
    assert isinstance(result, dict)
    assert "errors" in result


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_cleanup_removes_old_dirs(isolated_snapshots_dir):
    snaps = isolated_snapshots_dir
    snaps.mkdir(parents=True, exist_ok=True)
    # Make two snapshot dirs — one fresh, one ancient.
    fresh = snaps / "20991231T000000Z_pie_chart"
    old   = snaps / "20200101T000000Z_old_widget"
    fresh.mkdir()
    old.mkdir()
    (fresh / "manifest.json").write_text("{}", encoding="utf-8")
    (old / "manifest.json").write_text("{}", encoding="utf-8")

    removed = cleanup_old_snapshots(days=30)
    assert removed == 1
    assert fresh.exists()
    assert not old.exists()


def test_cleanup_handles_missing_dir(isolated_snapshots_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(extend_snapshot, "_SNAPSHOTS_DIR",
                         tmp_path / "does_not_exist")
    assert cleanup_old_snapshots(days=7) == 0


def test_cleanup_ignores_non_snapshot_dirs(isolated_snapshots_dir):
    snaps = isolated_snapshots_dir
    snaps.mkdir(parents=True, exist_ok=True)
    rogue = snaps / "not-a-snapshot-name"
    rogue.mkdir()
    (rogue / "stuff.txt").write_text("x", encoding="utf-8")
    cleanup_old_snapshots(days=1)
    assert rogue.exists()  # untouched


# ---------------------------------------------------------------------------
# Path-list shape
# ---------------------------------------------------------------------------


def test_allowed_paths_contain_camel_cased_component():
    paths = _allowed_paths_for("pie_chart")
    component = paths[4]  # 5th entry — the new tsx
    assert component.name == "PieChartWidget.tsx"

    paths2 = _allowed_paths_for("multi_word_thing")
    assert paths2[4].name == "MultiWordThingWidget.tsx"


def test_allowed_paths_contain_test_file():
    paths = _allowed_paths_for("pie_chart")
    test_path = paths[5]
    assert test_path.name == "test_extend_pie_chart.py"
