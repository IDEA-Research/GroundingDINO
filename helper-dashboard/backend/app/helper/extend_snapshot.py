"""M7 — snapshot + auto-rollback for the rescue_extend write set.

Before Big guy starts editing, we record the original content of every
file in the per-widget_type write allow-list. If the extend run ends
in failure (LLM rejected its own work, tests failed, runtime crashed,
gate post-hoc detected something bad), we restore the originals
surgically — file by file, no git operations.

Why not git stash / git reset?

  - The helper-dashboard tree is one subdirectory of a larger repo;
    git tags / resets would affect more than our extension.
  - The working tree often has unrelated uncommitted changes (logs,
    auto-generated artifacts). git stash would scoop those up too,
    making the safety boundary fuzzy.
  - Surgical per-file restore makes the audit log easy to verify:
    exactly the six write-listed paths are touched, never more.

The snapshot also persists to disk under
`backend/app/storage/extend_audit/snapshots/<ts>_<widget_type>/`
so a crashed Python process can still be recovered by a developer
running `restore_from(load_snapshot_from_disk(path))`.

M7 Phase 1 (this milestone): in-process snapshot + on-failure
auto-restore. Phase 2 (later): a developer-facing `/api/developer/
rollback` endpoint and a UI to list recent snapshots.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# `extend_snapshot.py` lives at:
#   helper-dashboard/backend/app/helper/extend_snapshot.py
# so parents[3] resolves to the helper-dashboard/ root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SNAPSHOTS_DIR = (
    _REPO_ROOT / "backend" / "app" / "storage" / "extend_audit" / "snapshots"
)

# Lowercase snake_case, 3-32 chars, must start with a letter. Mirrors
# the regex in extend_gate; redefined here so this module has no
# import-time coupling to other helpers.
_WIDGET_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{2,31}$")


def _camel(widget_type: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in widget_type.split("_") if p)


def _allowed_paths_for(widget_type: str) -> tuple[Path, ...]:
    """Six absolute paths the extend run is allowed to write — the
    same list bin/opencode's `_extend_write_allowlist` returns, kept
    in sync via the same conventions (NOT a shared import — the two
    are deliberately independent trust boundaries)."""
    cam = _camel(widget_type)
    return (
        _REPO_ROOT / "backend" / "app" / "specs" / "widget_spec.py",
        _REPO_ROOT / "backend" / "app" / "specs" / "widget_schema_doc.py",
        _REPO_ROOT / "frontend" / "lib" / "spec-schema.ts",
        _REPO_ROOT / "frontend" / "lib" / "renderer.tsx",
        _REPO_ROOT / "frontend" / "widget-toolkit" / f"{cam}Widget.tsx",
        _REPO_ROOT / "tests" / "spec_validation" / f"test_extend_{widget_type}.py",
    )


@dataclass
class SnapshotHandle:
    """An in-memory + on-disk record of file state before the
    extend run started.

    `files` maps absolute Path → (existed_before, original_bytes).
    `disk_dir` is the directory on disk where we also persisted the
    originals (one file each, mirroring the relative path).
    """

    widget_type: str
    files: dict[Path, tuple[bool, bytes]] = field(default_factory=dict)
    disk_dir: Path | None = None
    timestamp: str = ""


def take_snapshot(widget_type: str) -> SnapshotHandle:
    """Snapshot every allow-listed path. Files that don't exist yet
    are recorded as `(False, b"")` so restore can delete them on
    failure.

    Raises ValueError if widget_type fails the snake_case regex —
    the gate should have caught this, but we double-check because
    the snapshot creates a filesystem path from widget_type.
    """
    if not _WIDGET_TYPE_RE.match(widget_type):
        raise ValueError(f"invalid widget_type for snapshot: {widget_type!r}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    disk_dir = _SNAPSHOTS_DIR / f"{ts}_{widget_type}"

    handle = SnapshotHandle(widget_type=widget_type, timestamp=ts,
                              disk_dir=disk_dir)

    try:
        disk_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Disk snapshot is best-effort; in-memory is the source of truth.
        disk_dir = None
        handle.disk_dir = None

    manifest: list[dict[str, Any]] = []
    for path in _allowed_paths_for(widget_type):
        if path.exists():
            try:
                data = path.read_bytes()
            except Exception:
                # File exists but unreadable — record as missing-but-noted.
                data = b""
                existed = False
            else:
                existed = True
        else:
            data = b""
            existed = False

        handle.files[path] = (existed, data)

        if disk_dir is not None:
            try:
                rel = path.relative_to(_REPO_ROOT)
                dest = disk_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if existed:
                    dest.write_bytes(data)
                else:
                    # Empty sentinel file with a `.absent` suffix marks
                    # "this file did not exist at snapshot time".
                    absent = dest.with_suffix(dest.suffix + ".absent")
                    absent.write_text("absent at snapshot", encoding="utf-8")
            except Exception:
                # Disk snapshot failure does NOT compromise the
                # in-memory snapshot — restore still works.
                pass

        manifest.append({
            "path": str(path.relative_to(_REPO_ROOT)),
            "existed_before": existed,
            "bytes": len(data),
        })

    if disk_dir is not None:
        try:
            (disk_dir / "manifest.json").write_text(
                json.dumps({"widget_type": widget_type, "ts": ts,
                             "files": manifest}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    return handle


def _safe_relative(path: Path) -> str:
    """Best-effort `relative_to(_REPO_ROOT)`; falls back to str(path)
    for paths outside the repo. Used only in audit-log strings, never
    for security decisions."""
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def restore_from(handle: SnapshotHandle) -> dict[str, Any]:
    """Restore every snapshotted file to its pre-extend state.

    For files that didn't exist at snapshot time but exist now, the
    restore *deletes* them — they were created by Big guy. For files
    that existed, restore overwrites with the original bytes.

    Returns a small dict {restored, deleted, errors} for the audit log.
    Never raises (auto-rollback must not crash the request).
    """
    restored: list[str] = []
    deleted: list[str] = []
    errors: list[str] = []

    for path, (existed_before, original) in handle.files.items():
        try:
            if existed_before:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(original)
                restored.append(_safe_relative(path))
            else:
                if path.exists():
                    path.unlink()
                    deleted.append(_safe_relative(path))
        except Exception as exc:
            errors.append(
                f"{_safe_relative(path)}: {type(exc).__name__}: {exc}"
            )

    return {
        "widget_type": handle.widget_type,
        "timestamp": handle.timestamp,
        "restored": restored,
        "deleted": deleted,
        "errors": errors,
    }


def cleanup_old_snapshots(days: int = 7) -> int:
    """Remove snapshot directories older than `days` days. Returns
    the number of directories removed. Intended for a cron / startup
    hook; called manually in tests."""
    if not _SNAPSHOTS_DIR.exists():
        return 0
    now = datetime.now(timezone.utc)
    removed = 0
    for entry in _SNAPSHOTS_DIR.iterdir():
        if not entry.is_dir():
            continue
        # Directory name format: YYYYMMDDTHHMMSSZ_<widget_type>
        m = re.match(r"^(\d{8}T\d{6}Z)_", entry.name)
        if not m:
            continue
        try:
            stamp = datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc,
            )
        except ValueError:
            continue
        age_days = (now - stamp).total_seconds() / 86400.0
        if age_days > days:
            try:
                shutil.rmtree(entry)
                removed += 1
            except Exception:
                pass
    return removed
