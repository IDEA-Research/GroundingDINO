"""Isolated git worktrees for real-OpenCode `developer_fix` runs.

The DevFix runtime (see `runtime.py:DevFixOpenCodeCLIRuntime`) never
lets the OpenCode CLI touch the live tree. Instead each run gets a
detached worktree of the repo:

  1. `create()`  — `git worktree add --detach` from HEAD, then overlay
     the main tree's *uncommitted tracked* helper-dashboard changes
     (`git diff HEAD -- helper-dashboard` piped into `git apply`), a
     `node_modules` symlink so `npm test`/`node` work, and the ticket
     payload at `<wt>/helper-dashboard/.devfix/payload.json`.
     `.env` is deliberately NOT copied.
  2. `collect_changes()` — `git status --porcelain` inside the
     worktree, scoped to helper-dashboard, `.devfix/` excluded.
  3. `copy_back()` — byte-for-byte copy of changed files into the main
     tree, but ONLY if *every* changed path passes the copy-back
     allow-list. All-or-nothing: one violation means zero copies.
  4. `capture_diff()` — unified diff of the in-run changes (untracked
     files as new-file diffs), taken BEFORE the worktree is removed so
     every terminal state keeps a reviewable patch.
  5. `remove()` / `sweep_stale()` — cleanup, tolerant of already-gone.

The copy-back allow-list below mirrors `auto_fix.py`'s write scope
INDEPENDENTLY (no import — two trust boundaries, same philosophy as
auto_fix.py's own mirror of bin/helpercode's tool-level list). Keep
the three in sync by hand.

Pure git + filesystem. No LLM, no network, no imports from runtime.py
(runtime.py imports us; keep this module dependency-free).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

#   .../helper-dashboard/backend/app/helper/devfix_worktree.py
#   -> helper/ -> app/ -> backend/ -> helper-dashboard/
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# helper-dashboard is a subdirectory of the git repo.
_DEFAULT_GIT_ROOT = _PROJECT_ROOT.parent

# Name of the project subdirectory inside the repo (same in every
# worktree).
_PROJECT_SUBDIR = "helper-dashboard"

_PAYLOAD_REL = ".devfix/payload.json"

_GIT_TIMEOUT_S = 60.0

# ---------------------------------------------------------------------------
# Copy-back allow-list — mirrors auto_fix.py's write scope INDEPENDENTLY.
# (Two trust boundaries; do not import from auto_fix.)
# ---------------------------------------------------------------------------

_COPYBACK_FILES: tuple[str, ...] = (
    "backend/app/services/browser_evaluator.py",
    "frontend/lib/renderer.tsx",
    "frontend/lib/spec-schema.ts",
)

_COPYBACK_DIRS: tuple[str, ...] = (
    "frontend/widget-toolkit",
    "tests/browser_evaluation",
    "tests/review_loop",
    "tests/spec_validation",
)


def is_copyback_allowed(rel_path: str) -> bool:
    """True iff `rel_path` (project-relative, forward slashes) may be
    copied back into the main tree."""
    rel = rel_path.replace("\\", "/").lstrip("/")
    if not rel or ".." in rel.split("/"):
        return False
    if rel in _COPYBACK_FILES:
        return True
    return any(rel.startswith(d + "/") for d in _COPYBACK_DIRS)


class CopyBackViolation(Exception):
    """At least one changed path fails the copy-back allow-list.
    Nothing was copied (all-or-nothing)."""

    def __init__(self, bad_paths: list[str]):
        self.bad_paths = list(bad_paths)
        super().__init__(
            "devfix changes outside allow-list: " + ", ".join(self.bad_paths)
        )


class WorktreeError(Exception):
    """A git worktree operation failed."""


# ---------------------------------------------------------------------------
# Handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorktreeHandle:
    ticket_id: str
    git_root: Path       # main repo root
    worktree_root: Path  # the detached worktree dir
    project_dir: Path    # worktree_root / "helper-dashboard"
    # Content hashes of paths already dirty right after create() — the
    # overlaid main-tree diff. collect_changes() subtracts these so
    # only changes made DURING the run count (a dirty main tree must
    # not poison the all-or-nothing copy-back check).
    baseline: dict[str, str] = field(default_factory=dict)
    # Bytes of the same baseline paths, so capture_diff() can diff the
    # run's edits against what the agent actually started from (not
    # against HEAD, which would re-report the overlay).
    baseline_content: dict[str, bytes | None] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _git(
    cwd: Path, *args: str, input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - shell=False, fixed argv
        ["git", "-C", str(cwd), *args],
        input=input_text,
        text=True,
        capture_output=True,
        timeout=_GIT_TIMEOUT_S,
        check=False,
        shell=False,
    )


def _git_or_raise(
    cwd: Path, *args: str, input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = _git(cwd, *args, input_text=input_text)
    if proc.returncode != 0:
        raise WorktreeError(
            f"git {' '.join(args)} failed (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '').strip()[:500]}"
        )
    return proc


def _sanitize_ticket_id(ticket_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(ticket_id or "manual")).strip("-.")
    return (slug or "manual")[:64]


def worktree_parent(git_root: Path | None = None) -> Path:
    """Parent dir for devfix worktrees. Overridable via env."""
    root = Path(git_root) if git_root is not None else _DEFAULT_GIT_ROOT
    raw = (os.getenv("HELPER_DASHBOARD_DEVFIX_WORKTREE_DIR") or "").strip()
    if raw:
        return Path(raw)
    return root / ".devfix-worktrees"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def create(
    ticket_id: str,
    *,
    payload: dict[str, Any] | None = None,
    git_root: Path | None = None,
) -> WorktreeHandle:
    """Create a detached worktree for `ticket_id` and prepare it for a
    devfix run. See module docstring for the exact steps."""
    root = Path(git_root) if git_root is not None else _DEFAULT_GIT_ROOT
    parent = worktree_parent(root)
    parent.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    wt = parent / f"{_sanitize_ticket_id(ticket_id)}-{stamp}"
    # Avoid collisions when two runs share a second.
    n = 0
    while wt.exists():
        n += 1
        wt = parent / f"{_sanitize_ticket_id(ticket_id)}-{stamp}-{n}"

    _git_or_raise(root, "worktree", "add", "--detach", str(wt), "HEAD")

    handle = WorktreeHandle(
        ticket_id=str(ticket_id),
        git_root=root,
        worktree_root=wt,
        project_dir=wt / _PROJECT_SUBDIR,
    )

    try:
        # Overlay uncommitted *tracked* helper-dashboard modifications
        # so the agent sees the same code the live backend runs. Empty
        # diffs are skipped entirely (this box's git predates
        # `apply --allow-empty`).
        diff = _git_or_raise(root, "diff", "HEAD", "--", _PROJECT_SUBDIR)
        if (diff.stdout or "").strip():
            _git_or_raise(wt, "apply", "-", input_text=diff.stdout)

        # node_modules symlink (worktrees don't carry ignored dirs).
        main_nm = root / _PROJECT_SUBDIR / "frontend" / "node_modules"
        wt_nm = handle.project_dir / "frontend" / "node_modules"
        if main_nm.is_dir() and not wt_nm.exists():
            wt_nm.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(main_nm.resolve()), str(wt_nm))

        # Ticket payload. Never copy .env.
        if payload is not None:
            payload_path = handle.project_dir / _PAYLOAD_REL
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8",
            )

        # Baseline: everything already dirty at this point came from
        # the overlay, not from the run.
        for rel in _status_changes(handle):
            handle.baseline[rel] = _content_hash(handle.project_dir / rel)
            handle.baseline_content[rel] = _read_bytes_or_none(
                handle.project_dir / rel
            )
    except Exception:
        # Half-built worktree is useless — clean up before re-raising.
        remove(handle)
        raise

    return handle


def _parse_porcelain_line(line: str) -> str | None:
    """One `git status --porcelain` line -> repo-relative path (or
    None for lines we can't use). Renames report the NEW path."""
    if len(line) < 4:
        return None
    path = line[3:]
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    path = path.strip()
    if path.startswith('"') and path.endswith('"'):
        # git C-quotes unusual paths; minimal unquote is enough here.
        path = path[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return path or None


_MISSING_HASH = "<missing>"


def _content_hash(path: Path) -> str:
    try:
        if not path.is_file() or path.is_symlink():
            return _MISSING_HASH
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return _MISSING_HASH


def _read_bytes_or_none(path: Path) -> bytes | None:
    try:
        if not path.is_file() or path.is_symlink():
            return None
        return path.read_bytes()
    except OSError:
        return None


# Paths that mutate as SIDE EFFECTS of running code in the worktree
# (pytest appends the session log via app.helper.memory, audit stores,
# bytecode caches). They are never agent work: invisible to
# collect_changes and therefore to the copy-back guard.
_RUNTIME_ARTIFACT_FILES: tuple[str, ...] = (
    "docs/USER_SESSION_LOG.md",
    "frontend/tsconfig.tsbuildinfo",
)
_RUNTIME_ARTIFACT_PREFIXES: tuple[str, ...] = (
    "backend/app/storage/",
    "logs/",
    ".pytest_cache/",
    "frontend/.next/",
)


def _is_runtime_artifact(rel: str) -> bool:
    if rel in _RUNTIME_ARTIFACT_FILES:
        return True
    if any(rel.startswith(p) for p in _RUNTIME_ARTIFACT_PREFIXES):
        return True
    return "__pycache__" in rel or rel.endswith(".pyc")


def _status_changes(handle: WorktreeHandle) -> list[str]:
    """Raw `git status --porcelain -uall` paths (project-relative),
    `.devfix/` excluded. -uall expands untracked dirs into files."""
    proc = _git_or_raise(
        handle.worktree_root,
        "status", "--porcelain", "-uall", "--", _PROJECT_SUBDIR,
    )
    prefix = _PROJECT_SUBDIR + "/"
    out: list[str] = []
    for line in (proc.stdout or "").splitlines():
        path = _parse_porcelain_line(line)
        if path is None or not path.startswith(prefix):
            continue
        rel = path[len(prefix):]
        if rel == ".devfix" or rel.startswith(".devfix/"):
            continue
        if _is_runtime_artifact(rel):
            continue
        out.append(rel)
    return sorted(set(out))


def collect_changes(handle: WorktreeHandle) -> list[str]:
    """Files changed DURING the run (modified + untracked + deleted),
    as project-relative paths. Paths whose content still matches the
    post-create baseline (the overlaid main-tree diff) are excluded —
    they are inherited state, not agent work."""
    out: list[str] = []
    for rel in _status_changes(handle):
        current = _content_hash(handle.project_dir / rel)
        if handle.baseline.get(rel) == current:
            continue
        out.append(rel)
    return out


def copy_back(
    handle: WorktreeHandle,
    allowlist_check: Callable[[str], bool] | None = None,
) -> list[str]:
    """Copy every changed file from the worktree into the main tree,
    byte-for-byte. ALL changed paths are checked against the allow-list
    BEFORE any copy happens; one violation raises CopyBackViolation and
    copies nothing. Deletions in the worktree delete the main-tree file.

    Returns the list of project-relative paths applied."""
    check = allowlist_check or is_copyback_allowed
    changes = collect_changes(handle)

    bad = [p for p in changes if not check(p)]
    if bad:
        raise CopyBackViolation(bad)

    applied: list[str] = []
    main_project = handle.git_root / _PROJECT_SUBDIR
    for rel in changes:
        src = handle.project_dir / rel
        dst = main_project / rel
        if src.is_symlink() or (src.exists() and not src.is_file()):
            # Directories/symlinks never copy back (node_modules link
            # is gitignored and should not appear here anyway).
            continue
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
        elif dst.exists():
            dst.unlink()
        applied.append(rel)
    return applied


# ---------------------------------------------------------------------------
# Diff capture (read-only; pure difflib + `git show` for HEAD content)
# ---------------------------------------------------------------------------

_DIFF_MAX_FILE_BYTES = 2_000_000


def _git_show_head_bytes(handle: WorktreeHandle, rel: str) -> bytes | None:
    """Content of `rel` at the worktree's HEAD, or None when untracked."""
    try:
        proc = subprocess.run(  # noqa: S603 - shell=False, fixed argv
            [
                "git", "-C", str(handle.worktree_root),
                "show", f"HEAD:{_PROJECT_SUBDIR}/{rel}",
            ],
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _looks_binary(data: bytes | None) -> bool:
    return data is not None and b"\0" in data[:8000]


def _unified_diff(rel: str, old: bytes | None, new: bytes | None) -> tuple[str, int, int]:
    """git-style unified diff for one path -> (text, additions, deletions).
    Untracked files come out as new-file diffs against /dev/null."""
    header = [f"diff --git a/{rel} b/{rel}"]
    if old is None:
        header.append("new file mode 100644")
    if new is None:
        header.append("deleted file mode 100644")
    a_name = "/dev/null" if old is None else f"a/{rel}"
    b_name = "/dev/null" if new is None else f"b/{rel}"
    if _looks_binary(old) or _looks_binary(new):
        header.append(f"Binary files {a_name} and {b_name} differ")
        return "\n".join(header), 0, 0
    if (old is not None and len(old) > _DIFF_MAX_FILE_BYTES) or (
        new is not None and len(new) > _DIFF_MAX_FILE_BYTES
    ):
        header.append(f"(file too large to diff: {a_name} -> {b_name})")
        return "\n".join(header), 0, 0
    old_lines = old.decode("utf-8", "replace").splitlines() if old is not None else []
    new_lines = new.decode("utf-8", "replace").splitlines() if new is not None else []
    body = list(difflib.unified_diff(
        old_lines, new_lines, fromfile=a_name, tofile=b_name, lineterm="",
    ))
    additions = sum(1 for ln in body if ln.startswith("+") and not ln.startswith("+++"))
    deletions = sum(1 for ln in body if ln.startswith("-") and not ln.startswith("---"))
    return "\n".join(header + body), additions, deletions


def capture_diff(
    handle: WorktreeHandle, changes: list[str] | None = None,
) -> dict[str, Any]:
    """{patch, files:[{path, additions, deletions, status}]} for the
    files changed DURING the run (`collect_changes` semantics). Never
    mutates the worktree; safe to call in any terminal state."""
    rels = list(changes) if changes is not None else collect_changes(handle)
    chunks: list[str] = []
    files: list[dict[str, Any]] = []
    for rel in rels:
        new = _read_bytes_or_none(handle.project_dir / rel)
        if rel in handle.baseline_content:
            old = handle.baseline_content.get(rel)
        else:
            old = _git_show_head_bytes(handle, rel)
        if old is None and new is None:
            continue
        status = "added" if old is None else ("deleted" if new is None else "modified")
        text, additions, deletions = _unified_diff(rel, old, new)
        chunks.append(text)
        files.append({
            "path": rel,
            "additions": additions,
            "deletions": deletions,
            "status": status,
        })
    patch = "\n".join(chunks)
    if patch:
        patch += "\n"
    return {"patch": patch, "files": files}


def remove(handle: WorktreeHandle) -> None:
    """Remove the worktree + prune. Tolerates already-gone."""
    _remove_worktree_dir(handle.git_root, handle.worktree_root)


def _remove_worktree_dir(git_root: Path, wt: Path) -> None:
    try:
        _git(git_root, "worktree", "remove", "--force", str(wt))
    except Exception:
        pass
    if wt.exists():
        shutil.rmtree(wt, ignore_errors=True)
    try:
        _git(git_root, "worktree", "prune")
    except Exception:
        pass


def sweep_stale(
    parent: Path | None = None,
    max_age_h: float = 24.0,
    *,
    git_root: Path | None = None,
) -> list[str]:
    """Remove leftover worktree dirs under `parent` older than
    `max_age_h` (mtime). Returns the removed dir names. Never raises."""
    root = Path(git_root) if git_root is not None else _DEFAULT_GIT_ROOT
    parent_dir = Path(parent) if parent is not None else worktree_parent(root)
    removed: list[str] = []
    try:
        if not parent_dir.is_dir():
            return removed
        cutoff = time.time() - max_age_h * 3600.0
        for child in sorted(parent_dir.iterdir()):
            try:
                if not child.is_dir():
                    continue
                if child.stat().st_mtime >= cutoff:
                    continue
                _remove_worktree_dir(root, child)
                removed.append(child.name)
            except Exception:
                continue
    except Exception:
        pass
    return removed
