"""Disk-backed run log store for devfix ("Big Guy") runs.

Every `developer_fix` run — whether it executes inside the uvicorn
process (auto_fix daemon thread) or from a standalone script — streams
its OpenCode CLI events into
`backend/app/storage/devfix_runs/<run_id>/` AS THEY ARRIVE. Disk is the
source of truth: the read API (`/api/devfix`) needs no in-memory state
and sees runs from any process.

Files per run:

- ``meta.json``    {run_id, ticket_id, status, started_at, finished_at,
                    error_kind, model, instruction_preview}
- ``events.jsonl`` raw CLI events, one JSON object per line
- ``log.txt``      human-readable rendered lines, appended in lockstep

Every append is flushed immediately — the whole point is live tailing.

Rendering notes: the CLI may re-emit a growing ``text``/``reasoning``
part several times (accumulating snapshots). Those are buffered per
part and flushed as ONE line when the next different event arrives (or
on finish), so the log stays one-line-per-thought instead of spamming
partial repeats.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Sanitized run ids only — this doubles as the path-traversal guard for
# `read_run` (no "/", no "..", nothing outside [A-Za-z0-9_-]).
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,200}$")
_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")

_KEEP_LATEST = 50
_TERMINAL_STATUSES = frozenset({"resolved", "rejected", "needs_context", "failed"})

_TOOL_ICONS = {
    "edit": "✏️",
    "write": "✏️",
    "patch": "✏️",
    "bash": "▶️",
    "read": "📖",
}


def _runs_root() -> Path:
    """Storage root; env override so tests/scripts can redirect it."""
    override = (os.getenv("HELPER_DASHBOARD_DEVFIX_RUNS_DIR") or "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "storage" / "devfix_runs"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hhmmss(ts_ms: Any) -> str:
    """HH:MM:SS (server-local) from an event ms timestamp; falls back to
    'now' when the event carries no usable timestamp."""
    try:
        return datetime.fromtimestamp(float(ts_ms) / 1000.0).strftime("%H:%M:%S")
    except (TypeError, ValueError, OverflowError, OSError):
        return datetime.now().strftime("%H:%M:%S")


def _preview(value: Any, limit: int) -> str:
    """Single-line preview: collapse whitespace, hard-cap length."""
    if isinstance(value, str):
        s = " ".join(value.split())
    elif value is None:
        s = ""
    else:
        try:
            s = " ".join(json.dumps(value, ensure_ascii=False).split())
        except (TypeError, ValueError):
            s = " ".join(str(value).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _read_meta(run_dir: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------------------
# writer side
# ---------------------------------------------------------------------------


class RunLogger:
    """Append-only writer for one run. Thread-safe: the stdout reader
    thread calls `event()` while the runtime's main thread calls
    `note()`/`finish()`."""

    def __init__(self, run_dir: Path, meta: dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.run_id = str(meta["run_id"])
        self._meta = dict(meta)
        self._lock = threading.Lock()
        # Buffered streaming text/reasoning part:
        # (kind, part_id, first_seen_hhmmss, latest_text)
        self._pending: tuple[str, Any, str, str] | None = None
        self._finished = False
        self._events_f = open(  # noqa: SIM115 - long-lived, closed in finish()
            self.run_dir / "events.jsonl", "a", encoding="utf-8"
        )
        self._log_f = open(  # noqa: SIM115 - long-lived, closed in finish()
            self.run_dir / "log.txt", "a", encoding="utf-8"
        )
        self._write_meta()

    # -- public API --------------------------------------------------

    def event(self, raw: str | dict[str, Any]) -> None:
        """Record one raw CLI event (JSON line or already-parsed dict).
        Non-JSON noise lines are silently skipped."""
        if isinstance(raw, dict):
            ev: dict[str, Any] | None = raw
        elif isinstance(raw, str):
            s = raw.strip()
            if not s:
                return
            try:
                obj = json.loads(s)
            except (json.JSONDecodeError, ValueError):
                return
            ev = obj if isinstance(obj, dict) else None
        else:
            ev = None
        if ev is None:
            return
        with self._lock:
            if self._finished:
                return
            self._events_f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self._events_f.flush()
            self._render_event(ev)

    def note(self, text: str) -> None:
        """Runtime-side annotation (worktree path, copy-back result, …)."""
        with self._lock:
            if self._finished:
                return
            self._flush_pending()
            now = datetime.now().strftime("%H:%M:%S")
            self._append_line(now, f"ℹ️ {_preview(text, 400)}")

    def finish(self, status: str, error_kind: str | None = None) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._flush_pending()
            if status not in _TERMINAL_STATUSES:
                status = "failed"
            tail = f"🏁 run finished: {status}"
            if error_kind:
                tail += f" ({error_kind})"
            self._append_line(datetime.now().strftime("%H:%M:%S"), tail)
            self._meta["status"] = status
            self._meta["finished_at"] = _utc_now_iso()
            if error_kind is not None:
                self._meta["error_kind"] = str(error_kind)
            self._write_meta()
            for f in (self._events_f, self._log_f):
                try:
                    f.close()
                except OSError:  # pragma: no cover - defensive
                    pass

    # -- internals ---------------------------------------------------

    def _write_meta(self) -> None:
        tmp = self.run_dir / "meta.json.tmp"
        tmp.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.run_dir / "meta.json")

    def _append_line(self, ts: str, line: str) -> None:
        self._log_f.write(f"{ts} {line}\n")
        self._log_f.flush()

    def _flush_pending(self) -> None:
        if self._pending is None:
            return
        kind, _pid, ts, text = self._pending
        self._pending = None
        if kind == "reasoning":
            self._append_line(ts, f"💭 {_preview(text, 200)}")
        else:  # text
            self._append_line(ts, f"💬 {_preview(text, 300)}")

    def _render_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("type")
        ts = _hhmmss(ev.get("timestamp"))
        part = ev.get("part") if isinstance(ev.get("part"), dict) else {}

        if kind in ("text", "reasoning"):
            text = part.get("text")
            if not isinstance(text, str) or not text.strip():
                return
            pid = part.get("id")
            if self._pending is not None:
                p_kind, p_pid, p_ts, p_text = self._pending
                same_part = p_kind == kind and (
                    (pid is not None and pid == p_pid)
                    or (pid is None and p_pid is None and text.startswith(p_text))
                )
                if same_part:
                    # Accumulating snapshot of the same part: keep the
                    # first timestamp, latest text.
                    self._pending = (kind, pid, p_ts, text)
                    return
                self._flush_pending()
            self._pending = (str(kind), pid, ts, text)
            return

        # Any other event terminates a streaming text/reasoning part.
        self._flush_pending()
        line = self._render_other(str(kind), part)
        if line:
            self._append_line(ts, line)

    def _render_other(self, kind: str, part: dict[str, Any]) -> str | None:
        if kind == "tool_use":
            return self._render_tool_use(part)
        if kind == "error":
            msg: Any = None
            for cand in (part.get("message"), part.get("error")):
                if isinstance(cand, str) and cand.strip():
                    msg = cand
                    break
            if msg is None:
                data = part.get("data")
                if isinstance(data, dict) and isinstance(data.get("message"), str):
                    msg = data["message"]
            if msg is None:
                msg = part or "unknown error"
            return f"❌ {_preview(msg, 300)}"
        # step_start / step_finish and unknown event types: skipped for
        # compactness.
        return None

    @staticmethod
    def _render_tool_use(part: dict[str, Any]) -> str | None:
        tool = part.get("tool") if isinstance(part.get("tool"), str) else "tool"
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        status = state.get("status")
        if status in ("pending", "running"):
            return None  # only terminal states make a log line
        title = state.get("title") if isinstance(state.get("title"), str) else ""
        if not title:
            title = _preview(state.get("input"), 120)
        head = f"{tool} {title}".strip()
        if status == "error":
            err = ""
            for cand in (state.get("error"), state.get("message"), state.get("output")):
                if isinstance(cand, str) and cand.strip():
                    err = cand
                    break
            low = err.lower()
            if "permission" in low or "reject" in low or "denied" in low:
                return f"⛔ {head} → permission rejected"
            return f"❌ {head} → {_preview(err or 'error', 200)}"
        icon = _TOOL_ICONS.get(str(tool).lower(), "🔧")
        if status == "completed":
            return f"{icon} {head} ✓"
        return f"{icon} {head}"


# ---------------------------------------------------------------------------
# module API
# ---------------------------------------------------------------------------


def start_run(ticket_id: str, instruction: str, model: str) -> RunLogger:
    """Create the run directory + initial meta and return its logger.
    Also sweeps old runs (keep the newest ``_KEEP_LATEST``)."""
    root = _runs_root()
    root.mkdir(parents=True, exist_ok=True)
    slug = _UNSAFE_CHARS_RE.sub("-", str(ticket_id or "manual")).strip("-") or "run"
    slug = slug[:80]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"{slug}-{stamp}"
    run_dir = root / run_id
    n = 2
    while run_dir.exists():
        run_id = f"{slug}-{stamp}-{n}"
        run_dir = root / run_id
        n += 1
    run_dir.mkdir(parents=True)
    meta = {
        "run_id": run_id,
        "ticket_id": str(ticket_id),
        "status": "running",
        "started_at": _utc_now_iso(),
        "finished_at": None,
        "error_kind": None,
        "model": str(model),
        "instruction_preview": _preview(instruction or "", 200),
    }
    logger = RunLogger(run_dir, meta)
    try:
        keep_latest(_KEEP_LATEST)
    except OSError:  # pragma: no cover - retention is best-effort
        pass
    return logger


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Newest-first run metas (broken/partial dirs are skipped)."""
    root = _runs_root()
    if not root.is_dir():
        return []
    metas: list[dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_dir() or not _RUN_ID_RE.match(child.name):
            continue
        meta = _read_meta(child)
        if meta is not None:
            metas.append(meta)
    metas.sort(key=lambda m: str(m.get("started_at") or ""), reverse=True)
    return metas[: max(0, int(limit))]


def read_run(run_id: str, after: int = 0) -> dict[str, Any]:
    """Meta + rendered log lines from cursor `after` (line index).
    Raises KeyError for unknown or malformed run ids (the API maps that
    to 404); the run-id pattern doubles as the path-traversal guard."""
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise KeyError(f"invalid run_id: {run_id!r}")
    run_dir = _runs_root() / run_id
    meta = _read_meta(run_dir)
    if meta is None:
        raise KeyError(f"unknown run_id: {run_id!r}")
    lines: list[str] = []
    log_path = run_dir / "log.txt"
    if log_path.is_file():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    after = max(0, int(after))
    return {"meta": meta, "lines": lines[after:], "next_cursor": len(lines)}


def keep_latest(keep: int = _KEEP_LATEST) -> int:
    """Delete all but the `keep` newest run dirs; returns #removed."""
    root = _runs_root()
    if not root.is_dir():
        return 0
    entries: list[tuple[str, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not _RUN_ID_RE.match(child.name):
            continue  # never rmtree anything we did not name ourselves
        meta = _read_meta(child)
        key = str((meta or {}).get("started_at") or "")
        if not key:
            try:
                key = datetime.fromtimestamp(
                    child.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            except OSError:
                key = ""
        entries.append((key, child))
    entries.sort(key=lambda e: e[0], reverse=True)
    removed = 0
    for _, child in entries[max(0, int(keep)):]:
        try:
            shutil.rmtree(child)
            removed += 1
        except OSError:  # pragma: no cover - best-effort sweep
            pass
    return removed
