"""Disk-backed run log store for devfix ("Big Guy") runs.

Every `developer_fix` run — whether it executes inside the uvicorn
process (auto_fix daemon thread, the manual run queue) or from a
standalone script — streams its OpenCode CLI events into
`backend/app/storage/devfix_runs/<run_id>/` AS THEY ARRIVE. Disk is the
source of truth: the read API (`/api/devfix`) needs no in-memory state
and sees runs from any process.

Files per run:

- ``meta.json``    run metadata (see ``_meta_defaults`` for the full
                   additive schema: status, source, instruction, pid /
                   pgid, files_read / files_changed, tests, applied,
                   verification, failure_summary, report_status, ...)
- ``events.jsonl`` raw CLI events, one JSON object per line (scrubbed)
- ``log.txt``      human-readable rendered lines, appended in lockstep
- ``diff.patch``   unified diff of the worktree changes, captured
                   before the worktree is removed (any terminal state)

Status vocabulary (``meta.status``)::

    queued | running | waiting | success | failed | cancelled | timeout

``waiting`` = Big Guy reported ``rejected`` / ``needs_context`` and a
developer must decide; ``success`` = ``resolved`` AND changes applied.
The report-level conclusion is kept separately in ``report_status``
(resolved | rejected | needs_context | failed | timeout | cancelled).
Run dirs written by older versions (status resolved/rejected/...) are
coerced to this vocabulary on read — never rewritten on disk.

Redaction: everything that reaches disk (events, notes, instruction,
diff) passes through ``scrub_value`` — exact values of environment
secrets (env NAME matching KEY|TOKEN|SECRET|WEBHOOK|PASSWORD) plus the
usual secret shapes (``sk-...``, ``Bearer ...``, Discord webhooks,
``key=value`` assignments) become ``[REDACTED]``. Paths, commands and
test output are kept — developers need them.

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

# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------

ACTIVE_STATUSES = frozenset({"queued", "running"})
TERMINAL_STATUSES = frozenset({"success", "waiting", "failed", "cancelled", "timeout"})
STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES
# Backward-compatible alias (older code/tests referenced this name).
_TERMINAL_STATUSES = TERMINAL_STATUSES

REPORT_STATUSES = frozenset(
    {"resolved", "rejected", "needs_context", "failed", "timeout", "cancelled"}
)

# Report-level statuses (what the runtime historically passed to
# `finish`) -> meta.status.
_LEGACY_STATUS_MAP = {
    "resolved": "success",
    "rejected": "waiting",
    "needs_context": "waiting",
}

_TOOL_ICONS = {
    "edit": "✏️",
    "write": "✏️",
    "patch": "✏️",
    "bash": "▶️",
    "read": "📖",
}

_EDIT_TOOLS = frozenset({"edit", "write", "patch", "multiedit"})

# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

REDACTED = "[REDACTED]"
_SECRET_ENV_NAME_RE = re.compile(r"KEY|TOKEN|SECRET|WEBHOOK|PASSWORD", re.IGNORECASE)
# Env values shorter than this are never scrubbed as exact values: a
# one-character "secret" would erase every occurrence of that character
# from the log, and nothing that short is a real credential.
_MIN_SECRET_LEN = 6

_SHAPE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=\-]{8,}"),
    re.compile(
        r"https?://(?:[\w.-]+\.)?discord(?:app)?\.com/api/webhooks/[^\s\"'<>]+",
        re.IGNORECASE,
    ),
)
# `NAME=value` / `NAME: value` / `"name": "value"` where NAME smells
# like a credential. The name and separator survive; only the value is
# replaced, so the developer still sees WHICH variable was involved.
_KV_PATTERN = re.compile(
    r"(?i)\b([\w.\-]*(?:api[_-]?key|key|token|secret|password|passwd|webhook)[\w.\-]*)"
    r"([\"']?\s*[:=]\s*[\"']?)"
    r"([^\s\"',;]+)"
)


def _env_secret_values() -> list[str]:
    """Exact values of env vars whose NAME looks like a credential,
    longest first (so a value that contains another is scrubbed whole)."""
    values: set[str] = set()
    for name, value in os.environ.items():
        if not _SECRET_ENV_NAME_RE.search(name):
            continue
        v = (value or "").strip()
        if len(v) >= _MIN_SECRET_LEN:
            values.add(v)
    return sorted(values, key=len, reverse=True)


def scrub_text(text: str, secrets: list[str] | None = None) -> str:
    """Redact secrets from one string. Idempotent."""
    if not isinstance(text, str) or not text:
        return text
    out = text
    for value in (secrets if secrets is not None else _env_secret_values()):
        if value in out:
            out = out.replace(value, REDACTED)
    for pat in _SHAPE_PATTERNS:
        out = pat.sub(REDACTED, out)
    out = _KV_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", out)
    return out


def scrub_value(value: Any, secrets: list[str] | None = None) -> Any:
    """Recursively scrub every string inside dicts/lists/tuples. Keys
    are left alone (structure survives); only string VALUES change."""
    if secrets is None:
        secrets = _env_secret_values()
    if isinstance(value, str):
        return scrub_text(value, secrets)
    if isinstance(value, dict):
        return {k: scrub_value(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_value(v, secrets) for v in value]
    if isinstance(value, tuple):
        return tuple(scrub_value(v, secrets) for v in value)
    return value


# Backward-compatible private alias.
_scrub = scrub_text


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


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


def _meta_defaults() -> dict[str, Any]:
    """Additive meta fields with their neutral defaults. Applied on
    read (old run dirs) and at start_run (new ones)."""
    return {
        "source": "auto",
        "title": None,
        "instruction": "",
        "parent_run_id": None,
        "attempt": 1,
        "queued_at": None,
        "pid": None,
        "pgid": None,
        "files_read": [],
        "files_changed": [],
        "tests": {"ran": False, "passed": None, "failed": None, "summary": None},
        "applied": {"status": "n/a", "files": []},
        "verification": {"status": "none"},
        "failure_summary": None,
        "report_status": None,
        "safety_flags": [],
        "diff_files": [],
    }


def _normalize_terminal(
    status: Any, error_kind: Any, report_status: Any,
) -> tuple[str, str]:
    """(meta.status, report_status) for a terminal transition. Accepts
    both the report-level legacy names and the new vocabulary."""
    s = str(status or "")
    rs = str(report_status) if report_status else None
    if s in _LEGACY_STATUS_MAP:
        return _LEGACY_STATUS_MAP[s], rs or s
    if s == "failed":
        if error_kind == "timeout":
            return "timeout", rs or "timeout"
        if error_kind == "user_cancel":
            return "cancelled", rs or "cancelled"
        return "failed", rs or "failed"
    if s == "success":
        return "success", rs or "resolved"
    if s == "waiting":
        return "waiting", rs or "needs_context"
    if s == "timeout":
        return "timeout", rs or "timeout"
    if s == "cancelled":
        return "cancelled", rs or "cancelled"
    return "failed", rs or "failed"


def coerce_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible view of a meta dict: fills the additive
    fields and maps legacy statuses onto the current vocabulary."""
    out = _meta_defaults()
    out.update(meta)
    status = out.get("status")
    if status not in ACTIVE_STATUSES:
        st, rs = _normalize_terminal(status, out.get("error_kind"), out.get("report_status"))
        out["status"] = st
        out["report_status"] = rs
    if not out.get("instruction") and out.get("instruction_preview"):
        out["instruction"] = out["instruction_preview"]
    return out


# ---------------------------------------------------------------------------
# Test-output parsing (bash tool results)
# ---------------------------------------------------------------------------

_TEST_CMD_RE = re.compile(
    r"(?:^|\s|&&|;)(?:python3?\s+-m\s+pytest|pytest|npm\s+test|npx\s+tsc|npm\s+run\s+build)\b"
)
_PYTEST_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|errors?)\b")
_JEST_LINE_RE = re.compile(r"^\s*Tests:\s+(.*)$", re.MULTILINE)
_JEST_COUNT_RE = re.compile(r"(\d+)\s+(failed|passed)")
_TSC_ERROR_RE = re.compile(r"error TS\d+")
_TSC_FOUND_RE = re.compile(r"Found (\d+) errors?")


def parse_test_output(command: str, output: str, exit_code: int | None) -> dict[str, Any]:
    """Best-effort {passed, failed, summary} from a test/check command's
    output. Unknown counts stay None (never invented)."""
    text = output or ""
    passed: int | None = None
    failed: int | None = None
    if "tsc" in command:
        found = _TSC_FOUND_RE.search(text)
        n_err = int(found.group(1)) if found else len(_TSC_ERROR_RE.findall(text))
        failed = n_err
        if n_err == 0 and exit_code == 0:
            passed = 1  # one type-check, clean
    else:
        jest = _JEST_LINE_RE.search(text)
        if jest:
            # jest/vitest summary line ("Tests: 2 failed, 10 passed, 12 total").
            for n, kind in _JEST_COUNT_RE.findall(jest.group(1)):
                if kind == "passed":
                    passed = (passed or 0) + int(n)
                else:
                    failed = (failed or 0) + int(n)
        else:
            for n, kind in _PYTEST_COUNT_RE.findall(text):
                if kind == "passed":
                    passed = (passed or 0) + int(n)
                else:
                    failed = (failed or 0) + int(n)
        if failed is None and passed is not None:
            failed = 0
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    summary = _preview(lines[-1], 200) if lines else (
        f"exit {exit_code}" if exit_code is not None else None
    )
    return {"passed": passed, "failed": failed, "summary": summary}


# ---------------------------------------------------------------------------
# writer side
# ---------------------------------------------------------------------------


class RunLogger:
    """Append-only writer for one run. Thread-safe: the stdout reader
    thread calls `event()` while the runtime's main thread calls
    `note()`/`finish()`/`set_fields()`.

    Lifecycle: ``start_run`` (queued or running) -> optionally
    ``close()`` + ``resume_run`` in another thread/process ->
    ``mark_running`` -> events/notes -> ``finish`` -> meta-only updates
    (``set_fields``/``save_diff``) are still allowed after finish so the
    finally-block diff capture and auto_fix's post-hoc verdict land in
    the record."""

    def __init__(self, run_dir: Path, meta: dict[str, Any], *, write_meta: bool = True):
        self.run_dir = Path(run_dir)
        self.run_id = str(meta["run_id"])
        self._meta = dict(meta)
        self._lock = threading.RLock()
        # Buffered streaming text/reasoning part:
        # (kind, part_id, first_seen_hhmmss, latest_text)
        self._pending: tuple[str, Any, str, str] | None = None
        self._finished = self._meta.get("status") in TERMINAL_STATUSES
        self._closed = False
        self._project_dir: str | None = None
        self._last_error: str | None = None
        self._events_f = open(  # noqa: SIM115 - long-lived, closed in finish()
            self.run_dir / "events.jsonl", "a", encoding="utf-8"
        )
        self._log_f = open(  # noqa: SIM115 - long-lived, closed in finish()
            self.run_dir / "log.txt", "a", encoding="utf-8"
        )
        if write_meta:
            self._write_meta()

    # -- public API --------------------------------------------------

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def meta(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._meta))

    def event(self, raw: str | dict[str, Any]) -> None:
        """Record one raw CLI event (JSON line or already-parsed dict).
        Non-JSON noise lines are silently skipped. Secrets are scrubbed
        BEFORE anything touches the disk."""
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
        ev = scrub_value(ev)
        with self._lock:
            if self._finished or self._closed:
                return
            self._events_f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self._events_f.flush()
            self._render_event(ev)
            self._harvest_event(ev)

    def note(self, text: str) -> None:
        """Runtime-side annotation (worktree path, copy-back result, …)."""
        with self._lock:
            if self._finished or self._closed:
                return
            self._flush_pending()
            now = datetime.now().strftime("%H:%M:%S")
            self._append_line(now, f"ℹ️ {_preview(scrub_text(str(text)), 400)}")

    def set_process(self, pid: int | None, pgid: int | None) -> None:
        """Record the spawned CLI's identity so a cancel request can
        find it (in-process registry first, disk as the fallback)."""
        with self._lock:
            self._meta["pid"] = int(pid) if pid is not None else None
            self._meta["pgid"] = int(pgid) if pgid is not None else None
            self._write_meta()

    def set_project_dir(self, path: str | os.PathLike[str] | None) -> None:
        """Worktree project dir; absolute tool paths under it are
        stored project-relative in files_read / files_changed."""
        with self._lock:
            self._project_dir = (
                str(path).replace("\\", "/").rstrip("/") if path else None
            )

    def set_fields(self, **fields: Any) -> None:
        """Meta-only update (allowed after finish). Values are scrubbed."""
        with self._lock:
            self._meta.update(scrub_value(fields))
            self._write_meta()

    def mark_running(self, *, model: str | None = None) -> None:
        """queued -> running (called by the runtime when it picks the
        run up). No-op once terminal."""
        with self._lock:
            if self._finished:
                return
            self._meta["status"] = "running"
            self._meta["started_at"] = _utc_now_iso()
            if model:
                self._meta["model"] = str(model)
            self._write_meta()

    def add_changed_files(self, paths: list[str]) -> None:
        with self._lock:
            for p in paths:
                self._add_path("files_changed", p, write=False)
            self._write_meta()

    def save_diff(self, patch: str, files: list[dict[str, Any]] | None = None) -> None:
        """Persist the unified diff (scrubbed) as diff.patch and merge
        the per-file summary into meta. Allowed after finish."""
        with self._lock:
            safe_patch = scrub_text(patch or "")
            tmp = self.run_dir / "diff.patch.tmp"
            tmp.write_text(safe_patch, encoding="utf-8")
            os.replace(tmp, self.run_dir / "diff.patch")
            file_list = [dict(f) for f in (files or []) if isinstance(f, dict)]
            self._meta["diff_files"] = scrub_value(file_list)
            for f in file_list:
                p = f.get("path")
                if isinstance(p, str) and p:
                    self._add_path("files_changed", p, write=False)
            self._write_meta()

    def set_applied(self, status: str, files: list[str] | None = None) -> None:
        """applied | not_applied | violation | n/a."""
        with self._lock:
            self._meta["applied"] = {
                "status": str(status),
                "files": [str(p) for p in (files or [])],
            }
            self._write_meta()

    def finish(
        self,
        status: str,
        error_kind: str | None = None,
        *,
        report_status: str | None = None,
        failure_summary: str | None = None,
    ) -> None:
        """Terminal transition. `status` may be report-level
        (resolved/rejected/needs_context/failed) or the meta vocabulary
        (success/waiting/failed/cancelled/timeout); anything else is
        coerced to failed."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._flush_pending()
            st, rs = _normalize_terminal(status, error_kind, report_status)
            tail = f"🏁 run finished: {st}"
            if error_kind and error_kind != st:
                tail += f" ({error_kind})"
            elif rs and rs != st:
                tail += f" ({rs})"
            self._append_line(datetime.now().strftime("%H:%M:%S"), tail)
            self._meta["status"] = st
            self._meta["report_status"] = rs
            self._meta["finished_at"] = _utc_now_iso()
            if error_kind is not None:
                self._meta["error_kind"] = str(error_kind)
            if st in ("failed", "timeout", "cancelled"):
                summary = failure_summary
                if not summary:
                    summary = str(error_kind or st)
                    if self._last_error:
                        summary += f": {self._last_error}"
                self._meta["failure_summary"] = scrub_text(_preview(summary, 300))
            elif failure_summary:
                self._meta["failure_summary"] = scrub_text(_preview(failure_summary, 300))
            self._meta["verification"] = {"status": self._verification_status()}
            applied = self._meta.get("applied") or {}
            if st == "waiting" and applied.get("status") in (None, "n/a"):
                self._meta["applied"] = {"status": "not_applied", "files": []}
            self._write_meta()
            self._close_files()

    def close(self) -> None:
        """Release file handles WITHOUT finishing (queued hand-off: the
        runtime re-opens the same dir via `resume_run`)."""
        with self._lock:
            if self._closed:
                return
            self._flush_pending()
            self._closed = True
            self._close_files()

    # -- internals ---------------------------------------------------

    def _close_files(self) -> None:
        for f in (self._events_f, self._log_f):
            try:
                f.close()
            except OSError:  # pragma: no cover - defensive
                pass

    def _write_meta(self) -> None:
        tmp = self.run_dir / "meta.json.tmp"
        tmp.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.run_dir / "meta.json")

    def _append_line(self, ts: str, line: str) -> None:
        if self._closed:
            return
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
            self._last_error = _preview(msg, 200)
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

    # -- meta harvesting (files_read / files_changed / tests) ---------

    def _relativize(self, path: str) -> str:
        p = str(path).replace("\\", "/")
        if self._project_dir:
            prefix = self._project_dir + "/"
            if p.startswith(prefix):
                p = p[len(prefix):]
        return p[2:] if p.startswith("./") else p

    def _add_path(self, field: str, path: str, *, write: bool = True) -> None:
        rel = self._relativize(path)
        if not rel:
            return
        current = list(self._meta.get(field) or [])
        if rel not in current:
            current.append(rel)
            self._meta[field] = current
            if write:
                self._write_meta()

    def _harvest_event(self, ev: dict[str, Any]) -> None:
        if ev.get("type") != "tool_use":
            return
        part = ev.get("part") if isinstance(ev.get("part"), dict) else {}
        tool = str(part.get("tool") or "").lower()
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        status = state.get("status")
        if status not in ("completed", "error"):
            return
        inp = state.get("input") if isinstance(state.get("input"), dict) else {}
        path = None
        for key in ("filePath", "file_path", "path", "file"):
            if isinstance(inp.get(key), str) and inp.get(key):
                path = inp[key]
                break
        if status == "error":
            err = ""
            for cand in (state.get("error"), state.get("message"), state.get("output")):
                if isinstance(cand, str) and cand.strip():
                    err = cand
                    break
            self._last_error = _preview(f"{tool}: {err or 'error'}", 200)
        changed = False
        if tool == "read" and path:
            self._add_path("files_read", path, write=False)
            changed = True
        elif tool in _EDIT_TOOLS and path and status == "completed":
            self._add_path("files_changed", path, write=False)
            changed = True
        elif tool == "bash":
            command = ""
            for cand in (inp.get("command"), state.get("title")):
                if isinstance(cand, str) and cand.strip():
                    command = cand
                    break
            if command and _TEST_CMD_RE.search(" " + command):
                self._record_test(command, state)
                changed = True
        if changed:
            self._write_meta()

    def _record_test(self, command: str, state: dict[str, Any]) -> None:
        metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
        output = ""
        for cand in (state.get("output"), metadata.get("output")):
            if isinstance(cand, str) and cand:
                output = cand
                break
        exit_code: int | None = None
        for cand in (metadata.get("exit"), metadata.get("exitCode"), state.get("exit")):
            if isinstance(cand, int) and not isinstance(cand, bool):
                exit_code = cand
                break
        if exit_code is None and state.get("status") == "error":
            exit_code = 1
        parsed = parse_test_output(command, output, exit_code)
        tests = dict(self._meta.get("tests") or {})
        tests["ran"] = True
        if parsed["passed"] is not None:
            tests["passed"] = (tests.get("passed") or 0) + parsed["passed"]
        if parsed["failed"] is not None:
            tests["failed"] = (tests.get("failed") or 0) + parsed["failed"]
        tests["summary"] = parsed["summary"]
        commands = list(tests.get("commands") or [])
        commands.append({
            "command": _preview(command, 200),
            "exit": exit_code,
            "passed": parsed["passed"],
            "failed": parsed["failed"],
        })
        tests["commands"] = commands[-20:]
        self._meta["tests"] = tests

    def _verification_status(self) -> str:
        tests = self._meta.get("tests") or {}
        if not tests.get("ran"):
            return "none"
        commands = tests.get("commands") or []
        if (tests.get("failed") or 0) > 0:
            return "failed"
        if any(c.get("exit") not in (None, 0) for c in commands):
            return "failed"
        if (tests.get("passed") or 0) > 0 or any(c.get("exit") == 0 for c in commands):
            return "passed"
        return "none"


# ---------------------------------------------------------------------------
# module API
# ---------------------------------------------------------------------------


def _new_run_dir(ticket_id: str) -> tuple[str, Path]:
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
    return run_id, run_dir


def start_run(
    ticket_id: str,
    instruction: str,
    model: str,
    *,
    status: str = "running",
    source: str = "auto",
    title: str | None = None,
    parent_run_id: str | None = None,
    attempt: int = 1,
    safety_flags: list[str] | None = None,
) -> RunLogger:
    """Create the run directory + initial meta and return its logger.
    `status` is "running" (runtime starts now) or "queued" (manual queue:
    the dir exists immediately so the run shows up in history; the
    runtime later `resume_run`s it). Also sweeps old runs (keep the
    newest ``_KEEP_LATEST``)."""
    if status not in ("running", "queued"):
        status = "running"
    run_id, run_dir = _new_run_dir(ticket_id)
    now = _utc_now_iso()
    full_instruction = scrub_text(str(instruction or ""))
    meta = _meta_defaults()
    meta.update({
        "run_id": run_id,
        "ticket_id": str(ticket_id),
        "status": status,
        "started_at": now,
        "queued_at": now,
        "finished_at": None,
        "error_kind": None,
        "model": str(model),
        "instruction_preview": _preview(full_instruction, 200),
        "instruction": full_instruction[:4000],
        "source": "manual" if source == "manual" else "auto",
        "title": scrub_text(str(title))[:200] if title else None,
        "parent_run_id": str(parent_run_id) if parent_run_id else None,
        "attempt": max(1, int(attempt or 1)),
        "safety_flags": [str(f) for f in (safety_flags or [])],
    })
    logger = RunLogger(run_dir, meta)
    try:
        keep_latest(_KEEP_LATEST)
    except OSError:  # pragma: no cover - retention is best-effort
        pass
    return logger


def _validated_run_dir(run_id: str) -> Path:
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise KeyError(f"invalid run_id: {run_id!r}")
    return _runs_root() / run_id


def resume_run(run_id: str) -> RunLogger:
    """Re-open an existing run dir for appending (queued -> running
    hand-off, or a cancel of a queued run). KeyError when unknown."""
    run_dir = _validated_run_dir(run_id)
    meta = _read_meta(run_dir)
    if meta is None:
        raise KeyError(f"unknown run_id: {run_id!r}")
    return RunLogger(run_dir, meta, write_meta=False)


def read_meta(run_id: str) -> dict[str, Any] | None:
    """Coerced meta for one run, or None when unknown/malformed id."""
    try:
        run_dir = _validated_run_dir(run_id)
    except KeyError:
        return None
    meta = _read_meta(run_dir)
    return coerce_meta(meta) if meta is not None else None


def mark_running(run_id: str, *, model: str | None = None) -> bool:
    """queued -> running for a run that has no live logger (auto_fix does
    this before invoking the executor, so history shows the run as
    running even when the executor never touches the log). False when
    unknown or already terminal."""
    try:
        logger = resume_run(run_id)
    except KeyError:
        return False
    try:
        if logger.finished:
            return False
        logger.mark_running(model=model)
        return True
    finally:
        logger.close()


def update_meta(run_id: str, fields: dict[str, Any]) -> bool:
    """Merge `fields` into a run's meta.json (post-hoc verdicts from
    auto_fix). Returns False when the run is unknown."""
    try:
        run_dir = _validated_run_dir(run_id)
    except KeyError:
        return False
    meta = _read_meta(run_dir)
    if meta is None:
        return False
    meta.update(scrub_value(fields))
    tmp = run_dir / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / "meta.json")
    return True


def finalize_run(
    run_id: str,
    status: str,
    error_kind: str | None = None,
    *,
    report_status: str | None = None,
    failure_summary: str | None = None,
) -> bool:
    """Finish a run that has no live logger (queued cancel, runtime
    that never touched the log). No-op (False) when already terminal
    or unknown."""
    try:
        logger = resume_run(run_id)
    except KeyError:
        return False
    if logger.finished:
        logger.close()
        return False
    logger.finish(
        status, error_kind, report_status=report_status,
        failure_summary=failure_summary,
    )
    return True


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
            metas.append(coerce_meta(meta))
    metas.sort(
        key=lambda m: str(m.get("started_at") or m.get("queued_at") or ""),
        reverse=True,
    )
    return metas[: max(0, int(limit))]


def read_run(run_id: str, after: int = 0) -> dict[str, Any]:
    """Meta + rendered log lines from cursor `after` (line index).
    Raises KeyError for unknown or malformed run ids (the API maps that
    to 404); the run-id pattern doubles as the path-traversal guard."""
    run_dir = _validated_run_dir(run_id)
    meta = _read_meta(run_dir)
    if meta is None:
        raise KeyError(f"unknown run_id: {run_id!r}")
    lines: list[str] = []
    log_path = run_dir / "log.txt"
    if log_path.is_file():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    after = max(0, int(after))
    return {"meta": coerce_meta(meta), "lines": lines[after:], "next_cursor": len(lines)}


def read_events(run_id: str, after: int = 0) -> dict[str, Any]:
    """Raw (scrubbed) CLI events from cursor `after` (event index)."""
    run_dir = _validated_run_dir(run_id)
    meta = _read_meta(run_dir)
    if meta is None:
        raise KeyError(f"unknown run_id: {run_id!r}")
    raw_lines: list[str] = []
    path = run_dir / "events.jsonl"
    if path.is_file():
        raw_lines = [
            ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
    after = max(0, int(after))
    events: list[dict[str, Any]] = []
    secrets = _env_secret_values()
    for ln in raw_lines[after:]:
        try:
            obj = json.loads(ln)
        except ValueError:
            obj = {"type": "unparseable", "raw": ln[:500]}
        if isinstance(obj, dict):
            events.append(scrub_value(obj, secrets))
    return {"events": events, "next_cursor": len(raw_lines)}


def read_diff(run_id: str) -> dict[str, Any]:
    """{patch, files} for a run; empty when nothing was captured."""
    run_dir = _validated_run_dir(run_id)
    meta = _read_meta(run_dir)
    if meta is None:
        raise KeyError(f"unknown run_id: {run_id!r}")
    patch = ""
    path = run_dir / "diff.patch"
    if path.is_file():
        patch = scrub_text(path.read_text(encoding="utf-8", errors="replace"))
    files = meta.get("diff_files")
    if not isinstance(files, list):
        files = []
    return {"patch": patch, "files": scrub_value(files)}


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
        key = str((meta or {}).get("started_at") or (meta or {}).get("queued_at") or "")
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
