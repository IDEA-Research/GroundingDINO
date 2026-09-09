"""Manual developer runs — in-process FIFO queue for `developer_fix`.

`POST /api/devfix/runs` (dev-token gated, see api/devfix.py) lands here.
A manual run is the SAME pipeline as a chat-triggered auto-fix — a
`DeveloperTicket` (source "manual"), the auto_fix single-flight lock,
snapshot + protected-path guard + rollback, the OpenCode CLI inside a
detached worktree, copy-back through the independent allow-list — with
two differences: the instruction comes from a developer, and the run
record exists from the moment it is queued (status "queued") so the
dev panel shows it in history immediately.

Gates (all server-side, all fail closed):

  1. Kill switch: HELPER_DASHBOARD_DEVFIX_LOCAL_RUNS=1 (default off).
  2. Executor: HELPER_DASHBOARD_HELPERCODE != mock AND
     HELPER_DASHBOARD_DEVFIX_RUNTIME=opencode-cli AND the CLI binary
     exists — never "start" into the log-less helpercode/mock paths.
  3. Daily quota (shared with auto-fix).
  4. Queue bound: at most MAX_PENDING waiting, one active.

`extend_gate.check_user_message_safety` is ADVISORY for developer
instructions: flags are recorded in meta.safety_flags, nothing is
blocked (the OpenCode permission set and the copy-back allow-list are
the boundary — a regex never is).

No endpoint reaching this module accepts commands, paths or model
overrides: only `instruction` (+ optional `title`).
"""

from __future__ import annotations

import os
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..specs.developer_ticket import DeveloperTicket, TicketSeverity
from . import auto_fix, devfix_log, devfix_worktree
from . import runtime as runtime_mod
from .extend_gate import check_user_message_safety

MAX_PENDING = 5

_ENV_LOCAL_RUNS = "HELPER_DASHBOARD_DEVFIX_LOCAL_RUNS"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

INSTRUCTION_MAX = 2000
TITLE_MAX = 120


class LocalRunsUnavailable(Exception):
    """The local execution layer is off or misconfigured (reason code)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class QueueFull(Exception):
    pass


class QuotaExhausted(Exception):
    pass


class RunActive(Exception):
    """Retry refused: another run is active right now."""


@dataclass
class _Job:
    run_id: str
    ticket: DeveloperTicket
    instruction: str
    title: str | None
    parent_run_id: str | None
    attempt: int


_LOCK = threading.Lock()
_PENDING: deque[_Job] = deque()
_ACTIVE: _Job | None = None
_WORKER: threading.Thread | None = None

_RUNTIME: Any = None
_STORE: Any = None


# ---------------------------------------------------------------------------
# Backends (lazy; overridable for tests)
# ---------------------------------------------------------------------------


def _runtime() -> Any:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = runtime_mod.HelperCodeRuntime()
    return _RUNTIME


def _store() -> Any:
    global _STORE
    if _STORE is None:
        from ..services.dashboard_store import DashboardStore

        _STORE = DashboardStore()
    return _STORE


def set_backends_for_tests(*, runtime: Any = None, store: Any = None) -> None:
    global _RUNTIME, _STORE
    _RUNTIME = runtime
    _STORE = store


def reset_for_tests() -> None:
    """Drop queue state (does not join a live worker) and backends."""
    global _ACTIVE, _WORKER, _RUNTIME, _STORE
    with _LOCK:
        _PENDING.clear()
        _ACTIVE = None
        _WORKER = None
    _RUNTIME = None
    _STORE = None


# ---------------------------------------------------------------------------
# Gates / status
# ---------------------------------------------------------------------------


def local_runs_enabled() -> bool:
    """The env kill switch alone (start/retry/cancel all require it)."""
    return (os.getenv(_ENV_LOCAL_RUNS) or "").strip().lower() in _TRUTHY


def local_runs_status() -> tuple[bool, str | None]:
    """(ok, reason_code). The full start gate: kill switch + executor
    + binary. Reason codes are stable identifiers for the UI."""
    if not local_runs_enabled():
        return False, "local_runs_disabled"
    mode = (os.getenv("HELPER_DASHBOARD_HELPERCODE") or "mock").strip().lower()
    if mode == "mock":
        return False, "mock_mode"
    if runtime_mod._devfix_mode() != "opencode-cli":
        return False, "runtime_not_opencode_cli"
    bin_path = runtime_mod._resolve_devfix_bin()
    if not (os.path.isfile(bin_path) and os.access(bin_path, os.X_OK)):
        return False, "binary_missing"
    return True, None


def is_run_active() -> bool:
    """A developer_fix run is executing right now (manual or auto)."""
    with _LOCK:
        active = _ACTIVE is not None
    return active or auto_fix._RUN_LOCK.locked()


def snapshot() -> dict[str, Any]:
    with _LOCK:
        return {
            "active_run_id": _ACTIVE.run_id if _ACTIVE is not None else None,
            "pending_run_ids": [j.run_id for j in _PENDING],
            "max_pending": MAX_PENDING,
        }


def config_payload() -> dict[str, Any]:
    """Read-only preflight data for the dev panel (GET /api/devfix/config)."""
    bin_path = runtime_mod._resolve_devfix_bin()
    limit, used = auto_fix.quota_status()
    ok, reason = local_runs_status()
    return {
        "project_root": str(devfix_worktree._PROJECT_ROOT),
        "worktree_parent": str(devfix_worktree.worktree_parent()),
        "bin": bin_path,
        "bin_exists": bool(os.path.isfile(bin_path) and os.access(bin_path, os.X_OK)),
        "model": runtime_mod._resolve_devfix_model(),
        "timeout_seconds": runtime_mod._resolve_devfix_timeout(),
        "runtime_mode": runtime_mod.HelperCodeRuntime._mode(),
        "devfix_runtime": runtime_mod._devfix_mode(),
        "local_runs_enabled": ok,
        "local_runs_reason": reason,
        "auto_fix_enabled": auto_fix.is_enabled(),
        "daily_quota": {"limit": limit, "used": used},
        "queue": snapshot(),
    }


# ---------------------------------------------------------------------------
# Submit / cancel / retry
# ---------------------------------------------------------------------------


def _make_ticket(instruction: str, title: str | None) -> DeveloperTicket:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    summary = (title or instruction).strip().splitlines()[0][:256] or "manual run"
    return DeveloperTicket(
        ticket_id=f"manual-{stamp}-{uuid.uuid4().hex[:6]}",
        source_agent="developer",
        severity=TicketSeverity.medium,
        summary=summary,
        user_visible_effect="開發者手動啟動的本機執行（不是使用者回報的問題）",
        technical_evidence={
            "source": "manual",
            "title": title,
            "instruction_chars": len(instruction),
        },
        requested_action=instruction[:1024],
        safety_notes="",
    )


def submit(
    instruction: str,
    title: str | None = None,
    *,
    parent_run_id: str | None = None,
    attempt: int = 1,
    ticket: DeveloperTicket | None = None,
) -> dict[str, Any]:
    """Gate + enqueue. Returns {run_id, ticket_id, status: "queued"}.
    Raises LocalRunsUnavailable / QuotaExhausted / QueueFull."""
    instruction = (instruction or "").strip()
    if not instruction:
        raise ValueError("instruction is required")
    instruction = instruction[:INSTRUCTION_MAX]
    title = (title or "").strip()[:TITLE_MAX] or None

    ok, reason = local_runs_status()
    if not ok:
        raise LocalRunsUnavailable(reason or "local_runs_disabled")
    ok, _why = auto_fix._quota_available()
    if not ok:
        raise QuotaExhausted()
    with _LOCK:
        if len(_PENDING) >= MAX_PENDING:
            raise QueueFull()

    safety_flags: list[str] = []
    safe, why = check_user_message_safety(instruction)
    if not safe and why:
        safety_flags.append(why)

    store = _store()
    if ticket is None:
        ticket = _make_ticket(instruction, title)
    store.save_ticket(ticket)

    logger = devfix_log.start_run(
        ticket.ticket_id,
        instruction=instruction,
        model=runtime_mod._resolve_devfix_model(),
        status="queued",
        source="manual",
        title=title,
        parent_run_id=parent_run_id,
        attempt=attempt,
        safety_flags=safety_flags,
    )
    run_id = logger.run_id
    logger.close()

    auto_fix._consume_quota()
    job = _Job(
        run_id=run_id, ticket=ticket, instruction=instruction, title=title,
        parent_run_id=parent_run_id, attempt=attempt,
    )
    with _LOCK:
        if len(_PENDING) >= MAX_PENDING:
            # Lost the race with another submit — refuse honestly.
            devfix_log.finalize_run(
                run_id, "cancelled", "queue_full",
                failure_summary="queue full at submit time",
            )
            raise QueueFull()
        _PENDING.append(job)
        _ensure_worker_locked()
    return {"run_id": run_id, "ticket_id": ticket.ticket_id, "status": "queued"}


def cancel(run_id: str) -> str:
    """"removed" (was queued), "signalled" (running here), or
    "not_active" (no queued job and no live process in this backend)."""
    with _LOCK:
        for job in list(_PENDING):
            if job.run_id == run_id:
                _PENDING.remove(job)
                break
        else:
            job = None
    if job is not None:
        devfix_log.finalize_run(
            run_id, "cancelled", "user_cancel",
            failure_summary="cancelled by developer before start",
        )
        return "removed"
    if runtime_mod.request_devfix_cancel(run_id):
        return "signalled"
    return "not_active"


def retry(parent_run_id: str) -> dict[str, Any]:
    """New run with the parent's instruction; meta.parent_run_id links
    them and attempt = parent.attempt + 1. KeyError when the parent is
    unknown; RunActive while a run executes or the parent is still
    active; the submit() exceptions otherwise."""
    meta = devfix_log.read_meta(parent_run_id)
    if meta is None:
        raise KeyError(parent_run_id)
    if meta.get("status") in devfix_log.ACTIVE_STATUSES:
        raise RunActive("parent run is still active")
    if is_run_active():
        raise RunActive("another run is active")
    instruction = str(meta.get("instruction") or meta.get("instruction_preview") or "").strip()
    if not instruction:
        raise ValueError("parent run has no instruction to retry")
    title = meta.get("title") if isinstance(meta.get("title"), str) else None
    ticket = None
    ticket_id = meta.get("ticket_id")
    if isinstance(ticket_id, str) and ticket_id:
        try:
            ticket = _store().load_ticket(ticket_id)
        except Exception:
            ticket = None
    try:
        attempt = int(meta.get("attempt") or 1) + 1
    except (TypeError, ValueError):
        attempt = 2
    return submit(
        instruction, title, parent_run_id=parent_run_id, attempt=attempt,
        ticket=ticket,
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _ensure_worker_locked() -> None:
    """Caller holds _LOCK."""
    global _WORKER
    if _WORKER is not None and _WORKER.is_alive():
        return
    _WORKER = threading.Thread(
        target=_worker_loop, name="devfix-manual-queue", daemon=True,
    )
    _WORKER.start()


def _worker_loop() -> None:
    global _ACTIVE, _WORKER
    while True:
        with _LOCK:
            if not _PENDING:
                _WORKER = None
                return
            job = _PENDING.popleft()
            _ACTIVE = job
        try:
            auto_fix.run_manual(
                ticket=job.ticket,
                runtime=_runtime(),
                store=_store(),
                run_id=job.run_id,
                instruction=job.instruction,
                title=job.title,
            )
        except Exception as exc:  # pragma: no cover - auto_fix owns its errors
            devfix_log.finalize_run(
                job.run_id, "failed", "internal_error",
                failure_summary=f"{type(exc).__name__}: {exc}"[:300],
            )
        finally:
            with _LOCK:
                if _ACTIVE is job:
                    _ACTIVE = None
