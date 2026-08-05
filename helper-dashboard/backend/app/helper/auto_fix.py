"""Auto-fix pipeline — Big guy consumes diagnostic tickets automatically.

LD-1/LD-2 + user directive (2026-07-06): diagnostic tickets never wait
for a human. When the orchestrator persists a ticket, this module
schedules a background `developer_fix` run — Big guy in tool-using
mode — that attempts the real code fix immediately. The user's chat
turn is never blocked: the run happens on a daemon thread and the
ticket records the outcome.

Gate layers (every one fails closed):

  1. Enabled check — ON by default when the runtime mode is
     `opencode`/`auto`, OFF in `mock` (CI). `HELPER_DASHBOARD_AUTO_FIX`
     overrides either way; setting it to 0/false is the kill switch.
  2. Ticket-text safety — the prompt-injection signature check from
     `extend_gate` runs over every user-influenced ticket field.
  3. Daily quota — `HELPER_DASHBOARD_AUTO_FIX_DAILY_QUOTA` (default 10).
  4. Single-flight — at most one auto-fix runs at a time; a busy
     pipeline skips (ticket stays open) rather than queueing storms.
  5. Snapshot + rollback — the entire writable set is snapshotted
     before the run and restored if the run fails, the report is not
     `resolved`, or the guard below trips.
  6. Protected-path guard — the clinical anomaly files are byte-
     compared before/after. ANY difference restores the originals and
     rejects the run regardless of what the report claims. Big guy
     never touches the clinical safety code (CLAUDE.md protected paths).
  7. Audit — every decision and attempt appends one JSONL line to
     `backend/app/storage/auto_fix_audit/<date>.jsonl`.

The write scope is deliberately the render/product layer only, and is
mirrored INDEPENDENTLY in `bin/opencode`'s tool-level allow-list (two
trust boundaries, same philosophy as rescue_extend).

Backend code changed by a successful run needs a backend restart to
take effect (same C-5 caveat as rescue_extend); frontend changes need
a bundle rebuild. The audit entry records what was touched so the
operator knows.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..specs.developer_ticket import DeveloperTicket, TicketStatus
from .extend_gate import check_user_message_safety
from .runtime import RuntimeError_


_REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Write scope (mirrored independently in bin/opencode — keep in sync)
# ---------------------------------------------------------------------------

_WRITABLE_FILES: tuple[str, ...] = (
    "backend/app/services/browser_evaluator.py",
    "frontend/lib/renderer.tsx",
    "frontend/lib/spec-schema.ts",
)

_WRITABLE_DIRS: tuple[str, ...] = (
    "frontend/widget-toolkit",
    "tests/browser_evaluation",
    "tests/review_loop",
    "tests/spec_validation",
)


def _writable_file_paths() -> list[Path]:
    return [_REPO_ROOT / rel for rel in _WRITABLE_FILES]


def _writable_dir_paths() -> list[Path]:
    return [_REPO_ROOT / rel for rel in _WRITABLE_DIRS]


# ---------------------------------------------------------------------------
# Protected clinical paths — never writable, verified by byte-compare
# ---------------------------------------------------------------------------


def _protected_paths() -> list[Path]:
    """The clinical-safety file set from CLAUDE.md. Computed at call
    time so files added later are automatically covered."""
    services = _REPO_ROOT / "backend" / "app" / "services"
    prometheus = _REPO_ROOT / "backend" / "app" / "prometheus"
    out: list[Path] = sorted(services.glob("anomaly_*.py"))
    out.append(services / "alert_state_store.py")
    out.extend(sorted(prometheus.glob("*.py")))
    out.append(_REPO_ROOT / "backend" / "app" / "specs" / "alert_rule_spec.py")
    out.append(_REPO_ROOT / "backend" / "app" / "api" / "anomaly.py")
    return [p for p in out if p.exists()]


# ---------------------------------------------------------------------------
# Enabled / quota
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    raw = os.getenv("HELPER_DASHBOARD_AUTO_FIX")
    if raw is not None:
        return raw not in ("0", "", "false", "False", "no", "off")
    mode = (os.getenv("HELPER_DASHBOARD_OPENCODE") or "mock").strip().lower()
    return mode in {"opencode", "auto"}


_QUOTA_DEFAULT = 10
_QUOTA_LOCK = threading.Lock()
_QUOTA_STATE: dict[str, int] = {}


def _quota_max() -> int:
    raw = os.getenv(
        "HELPER_DASHBOARD_AUTO_FIX_DAILY_QUOTA", str(_QUOTA_DEFAULT),
    )
    try:
        n = int(raw)
    except ValueError:
        return _QUOTA_DEFAULT
    return max(0, min(n, 100))


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _quota_available() -> tuple[bool, str | None]:
    with _QUOTA_LOCK:
        used = _QUOTA_STATE.get(_today_key(), 0)
    cap = _quota_max()
    if used >= cap:
        return False, f"daily auto-fix quota exhausted: {used}/{cap}"
    return True, None


def _consume_quota() -> None:
    with _QUOTA_LOCK:
        key = _today_key()
        _QUOTA_STATE[key] = _QUOTA_STATE.get(key, 0) + 1


def reset_quota_for_tests() -> None:
    with _QUOTA_LOCK:
        _QUOTA_STATE.clear()


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


_AUDIT_DIR = _REPO_ROOT / "backend" / "app" / "storage" / "auto_fix_audit"
_AUDIT_LOCK = threading.Lock()


def _write_audit(entry: dict[str, Any]) -> None:
    """Append one structured line to today's audit file. Never raises."""
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    try:
        with _AUDIT_LOCK:
            _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
            path = _AUDIT_DIR / f"{_today_key()}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Snapshot / restore (in-memory, scoped to the writable + protected sets)
# ---------------------------------------------------------------------------


def _snapshot() -> dict[Path, bytes]:
    """Byte snapshot of every writable file, every existing file under
    the writable dirs, and every protected clinical file."""
    snap: dict[Path, bytes] = {}
    for p in _writable_file_paths():
        if p.exists():
            snap[p] = p.read_bytes()
    for d in _writable_dir_paths():
        if d.exists():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    snap[p] = p.read_bytes()
    for p in _protected_paths():
        snap[p] = p.read_bytes()
    return snap


def _created_files(snap: dict[Path, bytes]) -> list[Path]:
    """Files that exist now under the writable dirs but weren't in the
    snapshot — i.e. created by the run."""
    created: list[Path] = []
    for d in _writable_dir_paths():
        if d.exists():
            for p in sorted(d.rglob("*")):
                if p.is_file() and p not in snap:
                    created.append(p)
    return created


def _restore(snap: dict[Path, bytes]) -> dict[str, Any]:
    """Restore all snapshotted files and delete created ones. Returns
    {restored, deleted, errors} for the audit log. Never raises."""
    restored: list[str] = []
    deleted: list[str] = []
    errors: list[str] = []
    for p, original in snap.items():
        try:
            if not p.exists() or p.read_bytes() != original:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(original)
                restored.append(_rel(p))
        except Exception as exc:
            errors.append(f"{_rel(p)}: {type(exc).__name__}: {exc}")
    for p in _created_files(snap):
        try:
            p.unlink()
            deleted.append(_rel(p))
        except Exception as exc:
            errors.append(f"{_rel(p)}: {type(exc).__name__}: {exc}")
    return {"restored": restored, "deleted": deleted, "errors": errors}


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(_REPO_ROOT))
    except ValueError:
        return str(p)


def _protected_violations(snap: dict[Path, bytes]) -> list[str]:
    """Protected files whose bytes differ from the snapshot (or that
    vanished). Any entry here is a hard reject."""
    bad: list[str] = []
    for p in _protected_paths():
        original = snap.get(p)
        if original is None:
            # Protected file appeared after snapshot — treat as violation.
            bad.append(_rel(p))
            continue
        try:
            if not p.exists() or p.read_bytes() != original:
                bad.append(_rel(p))
        except Exception:
            bad.append(_rel(p))
    return bad


# ---------------------------------------------------------------------------
# Ticket helpers
# ---------------------------------------------------------------------------


def _ticket_text_safe(ticket: DeveloperTicket) -> tuple[bool, str | None]:
    """User text can flow into ticket fields (user_visible_effect quotes
    the chat message). Run the injection signature check over all of it."""
    blob = " ".join([
        ticket.summary or "",
        ticket.user_visible_effect or "",
        ticket.requested_action or "",
        ticket.safety_notes or "",
    ])
    return check_user_message_safety(blob)


def _with_evidence_note(
    ticket: DeveloperTicket, key: str, note: dict[str, Any],
) -> DeveloperTicket:
    ev = ticket.technical_evidence
    ev = dict(ev) if isinstance(ev, dict) else {"original_evidence": ev}
    attempts = list(ev.get(key) or [])
    attempts.append(note)
    ev[key] = attempts
    return ticket.model_copy(update={"technical_evidence": ev})


# ---------------------------------------------------------------------------
# Scheduling + run
# ---------------------------------------------------------------------------


# Single-flight: one auto-fix at a time on this box (Jetson-friendly).
_RUN_LOCK = threading.Lock()


def schedule_auto_fix(
    ticket: DeveloperTicket,
    *,
    runtime: Any,
    store: Any,
    user_intent: str = "",
) -> threading.Thread | None:
    """Gate + spawn a background developer_fix run for `ticket`.

    Returns the started Thread when scheduled (tests join() it), or
    None when a gate refused / the pipeline is busy. Never raises and
    never blocks the caller beyond the cheap gate checks.
    """
    if not is_enabled():
        return None

    safe, why = _ticket_text_safe(ticket)
    if not safe:
        _write_audit({
            "ticket_id": ticket.ticket_id, "allowed": False,
            "layer": "ticket_text_safety", "refusal_reason": why,
        })
        return None

    ok, why = _quota_available()
    if not ok:
        _write_audit({
            "ticket_id": ticket.ticket_id, "allowed": False,
            "layer": "quota", "refusal_reason": why,
        })
        return None

    if _RUN_LOCK.locked():
        _write_audit({
            "ticket_id": ticket.ticket_id, "allowed": False,
            "layer": "single_flight",
            "refusal_reason": "another auto-fix is already running",
        })
        return None

    _consume_quota()
    thread = threading.Thread(
        target=_run,
        kwargs={
            "ticket": ticket, "runtime": runtime, "store": store,
            "user_intent": user_intent,
        },
        name=f"auto-fix-{ticket.ticket_id}",
        daemon=True,
    )
    thread.start()
    return thread


def _run(
    *, ticket: DeveloperTicket, runtime: Any, store: Any, user_intent: str,
) -> None:
    """The background run. Own its errors completely — a crash here
    must never surface anywhere near the user path."""
    with _RUN_LOCK:
        t0 = time.monotonic()
        try:
            store.save_ticket(ticket.model_copy(
                update={"status": TicketStatus.in_progress},
            ))
            snap = _snapshot()

            report: dict[str, Any] | None = None
            error: str | None = None
            try:
                raw = runtime.invoke_operation(
                    "developer_fix",
                    {
                        "ticket_id": ticket.ticket_id,
                        "ticket": ticket.model_dump(mode="json"),
                        "instruction": ticket.requested_action,
                        "auto": True,
                    },
                    developer=True,
                )
                if isinstance(raw, dict) and raw.get("type") == "DeveloperReport":
                    report = raw
                else:
                    error = f"unexpected developer_fix response: {type(raw).__name__}"
            except RuntimeError_ as exc:
                error = f"runtime: {exc}"
            except Exception as exc:  # pragma: no cover - defensive
                error = f"unexpected: {type(exc).__name__}: {exc}"

            resolved = bool(report) and report.get("status") == "resolved"

            violations = _protected_violations(snap)
            if violations:
                # Hard reject no matter what the report says.
                resolved = False
                error = f"protected paths modified: {violations}"

            rollback: dict[str, Any] | None = None
            if not resolved:
                rollback = _restore(snap)

            note: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "resolved": resolved,
                "report_id": (report or {}).get("report_id"),
                "summary": str((report or {}).get("summary") or error or "")[:512],
                "actions_taken": list((report or {}).get("actions_taken") or [])[:8],
                "tests_run": list((report or {}).get("tests_run") or [])[:8],
            }
            if resolved:
                updated = _with_evidence_note(ticket, "auto_fix", note).model_copy(
                    update={
                        "status": TicketStatus.resolved,
                        "resolved_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            else:
                note["error"] = (error or "report not resolved")[:512]
                # Back to open: the diagnostic stays available for the
                # dev-time Claude agents; nothing waits on a human.
                updated = _with_evidence_note(
                    ticket, "auto_fix_attempts", note,
                ).model_copy(update={"status": TicketStatus.open})
            store.save_ticket(updated)

            _write_audit({
                "ticket_id": ticket.ticket_id,
                "allowed": True,
                "resolved": resolved,
                "error": error,
                "protected_violations": violations,
                "rollback": rollback,
                "report_id": (report or {}).get("report_id"),
                "actions_taken": list((report or {}).get("actions_taken") or [])[:8],
                "tests_run": list((report or {}).get("tests_run") or [])[:8],
                "user_intent_excerpt": (user_intent or "")[:240],
                "duration_ms": int((time.monotonic() - t0) * 1000),
            })
        except Exception as exc:  # pragma: no cover - last-ditch guard
            _write_audit({
                "ticket_id": ticket.ticket_id,
                "allowed": True,
                "resolved": False,
                "error": f"auto-fix crashed: {type(exc).__name__}: {exc}",
                "duration_ms": int((time.monotonic() - t0) * 1000),
            })
