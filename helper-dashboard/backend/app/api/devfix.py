"""Devfix ("Big Guy") run API — developer-gated.

EVERY route here requires `X-Developer-Token` (see `_auth.require_dev`):
the Jetson serves the user UI and the developer workspace from the same
process, so reads are gated too — run logs quote instructions, worktree
paths and model names.

Read side (disk-backed `app.helper.devfix_log`):
  GET /config               preflight (paths, bin, model, timeout, gates)
  GET /runs                 newest-first metas (limit 50)
  GET /runs/{id}?after=N    meta + rendered log lines from cursor
  GET /runs/{id}/events?after=N   raw scrubbed CLI events from cursor
  GET /runs/{id}/diff       unified diff captured before worktree removal

Execution side (`app.helper.devfix_queue`; kill switch
HELPER_DASHBOARD_DEVFIX_LOCAL_RUNS=1 + opencode-cli executor required,
else 409 {"detail": "本機執行層未啟用", "reason": <code>}):
  POST /runs                {instruction, title?} -> 202 {run_id, ticket_id, status}
  POST /runs/{id}/cancel    -> 202
  POST /runs/{id}/retry     -> 202 (new run linked via parent_run_id)

No route accepts commands, paths or model overrides (extra body keys
are rejected with 422).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..helper import devfix_log, devfix_queue
from ._auth import require_dev

router = APIRouter(dependencies=[Depends(require_dev)])

_LOCAL_RUNS_OFF = "本機執行層未啟用"


class StartRunRequest(BaseModel):
    """Only an instruction and an optional title — nothing else."""

    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(..., min_length=1, max_length=devfix_queue.INSTRUCTION_MAX)
    title: str | None = Field(default=None, max_length=devfix_queue.TITLE_MAX)


def _refuse(status_code: int, detail: str, reason: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail, "reason": reason})


def _local_runs_refusal() -> JSONResponse | None:
    ok, reason = devfix_queue.local_runs_status()
    if ok:
        return None
    return _refuse(409, _LOCAL_RUNS_OFF, reason or "local_runs_disabled")


def _meta_or_404(run_id: str) -> dict:
    meta = devfix_log.read_meta(run_id)
    if meta is None:
        # Unknown id AND malformed/traversal-shaped ids land here — the
        # store validates the run_id pattern before touching the disk.
        raise HTTPException(status_code=404, detail="run not found")
    return meta


# ---------------------------------------------------------------------------
# read side
# ---------------------------------------------------------------------------


@router.get("/config")
def read_config() -> dict:
    """Read-only preflight for the dev panel. Also the token-verification
    probe the frontend uses when the developer enters the token."""
    return devfix_queue.config_payload()


@router.get("/runs")
def list_runs() -> dict:
    """Newest-first run metas (limit 50)."""
    return {"runs": devfix_log.list_runs(limit=50)}


@router.get("/runs/{run_id}")
def read_run(run_id: str, after: int = 0) -> dict:
    """Meta + rendered log lines from cursor `after`; poll with
    `?after=<next_cursor>` to tail incrementally."""
    try:
        return devfix_log.read_run(run_id, after=after)
    except KeyError:
        raise HTTPException(status_code=404, detail="run not found")


@router.get("/runs/{run_id}/events")
def read_run_events(run_id: str, after: int = 0) -> dict:
    """Raw (scrubbed) CLI events from cursor `after` (event index)."""
    try:
        return devfix_log.read_events(run_id, after=after)
    except KeyError:
        raise HTTPException(status_code=404, detail="run not found")


@router.get("/runs/{run_id}/diff")
def read_run_diff(run_id: str) -> dict:
    """{patch, files} — the worktree diff captured on every terminal state."""
    try:
        return devfix_log.read_diff(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="run not found")


# ---------------------------------------------------------------------------
# execution side
# ---------------------------------------------------------------------------


def _submit_or_refuse(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except devfix_queue.LocalRunsUnavailable as exc:
        return _refuse(409, _LOCAL_RUNS_OFF, exc.reason)
    except devfix_queue.QueueFull:
        return _refuse(409, f"佇列已滿（最多 {devfix_queue.MAX_PENDING} 個待執行任務）", "queue_full")
    except devfix_queue.QuotaExhausted:
        return _refuse(409, "今日執行配額已用完", "quota_exhausted")
    except devfix_queue.RunActive:
        return _refuse(409, "已有任務執行中，請稍後再試", "run_active")


@router.post("/runs", status_code=202)
def start_run(req: StartRunRequest):
    refusal = _local_runs_refusal()
    if refusal is not None:
        return refusal
    instruction = req.instruction.strip()
    if not instruction:
        raise HTTPException(status_code=422, detail="任務說明不可為空白")
    title = (req.title or "").strip() or None
    return _submit_or_refuse(devfix_queue.submit, instruction, title)


@router.post("/runs/{run_id}/cancel", status_code=202)
def cancel_run(run_id: str):
    if not devfix_queue.local_runs_enabled():
        return _refuse(409, _LOCAL_RUNS_OFF, "local_runs_disabled")
    meta = _meta_or_404(run_id)
    result = devfix_queue.cancel(run_id)
    if result == "not_active":
        if meta.get("status") in devfix_log.TERMINAL_STATUSES:
            return _refuse(409, "任務已結束", "not_active")
        # Disk says running, but no live process in this backend process
        # (another worker or a stale record). Never fake a cancel.
        return _refuse(409, "找不到執行中的程序", "no_live_process")
    return {
        "run_id": run_id,
        "result": result,
        "status": "cancelled" if result == "removed" else "cancelling",
    }


@router.post("/runs/{run_id}/retry", status_code=202)
def retry_run(run_id: str):
    refusal = _local_runs_refusal()
    if refusal is not None:
        return refusal
    _meta_or_404(run_id)
    try:
        return _submit_or_refuse(devfix_queue.retry, run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="run not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
