"""Chat API — user-facing entry point.

POST /api/chat/message
    { session_id, message, current_dashboard_id? }
    -> { user_reply, intent, dashboard?, patch?, warnings }

The route is thin: it calls the orchestrator which owns the Helper
flow. Nothing here edits code, nothing here runs shell.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..helper.orchestrator import Orchestrator


router = APIRouter()
_orch = Orchestrator()
_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=4000)
    current_dashboard_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    # Optional direct assistant message from LLM/helper runtime.
    # Kept separate from user_reply so UI can choose to surface it explicitly.
    message_to_user: str | None = None
    user_reply: str
    intent_type: str
    dashboard: dict[str, Any] | None = None
    patch: dict[str, Any] | None = None
    warnings: list[str] = []
    # Runtime provenance — None in mock mode, "opencode" on real
    # success, "mock_fallback" when auto mode fell back. The UI can
    # use this for a "using fallback" badge.
    runtime_used: str | None = None
    fallback_reason: str | None = None
    # Pre-output review transcript (empty when review is off).
    review_trail: list[dict[str, Any]] = []
    # When the rescue step decided the user must clarify.
    clarification_questions: list[str] | None = None
    # When the orchestrator is asking "Would you like to save this?",
    # carries the dashboard_id the next user message will save.
    save_prompt_for: str | None = None
    # Internal artifacts (tickets, memory) are never returned here.


class ChatEnqueueResponse(BaseModel):
    job_id: str
    status: str


class ChatJobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress_message: str | None = None
    progress_pct: int | None = None
    result: ChatResponse | None = None
    error: str | None = None


def _run_job(job_id: str, req: ChatRequest) -> None:
    def _update_progress(msg: str, pct: int | None = None) -> None:
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["progress_message"] = msg
                if pct is not None:
                    _JOBS[job_id]["progress_pct"] = pct

    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["status"] = "running"
            _JOBS[job_id]["progress_message"] = "Starting…"
            _JOBS[job_id]["progress_pct"] = 5
    try:
        result = _orch.handle_user_message(
            session_id=req.session_id,
            message=req.message,
            current_dashboard_id=req.current_dashboard_id,
            progress_cb=_update_progress,
        )
        payload = ChatResponse(**result).model_dump(mode="json")
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "done"
            _JOBS[job_id]["result"] = payload
            _JOBS[job_id]["progress_message"] = "Completed"
            _JOBS[job_id]["progress_pct"] = 100
    except Exception as exc:
        err = f"chat pipeline error: {type(exc).__name__}: {exc}"
        print(f"[chat_api] background job error: {err}")
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["error"] = err
            _JOBS[job_id]["progress_message"] = "Failed"
    finally:
        with _JOBS_LOCK:
            _JOBS[job_id]["updated_at"] = time.time()


@router.post("/message", response_model=ChatEnqueueResponse)
def post_message(req: ChatRequest) -> ChatEnqueueResponse:
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress_message": "Queued…",
            "progress_pct": 0,
            "result": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    t = threading.Thread(target=_run_job, args=(job_id, req), daemon=True)
    t.start()
    return ChatEnqueueResponse(job_id=job_id, status="queued")


@router.get("/message/{job_id}", response_model=ChatJobStatusResponse)
def get_message_status(job_id: str) -> ChatJobStatusResponse:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    result_obj = ChatResponse(**job["result"]) if job.get("result") else None
    return ChatJobStatusResponse(
        job_id=job_id,
        status=str(job.get("status") or "unknown"),
        progress_message=job.get("progress_message"),
        progress_pct=job.get("progress_pct"),
        result=result_obj,
        error=job.get("error"),
    )
