"""Devfix run-log API — read-only.

Exposes the disk-backed run logs written by
`app.helper.devfix_log` so the frontend can show, live, what Big Guy
thinks and does while fixing the dashboard's own code. Read-only and
unauthenticated, like the rest of the read API: it leaks nothing that
isn't already on the operator's own disk, and it never triggers a run.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..helper import devfix_log

router = APIRouter()


@router.get("/runs")
def list_runs() -> dict:
    """Newest-first run metas (capped by the store's default limit)."""
    return {"runs": devfix_log.list_runs()}


@router.get("/runs/{run_id}")
def read_run(run_id: str, after: int = 0) -> dict:
    """Meta + rendered log lines from cursor `after`; poll with
    `?after=<next_cursor>` to tail incrementally."""
    try:
        return devfix_log.read_run(run_id, after=after)
    except KeyError:
        # Unknown id AND malformed/traversal-shaped ids land here — the
        # store validates the run_id pattern before touching the disk.
        raise HTTPException(status_code=404, detail="run not found")
