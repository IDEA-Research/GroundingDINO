"""Developer API — internal only.

This router is mounted but guarded. The user-facing chat flow never
reaches these endpoints. Access requires the developer header
`X-Developer-Token` matching the `HELPER_DASHBOARD_DEV_TOKEN` env var.

If the env var is unset, developer endpoints return 403. This keeps
production deployments from accidentally exposing them.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Header, HTTPException

from ..services.dashboard_store import DashboardStore
from ..specs.developer_ticket import DeveloperTicket, TicketStatus


router = APIRouter()
_store = DashboardStore()


def _require_dev(token: str | None) -> None:
    expected = os.getenv("HELPER_DASHBOARD_DEV_TOKEN")
    if not expected or token != expected:
        raise HTTPException(status_code=403, detail="developer access required")


@router.get("/tickets")
def list_tickets(x_developer_token: str | None = Header(default=None)) -> dict:
    _require_dev(x_developer_token)
    return {"tickets": _store.list_tickets()}


@router.get("/tickets/{ticket_id}")
def get_ticket(
    ticket_id: str, x_developer_token: str | None = Header(default=None)
) -> dict:
    _require_dev(x_developer_token)
    ticket = _store.load_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    return {"ticket": ticket.model_dump(mode="json")}


@router.post("/tickets/{ticket_id}/status")
def set_ticket_status(
    ticket_id: str,
    payload: dict,
    x_developer_token: str | None = Header(default=None),
) -> dict:
    _require_dev(x_developer_token)
    status = payload.get("status")
    if status not in {s.value for s in TicketStatus}:
        raise HTTPException(status_code=400, detail="invalid status")
    ticket = _store.load_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    ticket = ticket.model_copy(update={"status": TicketStatus(status)})
    _store.save_ticket(ticket)
    return {"ok": True, "ticket": ticket.model_dump(mode="json")}
