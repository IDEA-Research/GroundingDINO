"""Developer API — internal only.

This router is mounted but guarded. The user-facing chat flow never
reaches these endpoints. Access requires the developer header
`X-Developer-Token` matching the `HELPER_DASHBOARD_DEV_TOKEN` env var
(shared dependency `_auth.require_dev`, constant-time compare).

If the env var is unset, developer endpoints return 403. This keeps
production deployments from accidentally exposing them.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..services.dashboard_store import DashboardStore
from ..specs.developer_ticket import DeveloperTicket, TicketStatus
from ._auth import require_dev


router = APIRouter(dependencies=[Depends(require_dev)])
_store = DashboardStore()


@router.get("/tickets")
def list_tickets() -> dict:
    return {"tickets": _store.list_tickets()}


@router.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str) -> dict:
    ticket = _store.load_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    return {"ticket": ticket.model_dump(mode="json")}


@router.post("/tickets/{ticket_id}/status")
def set_ticket_status(ticket_id: str, payload: dict) -> dict:
    status = payload.get("status")
    if status not in {s.value for s in TicketStatus}:
        raise HTTPException(status_code=400, detail="invalid status")
    ticket = _store.load_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    ticket = ticket.model_copy(update={"status": TicketStatus(status)})
    _store.save_ticket(ticket)
    return {"ok": True, "ticket": ticket.model_dump(mode="json")}
