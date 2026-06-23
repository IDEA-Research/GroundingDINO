"""DeveloperTicket — the only channel into Big guy.

Written by Helper agents or the browser evaluator when a problem
requires real code changes. Stored in
`backend/app/storage/tickets/<ticket_id>.json`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TicketSeverity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class TicketStatus(str, Enum):
    open = "open"
    in_progress = "in_progress"
    resolved = "resolved"
    rejected = "rejected"


class DeveloperTicket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    source_agent: str = Field(..., max_length=64)
    severity: TicketSeverity = TicketSeverity.medium
    summary: str = Field(..., min_length=1, max_length=256)
    user_visible_effect: str = Field(..., min_length=1, max_length=512)
    technical_evidence: dict[str, Any] | str = Field(default_factory=dict)
    requested_action: str = Field(..., min_length=1, max_length=1024)
    safety_notes: str = Field(default="", max_length=1024)
    status: TicketStatus = TicketStatus.open
    created_at: str | None = None
    resolved_at: str | None = None
