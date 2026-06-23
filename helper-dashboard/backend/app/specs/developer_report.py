"""DeveloperReport — output contract for the `developer_fix` operation.

Written by `big-guy-developer-agent` via the developer-only
`/api/developer/tickets/{id}/...` channel. Never produced by Helper
agents and never surfaced through the user-facing chat flow.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DeveloperReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    ticket_id: str | None = Field(default=None, max_length=64)
    instruction: str | None = Field(default=None, max_length=1024)
    actions_taken: list[str] = Field(default_factory=list, max_length=64)
    tests_run: list[str] = Field(default_factory=list, max_length=64)
    summary: str = Field(..., min_length=1, max_length=2048)
    status: Literal["resolved", "in_progress", "rejected"] = "resolved"
    created_at: str | None = None
