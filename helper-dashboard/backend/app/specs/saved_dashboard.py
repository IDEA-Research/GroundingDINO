"""SavedDashboard — user library of reusable dashboards.

After a dashboard passes the review loop, the orchestrator asks the
user whether to save it. If yes (with a name), we write one file per
entry here. Helper gets the list as context at session start and can
semantically suggest reuse.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .dashboard_spec import DashboardSpec


class SavedDashboard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    saved_id: str = Field(
        ..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$"
    )
    name: str = Field(..., min_length=1, max_length=96)
    description: str = Field(default="", max_length=512)
    tags: list[str] = Field(default_factory=list, max_length=16)
    # Snapshot of the DashboardSpec at save time. Future loads clone
    # this and give the clone a fresh dashboard_id.
    spec: DashboardSpec
    # Convenience summary Helper can use without loading the whole
    # spec: widget types and their promql strings.
    summary: dict = Field(default_factory=dict)
    created_at: str | None = None
    used_count: int = 0


class SavedDashboardSummary(BaseModel):
    """The lightweight shape that gets injected into Helper's prompt."""

    model_config = ConfigDict(extra="forbid")

    saved_id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    widget_types: list[str] = Field(default_factory=list)
    promqls: list[str] = Field(default_factory=list)
    used_count: int = 0
