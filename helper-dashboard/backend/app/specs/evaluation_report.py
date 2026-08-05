"""BrowserEvaluationReport — output of the Playwright evaluator."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BrowserEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dashboard_id: str = Field(..., min_length=1, max_length=64)
    page_loaded: bool
    widgets_rendered: list[str] = Field(default_factory=list, max_length=64)
    missing_widgets: list[str] = Field(default_factory=list, max_length=64)
    console_errors: list[str] = Field(default_factory=list, max_length=200)
    layout_errors: list[str] = Field(default_factory=list, max_length=64)
    prometheus_errors: list[str] = Field(default_factory=list, max_length=64)
    screenshot_path: str | None = Field(default=None, max_length=512)
    recommendation: str = Field(default="", max_length=512)
    # Populated by the orchestrator, not by Playwright itself.
    created_at: str | None = None
