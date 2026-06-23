"""BugReport — structured record of a dashboard malfunction."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class BugSeverity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class SuggestedFixType(str, Enum):
    patch = "patch"           # Fix by PatchSpec
    code = "code"             # Needs a DeveloperTicket
    promql = "promql"         # Query change, still a patch
    data_source = "data_source"  # Prometheus / infra
    unknown = "unknown"


class BugEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    console_errors: list[str] = Field(default_factory=list, max_length=100)
    missing_widgets: list[str] = Field(default_factory=list, max_length=64)
    layout_errors: list[str] = Field(default_factory=list, max_length=64)
    prometheus_errors: list[str] = Field(default_factory=list, max_length=64)
    screenshot_path: str | None = Field(default=None, max_length=512)


class BugReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bug_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    source: str = Field(..., max_length=64)
    severity: BugSeverity
    summary: str = Field(..., min_length=1, max_length=256)
    evidence: BugEvidence = Field(default_factory=BugEvidence)
    suspected_cause: str = Field(default="", max_length=512)
    suggested_fix_type: SuggestedFixType = SuggestedFixType.unknown
