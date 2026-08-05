"""ClarificationRequest — user-facing ask-the-user envelope.

Produced by the orchestrator when rescue_review decides the user
must clarify. The UI renders these with the question list visible.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ClarificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_to_user: str = Field(..., min_length=1, max_length=512)
    questions: list[str] = Field(..., min_length=1, max_length=8)
    origin: str = Field(
        default="rescue_review",
        max_length=32,
        description="Which stage requested the clarification.",
    )
