"""ReviewDecision and RescueDecision schemas.

Produced by the pre-output review loop:

- Helper's `review_rendered` operation emits ReviewDecision.
- Big guy's `rescue_review` operation emits RescueDecision.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .patch_spec import PatchSpec
from .developer_ticket import DeveloperTicket


class ReviewDecision(BaseModel):
    """Helper's verdict on a rendered dashboard.

    `decision`:
      - approve: the dashboard matches user intent; promote and return.
      - patch: author the attached PatchSpec and re-review.
      - escalate: Helper cannot fix it; call Big guy rescue.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "patch", "escalate"]
    rationale: str = Field(..., min_length=1, max_length=1024)
    patch: PatchSpec | None = None


class ExtendRequest(BaseModel):
    """A request for Big guy to extend the widget toolkit with a new
    widget type. Carried inside `RescueDecision.extend` when
    `kind == "extend"`.

    Big guy, in rescue mode, can recognize that the failed dashboard
    came from a missing widget type and propose an extension. M4 routes
    accepted extends to the `rescue_extend` operation, which runs with
    elevated permissions (gated by M5 protections) to make the 5-layer
    code change.

    Fields here are intentionally minimal — the actual code change is
    decided by Big guy in rescue_extend mode, not encoded here.
    """

    model_config = ConfigDict(extra="forbid")

    # Lowercase snake_case, 3-32 chars, must start with a letter.
    # Enforced by Pydantic so the contract layer doesn't need to repeat
    # the regex. M5 will additionally enforce a denylist of forbidden
    # names (script, iframe, ...) at the runtime gate.
    widget_type: str = Field(..., pattern=r"^[a-z][a-z0-9_]{2,31}$")
    rationale: str = Field(..., min_length=1, max_length=512)
    # Free-text hint for the implementer (e.g. "use recharts PieChart",
    # "donut variant allowed"). Optional — Big guy decides defaults.
    component_hint: str | None = Field(default=None, max_length=512)


class RescueDecision(BaseModel):
    """Big guy's verdict when Helper escalated.

    `kind`:
      - patch: Big guy proposes a final PatchSpec (one last iteration).
      - ask_user: the user needs to clarify; Helper will relay questions.
      - ticket: the issue is a code-level bug; file a DeveloperTicket
        and show the user a safe error message.
      - extend: the failure is because the widget type is missing from
        the toolkit; M4 routes this to `rescue_extend` (gated by M5
        protections) to auto-build the missing widget, then retries.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["patch", "ask_user", "ticket", "extend"]
    rationale: str = Field(..., min_length=1, max_length=1024)
    patch: PatchSpec | None = None
    questions: list[str] | None = Field(default=None, max_length=8)
    ticket: DeveloperTicket | None = None
    extend: ExtendRequest | None = None
