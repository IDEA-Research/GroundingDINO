"""Auto-debug loop.

Given a validation error, a patch application error, or a browser
evaluation report, decides whether the problem can be fixed by JSON
(ask `patch-agent` for a PatchSpec) or requires code changes (write
a DeveloperTicket for Big guy).

This module does **not** call Big guy. It only produces tickets on
disk. Big guy is invoked separately by developer tooling.
"""

from __future__ import annotations

import uuid
from typing import Any

from ..specs import BrowserEvaluationReport, DashboardSpec
from ..specs.developer_ticket import DeveloperTicket, TicketSeverity
from .runtime import OpenCodeRuntime


class DebugLoop:
    def __init__(self, runtime: OpenCodeRuntime) -> None:
        self._rt = runtime

    def from_validation_errors(
        self, errors: list[str], *, source: str
    ) -> DeveloperTicket:
        return DeveloperTicket(
            ticket_id=f"tkt-{uuid.uuid4().hex[:8]}",
            source_agent=source,
            severity=TicketSeverity.medium,
            summary=f"spec validation failure from {source}",
            user_visible_effect="user's dashboard could not be saved",
            technical_evidence={"validation_errors": errors[:20]},
            requested_action=(
                "tighten Helper prompt and/or extend validator; see errors."
            ),
            safety_notes="never weaken validator without explicit dev instruction",
        )

    def from_browser_report(
        self, report: BrowserEvaluationReport, *, spec: DashboardSpec
    ) -> dict[str, Any]:
        """Ask `browser-eval-agent` for a structured decision."""
        return self._rt.invoke_helper(
            "browser-eval-agent",
            {"report": report.model_dump(mode="json"), "dashboard": spec.model_dump(mode="json")},
        )
