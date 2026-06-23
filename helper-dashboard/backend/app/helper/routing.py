"""Routing is now inlined in `orchestrator._dispatch`.

This module is kept as a stable import path. New code should call
`OpenCodeRuntime.invoke_operation` directly.
"""

from __future__ import annotations

from typing import Any

from .runtime import OpenCodeRuntime


class Routing:
    """Deprecated thin shim around `OpenCodeRuntime.invoke_operation`.

    Kept for any external callers; the orchestrator no longer uses it.
    """

    def __init__(self, runtime: OpenCodeRuntime):
        self._rt = runtime

    def dispatch(
        self,
        intent: dict[str, Any],
        *,
        current_dashboard: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        t = intent.get("type")
        if t == "DashboardIntent":
            return self._rt.invoke_operation(
                "generate_dashboard",
                {"intent": intent, "previous_dashboard": current_dashboard},
            )
        if t == "PatchIntent":
            if current_dashboard is None:
                return {
                    "type": "UserResponse",
                    "message": "I don't have a dashboard loaded yet — create one first.",
                }
            return self._rt.invoke_operation(
                "patch_dashboard",
                {"intent": intent, "dashboard": current_dashboard},
            )
        if t == "PrometheusIntent":
            return self._rt.invoke_operation(
                "prometheus_query",
                {"question": intent.get("question", "")},
            )
        return intent
