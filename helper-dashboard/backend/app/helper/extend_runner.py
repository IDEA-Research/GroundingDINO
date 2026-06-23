"""Run the `rescue_extend` operation and surface the result.

Thin wrapper around `OpenCodeRuntime.invoke_operation("rescue_extend",
...)` that exists so both the orchestrator's DeveloperTicket branch
and the review_loop's `_run_rescue(kind="extend")` branch share the
same invocation path, error handling, and audit trail.

The runner deliberately keeps the policy thin:
- Validates the ExtendRequest shape via Pydantic.
- Runs the safety gates (denylist, prompt-injection, quota, audit).
- Calls the runtime.
- Returns the parsed DeveloperReport on success, or an ExtendOutcome
  with `.error` set on any failure (caller decides how to degrade —
  usually to an honest "tried to build it, didn't pass" message).

Extend is the **default path** for unknown widget types: the toolkit
is a cache, not a fence. Callers do not need to check any opt-in flag
before invoking the runner — they just call `run()` and let the
safety gates inside decide. See `extend_gate.py` for the gate layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
import time
from typing import Any

from ..specs import ExtendRequest
from .extend_gate import (
    GateDecision,
    consume_quota,
    evaluate_gates,
    write_audit_entry,
)
from .extend_request import ExtendRequest as ParsedExtendRequest
from .extend_snapshot import restore_from, take_snapshot
from .runtime import OpenCodeRuntime, RuntimeError_


# After a successful rescue_extend, these modules have stale class
# references in memory (e.g. WidgetType still has the old 5 members).
# Reloading in dependency order lets the live uvicorn worker pick up
# the new widget type without a full restart. Best-effort; downstream
# orchestrator separately recreates its SpecValidator / PatchService.
_RELOAD_AFTER_EXTEND_MODULES: tuple[str, ...] = (
    "app.specs.widget_spec",
    "app.specs.widget_schema_doc",
    "app.specs.dashboard_spec",
    "app.specs.patch_spec",
    "app.specs",
    "app.services.spec_validator",
    "app.services.patch_service",
)


def _reload_widget_modules() -> list[str]:
    """Best-effort reload of widget-related modules so the live process
    picks up newly-added enum members. Returns the list of module names
    that were successfully reloaded, for audit logging."""
    reloaded: list[str] = []
    for name in _RELOAD_AFTER_EXTEND_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        try:
            importlib.reload(mod)
            reloaded.append(name)
        except Exception:  # pragma: no cover - defensive
            pass
    return reloaded


@dataclass
class ExtendOutcome:
    """Result of one `rescue_extend` invocation.

    `report` is set on success; `error` is set on any failure (LLM
    refused, runtime crash, bad output shape). The caller can degrade
    on `report is None`.
    """

    report: dict[str, Any] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.report is not None and self.report.get("status") == "resolved"


class ExtendRunner:
    def __init__(self, runtime: OpenCodeRuntime) -> None:
        self._runtime = runtime

    # ------------------------------------------------------------
    def run(
        self,
        extend: ParsedExtendRequest,
        *,
        user_intent: str = "",
        original_args: dict[str, Any] | None = None,
    ) -> ExtendOutcome:
        """Invoke `rescue_extend` for the given ExtendRequest.

        Args:
          extend: a parsed extend request (from parse_extend_ticket or
            a RescueDecision.extend field).
          user_intent: the user's original message — passed to Big guy
            so it can pick reasonable defaults (e.g. donut vs full pie).
            Also fed to the gate's prompt-injection check.
          original_args: optional context bag (e.g. the failing spec)
            so Big guy can see what shape the user wanted.

        Returns an ExtendOutcome. Never raises.
        """
        # Belt-and-braces: re-validate via the Pydantic model. Even though
        # parse_extend_ticket / RescueDecision already did this, the
        # runner runs as the single trust boundary into rescue_extend.
        try:
            validated = ExtendRequest(
                widget_type=extend.widget_type,
                rationale=extend.summary or extend.user_visible_effect
                          or "extend toolkit",
            )
        except Exception as exc:
            return ExtendOutcome(error=f"extend request invalid: {exc}")

        # Safety gates. Refusals never reach the runtime.
        decision = evaluate_gates(
            widget_type=validated.widget_type,
            user_message=user_intent,
        )
        if not decision.allowed:
            return ExtendOutcome(
                error=f"gate refused ({decision.layer}): {decision.refusal_reason}"
            )
        # Quota is consumed atomically here — only counts toward the
        # day's budget if we actually about to invoke the runtime.
        consume_quota()

        args: dict[str, Any] = {
            "extend": validated.model_dump(mode="json"),
            "user_intent": user_intent[:1024] if user_intent else "",
        }
        if original_args is not None:
            args["context"] = {
                k: v for k, v in original_args.items()
                if k in {"intent", "requested_changes", "current_dashboard_id"}
            }

        # M7: take a per-file snapshot of the six write-listed paths
        # BEFORE invoking the runtime. Used by `_finalize` to surgically
        # restore originals if the run ends in failure (LLM rejection,
        # test fail, runtime crash, unexpected output shape).
        try:
            snapshot = take_snapshot(validated.widget_type)
        except Exception as exc:
            # Snapshot is best-effort but treated as required for safety:
            # without it we can't roll back on failure. Refuse to proceed.
            return ExtendOutcome(
                error=f"snapshot failed: {type(exc).__name__}: {exc}",
            )

        t0 = time.monotonic()

        def _finalize(*, outcome: ExtendOutcome, extra: dict[str, Any]) -> ExtendOutcome:
            extra.setdefault(
                "duration_ms", int((time.monotonic() - t0) * 1000),
            )
            if not outcome.ok:
                rollback = restore_from(snapshot)
                extra["rollback"] = {
                    "restored": rollback["restored"],
                    "deleted": rollback["deleted"],
                    "errors": rollback["errors"],
                }
            write_audit_entry(
                GateDecision(
                    allowed=True,
                    widget_type=validated.widget_type,
                    user_message_excerpt=(user_intent or "")[:240],
                ),
                extra=extra,
            )
            return outcome

        try:
            raw = self._runtime.invoke_operation("rescue_extend", args)
        except RuntimeError_ as exc:
            return _finalize(
                outcome=ExtendOutcome(error=f"runtime: {exc}"),
                extra={"error": f"runtime: {exc}"},
            )
        except Exception as exc:  # pragma: no cover - defensive
            return _finalize(
                outcome=ExtendOutcome(
                    error=f"unexpected: {type(exc).__name__}: {exc}"
                ),
                extra={"error": f"unexpected: {type(exc).__name__}: {exc}"},
            )

        if not isinstance(raw, dict):
            err = f"non-dict response: {type(raw).__name__}"
            return _finalize(
                outcome=ExtendOutcome(error=err),
                extra={"error": err},
            )
        if raw.get("type") != "DeveloperReport":
            err = (
                f"unexpected type {raw.get('type')!r}; expected DeveloperReport"
            )
            return _finalize(
                outcome=ExtendOutcome(error=err),
                extra={"error": err},
            )

        report_status = str(raw.get("status") or "")
        outcome = ExtendOutcome(report=raw)
        extra = {
            "status": report_status,
            "report_id": str(raw.get("report_id") or ""),
            "actions_taken": list(raw.get("actions_taken") or [])[:5],
            "tests_run": list(raw.get("tests_run") or [])[:5],
        }
        # C-5: importlib.reload of widget modules after extend is
        # POSSIBLE (see _reload_widget_modules helper above) but
        # creates two co-existing class objects in memory (old + new),
        # which breaks isinstance checks and corrupts pytest's
        # cross-test state. Gated behind explicit env opt-in so the
        # default behaviour is safe: operator restarts the backend
        # after the extend audit log shows a new widget type.
        #
        # Enable with HELPER_DASHBOARD_AUTO_EXTEND_RELOAD_MODULES=true
        # for live-demo scenarios where a manual restart is impractical.
        import os
        if outcome.ok and os.getenv(
            "HELPER_DASHBOARD_AUTO_EXTEND_RELOAD_MODULES", ""
        ).strip().lower() in {"1", "true", "yes", "on"}:
            reloaded = _reload_widget_modules()
            if reloaded:
                extra["reloaded_modules"] = reloaded
        return _finalize(outcome=outcome, extra=extra)
