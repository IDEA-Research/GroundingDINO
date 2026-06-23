"""Pre-output review loop.

Runs after Helper authors a `DashboardSpec` or `PatchSpec`, before
the result is surfaced to the user. Synchronous; the chat request
blocks while this runs.

Flow:
  1. Render draft in headless browser (Playwright, with a safe
     fallback when Playwright isn't installed).
  2. Call `review_rendered` (Helper small model). It emits
     ReviewDecision: approve | patch | escalate.
  3. If `patch`, apply → go to 1. Max 3 Helper iterations.
     If the patch is malformed (Pydantic rejects it), feed the
     errors back as `_prior_errors` / `_prior_feedback_message`
     on the next iteration instead of escalating immediately.
  4. If `escalate` (or retries exhausted), call `rescue_review`
     (Big guy, JSON-only, no code edits). It emits RescueDecision:
     patch | ask_user | ticket | extend. Rescue itself retries up
     to `_RESCUE_MAX_ATTEMPTS` times with structured Pydantic
     errors when the response is malformed; out of budget degrades
     to a user-clarify (never RuntimeError).
  5. If rescue `patch`, apply → render once more to confirm. If
     still bad, fall through to ask_user.

The loop returns a `ReviewOutcome` to the orchestrator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from ..services.browser_evaluator import BrowserEvaluator
from ..services.dashboard_store import DashboardStore
from ..services.patch_service import PatchApplicationError, PatchService
from ..services.spec_validator import SpecValidationError, SpecValidator
from ..specs import DashboardSpec, PatchSpec
from ..specs.developer_ticket import DeveloperTicket
from .extend_request import ExtendRequest as ParsedExtendRequest
from .extend_runner import ExtendRunner
from .runtime import OpenCodeRuntime, RuntimeError_


HELPER_MAX_ATTEMPTS = 3
# Rescue is "one-shot-ish": one initial call + at most one feedback
# retry. The retry only triggers when the first response was
# structurally invalid (PatchSpec or DeveloperTicket rejected by
# Pydantic), giving the LLM a concrete repair target instead of
# crashing the turn into a user-visible RuntimeError.
_RESCUE_MAX_ATTEMPTS = 2


OutcomeKind = Literal["approved", "clarify", "ticket", "failed"]


@dataclass
class ReviewOutcome:
    kind: OutcomeKind
    dashboard: DashboardSpec | None = None
    patches_applied: list[PatchSpec] = field(default_factory=list)
    questions: list[str] | None = None
    ticket: DeveloperTicket | None = None
    # Transcript of what happened, for the Inspector "Review" tab.
    trail: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def is_enabled() -> bool:
    """Feature flag. On by default in opencode/auto; off in mock."""
    raw = os.getenv("HELPER_DASHBOARD_PRE_OUTPUT_REVIEW")
    if raw is not None:
        return raw not in ("0", "", "false", "False", "no", "off")
    mode = (os.getenv("HELPER_DASHBOARD_OPENCODE") or "mock").strip().lower()
    return mode in {"opencode", "auto"}


class ReviewLoop:
    def __init__(
        self,
        runtime: OpenCodeRuntime,
        evaluator: BrowserEvaluator | None = None,
        validator: SpecValidator | None = None,
        patch_service: PatchService | None = None,
        store: DashboardStore | None = None,
        extend_runner: ExtendRunner | None = None,
    ) -> None:
        self._runtime = runtime
        self._evaluator = evaluator or BrowserEvaluator()
        self._validator = validator or SpecValidator()
        self._patch_service = patch_service or PatchService(self._validator)
        self._store = store or DashboardStore()
        # M4: shared ExtendRunner for the rescue_extend escape hatch.
        self._extend_runner = extend_runner or ExtendRunner(runtime)

    # --------------------------------------------------------------
    def run(
        self,
        draft: DashboardSpec,
        *,
        user_intent: str,
    ) -> ReviewOutcome:
        trail: list[dict[str, Any]] = []
        patches_applied: list[PatchSpec] = []
        current = draft

        # Expose the draft to the frontend while we render it. The
        # shadow lives in memory + on disk (`_drafts/`) so a different
        # uvicorn worker handling the GET can still see it.
        self._store.set_draft(current)
        try:
            return self._run_loop(current, user_intent, trail, patches_applied)
        finally:
            self._store.clear_draft(draft.dashboard_id)

    def _run_loop(
        self,
        current: DashboardSpec,
        user_intent: str,
        trail: list[dict[str, Any]],
        patches_applied: list[PatchSpec],
    ) -> ReviewOutcome:

        # When a previous attempt's PatchSpec failed Python schema
        # validation, we surface the errors to the next review call
        # as structured `_prior_errors` + `_prior_feedback_message`.
        # The Python validator is authoritative; the LLM does NOT get
        # to decide whether a spec/patch is well-formed. This is the
        # same `_prior_errors` contract the user-originated retry
        # wrapper uses, but applied locally inside the review loop —
        # no nested wrapper, so no combinatorial blow-up.
        pending_feedback: dict[str, Any] | None = None

        # --- Helper loop ----------------------------------------------------
        for attempt in range(1, HELPER_MAX_ATTEMPTS + 1):
            report = self._evaluator.evaluate(current)
            trail.append(
                {
                    "stage": "render",
                    "attempt": attempt,
                    "page_loaded": report.page_loaded,
                    "missing": report.missing_widgets,
                    "console_errors": len(report.console_errors),
                }
            )

            review_args: dict[str, Any] = {
                "dashboard": current.model_dump(mode="json"),
                "report": report.model_dump(mode="json"),
                "user_intent": user_intent,
                "attempt": attempt,
                "history": [t for t in trail if t.get("stage") == "review"],
            }
            if pending_feedback is not None:
                review_args["_prior_errors"] = pending_feedback["errors"]
                review_args["_prior_feedback_message"] = pending_feedback["message"]
            try:
                review = self._runtime.invoke_operation(
                    "review_rendered", review_args,
                )
            except RuntimeError_ as exc:
                return _failed(trail, f"review_rendered failed: {exc}")

            # Feedback is one-shot: consumed by this attempt.
            pending_feedback = None

            trail.append(
                {
                    "stage": "review",
                    "attempt": attempt,
                    "decision": review.get("decision"),
                    "rationale": str(review.get("rationale", ""))[:160],
                }
            )

            decision = review.get("decision")
            if decision == "approve":
                return ReviewOutcome(
                    kind="approved",
                    dashboard=current,
                    patches_applied=patches_applied,
                    trail=trail,
                )

            if decision == "patch":
                patch_dict = review.get("patch") or {}
                try:
                    patch = self._validator.validate_patch(patch_dict)
                    current = self._patch_service.apply(current, patch)
                except (SpecValidationError, PatchApplicationError) as exc:
                    errs = (
                        exc.errors
                        if isinstance(exc, SpecValidationError)
                        else [str(exc)]
                    )
                    trail.append(
                        {
                            "stage": "patch_validation_failed",
                            "attempt": attempt,
                            "errors": errs[:6],
                        }
                    )
                    # On the last attempt there's no budget left for a
                    # retry — escalate. Otherwise stash the errors as
                    # structured feedback for the next iteration so the
                    # LLM gets a concrete repair target, not a guess.
                    if attempt >= HELPER_MAX_ATTEMPTS:
                        break
                    pending_feedback = {
                        "errors": errs[:8],
                        "message": _build_patch_feedback_message(errs[:8]),
                    }
                    continue  # next render + review
                patches_applied.append(patch)
                self._store.set_draft(current)  # keep shadow current
                continue  # next render + review

            if decision == "escalate":
                break

            return _failed(trail, f"unknown review decision: {decision!r}")

        # --- Big guy rescue --------------------------------------------------
        # Rescue is "one-shot-ish": we allow at most one retry, and only
        # when the LLM returned a structurally-broken patch or ticket
        # that Python rejected. The retry surfaces the Pydantic errors
        # via `_prior_errors` / `_prior_feedback_message` (same contract
        # as the Helper loop). If the second attempt is still bad we
        # degrade to a user-clarify message — NEVER to a RuntimeError —
        # so the UI shows something actionable instead of "Something
        # went wrong".
        return self._run_rescue(
            current=current,
            user_intent=user_intent,
            trail=trail,
            patches_applied=patches_applied,
        )

    # --------------------------------------------------------------
    def _run_rescue(
        self,
        *,
        current: DashboardSpec,
        user_intent: str,
        trail: list[dict[str, Any]],
        patches_applied: list[PatchSpec],
    ) -> ReviewOutcome:
        last_report = self._evaluator.evaluate(current)
        trail.append({"stage": "escalate", "to": "big-guy-developer-agent"})

        rescue_pending: dict[str, Any] | None = None
        rescue: dict[str, Any] | None = None
        for rescue_attempt in range(1, _RESCUE_MAX_ATTEMPTS + 1):
            args: dict[str, Any] = {
                "dashboard": current.model_dump(mode="json"),
                "report": last_report.model_dump(mode="json"),
                "user_intent": user_intent,
                "trail": trail,
            }
            if rescue_pending is not None:
                args["_prior_errors"] = rescue_pending["errors"]
                args["_prior_feedback_message"] = rescue_pending["message"]
            try:
                rescue = self._runtime.invoke_operation("rescue_review", args)
            except RuntimeError_ as exc:
                trail.append(
                    {"stage": "rescue_runtime_error",
                     "attempt": rescue_attempt,
                     "error": str(exc)[:240]}
                )
                # Runtime-level failure (provider down, JSON parse, ...) —
                # don't bury this; degrade to clarify so the user has a
                # chance to retry with different wording.
                return _ask_user_fallback(
                    trail,
                    ["I had trouble finalizing that dashboard. "
                     "Could you describe what you'd like to see more "
                     "concretely (which metrics, which time range)?"],
                )

            rescue_pending = None
            kind = rescue.get("kind")
            trail.append(
                {"stage": "rescue", "attempt": rescue_attempt, "kind": kind,
                 "rationale": str(rescue.get("rationale", ""))[:160]}
            )

            if kind == "patch":
                try:
                    patch = self._validator.validate_patch(rescue.get("patch") or {})
                    current = self._patch_service.apply(current, patch)
                    patches_applied.append(patch)
                    self._store.set_draft(current)
                except (SpecValidationError, PatchApplicationError) as exc:
                    errs = (
                        exc.errors
                        if isinstance(exc, SpecValidationError)
                        else [str(exc)]
                    )
                    trail.append(
                        {"stage": "rescue_patch_validation_failed",
                         "attempt": rescue_attempt, "errors": errs[:6]}
                    )
                    if rescue_attempt < _RESCUE_MAX_ATTEMPTS:
                        rescue_pending = {
                            "errors": errs[:8],
                            "message": _build_patch_feedback_message(errs[:8]),
                        }
                        continue  # retry rescue with feedback
                    # Out of rescue budget — clarify rather than fail.
                    return _ask_user_fallback(
                        trail,
                        ["I built a draft dashboard but couldn't finalize "
                         "the last automated fix. Can you describe what you'd "
                         "like to see more precisely?"],
                    )

                # Patch applied — one confirmation render.
                confirm = self._evaluator.evaluate(current)
                trail.append(
                    {"stage": "rescue_confirm",
                     "page_loaded": confirm.page_loaded,
                     "missing": confirm.missing_widgets}
                )
                if confirm.page_loaded and not confirm.missing_widgets \
                   and not confirm.console_errors:
                    return ReviewOutcome(
                        kind="approved",
                        dashboard=current,
                        patches_applied=patches_applied,
                        trail=trail,
                    )
                return _ask_user_fallback(
                    trail,
                    ["I tried a couple of approaches but the dashboard still "
                     "doesn't render cleanly. Can you clarify what you need?"],
                )

            if kind == "ask_user":
                questions = rescue.get("questions") or []
                if not questions:
                    questions = [
                        "Can you tell me more about the dashboard you'd like?"
                    ]
                return ReviewOutcome(
                    kind="clarify",
                    questions=questions,
                    trail=trail,
                )

            if kind == "ticket":
                try:
                    ticket = _parse_ticket(rescue.get("ticket") or {})
                except Exception as exc:
                    errs = _flatten_pydantic_errors(exc)
                    trail.append(
                        {"stage": "rescue_ticket_validation_failed",
                         "attempt": rescue_attempt, "errors": errs[:6]}
                    )
                    if rescue_attempt < _RESCUE_MAX_ATTEMPTS:
                        rescue_pending = {
                            "errors": errs[:8],
                            "message": _build_ticket_feedback_message(errs[:8]),
                        }
                        continue  # retry rescue with feedback
                    # Bad ticket on the last try — degrade to clarify
                    # rather than crash the whole turn into RuntimeError.
                    return _ask_user_fallback(
                        trail,
                        ["I hit an issue while finalizing the dashboard. "
                         "Could you rephrase what you'd like to see "
                         "(metrics, layout, refresh interval)?"],
                    )
                return ReviewOutcome(kind="ticket", ticket=ticket, trail=trail)

            if kind == "extend":
                # M4: Big guy asks to extend the toolkit. Run extend
                # runner; on success degrade to a clarify message so
                # the user can re-issue (the current draft doesn't
                # use the new widget yet).
                extend_raw = rescue.get("extend") or {}
                widget_type = str(extend_raw.get("widget_type") or "").strip()
                if not widget_type:
                    trail.append({"stage": "rescue_extend_invalid",
                                  "error": "missing extend.widget_type"})
                    return _ask_user_fallback(
                        trail,
                        ["I couldn't finalize the dashboard. "
                         "Could you describe what you'd like to see?"],
                    )
                trail.append({"stage": "rescue_extend_start",
                              "widget_type": widget_type})
                parsed = ParsedExtendRequest(
                    widget_type=widget_type,
                    source_agent="big-guy-developer-agent",
                    summary=str(extend_raw.get("rationale") or "")[:256],
                    user_visible_effect=user_intent[:512] if user_intent else "",
                )
                outcome = self._extend_runner.run(
                    parsed,
                    user_intent=user_intent,
                    original_args={"dashboard": current.model_dump(mode="json")},
                )
                if outcome.ok:
                    trail.append({"stage": "rescue_extend_resolved",
                                  "widget_type": widget_type,
                                  "actions": (outcome.report or {}).get(
                                      "actions_taken", [])[:5]})
                    return _ask_user_fallback(
                        trail,
                        [
                            f"I've added a `{widget_type}` widget to the "
                            f"toolkit. Could you re-issue your last request "
                            f"so I can use it?"
                        ],
                    )
                trail.append({"stage": "rescue_extend_failed",
                              "widget_type": widget_type,
                              "error": (outcome.error or "no report")[:240]})
                return _ask_user_fallback(
                    trail,
                    ["I tried to extend the toolkit but couldn't finish. "
                     "Could you describe what you'd like to see with the "
                     "existing widget types (line, stat, gauge, table, alerts)?"],
                )

            # Unknown kind — clarify, don't crash.
            trail.append(
                {"stage": "rescue_unknown_kind", "kind": kind}
            )
            return _ask_user_fallback(
                trail,
                ["I'm not sure how to finalize that dashboard. "
                 "Could you describe what you'd like to see?"],
            )

        # Should be unreachable — the loop body returns or sets pending
        # then continues. Defensive fallback so we never crash.
        return _ask_user_fallback(
            trail,
            ["I couldn't finalize that dashboard. Could you describe "
             "what you'd like to see?"],
        )


def _failed(trail: list[dict[str, Any]], err: str) -> ReviewOutcome:
    trail.append({"stage": "failed", "error": err[:240]})
    return ReviewOutcome(kind="failed", trail=trail, error=err)


def _ask_user_fallback(trail: list[dict[str, Any]], questions: list[str]) -> ReviewOutcome:
    trail.append({"stage": "fallback_ask_user"})
    return ReviewOutcome(kind="clarify", questions=questions, trail=trail)


def _parse_ticket(raw: dict[str, Any]) -> DeveloperTicket:
    return DeveloperTicket.model_validate(raw)


def _flatten_pydantic_errors(exc: Exception) -> list[str]:
    """Turn a ValidationError (or anything resembling one) into a
    short, LLM-readable list of `<loc> <msg>` strings. Falls back to
    `str(exc)` for non-Pydantic exceptions."""
    if isinstance(exc, ValidationError):
        out: list[str] = []
        for e in exc.errors():
            loc = ".".join(str(p) for p in (e.get("loc") or ()))
            msg = str(e.get("msg") or "")
            out.append(f"{loc}: {msg}" if loc else msg)
        return out
    return [str(exc)[:240]]


def _build_patch_feedback_message(errors: list[str]) -> str:
    """Frame a Python-validator rejection so the LLM treats it as
    authoritative and produces a corrected PatchSpec rather than a
    rationalization. Mirrors the framing used by the user-facing
    retry wrapper so the LLM sees a consistent contract."""
    bullets = "\n- ".join(errors) if errors else "(no detail)"
    return (
        "Your previous PatchSpec was rejected by the Python schema "
        "validator. Treat these errors as authoritative — they describe "
        "exact field-level violations that you must fix, not opinions:\n"
        f"- {bullets}\n\n"
        "Return a corrected PatchSpec (same envelope, fixed shape), or "
        "if the requested change genuinely cannot be expressed as a "
        "PatchSpec, return decision=escalate. Do not argue with the "
        "validator and do not invent fields or widget types."
    )


def _build_ticket_feedback_message(errors: list[str]) -> str:
    """Same shape as the patch-feedback message but tailored to a
    DeveloperTicket envelope rejection. This catches the
    title/component/description hallucination that broke the live
    walk-through."""
    bullets = "\n- ".join(errors) if errors else "(no detail)"
    return (
        "Your previous DeveloperTicket was rejected by the Python schema "
        "validator. The required fields are: ticket_id, source_agent, "
        "summary, user_visible_effect, requested_action. Do NOT use "
        "title/component/description/affected_files/suggested_fix — "
        "those keys do not exist on DeveloperTicket. Errors:\n"
        f"- {bullets}\n\n"
        "Return a corrected RescueDecision with the inner ticket using "
        "the right field names. If you do not have a code-level bug to "
        "file, change kind to ask_user with a useful clarifying question."
    )
