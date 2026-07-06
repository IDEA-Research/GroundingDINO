"""Orchestrator — top-level Helper flow.

Pipeline per chat message:

  1. Preload saved dashboards (library context for Helper).
  2. Classify intent via `user_message` operation.
  3. Dispatch to specialist (generate/patch/prometheus).
  4. Validate + save the draft.
  5. Pre-output review loop (if enabled) — render, Helper review,
     escalate to Big guy, etc. See `review_loop.py`.
  6. If review approved → save final, ask user if they want to save
     it to their library.
  7. If review clarified → return clarification questions to user.
  8. If review ticketed → persist the diagnostic, deliver the
     validated draft with a caveat, and schedule a background Big-guy
     auto-fix (`auto_fix.py`). No human interrupt anywhere (LD-1/LD-2).

`developer_fix` runs from this pipeline only via `auto_fix.py`'s gated
background scheduler — never synchronously, never blocking the user.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from ..services.dashboard_store import DashboardStore
from ..services.patch_service import PatchApplicationError, PatchService
from ..services.saved_dashboard_store import SavedDashboardStore
from ..services.spec_validator import SpecValidationError, SpecValidator
from ..specs import DashboardSpec
from . import auto_fix
from .debug_loop import DebugLoop
from .extend_request import parse_extend_ticket
from .extend_runner import ExtendRunner
from .memory import SessionMemory
from .retry_loop import AgentValidationRetryLoop, RetryOutcome
from .review_loop import ReviewLoop, ReviewOutcome, is_enabled as review_enabled
from .runtime import OpenCodeRuntime, RuntimeError_
from .semantic_contracts import (
    AnyValidPatchContract,
    contract_for_generate,
    contract_for_patch,
)


# Session-scoped pending save prompts keyed by session_id.
# When the orchestrator completes a review-approved dashboard, it asks
# the user "Would you like to save this?" and remembers the id until
# the user's next message answers yes/no.
_PENDING_SAVE: dict[str, str] = {}


class Orchestrator:
    def __init__(self) -> None:
        self._runtime = OpenCodeRuntime()
        self._memory = SessionMemory()
        self._validator = SpecValidator()
        self._patch_service = PatchService(self._validator)
        self._store = DashboardStore()
        self._saved = SavedDashboardStore()
        self._debug = DebugLoop(self._runtime)
        # M4: shared runner for the rescue_extend operation. Both the
        # DeveloperTicket branch (specialist agent declined toolkit-
        # missing widget) and the review_loop rescue path use this.
        # Constructed before ReviewLoop so we can inject it.
        self._extend_runner = ExtendRunner(self._runtime)
        self._review = ReviewLoop(
            self._runtime,
            validator=self._validator,
            patch_service=self._patch_service,
            store=self._store,
            extend_runner=self._extend_runner,
        )
        # Validation-retry is applied only to user-originated
        # generate/patch calls (Decision 1b + 2c). Never stacked
        # inside the review loop.
        self._retry = AgentValidationRetryLoop(
            self._runtime,
            validator=self._validator,
            patch_service=self._patch_service,
        )

    # ---------------------------------------------------------------
    def _rebuild_after_extend(self) -> None:
        """C-5 fix: after a successful rescue_extend, the Python class
        objects in widget_spec / dashboard_spec / patch_spec have been
        reloaded by `extend_runner._reload_widget_modules`. But this
        orchestrator's bound `self._validator`, `self._patch_service`,
        `self._retry`, and `self._review` still hold references to the
        OLD class objects (Python's `from X import Y` captures
        binding-at-import-time, not the live module globals).

        To pick up the new enum members (e.g. the freshly-added
        `pie_chart` WidgetType) without restarting the uvicorn worker,
        we re-import the relevant classes after the reload and rebuild
        every cached instance.
        """
        import importlib
        # Re-import the relevant classes from the now-reloaded modules.
        spec_validator_mod = importlib.import_module(
            "app.services.spec_validator"
        )
        patch_service_mod = importlib.import_module(
            "app.services.patch_service"
        )
        SpecValidator = spec_validator_mod.SpecValidator
        PatchService = patch_service_mod.PatchService

        self._validator = SpecValidator()
        self._patch_service = PatchService(self._validator)
        self._retry = AgentValidationRetryLoop(
            self._runtime,
            validator=self._validator,
            patch_service=self._patch_service,
        )
        self._review = ReviewLoop(
            self._runtime,
            validator=self._validator,
            patch_service=self._patch_service,
            store=self._store,
            extend_runner=self._extend_runner,
        )

    # ---------------------------------------------------------------
    def _rebuild_frontend_bundle(self) -> bool:
        """C-7 fix: after a successful rescue_extend the LLM has written
        a new `<NewType>Widget.tsx` file and updated `renderer.tsx` /
        `spec-schema.ts`. But the `next start` process is serving the
        pre-built bundle from `frontend/.next/` — the new .tsx file is
        NOT in that bundle. We need to:

          1. Run `npm run build` to recompile.
          2. Bounce the `next-server` process so it loads the new
             bundle (next start caches the manifest at startup).
          3. Wait for the new next-server to be ready.

        Returns True iff the rebuild + restart fully succeeded.
        Failures are logged but never raised — the caller falls back
        to the existing degrade-to-clarify path so the user gets an
        actionable message instead of a crash.

        Guarded by env `HELPER_DASHBOARD_AUTO_EXTEND_REBUILD_FRONTEND`
        (default true when AUTO_EXTEND is true). Operators who manage
        the frontend with `next dev` (auto-watching) or a different
        process supervisor can disable this with =false.
        """
        import os as _os
        import signal as _signal
        import subprocess
        import time as _time
        from pathlib import Path

        gate = _os.getenv(
            "HELPER_DASHBOARD_AUTO_EXTEND_REBUILD_FRONTEND", "true"
        ).strip().lower()
        if gate in {"0", "false", "no", "off"}:
            print("[orchestrator] frontend rebuild disabled by env; skipping")
            return False

        # Locate project root from this module's path so dev / prod
        # both work.
        root = Path(__file__).resolve().parents[3]
        frontend = root / "frontend"
        if not (frontend / "package.json").exists():
            print(f"[orchestrator] no package.json at {frontend}; skip rebuild")
            return False

        # 1. Rebuild
        backend_env = _os.environ.copy()
        backend_env.setdefault(
            "NEXT_PUBLIC_API_BASE",
            backend_env.get("NEXT_PUBLIC_API_BASE", "http://127.0.0.1:8000"),
        )
        print("[orchestrator] running `npm run build` in frontend/ (this may take 30-60s)")
        try:
            build = subprocess.run(
                ["npm", "run", "build"],
                cwd=str(frontend),
                env=backend_env,
                capture_output=True,
                text=True,
                timeout=180,
                shell=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            print(f"[orchestrator] npm run build failed to launch: {exc}")
            return False
        if build.returncode != 0:
            print(
                f"[orchestrator] npm run build exited rc={build.returncode}; "
                f"stderr tail: {build.stderr[-500:]!r}"
            )
            return False
        print("[orchestrator] npm run build succeeded")

        # Update the sentinel so dev.sh next time doesn't rebuild again.
        try:
            (frontend / ".next").mkdir(exist_ok=True)
            (frontend / ".next" / ".api_base_baked").write_text(
                backend_env.get("NEXT_PUBLIC_API_BASE", "http://127.0.0.1:8000")
            )
        except Exception:  # pragma: no cover
            pass

        # 2. Find and kill the running next-server. `pgrep` is in PATH
        # on every dev/prod host we ship to; falling back to a Python
        # scan of /proc would be more portable but is rarely needed.
        next_pids: list[int] = []
        try:
            pg = subprocess.run(
                ["pgrep", "-f", "next-server"],
                capture_output=True, text=True, timeout=5, shell=False,
            )
            if pg.returncode == 0:
                for line in pg.stdout.splitlines():
                    line = line.strip()
                    if line.isdigit():
                        next_pids.append(int(line))
        except Exception as exc:
            print(f"[orchestrator] pgrep next-server failed: {exc}")

        for pid in next_pids:
            try:
                _os.kill(pid, _signal.SIGTERM)
                print(f"[orchestrator] SIGTERM next-server pid={pid}")
            except ProcessLookupError:
                pass
            except Exception as exc:  # pragma: no cover
                print(f"[orchestrator] kill pid={pid} failed: {exc}")

        # Give the process a moment to release :3050.
        _time.sleep(2)

        # 3. Spawn a fresh next-server. `start_new_session=True` makes
        # it survive the orchestrator's parent (dev.sh) tearing down,
        # which would otherwise SIGTERM us too when its `wait -n` sees
        # the old next-server exit.
        port = _os.getenv("HELPER_DASHBOARD_FRONTEND_PORT", "3050")
        try:
            log_path = _os.getenv(
                "HELPER_DASHBOARD_NEXT_LOG", "/tmp/next.log"
            )
            log_fh = open(log_path, "ab")
            subprocess.Popen(  # noqa: S603 shell=False, fixed argv
                ["npx", "next", "start", "-p", port],
                cwd=str(frontend),
                env=backend_env,
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
                shell=False,
            )
            print(
                f"[orchestrator] spawned fresh next-server on port {port}; "
                f"logs at {log_path}"
            )
        except Exception as exc:
            print(f"[orchestrator] failed to spawn next-server: {exc}")
            return False

        # 4. Wait for it to come up (max 30s).
        import urllib.request
        deadline = _time.time() + 30
        while _time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/", timeout=2
                ) as resp:
                    if 200 <= resp.status < 500:
                        print(
                            f"[orchestrator] next-server ready on :{port} "
                            f"after {(30 - (deadline - _time.time())):.1f}s"
                        )
                        return True
            except Exception:
                pass
            _time.sleep(1)
        print(f"[orchestrator] next-server on :{port} did not become ready in 30s")
        return False

    # ---------------------------------------------------------------
    def handle_user_message(
        self,
        *,
        session_id: str,
        message: str,
        current_dashboard_id: str | None = None,
        progress_cb: Callable[[str, int | None], None] | None = None,
    ) -> dict[str, Any]:
        def _progress(msg: str, pct: int | None = None) -> None:
            if progress_cb is None:
                return
            try:
                progress_cb(msg, pct)
            except Exception:
                pass

        _progress("Preparing context…", 10)
        print(
            "[orchestrator] handle_user_message:start "
            f"session_id={session_id!r} "
            f"current_dashboard_id={current_dashboard_id!r} "
            f"message_len={len(message)}"
        )
        current_dashboard_id = (
            current_dashboard_id
            or self._memory.get_current_dashboard_id(session_id)
        )
        current_dashboard = None
        if current_dashboard_id:
            spec = self._store.load_dashboard(current_dashboard_id)
            if spec is not None:
                # by_alias=True keeps the LLM's context canonical (`from`, not
                # `from_`) so authored patches copy the documented key shape.
                current_dashboard = spec.model_dump(mode="json", by_alias=True)

        # Handle a pending save prompt before asking Helper anything.
        _progress("Checking pending save prompt…", 15)
        print("[orchestrator] checking pending save answer")
        save_reply = self._maybe_handle_save_answer(session_id, message)
        if save_reply is not None:
            _progress("Handled save prompt", 100)
            print("[orchestrator] pending save answer handled; early return")
            return save_reply

        # 1. Classify the user message with library context.
        try:
            _progress("Classifying request…", 25)
            _progress("OpenCode: running user_message…", 30)
            print("[orchestrator] invoking user_message classifier")
            intent = self._runtime.invoke_operation(
                "user_message",
                {
                    "message": message,
                    "session_id": session_id,
                    "current_dashboard_id": current_dashboard_id,
                    "saved_dashboards": [
                        s.model_dump(mode="json")
                        for s in self._saved.summaries()
                    ],
                },
            )
            print(
                "[orchestrator] classifier result "
                f"type={intent.get('type')!r} "
                f"runtime_used={intent.get('runtime_used')!r}"
            )
            _progress(
                "OpenCode: user_message completed "
                f"({intent.get('runtime_used') or 'unknown runtime'})",
                40,
            )
            _progress(f"Classified as {intent.get('type') or 'Unknown'}", 40)
        except RuntimeError_ as exc:
            _progress(f"OpenCode error during user_message: {exc}", 100)
            print(f"[orchestrator] classifier runtime error: {exc}")
            return self._runtime_error_response(str(exc))

        runtime_meta = _runtime_meta(intent)

        # 2. Dispatch to specialist — wrapped in the validation-retry
        # loop for generate/patch operations, direct passthrough for
        # the rest.
        try:
            _progress("Running specialist agent…", 60)
            _progress("OpenCode: running specialist operation…", 65)
            print("[orchestrator] dispatch_with_retry:start")
            result, retry_outcome = self._dispatch_with_retry(
                intent, current_dashboard=current_dashboard,
            )
            result_type = (
                result.get("type") if isinstance(result, dict) else type(result).__name__
            )
            print(
                "[orchestrator] dispatch_with_retry:done "
                f"result_type={result_type!r} "
                f"retry_used={retry_outcome.attempts_used if retry_outcome else None}"
            )
            _progress(
                "OpenCode: specialist operation completed "
                f"({result.get('runtime_used') if isinstance(result, dict) else 'unknown runtime'})",
                80,
            )
            _progress("Specialist step completed", 80)
        except RuntimeError_ as exc:
            _progress(f"OpenCode error during specialist step: {exc}", 100)
            print(f"[orchestrator] dispatch runtime error: {exc}")
            return self._runtime_error_response(str(exc))
        except Exception as exc:
            print(
                "[orchestrator] dispatch unexpected error: "
                f"{type(exc).__name__}: {exc}"
            )
            return self._runtime_error_response(
                f"unexpected dispatch error: {type(exc).__name__}: {exc}"
            )

        if isinstance(result, dict) and result is not intent:
            specialist_meta = _runtime_meta(result)
            if specialist_meta["runtime_used"] is not None:
                runtime_meta = specialist_meta

        if not isinstance(result, dict):
            print(
                "[orchestrator] non-dict result from dispatch; "
                f"type={type(result).__name__}"
            )
            ticket = self._debug.from_validation_errors(
                [f"dispatch returned non-dict result: {type(result).__name__}"],
                source="orchestrator",
            )
            self._save_ticket_and_schedule_fix(ticket, user_intent=message)
            return self._response(
                user_reply=_safe_user_reply_on_error(),
                intent_type="RuntimeError",
                warnings=["internal error — diagnostic logged"],
                runtime_meta=runtime_meta,
            )

        rtype = result.get("type")
        print(f"[orchestrator] routing result type={rtype!r}")
        _progress(f"Finalizing {rtype or 'response'}…", 90)

        # ----- Auto-extend toolkit if specialist agent declined --------
        # The widget toolkit is a cache of pre-built widgets, not a fence
        # on what can be built. If the specialist agent emitted a
        # DeveloperTicket for a missing widget type, we ask Big guy to
        # extend the toolkit and re-dispatch the same intent ONCE.
        # Safety still applies — ExtendRunner.run enforces denylist,
        # prompt-injection check, daily quota, and audit log. Extend
        # happens at most once per chat turn; if the second attempt
        # also fails, fall through to normal handling.
        extend_attempt: dict[str, str] | None = None
        if rtype == "DeveloperTicket":
            ext = parse_extend_ticket(result)
            if ext is not None:
                print(f"[orchestrator] extend ticket detected widget_type={ext.widget_type!r}")
                _progress(
                    f"Extending toolkit with {ext.widget_type}…", 70,
                )
                outcome = self._extend_runner.run(
                    ext, user_intent=message, original_args=intent,
                )
                if outcome.ok:
                    print(
                        f"[orchestrator] extend succeeded for "
                        f"{ext.widget_type!r}; re-dispatching"
                    )
                    _progress("Toolkit extended; rebuilding dashboard…", 80)
                    # C-5: only meaningful to rebuild instances if
                    # extend_runner actually reloaded the underlying
                    # modules. That reload is itself opt-in via
                    # HELPER_DASHBOARD_AUTO_EXTEND_RELOAD_MODULES
                    # (default off because importlib.reload creates
                    # two class objects in memory and breaks
                    # isinstance checks elsewhere). When the flag is
                    # off, the running validator still has the old
                    # WidgetType — backend restart is the right fix
                    # and is documented in docs/SECURITY_BOUNDARIES.md.
                    import os as _os
                    if _os.getenv(
                        "HELPER_DASHBOARD_AUTO_EXTEND_RELOAD_MODULES", ""
                    ).strip().lower() in {"1", "true", "yes", "on"}:
                        try:
                            self._rebuild_after_extend()
                            print(
                                "[orchestrator] rebuilt validator/patch/review "
                                "with reloaded widget modules"
                            )
                        except Exception as exc:
                            print(
                                f"[orchestrator] _rebuild_after_extend failed: "
                                f"{type(exc).__name__}: {exc} "
                                f"(proceeding; backend may need manual restart)"
                            )
                    # C-7: rebuild frontend bundle so the newly-written
                    # <NewType>Widget.tsx is actually served. Blocks
                    # for up to ~60s (npm run build is the slow path).
                    # Defaults on; toggle off with env when running
                    # `next dev` (which auto-watches files instead).
                    _progress("Rebuilding frontend bundle…", 85)
                    try:
                        ok = self._rebuild_frontend_bundle()
                        if ok:
                            print("[orchestrator] frontend bundle rebuilt + next-server bounced")
                        else:
                            print(
                                "[orchestrator] frontend rebuild incomplete; "
                                "the new widget may not render in the browser"
                            )
                    except Exception as exc:
                        print(
                            f"[orchestrator] _rebuild_frontend_bundle raised: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    try:
                        result, retry_outcome = self._dispatch_with_retry(
                            intent, current_dashboard=current_dashboard,
                        )
                    except (RuntimeError_, Exception) as exc:
                        print(
                            f"[orchestrator] re-dispatch after extend failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        rtype = (
                            result.get("type") if isinstance(result, dict)
                            else type(result).__name__
                        )
                        new_meta = (
                            _runtime_meta(result) if isinstance(result, dict)
                            else None
                        )
                        if new_meta and new_meta.get("runtime_used"):
                            runtime_meta = new_meta
                        print(
                            f"[orchestrator] re-dispatch result type={rtype!r}"
                        )
                else:
                    print(
                        f"[orchestrator] extend failed for "
                        f"{ext.widget_type!r}: {outcome.error or 'no report'}"
                    )
                    extend_attempt = {
                        "widget_type": ext.widget_type,
                        "error": outcome.error or "no report",
                    }

        # ----- non-spec types pass straight through --------------------
        if rtype == "UserResponse":
            print("[orchestrator] branch=UserResponse")
            _progress("Completed", 100)
            return self._response(
                user_reply=str(result.get("message", "")),
                intent_type="UserResponse",
                runtime_meta=runtime_meta,
            )

        if rtype == "ClarificationRequest":
            print("[orchestrator] branch=ClarificationRequest")
            _progress("Completed", 100)
            qs = [
                str(q)[:512]
                for q in (result.get("questions") or [])
                if isinstance(q, str)
            ] or ["Could you clarify what dashboard you'd like?"]
            return self._response(
                user_reply=str(result.get("message_to_user", "")) or qs[0],
                intent_type="ClarificationRequest",
                clarification=qs,
                runtime_meta=runtime_meta,
            )

        if rtype == "PrometheusQueryReport":
            print("[orchestrator] branch=PrometheusQueryReport")
            _progress("Completed", 100)
            return self._response(
                user_reply=_summarize_prometheus(result.get("report", {})),
                intent_type="PrometheusQueryReport",
                runtime_meta=runtime_meta,
            )

        if rtype == "SystemAlertRule":
            print("[orchestrator] branch=SystemAlertRule")
            _progress("Validating alert rule…", 90)
            return self._handle_system_alert_rule(
                result, runtime_meta=runtime_meta,
            )

        if rtype == "DeveloperTicket":
            print("[orchestrator] branch=DeveloperTicket")
            _progress("Completed with ticket", 100)
            # source_agent inferred from the ticket itself when available
            # (specialist agents stamp their own name); fall back to the
            # classifier when the ticket came from helper-chat-agent.
            fallback_source = (
                "dashboard-spec-agent"
                if intent.get("type") == "DashboardIntent"
                else "patch-agent"
                if intent.get("type") == "PatchIntent"
                else "helper-chat-agent"
            )
            ticket = _ticket_from_raw(result, source_agent=fallback_source)
            self._save_ticket_and_schedule_fix(ticket, user_intent=message)

            # Honest fallback when we land here after extend was already
            # attempted: tell the user what we tried, why it didn't work,
            # and suggest a cached alternative. The toolkit is a cache,
            # not a fence — so we never tell the user "X isn't supported";
            # we tell them what actually went wrong.
            ext = parse_extend_ticket(result)
            if extend_attempt is not None:
                err = extend_attempt["error"]
                # Distinguish safety refusal (gate said no) from build
                # failure (LLM tried, didn't pass). The gate error string
                # starts with "gate refused (...)"; everything else is a
                # real failure.
                if err.startswith("gate refused"):
                    user_reply = (
                        f"I can't add a `{extend_attempt['widget_type']}` "
                        f"widget right now ({err}). Try `bar_chart` or "
                        f"`table` for the same data."
                    )
                else:
                    user_reply = (
                        f"I tried to add a `{extend_attempt['widget_type']}` "
                        f"widget but the build didn't pass — try `bar_chart` "
                        f"or `table` for the same data."
                    )
            elif ext is not None:
                # Extend wasn't attempted (e.g. ticket parsed but extend
                # branch didn't run). Still avoid "not in the toolkit" —
                # frame it as a build that didn't happen.
                user_reply = (
                    f"I couldn't build the `{ext.widget_type}` widget on "
                    f"this turn — try `bar_chart` or `table` for the same "
                    f"data, or ask again."
                )
            else:
                user_reply = (
                    "That isn't something I could build on this turn — "
                    "I've logged a diagnostic and will keep improving. "
                    "Try describing it differently or ask again."
                )
            return self._response(
                user_reply=user_reply,
                intent_type="DeveloperTicket",
                runtime_meta=runtime_meta,
            )

        # ----- DashboardSpec / PatchSpec go through review loop --------
        if rtype == "DashboardSpec":
            print("[orchestrator] branch=DashboardSpec")
            return self._handle_dashboard_spec(
                session_id=session_id,
                message=message,
                intent=intent,
                result=result,
                runtime_meta=runtime_meta,
                retry_outcome=retry_outcome,
            )

        if rtype == "PatchSpec":
            print("[orchestrator] branch=PatchSpec")
            return self._handle_patch_spec(
                session_id=session_id,
                message=message,
                intent=intent,
                result=result,
                current_dashboard=current_dashboard,
                runtime_meta=runtime_meta,
                retry_outcome=retry_outcome,
            )

        # Unknown — output contract should have caught this already.
        print(f"[orchestrator] branch=Unknown type={rtype!r}; creating ticket")
        ticket = self._debug.from_validation_errors(
            [f"unknown result type: {rtype!r}"], source="orchestrator",
        )
        self._save_ticket_and_schedule_fix(ticket, user_intent=message)
        return self._response(
            user_reply=_safe_user_reply_on_error(),
            intent_type=rtype or "Unknown",
            warnings=["internal error — diagnostic logged"],
            runtime_meta=runtime_meta,
        )

    # ---------------------------------------------------------------
    def _handle_dashboard_spec(
        self,
        *,
        session_id: str,
        message: str,
        intent: dict[str, Any],
        result: dict[str, Any],
        runtime_meta: dict[str, Any],
        retry_outcome: RetryOutcome | None = None,
    ) -> dict[str, Any]:
        # If the retry loop ran and failed, `result` will be the last
        # (rejected) attempt's output. File a ticket with the attempt
        # log and return a safe error to the user.
        if retry_outcome and retry_outcome.accepted is None:
            self._file_retry_ticket(retry_outcome, source="dashboard-spec-agent")
            return self._response(
                user_reply=_safe_user_reply_on_error(),
                intent_type="DashboardSpec",
                warnings=[
                    f"generation rejected after "
                    f"{retry_outcome.attempts_used} attempts — diagnostic logged"
                ],
                runtime_meta=runtime_meta,
            )

        try:
            spec = self._validator.validate_dashboard(result.get("spec", {}))
        except SpecValidationError as exc:
            ticket = self._debug.from_validation_errors(
                exc.errors, source="dashboard-spec-agent",
            )
            self._save_ticket_and_schedule_fix(ticket, user_intent=message)
            return self._response(
                user_reply=_safe_user_reply_on_error(),
                intent_type="DashboardSpec",
                warnings=["internal validation error — diagnostic logged"],
                runtime_meta=runtime_meta,
            )

        final_spec, outcome = self._review_or_passthrough(
            spec, user_intent=message
        )
        warnings = _retry_warnings(retry_outcome)
        return self._finalize(
            session_id=session_id,
            message=message,
            intent=intent,
            spec=final_spec,
            outcome=outcome,
            action="created",
            runtime_meta=runtime_meta,
            extra_warnings=warnings,
        )

    def _handle_patch_spec(
        self,
        *,
        session_id: str,
        message: str,
        intent: dict[str, Any],
        result: dict[str, Any],
        current_dashboard: dict[str, Any] | None,
        runtime_meta: dict[str, Any],
        retry_outcome: RetryOutcome | None = None,
    ) -> dict[str, Any]:
        if current_dashboard is None:
            return self._response(
                user_reply="I don't have a dashboard loaded to modify yet.",
                intent_type="PatchSpec",
                runtime_meta=runtime_meta,
            )

        if retry_outcome and retry_outcome.accepted is None:
            self._file_retry_ticket(retry_outcome, source="patch-agent")
            return self._response(
                user_reply=(
                    "I couldn't apply that change cleanly — I've logged a "
                    "diagnostic. Try rephrasing the change."
                ),
                intent_type="PatchSpec",
                warnings=[
                    f"patch rejected after {retry_outcome.attempts_used} "
                    "attempts — diagnostic logged"
                ],
                runtime_meta=runtime_meta,
            )

        try:
            patch = self._validator.validate_patch(result.get("spec", {}))
            current_spec = self._validator.validate_dashboard(current_dashboard)
            new_spec = self._patch_service.apply(current_spec, patch)
        except (SpecValidationError, PatchApplicationError) as exc:
            errors = (
                exc.errors if isinstance(exc, SpecValidationError)
                else [str(exc)]
            )
            ticket = self._debug.from_validation_errors(
                errors, source="patch-agent",
            )
            self._save_ticket_and_schedule_fix(ticket, user_intent=message)
            return self._response(
                user_reply=(
                    "I couldn't apply that change cleanly — I've logged a "
                    "diagnostic. Try rephrasing the change."
                ),
                intent_type="PatchSpec",
                warnings=["patch failed validation"],
                runtime_meta=runtime_meta,
            )

        final_spec, outcome = self._review_or_passthrough(
            new_spec, user_intent=message,
        )
        warnings = _retry_warnings(retry_outcome)
        return self._finalize(
            session_id=session_id,
            message=message,
            intent=intent,
            spec=final_spec,
            outcome=outcome,
            action="patched",
            # by_alias=True: patch ops can embed widgets whose decision_flow
            # edges must serialize as `from` (canonical), matching the
            # dashboard dump in _finalize.
            patch=patch.model_dump(mode="json", by_alias=True),
            runtime_meta=runtime_meta,
            extra_warnings=warnings,
        )

    # ---------------------------------------------------------------
    def _handle_system_alert_rule(
        self, result: dict[str, Any], *, runtime_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate + install a Helper-authored system alert rule.

        The Python validator is authoritative (the spec forces SHADOW and
        the curated metric catalog — see specs/system_rule_spec.py). On
        success the rule is persisted and hot-added to the running
        evaluator; the reply is honest about shadow mode and about the
        decision-support disclaimer."""
        from pydantic import ValidationError

        from ..services import system_rules
        from ..specs.system_rule_spec import SystemAlertRuleSpec

        raw = result.get("spec") or {}
        try:
            rule = SystemAlertRuleSpec.model_validate(raw)
        except ValidationError as exc:
            errs = "; ".join(
                f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg')}"
                for e in exc.errors()[:4]
            )
            return self._response(
                user_reply=(
                    "I couldn't turn that into a valid alert rule "
                    f"({errs}). Try naming the metric (CPU, disk, memory, "
                    "load), a threshold, and a duration — e.g. 'alert me "
                    "when CPU is above 90% for 5 minutes'."
                ),
                intent_type="AlertRuleIntent",
                warnings=["alert rule failed validation"],
                runtime_meta=runtime_meta,
            )

        system_rules.get_service().add_rule(rule)
        return self._response(
            user_reply=(
                f"Alert rule created: {rule.human_summary()}.\n\n"
                "It starts in SHADOW mode — breaches are recorded as "
                "would-fire events (visible under /api/system-rules and "
                "its alerts feed) but nothing pages. If the metric source "
                "is unreachable or stale the rule reports SIGNAL_LOST "
                "instead of guessing. This is decision support, not a "
                "diagnosis."
            ),
            intent_type="SystemAlertRule",
            runtime_meta=runtime_meta,
        )

    # ---------------------------------------------------------------
    def _review_or_passthrough(
        self, draft: DashboardSpec, *, user_intent: str,
    ) -> tuple[DashboardSpec | None, ReviewOutcome | None]:
        if not review_enabled():
            return draft, None
        outcome = self._review.run(draft, user_intent=user_intent)
        if outcome.kind == "approved":
            return outcome.dashboard, outcome
        if outcome.kind == "ticket":
            # LD-1/LD-2: a rescue ticket is a diagnostic record, not a
            # hand-off to humans. The draft passed Python validation —
            # deliver it; _finalize adds the caveat + logs the ticket.
            return outcome.dashboard, outcome
        return None, outcome

    def _finalize(
        self,
        *,
        session_id: str,
        message: str,
        intent: dict[str, Any],
        spec: DashboardSpec | None,
        outcome: ReviewOutcome | None,
        action: str,
        patch: dict[str, Any] | None = None,
        runtime_meta: dict[str, Any],
        extra_warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        # Save prompt — ask the user whether to save, after they've
        # successfully produced a dashboard through review.
        pending_save_question = None
        if spec is not None:
            self._store.save_dashboard(spec)
            self._memory.set_current_dashboard_id(session_id, spec.dashboard_id)
            self._memory.record(
                session_id, action=action,
                dashboard_id=spec.dashboard_id, summary=message,
            )
            _PENDING_SAVE[session_id] = spec.dashboard_id
            pending_save_question = (
                f"Would you like to save this dashboard to your library "
                f"for later? Reply with a name (e.g. 'prod CPU'), or 'no' "
                f"to skip."
            )

        review_trail = outcome.trail if outcome else []

        if outcome is not None and outcome.kind == "clarify":
            return self._response(
                user_reply=(
                    outcome.questions[0]
                    if outcome.questions
                    else "Could you clarify what you'd like?"
                ),
                intent_type="ClarificationRequest",
                clarification=outcome.questions,
                review_trail=review_trail,
                runtime_meta=runtime_meta,
            )

        # LD-1/LD-2: no ticket-and-wait. The ticket is persisted purely
        # as a diagnostic record; the validated dashboard (saved above
        # when present) is delivered with an honest caveat. No human is
        # notified and nothing blocks on one.
        render_caveat: str | None = None
        if outcome is not None and outcome.kind == "ticket":
            fix_scheduled = False
            if outcome.ticket is not None:
                fix_scheduled = self._save_ticket_and_schedule_fix(
                    outcome.ticket, user_intent=message,
                )
            if spec is None:
                # Defensive: a ticket outcome without a deliverable
                # draft. Stay non-blocking — invite a retry, and be
                # truthful that diagnosis is automated.
                return self._response(
                    user_reply=(
                        "I couldn't finish building that dashboard this "
                        "turn — I've logged a diagnostic for the automated "
                        "pipeline. Please ask again, or describe what you "
                        "need differently."
                    ),
                    intent_type="UserResponse",
                    warnings=["render check failed — diagnostic logged"],
                    review_trail=review_trail,
                    runtime_meta=runtime_meta,
                )
            render_caveat = (
                "Note: my automated render check couldn't confirm every "
                "widget drew correctly, so I've logged a diagnostic"
                + (" and started an automatic background fix"
                   if fix_scheduled else "")
                + ". If anything looks empty, refresh or ask me to "
                "rebuild it."
            )

        if outcome is not None and outcome.kind == "failed":
            return self._response(
                user_reply=_safe_user_reply_on_error(),
                intent_type="RuntimeError",
                warnings=[outcome.error or "review failed"],
                review_trail=review_trail,
                runtime_meta=runtime_meta,
            )

        # Approved (or passthrough when review disabled, or ticket
        # outcome delivered-with-caveat).
        base_reply = (
            intent.get("message_to_user")
            or (
                f"Here is your '{spec.title}' dashboard."
                if action == "created" and spec
                else "Updated."
            )
        )
        reply = base_reply
        if render_caveat:
            reply = f"{reply}\n\n{render_caveat}"
        if pending_save_question:
            reply = f"{reply}\n\n{pending_save_question}"

        return self._response(
            user_reply=reply,
            intent_type="DashboardSpec" if action == "created" else "PatchSpec",
            # by_alias=True so decision_flow edges serialize their `from` key —
            # the frontend renders this dump directly (page.tsx setSpec) and
            # silently drops every branch edge if it sees `from_` instead.
            dashboard=spec.model_dump(mode="json", by_alias=True) if spec else None,
            patch=patch,
            review_trail=review_trail,
            save_prompt=(spec.dashboard_id if spec else None),
            runtime_meta=runtime_meta,
            warnings=(extra_warnings or []) + (
                ["render check unconfirmed — diagnostic logged"]
                if render_caveat else []
            ),
        )

    # ---------------------------------------------------------------
    def _maybe_handle_save_answer(
        self, session_id: str, message: str
    ) -> dict[str, Any] | None:
        """If a save prompt is pending and the message looks like an
        answer (yes/no/name), act on it and return a response;
        otherwise drop the prompt and let the normal flow proceed."""
        pending = _PENDING_SAVE.get(session_id)
        if not pending:
            return None
        clean = message.strip()
        low = clean.lower()
        if low in {"no", "nope", "n", "skip", "not now", "cancel", "not save"}:
            _PENDING_SAVE.pop(session_id, None)
            return self._response(
                user_reply="Okay — not saving.",
                intent_type="UserResponse",
                runtime_meta={"runtime_used": None, "fallback_reason": None},
            )
        # If the message looks like a short name (<= 64 chars, no "show me"
        # / "add a" triggers), treat it as the library name.
        if (
            1 <= len(clean) <= 64
            and not any(k in low for k in [
                "show me", "add a ", "remove", "delete", "rename",
                "change ", "give me ", "build ", "make ", "dashboard",
                "chart", "graph", "metric", "query",
            ])
            and ("\n" not in clean)
        ):
            spec = self._store.load_dashboard(pending)
            _PENDING_SAVE.pop(session_id, None)
            if spec is None:
                return self._response(
                    user_reply=(
                        "I couldn't find the dashboard to save. "
                        "Let's start fresh."
                    ),
                    intent_type="UserResponse",
                    runtime_meta={"runtime_used": None, "fallback_reason": None},
                )
            entry = self._saved.save(spec, name=clean)
            return self._response(
                user_reply=(
                    f"Saved as {entry.name!r}. Ask for it again anytime."
                ),
                intent_type="UserResponse",
                runtime_meta={"runtime_used": None, "fallback_reason": None},
            )
        # Otherwise, drop the prompt and continue normally.
        _PENDING_SAVE.pop(session_id, None)
        return None

    # ---------------------------------------------------------------
    def _dispatch(
        self,
        intent: dict[str, Any],
        *,
        current_dashboard: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Passthrough dispatch WITHOUT validation retry. Kept for
        back-compat with the review loop and any external caller."""
        t = intent.get("type")
        if t == "DashboardIntent":
            return self._runtime.invoke_operation(
                "generate_dashboard",
                {"intent": intent, "previous_dashboard": current_dashboard},
            )
        if t == "PatchIntent":
            if current_dashboard is None:
                return {
                    "type": "UserResponse",
                    "message": "I don't have a dashboard loaded yet — create one first.",
                }
            return self._runtime.invoke_operation(
                "patch_dashboard",
                {"intent": intent, "dashboard": current_dashboard},
            )
        if t == "PrometheusIntent":
            return self._runtime.invoke_operation(
                "prometheus_query",
                {"question": intent.get("question", "")},
            )
        if t == "AlertRuleIntent":
            return self._runtime.invoke_operation(
                "author_alert_rule", {"intent": intent},
            )
        return intent

    def _dispatch_with_retry(
        self,
        intent: dict[str, Any],
        *,
        current_dashboard: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], RetryOutcome | None]:
        """Dispatch with validation-retry for generate/patch; straight
        passthrough for everything else."""
        t = intent.get("type")
        if t == "DashboardIntent":
            requirements = (intent.get("requirements") or {})
            contract = contract_for_generate(requirements)
            outcome = self._retry.run(
                "generate_dashboard",
                {"intent": intent, "previous_dashboard": current_dashboard},
                contract=contract,
            )
            # Prefer the accepted output; if none, return the last
            # attempt's output if any (so the orchestrator can still
            # surface it as a warning) — but handler will see None
            # via retry_outcome and short-circuit.
            if outcome.accepted is not None:
                return outcome.accepted, outcome
            # No accepted output: synthesize a placeholder with type
            # DashboardSpec + empty spec so the downstream type check
            # routes through _handle_dashboard_spec's failure path.
            return {"type": "DashboardSpec", "spec": {},
                    "runtime_used": None, "fallback_reason": None}, outcome

        if t == "PatchIntent":
            if current_dashboard is None:
                return {
                    "type": "UserResponse",
                    "message": "I don't have a dashboard loaded yet — create one first.",
                }, None
            requested = intent.get("requested_changes") or []
            contract = contract_for_patch(requested, current_dashboard)
            outcome = self._retry.run(
                "patch_dashboard",
                {"intent": intent, "dashboard": current_dashboard},
                contract=contract,
            )
            if outcome.accepted is not None:
                return outcome.accepted, outcome
            return {"type": "PatchSpec", "spec": {},
                    "runtime_used": None, "fallback_reason": None}, outcome

        # Non-retry operations passthrough.
        return self._dispatch(intent, current_dashboard=current_dashboard), None

    # ---------------------------------------------------------------
    @staticmethod
    def _response(
        *,
        user_reply: str,
        intent_type: str,
        dashboard: dict[str, Any] | None = None,
        patch: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        runtime_meta: dict[str, Any] | None = None,
        review_trail: list[dict[str, Any]] | None = None,
        clarification: list[str] | None = None,
        save_prompt: str | None = None,
    ) -> dict[str, Any]:
        meta = runtime_meta or {"runtime_used": None, "fallback_reason": None}
        return {
            "user_reply": user_reply,
            "intent_type": intent_type,
            "dashboard": dashboard,
            "patch": patch,
            "warnings": warnings or [],
            "runtime_used": meta.get("runtime_used"),
            "fallback_reason": meta.get("fallback_reason"),
            "review_trail": review_trail or [],
            "clarification_questions": clarification,
            "save_prompt_for": save_prompt,
        }

    def _save_ticket_and_schedule_fix(
        self, ticket, *, user_intent: str = "",
    ) -> bool:
        """Persist a diagnostic ticket and hand it to the background
        Big-guy auto-fix pipeline (LD-1/LD-2: automated diagnosis AND
        automated repair; no human interrupt). Returns True when a fix
        run was actually scheduled — gates may refuse. Never raises."""
        self._store.save_ticket(ticket)
        try:
            thread = auto_fix.schedule_auto_fix(
                ticket, runtime=self._runtime, store=self._store,
                user_intent=user_intent,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[orchestrator] auto-fix scheduling failed: {exc}")
            return False
        if thread is not None:
            print(
                f"[orchestrator] auto-fix scheduled for ticket "
                f"{ticket.ticket_id!r}"
            )
            return True
        return False

    def _runtime_error_response(self, msg: str) -> dict[str, Any]:
        ticket = self._debug.from_validation_errors([msg], source="runtime")
        self._save_ticket_and_schedule_fix(ticket)
        return self._response(
            user_reply=_safe_user_reply_on_error(),
            intent_type="RuntimeError",
            warnings=["runtime error — diagnostic logged"],
        )

    def _file_retry_ticket(
        self, outcome: RetryOutcome, *, source: str,
    ) -> None:
        errors_blob: list[str] = []
        for a in outcome.attempts:
            errors_blob.append(
                f"attempt {a.attempt}/{outcome.attempts_used} "
                f"runtime={a.runtime_used or 'n/a'} "
                f"type={a.output_type or 'n/a'}: "
                + "; ".join(a.errors[:3])
            )
        ticket = self._debug.from_validation_errors(
            errors_blob, source=source,
        )
        self._save_ticket_and_schedule_fix(ticket)


def _runtime_meta(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "runtime_used": result.get("runtime_used"),
        "fallback_reason": result.get("fallback_reason"),
    }


def _retry_warnings(outcome: RetryOutcome | None) -> list[str]:
    """When the validation-retry loop needed more than one attempt,
    surface it as a warning on the response so the UI and tests can
    see the quality drift."""
    if outcome is None or outcome.attempts_used <= 1:
        return []
    max_attempts = outcome.attempts_used
    return [
        f"validation retry used {max_attempts} attempts "
        f"(contract={outcome.contract_name})"
    ]


def _safe_user_reply_on_error() -> str:
    return (
        "Something went wrong while putting that together. "
        "I've logged a diagnostic — please try asking again."
    )


def _summarize_prometheus(report: dict[str, Any]) -> str:
    suggestions = report.get("metric_suggestions") or []
    if not suggestions:
        return report.get("notes") or "I couldn't find useful metric suggestions."
    names = ", ".join(s.get("metric", "") for s in suggestions[:5])
    return f"Some metrics that might help: {names}."


def _ticket_from_raw(raw: dict[str, Any], *, source_agent: str):
    from ..specs.developer_ticket import DeveloperTicket, TicketSeverity

    summary = raw.get("summary") or raw.get("title") or "helper-reported issue"
    user_visible_effect = (
        raw.get("user_visible_effect")
        or raw.get("description")
        or "user could not complete request"
    )
    requested_action = (
        raw.get("requested_action")
        or raw.get("requested_fix")
        or "investigate"
    )

    technical_evidence = raw.get("technical_evidence")
    if not technical_evidence:
        evidence: dict[str, Any] = {}
        if raw.get("reproduction_steps"):
            evidence["reproduction_steps"] = raw.get("reproduction_steps")
        if raw.get("affected_files"):
            evidence["affected_files"] = raw.get("affected_files")
        if raw.get("console_errors"):
            evidence["console_errors"] = raw.get("console_errors")
        technical_evidence = evidence or {}

    ticket_id = f"tkt-{uuid.uuid4().hex[:8]}"
    severity = raw.get("severity") or "medium"
    try:
        severity_enum = TicketSeverity(severity)
    except ValueError:
        severity_enum = TicketSeverity.medium
    return DeveloperTicket(
        ticket_id=ticket_id,
        source_agent=raw.get("source_agent") or source_agent,
        severity=severity_enum,
        summary=str(summary)[:256],
        user_visible_effect=str(user_visible_effect)[:512],
        technical_evidence=technical_evidence,
        requested_action=str(requested_action)[:1024],
        safety_notes=str(raw.get("safety_notes") or "")[:1024],
    )
