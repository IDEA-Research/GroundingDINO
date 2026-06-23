"""Agent validation retry loop.

Wraps `OpenCodeRuntime.invoke_operation` with a bounded
validate-then-retry-with-feedback loop. Used only by the orchestrator
for user-originated `generate_dashboard` / `patch_dashboard` (per
Decision 1b + 2c — NOT inside the review loop, not inside runtime).

Key invariants:
- Does not weaken any Pydantic schema — schema errors become retry
  feedback, they never pass through.
- Does not widen the operation allow-list.
- Does not print secrets. Attempt logs never serialize env values.
- Short-circuits deterministic providers (mock / mock_fallback /
  heuristic) after one failed attempt.
- `HELPER_AGENT_MAX_ATTEMPTS` clamped to [1, 5], default 3.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..services.patch_service import PatchApplicationError, PatchService
from ..services.spec_validator import SpecValidationError, SpecValidator
from ..specs import DashboardSpec, PatchSpec
from .runtime import EXPECTED_OUTPUT_TYPES, OpenCodeRuntime, RuntimeError_
from .semantic_contracts import ContractCheck, SemanticContract


_MIN_ATTEMPTS = 1
_MAX_ATTEMPTS_CAP = 5
_DEFAULT_MAX_ATTEMPTS = 3

_RETRY_LOGS_DIR = (
    Path(__file__).resolve().parent.parent / "storage" / "retry_logs"
)


def resolve_max_attempts() -> int:
    raw = os.getenv("HELPER_AGENT_MAX_ATTEMPTS")
    if raw is None or raw.strip() == "":
        return _DEFAULT_MAX_ATTEMPTS
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_MAX_ATTEMPTS
    if n < _MIN_ATTEMPTS:
        return _MIN_ATTEMPTS
    if n > _MAX_ATTEMPTS_CAP:
        return _MAX_ATTEMPTS_CAP
    return n


@dataclass
class AttemptLog:
    attempt: int
    elapsed_s: float
    output_type: str | None
    op_summary: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    accepted: bool = False
    runtime_used: str | None = None
    fallback_reason: str | None = None


@dataclass
class RetryOutcome:
    accepted: dict | None
    attempts: list[AttemptLog]
    contract_name: str
    operation: str

    @property
    def attempts_used(self) -> int:
        return len(self.attempts)


# Keys we'll never serialize into retry logs.
_SECRET_PATTERNS = (
    re.compile(r"(?i)api[_-]?key"),
    re.compile(r"(?i)token"),
    re.compile(r"(?i)secret"),
    re.compile(r"(?i)password"),
    re.compile(r"sk-[a-z0-9-]+", re.IGNORECASE),
)


def _contains_secret(value: str) -> bool:
    return any(p.search(value) for p in _SECRET_PATTERNS)


def _scrub(text: str | None) -> str | None:
    """Replace anything that looks like a secret with a masked
    placeholder. Used only on log payloads."""
    if not text:
        return text
    if _contains_secret(text):
        return "<redacted>"
    return text


class AgentValidationRetryLoop:
    """Bounded validate-then-retry-with-feedback wrapper.

    The runtime call itself is unchanged — this class composes
    semantic-contract checking, structured feedback injection, and
    persistence of attempt logs.
    """

    def __init__(
        self,
        runtime: OpenCodeRuntime,
        *,
        validator: SpecValidator | None = None,
        patch_service: PatchService | None = None,
        max_attempts: int | None = None,
        persist_logs: bool | None = None,
    ) -> None:
        self._runtime = runtime
        self._validator = validator or SpecValidator()
        self._patch_service = patch_service or PatchService(self._validator)
        self._max_attempts = (
            max(
                _MIN_ATTEMPTS,
                min(_MAX_ATTEMPTS_CAP, int(max_attempts)),
            ) if max_attempts is not None
            else resolve_max_attempts()
        )
        if persist_logs is None:
            persist_logs = os.getenv(
                "HELPER_DASHBOARD_PERSIST_RETRY_LOGS", "0"
            ) not in ("", "0", "false", "False", "no", "off")
        self._persist_logs = persist_logs

    # --------------------------------------------------------------
    def run(
        self,
        operation: str,
        args: dict[str, Any],
        *,
        contract: SemanticContract,
    ) -> RetryOutcome:
        attempts: list[AttemptLog] = []
        accepted: dict | None = None
        # We mutate a copy so the caller's args dict isn't polluted.
        call_args = dict(args)

        for n in range(1, self._max_attempts + 1):
            if n > 1:
                call_args["_retry_attempt"] = n
                call_args["_prior_errors"] = attempts[-1].errors[:6]
                call_args["_prior_feedback_message"] = \
                    contract.feedback_message(attempts[-1].errors, args)

            t0 = time.monotonic()
            output_type: str | None = None
            runtime_used: str | None = None
            fallback_reason: str | None = None
            op_summary: list[str] = []
            errors: list[str] = []

            try:
                result = self._runtime.invoke_operation(
                    operation, call_args, developer=False,
                )
                output_type = result.get("type")
                runtime_used = result.get("runtime_used")
                fallback_reason = result.get("fallback_reason")
            except RuntimeError_ as exc:
                errors.append(f"runtime error: {exc}")
                result = None

            if result is not None and not errors:
                # 1. Schema validation — varies by operation.
                schema_errors, parsed_kind = self._validate_schema(
                    operation, result,
                )
                errors.extend(schema_errors)

                # 1b. Terminal fallback envelope: if the specialist agent
                # emitted a DeveloperTicket (allowed in EXPECTED_OUTPUT_TYPES
                # for generate/patch operations), treat it as a deliberate
                # "I cannot satisfy this with the current toolkit" signal
                # and accept it without further semantic-contract retries.
                # Without this, dashboard-spec-agent's well-formed
                # toolkit-extension ticket would be discarded and the
                # orchestrator would never see it. See helper/extend_request.py.
                output_type_now = result.get("type")
                fallback_envelopes = (
                    EXPECTED_OUTPUT_TYPES.get(operation, frozenset())
                    - {"DashboardSpec", "PatchSpec"}
                )
                if (
                    not schema_errors
                    and output_type_now == "DeveloperTicket"
                    and "DeveloperTicket" in fallback_envelopes
                ):
                    op_summary = ["fallback_envelope=DeveloperTicket"]
                # 2. Semantic contract (skipped on terminal fallback).
                else:
                    check = contract.check(result, args)
                    op_summary = check.op_summary
                    if not check.ok:
                        errors.extend(check.errors)

            elapsed = time.monotonic() - t0
            ok = not errors and result is not None
            attempts.append(AttemptLog(
                attempt=n,
                elapsed_s=round(elapsed, 3),
                output_type=output_type,
                op_summary=op_summary,
                errors=[_scrub(e) or "" for e in errors],
                accepted=ok,
                runtime_used=runtime_used,
                fallback_reason=_scrub(fallback_reason),
            ))
            if ok:
                accepted = result
                break

            # Short-circuit on deterministic providers (Decision 6).
            if runtime_used in ("mock", "mock_fallback", "heuristic"):
                break

        outcome = RetryOutcome(
            accepted=accepted,
            attempts=attempts,
            contract_name=contract.name,
            operation=operation,
        )
        if self._persist_logs:
            self._persist(outcome)
        return outcome

    # --------------------------------------------------------------
    def _validate_schema(
        self, operation: str, result: dict,
    ) -> tuple[list[str], str | None]:
        """Returns (errors, parsed_kind). Does NOT mutate `result`."""
        t = result.get("type")
        errors: list[str] = []
        parsed_kind: str | None = None
        try:
            if operation == "generate_dashboard" and t == "DashboardSpec":
                self._validator.validate_dashboard(result.get("spec") or {})
                parsed_kind = "DashboardSpec"
            elif operation == "patch_dashboard" and t == "PatchSpec":
                self._validator.validate_patch(result.get("spec") or {})
                parsed_kind = "PatchSpec"
            # Other envelope types (DeveloperTicket, UserResponse, ...)
            # pass through without schema validation here; the
            # semantic contract will decide whether they satisfy the
            # user's request.
        except SpecValidationError as exc:
            errors.extend(exc.errors[:8])
        return errors, parsed_kind

    # --------------------------------------------------------------
    def _persist(self, outcome: RetryOutcome) -> Path | None:
        """Write the attempt log to retry_logs/. Returns the path or
        None on failure (failures never propagate)."""
        try:
            _RETRY_LOGS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            suffix = uuid.uuid4().hex[:6]
            path = _RETRY_LOGS_DIR / f"{stamp}_{outcome.operation}_{suffix}.json"
            payload = {
                "operation": outcome.operation,
                "contract_name": outcome.contract_name,
                "max_attempts": self._max_attempts,
                "attempts": [asdict(a) for a in outcome.attempts],
                "accepted_attempt": (
                    outcome.attempts[-1].attempt
                    if outcome.accepted and outcome.attempts else None
                ),
            }
            # Defensive secret scrub across the whole serialization.
            dump = json.dumps(payload, default=str)
            if _contains_secret(dump):
                # Rebuild with scrubbed strings (should rarely trigger
                # — the attempt log already scrubs).
                payload = json.loads(
                    re.sub(r"sk-[A-Za-z0-9-]+", "<redacted>", dump)
                )
            path.write_text(json.dumps(payload, indent=2))
            return path
        except OSError:
            return None
