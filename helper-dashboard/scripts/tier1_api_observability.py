"""Tier 1 — API observability scenario (CLI-only, real model).

Invokes `bin/opencode` directly for each operation to confirm the
configured LLM provider can produce a *usable* API observability
dashboard, and that patch rephrasings semantically satisfy the user
request (with bounded retry + feedback, mirroring the orchestrator's
validation-retry layer).

Scope:
- generate_dashboard for the API observability prompt.
- Four rephrasing scenarios:
    * "move critical widgets to the top"   -> reorder_widgets
    * "make latency more prominent"        -> update_widget on latency
    * "add error-rate threshold at 2%"     -> update_widget on error
    * "change CPU chart to memory chart"   -> update_widget on cpu
- Each patch check is allowed up to HELPER_AGENT_MAX_ATTEMPTS retries
  with structured feedback injected via args. The check passes only
  if some attempt produced a valid PatchSpec.
- Heuristic provider short-circuits after one failed attempt.

Usage:
    cd helper-dashboard
    set -a; source .env; set +a
    python3 scripts/tier1_api_observability.py
    OPENCODE_LLM_PROVIDER=heuristic python3 scripts/tier1_api_observability.py

Exit code: 0 if every check passes (possibly with retry WARNs);
           1 otherwise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BIN = Path(os.environ.get(
    "HELPER_DASHBOARD_OPENCODE_BIN", str(ROOT / "bin" / "opencode")
))
PER_CALL_TIMEOUT = int(os.environ.get("TIER1_TIMEOUT", "180"))

# Mirrors backend/app/helper/retry_loop.py clamp.
_MIN = 1
_MAX = 5


def resolve_max_attempts() -> int:
    raw = os.environ.get("HELPER_AGENT_MAX_ATTEMPTS")
    if not raw:
        return 3
    try:
        n = int(raw)
    except ValueError:
        return 3
    return max(_MIN, min(_MAX, n))


def call(payload: dict) -> tuple[bool, dict | None, str, float, str | None]:
    """Return (ok, parsed, err, elapsed, runtime_used)."""
    t0 = time.monotonic()
    try:
        p = subprocess.run(
            [str(BIN)], input=json.dumps(payload), text=True,
            capture_output=True, timeout=PER_CALL_TIMEOUT, check=False,
        )
    except subprocess.TimeoutExpired:
        return (False, None, f"TIMEOUT {PER_CALL_TIMEOUT}s",
                time.monotonic() - t0, None)
    t = time.monotonic() - t0
    if p.returncode != 0:
        return (False, None, f"rc={p.returncode} err={p.stderr[:200]}", t,
                None)
    try:
        parsed = json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        return False, None, f"non-JSON: {exc.msg}", t, None
    return True, parsed, p.stderr, t, parsed.get("runtime_used")


def fmt(ok: bool, warn: bool = False) -> str:
    if warn and ok:
        return "\033[93mPASS (WARN)\033[0m"
    return "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"


ALLOWED_TYPES = {"line_chart", "stat_card", "gauge", "table", "alert_list"}


def main() -> int:
    if not BIN.exists():
        print(f"ERR: bin not found at {BIN}")
        return 2

    sys.path.insert(0, str(ROOT / "backend"))
    from app.services.spec_validator import SpecValidationError, SpecValidator
    from app.helper.semantic_contracts import (
        contract_for_generate, contract_for_patch,
    )

    max_attempts = resolve_max_attempts()
    provider = os.environ.get("OPENCODE_LLM_PROVIDER", "openrouter")

    print("Tier 1 — API observability scenario")
    print(f"  bin:          {BIN}")
    print(f"  provider:     {provider}")
    print(f"  max_attempts: {max_attempts}")
    print()

    passes = 0
    total = 0
    total_attempts = 0
    warn_count = 0

    # -----------------------------------------------------------------
    # 1. generate_dashboard — with retry
    # -----------------------------------------------------------------
    print("[generate_dashboard] API observability scenario")
    intent = {
        "type": "DashboardIntent",
        "summary": "API observability",
        "requirements": {
            "title": "API Observability",
            "goal": (
                "Show HTTP request rate, 5xx error rate, p95 and p99 "
                "request latency, service availability, CPU, memory, "
                "and firing alerts for the api service."
            ),
            "metrics_hints": [
                "http_requests_total", "http_request_duration_seconds",
                "process_cpu_seconds_total", "process_resident_memory_bytes",
                "up", "ALERTS",
            ],
            "widget_hints": ["line_chart", "gauge", "alert_list"],
            "refresh_interval_hint": "30s",
        },
    }
    generate_args = {"intent": intent}
    generate_contract = contract_for_generate(intent["requirements"])

    spec = None
    spec_dict = {}
    gen_attempts = 0
    gen_elapsed_total = 0.0
    last_gen_errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        gen_attempts += 1
        payload = {
            "operation": "generate_dashboard",
            "agent": "dashboard-spec-agent",
            "args": _inject_feedback(generate_args, attempt, last_gen_errors,
                                      generate_contract, intent),
        }
        ok, parsed, err, t, rtu = call(payload)
        gen_elapsed_total += t
        total_attempts += 1
        if not ok or not parsed or parsed.get("type") != "DashboardSpec":
            last_gen_errors = [err or f"unexpected type {parsed.get('type') if parsed else '?'}"]
            print(f"  attempt {attempt}: runtime={rtu} errors={last_gen_errors[:1]}  ({t:.1f}s)")
            if rtu in ("mock", "mock_fallback", "heuristic"):
                break
            continue
        # Schema + contract.
        spec_dict = parsed.get("spec") or {}
        try:
            spec = SpecValidator().validate_dashboard(spec_dict)
            schema_errors: list[str] = []
        except SpecValidationError as exc:
            spec = None
            schema_errors = exc.errors[:6]

        if schema_errors:
            last_gen_errors = schema_errors
            print(f"  attempt {attempt}: schema errors {schema_errors[:2]}  ({t:.1f}s)")
            if rtu in ("mock", "mock_fallback", "heuristic"):
                break
            continue

        contract_result = generate_contract.check(parsed, generate_args)
        if contract_result.ok:
            print(f"  attempt {attempt}: accepted  ({t:.1f}s)")
            last_gen_errors = []
            break
        last_gen_errors = contract_result.errors
        print(f"  attempt {attempt}: contract errors {contract_result.errors[:2]}  ({t:.1f}s)")
        if rtu in ("mock", "mock_fallback", "heuristic"):
            break

    total += 1
    if spec is None or last_gen_errors:
        print(f"  {fmt(False)}  generate_dashboard failed after "
              f"{gen_attempts} attempt(s); last errors: {last_gen_errors[:3]}")
        return _report_failure(passes, total, total_attempts, warn_count)
    passes += 1
    if gen_attempts > 1:
        warn_count += 1
    print(f"  {fmt(True, warn=gen_attempts > 1)}  DashboardSpec  "
          f"widgets={len(spec.widgets)}  attempts_used={gen_attempts}/{max_attempts}  "
          f"({gen_elapsed_total:.1f}s total)")

    # -- scenario assertions (no retry — the spec is already accepted)
    ids = [w.id for w in spec.widgets]
    types = [w.type.value for w in spec.widgets]
    print(f"  widgets: {ids}")
    print(f"  types:   {types}")

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal passes, total
        total += 1
        print(f"  {fmt(cond)}  {label}")
        if detail:
            print(f"        {detail}")
        if cond:
            passes += 1

    check("at least 6 widgets", len(spec.widgets) >= 6,
          f"got {len(spec.widgets)}")
    check("every widget uses an allowed type",
          all(t in ALLOWED_TYPES for t in types),
          f"types={types}")
    check("includes request-rate PromQL",
          any("rate(http_requests_total" in w.query.promql for w in spec.widgets))
    check("includes latency percentile PromQL (histogram_quantile + by (le))",
          any("histogram_quantile" in w.query.promql and "by (le)" in w.query.promql
              for w in spec.widgets))
    check("at least one widget has a threshold",
          any(len(w.thresholds) > 0 for w in spec.widgets))

    dashboard_for_patches = spec.model_dump(mode="json")

    # -----------------------------------------------------------------
    # 2. patch scenarios — each with retry
    # -----------------------------------------------------------------
    print()
    print(f"[patch_dashboard] four rephrasing scenarios (max_attempts={max_attempts})")

    def patch_with_retry(change: str) -> tuple[dict | None, int, float,
                                                 list[str], str | None]:
        """Returns (accepted_output, attempts_used, total_elapsed,
        last_errors, runtime_used_last)."""
        contract = contract_for_patch([change], dashboard_for_patches)
        base_args = {
            "intent": {
                "type": "PatchIntent",
                "target_dashboard_id": dashboard_for_patches["dashboard_id"],
                "requested_changes": [change],
                "message_to_user": "",
            },
            "dashboard": dashboard_for_patches,
        }
        last_errors: list[str] = []
        attempts = 0
        total_t = 0.0
        rtu = None
        for attempt in range(1, max_attempts + 1):
            attempts += 1
            args = _inject_feedback(base_args, attempt, last_errors,
                                     contract, base_args.get("intent"))
            payload = {
                "operation": "patch_dashboard",
                "agent": "patch-agent",
                "args": args,
            }
            ok, parsed, err, t, rtu = call(payload)
            total_t += t
            if not ok or not parsed:
                last_errors = [err or "no output"]
                if rtu in ("mock", "mock_fallback", "heuristic"):
                    break
                continue
            if parsed.get("type") != "PatchSpec":
                last_errors = [f"wrong envelope type {parsed.get('type')!r}"]
                if rtu in ("mock", "mock_fallback", "heuristic"):
                    break
                continue
            # Schema
            try:
                SpecValidator().validate_patch(parsed.get("spec") or {})
                schema_errors: list[str] = []
            except SpecValidationError as exc:
                schema_errors = exc.errors[:4]
            if schema_errors:
                last_errors = schema_errors
                if rtu in ("mock", "mock_fallback", "heuristic"):
                    break
                continue
            # Contract
            r = contract.check(parsed, base_args)
            if r.ok:
                return parsed, attempts, total_t, [], rtu
            last_errors = r.errors
            if rtu in ("mock", "mock_fallback", "heuristic"):
                break
        return None, attempts, total_t, last_errors, rtu

    scenarios = [
        ("move critical widgets to the top", "reorder_widgets in ops"),
        ("make latency more prominent", "update_widget on latency widget"),
        ("add error-rate threshold at 2%",
         "update_widget on error widget with 0.02 threshold"),
        ("change CPU chart to memory chart",
         "update_widget on cpu widget with memory PromQL"),
    ]
    for change, _desc in scenarios:
        accepted, attempts, elapsed, errors, rtu = patch_with_retry(change)
        total_attempts += attempts
        total += 1
        ok = accepted is not None
        if ok:
            passes += 1
            if attempts > 1:
                warn_count += 1
            label = f"PASS (WARN attempts_used={attempts}/{max_attempts})" \
                if attempts > 1 else f"PASS attempts_used={attempts}/{max_attempts}"
            print(f"  {fmt(True, warn=attempts>1)}  {change!r}  "
                  f"{label}  runtime={rtu}  ({elapsed:.1f}s)")
        else:
            print(f"  {fmt(False)}  {change!r}  all {attempts} attempt(s) failed  "
                  f"runtime={rtu}  ({elapsed:.1f}s)")
            for i, e in enumerate(errors[:4], 1):
                print(f"        err[{i}]: {e[:160]}")
            return _report_failure(passes, total, total_attempts, warn_count)

    # -----------------------------------------------------------------
    print()
    return _report_summary(passes, total, total_attempts, warn_count,
                             max_attempts)


def _inject_feedback(
    args: dict, attempt: int, prior_errors: list[str], contract, intent,
) -> dict:
    """Return a fresh args dict with retry feedback when attempt > 1."""
    if attempt == 1 or not prior_errors:
        return dict(args)
    out = dict(args)
    out["_retry_attempt"] = attempt
    out["_prior_errors"] = prior_errors[:6]
    out["_prior_feedback_message"] = contract.feedback_message(
        prior_errors, args,
    )
    return out


def _report_failure(passes, total, total_attempts, warn_count) -> int:
    print()
    print(f"Tier 1 API observability: {passes}/{total} checks  "
          f"total_attempts_used={total_attempts}  warns={warn_count}")
    print(fmt(False) + "  scenario degraded")
    return 1


def _report_summary(passes, total, total_attempts, warn_count, max_attempts) -> int:
    print(f"Tier 1 API observability: {passes}/{total} checks  "
          f"total_attempts_used={total_attempts}  warns={warn_count}  "
          f"max_attempts={max_attempts}")
    if passes == total:
        if warn_count:
            print(fmt(True, warn=True) + "  all checks passed "
                  f"({warn_count} needed retry)")
        else:
            print(fmt(True) + "  all checks passed on first attempt")
        return 0
    print(fmt(False) + "  scenario degraded")
    return 1


if __name__ == "__main__":
    sys.exit(main())
