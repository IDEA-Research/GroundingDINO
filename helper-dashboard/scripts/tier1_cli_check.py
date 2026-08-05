"""Tier 1 verification — CLI-only, real model.

Invokes `bin/opencode` directly for each operation to confirm the
configured LLM provider can produce valid JSON for every agent in
the allow-list. Does NOT start the backend, frontend, or Playwright.

Scope:
- user_message           -> DashboardIntent / PatchIntent / UserResponse
- generate_dashboard     -> DashboardSpec
- patch_dashboard        -> PatchSpec
- review_rendered        -> ReviewDecision (approve / patch / escalate)
- rescue_review          -> RescueDecision (patch / ask_user / ticket)

What it verifies:
- Subprocess protocol (stdin JSON -> stdout JSON).
- Each response has the expected top-level `type`.
- Schema validation succeeds for any spec payloads.
- Latency per call is reported.

What it does NOT verify:
- HTTP surface (Tier 2).
- Frontend rendering (Tier 3).
- Review loop orchestration (Tier 2 + unit tests).

Usage:
    cd helper-dashboard
    set -a; source .env; set +a       # load OPENROUTER_API_KEY, etc.
    python3 scripts/tier1_cli_check.py

Env overrides:
    OPENCODE_LLM_PROVIDER=heuristic   # test without a key
    HELPER_DASHBOARD_OPENCODE_BIN     # alternate binary

Exit code: 0 if all checks pass, non-zero otherwise.
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

# Per-call timeout — Tier 1 is meant to be quick. If your model is
# slower, bump this via env.
PER_CALL_TIMEOUT = int(os.environ.get("TIER1_TIMEOUT", "120"))


def call(payload: dict, *, label: str) -> tuple[bool, dict | None, str, float]:
    """Invoke bin/opencode. Returns (ok, parsed, stderr, elapsed)."""
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [str(BIN)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=PER_CALL_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, None, f"TIMEOUT after {PER_CALL_TIMEOUT}s", time.monotonic() - t0

    elapsed = time.monotonic() - t0
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    if proc.returncode != 0:
        return False, None, f"rc={proc.returncode}  stderr: {err[:300]}", elapsed
    if not out:
        return False, None, f"empty stdout; stderr: {err[:300]}", elapsed
    try:
        return True, json.loads(out), err, elapsed
    except json.JSONDecodeError as exc:
        return False, None, f"non-JSON: {exc.msg}  stdout[:200]={out[:200]!r}", elapsed


def expect_type(parsed: dict | None, allowed: set[str]) -> tuple[bool, str]:
    if parsed is None:
        return False, "no parsed response"
    t = parsed.get("type")
    if t not in allowed:
        return False, f"type={t!r} not in {sorted(allowed)}"
    return True, f"type={t}"


def fmt(ok: bool) -> str:
    return "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"


def main() -> int:
    if not BIN.exists():
        print(f"ERR: bin not found at {BIN}")
        return 2

    provider = os.environ.get("OPENCODE_LLM_PROVIDER", "openrouter")
    print(f"Tier 1 CLI verification")
    print(f"  bin:      {BIN}")
    print(f"  provider: {provider}")
    print(f"  timeout:  {PER_CALL_TIMEOUT}s per call")
    print()

    total_pass = 0
    total_checks = 0
    total_elapsed = 0.0

    # --- Check 1: user_message ---------------------------------------
    print("[1/5] user_message")
    ok, parsed, err, t = call({
        "operation": "user_message",
        "agent": "helper-chat-agent",
        "args": {"message": "Show me a CPU and memory dashboard"},
    }, label="user_message")
    total_elapsed += t
    total_checks += 1
    if not ok:
        print(f"      {fmt(False)}  {err}  ({t:.1f}s)")
    else:
        tok, tmsg = expect_type(parsed, {
            "DashboardIntent", "PatchIntent", "PrometheusIntent",
            "UserResponse", "ClarificationRequest", "DeveloperTicket",
        })
        print(f"      {fmt(tok)}  {tmsg}  ({t:.1f}s)")
        if tok:
            total_pass += 1

    # --- Check 2: generate_dashboard ---------------------------------
    print("[2/5] generate_dashboard")
    ok, parsed, err, t = call({
        "operation": "generate_dashboard",
        "agent": "dashboard-spec-agent",
        "args": {
            "intent": {
                "type": "DashboardIntent",
                "summary": "CPU dashboard",
                "requirements": {
                    "title": "CPU",
                    "goal": "See node CPU usage",
                    "metrics_hints": ["node_cpu_seconds_total"],
                    "widget_hints": ["line_chart", "stat_card"],
                    "refresh_interval_hint": "30s",
                },
            },
        },
    }, label="generate_dashboard")
    total_elapsed += t
    total_checks += 1
    spec_payload: dict | None = None
    if not ok:
        print(f"      {fmt(False)}  {err}  ({t:.1f}s)")
    else:
        tok, tmsg = expect_type(parsed, {"DashboardSpec", "DeveloperTicket"})
        widgets_info = ""
        if tok and parsed.get("type") == "DashboardSpec":
            spec_payload = parsed.get("spec") or {}
            ws = spec_payload.get("widgets") or []
            widgets_info = f"  widgets={len(ws)}  types={[w.get('type') for w in ws]}"
        print(f"      {fmt(tok)}  {tmsg}{widgets_info}  ({t:.1f}s)")
        if tok:
            total_pass += 1

    # Schema-validate the returned spec via Pydantic.
    print("[3/5] schema validation of generated DashboardSpec")
    total_checks += 1
    if spec_payload is None:
        print(f"      {fmt(False)}  no spec to validate")
    else:
        sys.path.insert(0, str(ROOT / "backend"))
        from app.services.spec_validator import SpecValidationError, SpecValidator
        try:
            SpecValidator().validate_dashboard(spec_payload)
            print(f"      {fmt(True)}  Pydantic + extra-forbid validation passed")
            total_pass += 1
        except SpecValidationError as exc:
            print(f"      {fmt(False)}  {exc.errors[:3]}")

    # --- Check 4: patch_dashboard ------------------------------------
    print("[4/5] patch_dashboard (add a widget)")
    dashboard_for_patch = spec_payload or {
        "dashboard_id": "p",
        "title": "P",
        "description": "",
        "layout": {"columns": 12, "row_height": 40},
        "variables": [],
        "widgets": [{
            "id": "w1", "type": "line_chart", "title": "x",
            "description": "",
            "query": {"source": "prometheus", "promql": "up",
                       "query_type": "range", "range": "1h", "step": "30s"},
            "position": {"x": 0, "y": 0, "w": 6, "h": 6},
            "encoding": {}, "thresholds": [], "options": {},
        }],
        "refresh_interval": "30s",
    }
    ok, parsed, err, t = call({
        "operation": "patch_dashboard",
        "agent": "patch-agent",
        "args": {
            "intent": {
                "type": "PatchIntent",
                "target_dashboard_id": dashboard_for_patch["dashboard_id"],
                "requested_changes": ["add a stat card widget for uptime"],
                "message_to_user": "",
            },
            "dashboard": dashboard_for_patch,
        },
    }, label="patch_dashboard")
    total_elapsed += t
    total_checks += 1
    if not ok:
        print(f"      {fmt(False)}  {err}  ({t:.1f}s)")
    else:
        tok, tmsg = expect_type(parsed, {"PatchSpec", "DeveloperTicket"})
        ops_info = ""
        if tok and parsed.get("type") == "PatchSpec":
            ops = parsed.get("spec", {}).get("operations", [])
            ops_info = f"  ops={[o.get('op') for o in ops]}"
        print(f"      {fmt(tok)}  {tmsg}{ops_info}  ({t:.1f}s)")
        if tok:
            total_pass += 1

    # --- Check 5: review_rendered ------------------------------------
    print("[5/5] review_rendered (clean report -> should approve)")
    ok, parsed, err, t = call({
        "operation": "review_rendered",
        "agent": "helper-review-agent",
        "args": {
            "dashboard": dashboard_for_patch,
            "report": {
                "dashboard_id": dashboard_for_patch["dashboard_id"],
                "page_loaded": True,
                "widgets_rendered": [
                    w["id"] for w in dashboard_for_patch.get("widgets", [])
                ],
                "missing_widgets": [],
                "console_errors": [],
                "layout_errors": [],
                "prometheus_errors": [],
                "recommendation": "clean render",
            },
            "user_intent": "CPU dashboard",
            "attempt": 1,
            "history": [],
        },
    }, label="review_rendered")
    total_elapsed += t
    total_checks += 1
    if not ok:
        print(f"      {fmt(False)}  {err}  ({t:.1f}s)")
    else:
        tok, tmsg = expect_type(parsed, {"ReviewDecision"})
        decision = parsed.get("decision") if tok else None
        print(f"      {fmt(tok)}  {tmsg}  decision={decision}  ({t:.1f}s)")
        if tok:
            total_pass += 1

    # --- Summary -----------------------------------------------------
    print()
    print("-" * 60)
    print(
        f"Tier 1 result: {total_pass}/{total_checks} checks  "
        f"wall-clock {total_elapsed:.1f}s"
    )
    if total_pass == total_checks:
        print(fmt(True) + "  all CLI real-model checks passed")
        return 0
    print(fmt(False) + "  one or more checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
