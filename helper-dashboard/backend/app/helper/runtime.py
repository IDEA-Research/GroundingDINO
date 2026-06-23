"""OpenCode runtime wrapper.

Responsibilities:

- Expose a fixed, allow-listed set of *operations* (not arbitrary
  agent names) to the rest of the backend. Operations are the only
  way to reach an agent.
- Enforce which operations the user-facing flow is allowed to invoke
  vs. developer-only.
- Validate the output contract per operation (must be a known
  response `type`).
- Dispatch to a mock runtime or to a subprocess OpenCode CLI
  runtime depending on mode.

Safety model (defense in depth — this file is **not** the primary
enforcement point; see `docs/SECURITY_BOUNDARIES.md`):

- Primary agent permissions live in `opencode.json` and in
  `.opencode/agent/*.md`. We do not pass permission flags on the
  CLI — OpenCode is not assumed to support them.
- The backend operation allow-list prevents arbitrary agent names.
- Pydantic schemas reject unsafe output.
- `/api/developer/*` endpoints are the only channel into Big guy.

Modes (selected by `HELPER_DASHBOARD_OPENCODE`):

- `mock` (default): always use `MockHelperRuntime`.
- `opencode`: must use the subprocess adapter. If the CLI is missing,
  times out, or returns invalid JSON, a `RuntimeError_` is raised.
  Never silently falls back to mock.
- `auto`: optional development convenience. Tries the subprocess
  adapter; on failure falls back to mock but the response dict
  carries `runtime_used="mock_fallback"` and `fallback_reason`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..specs.widget_spec import WidgetType


# ---------------------------------------------------------------------------
# Operation allow-list
# ---------------------------------------------------------------------------
#
# Every agent invocation flows through an *operation* defined here.
# The backend never accepts a free-form agent name from callers.
# Adding an operation is a Big guy task.

OPERATIONS: dict[str, dict[str, str | None]] = {
    "user_message":       {"agent": "helper-chat-agent",       "command": "user-message"},
    "generate_dashboard": {"agent": "dashboard-spec-agent",    "command": "generate-dashboard"},
    "patch_dashboard":    {"agent": "patch-agent",             "command": "patch-dashboard"},
    # `prometheus_query` is an internal sub-operation used after a
    # user_message classifies a PrometheusIntent. It has no
    # user-facing slash command.
    "prometheus_query":   {"agent": "prometheus-agent",        "command": None},
    "evaluate_dashboard": {"agent": "browser-eval-agent",      "command": "evaluate-dashboard"},
    # Pre-output review operations. `review_rendered` is Helper's
    # self-check (small model), `rescue_review` escalates to Big guy
    # (bigger model, same internal agent file but review-mode prompt).
    "review_rendered":    {"agent": "helper-review-agent",     "command": None},
    "rescue_review":      {"agent": "big-guy-developer-agent", "command": None},
    # `rescue_extend` is Big guy in "extend the widget toolkit" mode —
    # reached whenever the orchestrator or review loop sees a missing
    # widget type. The toolkit is a cache, not a fence; extend is the
    # default path for unknown types. Big guy gets edit + restricted
    # shell here (all other paths keep him locked to JSON-only). Safety
    # is enforced by `extend_gate.py` (denylist, prompt-injection
    # check, daily quota, audit log) and the per-path write allowlist
    # in `bin/opencode`. See docs/SECURITY_BOUNDARIES.md.
    "rescue_extend":      {"agent": "big-guy-developer-agent", "command": "rescue-extend"},
    "developer_fix":      {"agent": "big-guy-developer-agent", "command": "developer-fix"},
}

# Only these operations are reachable from the user-facing API.
# `rescue_review` is in the user hot path too: it's Big guy in
# review-only mode, never edits code (see docs/SECURITY_BOUNDARIES.md).
# `rescue_extend` is also user-facing — safety gates in
# `extend_gate.py` keep it bounded.
USER_OPERATIONS: frozenset[str] = frozenset({
    "user_message",
    "generate_dashboard",
    "patch_dashboard",
    "prometheus_query",
    "evaluate_dashboard",
    "review_rendered",
    "rescue_review",
    "rescue_extend",
})

# developer_fix is the only developer-only operation. It requires
# `developer=True` to be passed to `invoke_operation` AND it is only
# reachable through the dev-token-gated `/api/developer/*` endpoints.
DEVELOPER_OPERATIONS: frozenset[str] = frozenset({"developer_fix"})

assert USER_OPERATIONS.isdisjoint(DEVELOPER_OPERATIONS), \
    "an operation cannot be both user-facing and developer-only"
assert USER_OPERATIONS | DEVELOPER_OPERATIONS == frozenset(OPERATIONS.keys()), \
    "every operation must be classified as user or developer"

# Expected response `type` values per operation. The runtime rejects
# any agent output whose `type` is not in the set. These map to the
# envelope types defined in `.opencode/agent/*.md`.
EXPECTED_OUTPUT_TYPES: dict[str, frozenset[str]] = {
    "user_message": frozenset({
        "UserResponse",
        "DashboardIntent",
        "PatchIntent",
        "PrometheusIntent",
        "ClarificationRequest",
        "DeveloperTicket",
    }),
    "generate_dashboard": frozenset({"DashboardSpec", "DeveloperTicket"}),
    "patch_dashboard": frozenset({"PatchSpec", "DeveloperTicket"}),
    "prometheus_query": frozenset({"PrometheusQueryReport", "DeveloperTicket"}),
    "evaluate_dashboard": frozenset({
        "BrowserEvaluationReport",
        "BugReport",
        "PatchSpec",
        "DeveloperTicket",
    }),
    "review_rendered": frozenset({"ReviewDecision"}),
    "rescue_review": frozenset({"RescueDecision"}),
    "rescue_extend": frozenset({"DeveloperReport"}),
    "developer_fix": frozenset({"DeveloperReport"}),
}


# ---------------------------------------------------------------------------
# Backwards-compat helpers (used by older code paths and tests).
# ---------------------------------------------------------------------------

HELPER_AGENTS: frozenset[str] = frozenset(
    OPERATIONS[op]["agent"]
    for op in OPERATIONS
    if op not in DEVELOPER_OPERATIONS
    and OPERATIONS[op]["agent"] != "big-guy-developer-agent"
)
BIG_GUY: str = "big-guy-developer-agent"
assert BIG_GUY not in HELPER_AGENTS

_AGENT_TO_USER_OP: dict[str, str] = {}
for op, meta in OPERATIONS.items():
    agent = meta["agent"]
    if (
        op in USER_OPERATIONS
        and agent != BIG_GUY
        and agent not in _AGENT_TO_USER_OP
    ):
        _AGENT_TO_USER_OP[agent] = op


class RuntimeError_(Exception):
    """Raised for contract violations (bad agent/operation name,
    invalid output type, opencode-mode failures, etc.)."""


# ---------------------------------------------------------------------------
# Subprocess adapter
# ---------------------------------------------------------------------------


class _OpenCodeUnavailable(Exception):
    """Binary missing, not executable, or timed out. Structured error
    for the runtime wrapper to decide whether to raise or fall back."""

    def __init__(self, reason: str, *, kind: str):
        super().__init__(reason)
        self.reason = reason
        self.kind = kind


class _OpenCodeInvalidOutput(Exception):
    """Binary ran but returned non-JSON, empty output, or a non-dict
    value, or a dict without a `type` field."""

    def __init__(self, reason: str, *, kind: str = "invalid_output",
                 stderr: str = "", stdout: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.kind = kind
        self.stderr = stderr[:1000]
        self.stdout = stdout[:1000]


# Project root used as subprocess cwd. Resolved from this module's
# location so tests and prod agree.
#   .../helper-dashboard/backend/app/helper/runtime.py
#   -> helper/ -> app/ -> backend/ -> helper-dashboard/
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Default subprocess argv. The operator may override via
# `HELPER_DASHBOARD_OPENCODE_CMD` (a JSON array of strings). Only
# `{bin}` and `{agent}` are substituted, and `{agent}` is always one
# of our allow-listed values.
_DEFAULT_CMD_TEMPLATE: list[str] = ["{bin}", "run", "--agent", "{agent}", "--json"]

_DEFAULT_TIMEOUT_S: float = 120.0


_ALLOWED_AGENTS: frozenset[str] = frozenset(OPERATIONS[op]["agent"] for op in OPERATIONS)


def _trace_enabled() -> bool:
    return (os.getenv("HELPER_DASHBOARD_OPENCODE_TRACE", "").strip().lower()
            in {"1", "true", "yes", "on"})


def _trace(msg: str) -> None:
    if not _trace_enabled():
        return
    line = f"[helper-dashboard][opencode-trace] {msg}"
    print(line, file=sys.stderr)
    path = (os.getenv("OPENCODE_TRACE_FILE") or "").strip()
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def _safe_keys(d: dict[str, Any]) -> list[str]:
    try:
        return sorted(str(k) for k in d.keys())
    except Exception:
        return ["<unavailable>"]


class RealOpenCodeRuntime:
    """Subprocess adapter for the OpenCode CLI.

    Invariants:

    - `shell=False` always.
    - Command tokens come from an operator-configured template, never
      from user input.
    - Agent name is substituted from the allow-list — not the caller.
    - cwd is pinned to the project root.
    - User args go on stdin as a JSON object.
    """

    def invoke(self, operation: str, agent: str, args: dict[str, Any]) -> dict[str, Any]:
        if operation not in OPERATIONS:
            raise RuntimeError_(f"unknown operation: {operation!r}")
        if agent != OPERATIONS[operation]["agent"]:
            raise RuntimeError_(
                f"operation/agent mismatch: {operation!r} -> {OPERATIONS[operation]['agent']!r} "
                f"but caller passed {agent!r}"
            )
        if agent not in _ALLOWED_AGENTS:  # pragma: no cover - defensive
            raise RuntimeError_(f"agent {agent!r} not in allow-list")

        cmd = _resolve_cmd(agent)
        if cmd is None:
            raise _OpenCodeUnavailable(
                "OpenCode binary not found or not executable",
                kind="binary_missing",
            )

        stdin_payload = json.dumps({
            "operation": operation,
            "agent": agent,
            # Compatibility: some opencode builds expect `command` to be
            # a string and call `.strip()` internally.
            # For operations without a slash command, pass empty string
            # instead of null/omitted to avoid NoneType.strip crashes.
            "command": str(OPERATIONS[operation].get("command") or ""),
            "args": args,
        })

        timeout = _resolve_timeout()
        t0 = time.monotonic()
        _trace(
            f"invoke:start operation={operation!r} agent={agent!r} "
            f"cmd={cmd!r} timeout={timeout}s args_keys={_safe_keys(args)!r}"
        )

        try:
            proc = subprocess.run(  # noqa: S603  shell=False, fixed cwd
                cmd,
                input=stdin_payload,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
                shell=False,
                cwd=str(_PROJECT_ROOT),
            )
        except FileNotFoundError as exc:
            raise _OpenCodeUnavailable(str(exc), kind="binary_missing") from exc
        except subprocess.TimeoutExpired as exc:
            raise _OpenCodeUnavailable(
                f"timeout after {timeout}s", kind="timeout"
            ) from exc

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        elapsed = time.monotonic() - t0
        _trace(
            f"invoke:done rc={proc.returncode} elapsed={elapsed:.2f}s "
            f"stdout_len={len(stdout)} stderr_len={len(stderr)}"
        )

        if proc.returncode != 0:
            import signal as _sig
            sig_name = None
            if proc.returncode < 0:
                try:
                    sig_name = _sig.Signals(-proc.returncode).name
                except Exception:
                    sig_name = f"signal {-proc.returncode}"

            # Classify common CLI failure modes into clearer `kind`
            # values so tickets tell the operator what to do.
            kind = "nonzero_exit"
            stderr_l = stderr.lower()
            if "credits exhausted" in stderr_l or "http 402" in stderr_l:
                kind = "credits_exhausted"
            elif "rejected the api key" in stderr_l or "http 401" in stderr_l:
                kind = "bad_api_key"
            elif "rate-limited" in stderr_l or "http 429" in stderr_l:
                kind = "rate_limited"
            elif sig_name:
                kind = f"killed_by_{sig_name.lower()}"

            _warn(
                f"opencode subprocess {kind}: rc={proc.returncode} "
                f"sig={sig_name} stderr={stderr[:200]!r}"
            )
            raise _OpenCodeInvalidOutput(
                f"exit {proc.returncode}"
                + (f" ({sig_name})" if sig_name else ""),
                kind=kind,
                stderr=stderr,
                stdout=stdout,
            )
        if not stdout:
            raise _OpenCodeInvalidOutput(
                "empty stdout", kind="empty_output", stderr=stderr
            )
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise _OpenCodeInvalidOutput(
                f"non-JSON stdout: {exc.msg}",
                kind="non_json",
                stdout=stdout,
                stderr=stderr,
            ) from exc

        if not isinstance(result, dict) or "type" not in result:
            raise _OpenCodeInvalidOutput(
                "response is not a typed JSON object",
                kind="missing_type",
                stdout=stdout,
                stderr=stderr,
            )

        _trace(
            f"invoke:parsed type={result.get('type')!r} runtime_used={result.get('runtime_used')!r}"
        )
        return result


def _resolve_cmd(agent: str) -> list[str] | None:
    bin_name = os.getenv("HELPER_DASHBOARD_OPENCODE_BIN", "opencode")
    raw_template = os.getenv("HELPER_DASHBOARD_OPENCODE_CMD")

    if raw_template:
        try:
            template = json.loads(raw_template)
            if (
                not isinstance(template, list)
                or not template
                or not all(isinstance(t, str) for t in template)
            ):
                raise ValueError("template must be a non-empty JSON array of strings")
        except Exception as exc:
            _warn(
                f"HELPER_DASHBOARD_OPENCODE_CMD is invalid ({exc}); "
                "using default template."
            )
            template = list(_DEFAULT_CMD_TEMPLATE)
    else:
        template = list(_DEFAULT_CMD_TEMPLATE)

    # Defense-in-depth: only `{bin}` and `{agent}` substitutions are
    # allowed, and `{agent}` must be in the allow-list.
    cmd: list[str] = []
    for tok in template:
        if "{agent}" in tok and agent not in _ALLOWED_AGENTS:  # pragma: no cover
            return None
        tok = tok.replace("{bin}", bin_name).replace("{agent}", agent)
        cmd.append(tok)

    head = cmd[0]
    if not head:
        return None
    if not os.path.isabs(head) and "/" not in head:
        if shutil.which(head) is None:
            return None
    else:
        # Resolve relative paths against the project root, which is
        # the subprocess's cwd. Caller's cwd may be different
        # (e.g. uvicorn launched from backend/).
        if not os.path.isabs(head):
            head_abs = str((_PROJECT_ROOT / head).resolve())
            cmd[0] = head_abs
        else:
            head_abs = head
        if not os.path.exists(head_abs) or not os.access(head_abs, os.X_OK):
            return None
    return cmd


def _resolve_timeout() -> float:
    raw = os.getenv("HELPER_DASHBOARD_OPENCODE_TIMEOUT_SECONDS")
    if not raw:
        return _DEFAULT_TIMEOUT_S
    try:
        v = float(raw)
        return v if v > 0 else _DEFAULT_TIMEOUT_S
    except ValueError:
        return _DEFAULT_TIMEOUT_S


def _warn(msg: str) -> None:
    print(f"[helper-dashboard] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public runtime wrapper
# ---------------------------------------------------------------------------


class OpenCodeRuntime:
    """Primary entry point for invoking agents.

    Callers use `invoke_operation(operation, args, *, developer=False)`.
    The legacy `invoke_helper(agent, args)` remains as a thin
    compatibility wrapper for tests and older code paths.
    """

    def __init__(self) -> None:
        self._mock = MockHelperRuntime()
        self._real = RealOpenCodeRuntime()

    # --- mode selection ---------------------------------------------
    @staticmethod
    def _mode() -> str:
        return (os.getenv("HELPER_DASHBOARD_OPENCODE") or "mock").strip().lower()

    # --- primary API ------------------------------------------------
    def invoke_operation(
        self,
        operation: str,
        args: dict[str, Any],
        *,
        developer: bool = False,
    ) -> dict[str, Any]:
        if operation not in OPERATIONS:
            raise RuntimeError_(f"unknown operation: {operation!r}")
        if operation in DEVELOPER_OPERATIONS and not developer:
            raise RuntimeError_(
                f"operation {operation!r} is developer-only; the user chat flow "
                "must never invoke it"
            )

        agent = OPERATIONS[operation]["agent"]
        mode = self._mode()

        if mode == "mock":
            result = self._mock.invoke(agent, args, operation=operation)
            result.setdefault("runtime_used", "mock")

        elif mode == "opencode":
            try:
                result = self._real.invoke(operation, agent, args)
            except (_OpenCodeUnavailable, _OpenCodeInvalidOutput) as exc:
                # Strict by design — no silent fallback.
                raise RuntimeError_(
                    f"opencode runtime failure ({exc.kind}): {exc}"
                ) from exc
            result.setdefault("runtime_used", "opencode")

        elif mode == "auto":
            try:
                result = self._real.invoke(operation, agent, args)
                result.setdefault("runtime_used", "opencode")
            except (_OpenCodeUnavailable, _OpenCodeInvalidOutput) as exc:
                reason = f"{exc.kind}:{exc}"
                _warn(f"auto mode: falling back to mock ({reason})")
                result = self._mock.invoke(agent, args, operation=operation)
                # Overwrite any inner value — the marker must be explicit.
                result["runtime_used"] = "mock_fallback"
                result["fallback_reason"] = reason

        else:
            raise RuntimeError_(
                f"unknown HELPER_DASHBOARD_OPENCODE mode: {mode!r} "
                "(expected 'mock', 'opencode', or 'auto')"
            )

        self._validate_output_contract(operation, result)
        return result

    # --- compat API ------------------------------------------------
    def invoke_helper(self, agent: str, args: dict[str, Any]) -> dict[str, Any]:
        """Legacy entry point. Prefer `invoke_operation`.

        - Rejects Big guy explicitly (Big guy is not a Helper agent).
        - Maps the Helper agent back to its user-operation and
          delegates.
        """
        if agent == BIG_GUY or agent not in HELPER_AGENTS:
            raise RuntimeError_(f"agent {agent!r} is not a Helper agent")
        op = _AGENT_TO_USER_OP.get(agent)
        if op is None:  # pragma: no cover - defensive
            raise RuntimeError_(f"no user operation mapped to agent {agent!r}")
        return self.invoke_operation(op, args, developer=False)

    # --- output contract -------------------------------------------
    @staticmethod
    def _validate_output_contract(operation: str, result: dict[str, Any]) -> None:
        expected = EXPECTED_OUTPUT_TYPES.get(operation)
        if expected is None:  # pragma: no cover - defensive
            raise RuntimeError_(f"no expected output types for operation {operation!r}")
        t = result.get("type")
        if t not in expected:
            raise RuntimeError_(
                f"invalid output for {operation!r}: type={t!r} "
                f"not in {sorted(expected)}"
            )


# ---------------------------------------------------------------------------
# Mock runtime
# ---------------------------------------------------------------------------


class MockHelperRuntime:
    """Deterministic, in-process stand-in for the real OpenCode runtime.

    Mirrors the output shapes described in `.opencode/agent/*.md`.
    Never edits code, never hits the shell, never calls the network.

    Accepts an optional `operation` kwarg so new operations that don't
    map cleanly to one agent name (e.g. `rescue_review` shares Big
    guy's agent file with `developer_fix`) can dispatch correctly.
    """

    def invoke(
        self,
        agent: str,
        args: dict[str, Any],
        operation: str | None = None,
    ) -> dict[str, Any]:
        # Operation-first dispatch for new review operations.
        if operation == "review_rendered":
            return self._review_rendered(args)
        if operation == "rescue_review":
            return self._rescue_review(args)
        if operation == "developer_fix":
            return self._developer_fix(args)

        # Legacy agent-based dispatch for user-facing Helpers.
        if agent == "helper-chat-agent":
            return self._chat(args)
        if agent == "dashboard-spec-agent":
            return self._dashboard_spec(args)
        if agent == "patch-agent":
            return self._patch(args)
        if agent == "prometheus-agent":
            return self._prometheus(args)
        if agent == "browser-eval-agent":
            return self._browser_eval(args)
        if agent == BIG_GUY:
            # Runtime wrapper gates this path — in mock mode Big guy
            # is only reachable via the `operation` kwarg above.
            raise RuntimeError_("mock runtime will not impersonate Big guy")
        raise RuntimeError_(f"mock runtime has no handler for {agent!r}")

    # -- helper-chat-agent ---------------------------------------------------
    def _chat(self, args: dict[str, Any]) -> dict[str, Any]:
        msg = (args.get("message") or "").strip()
        low = msg.lower()
        current = args.get("current_dashboard_id")

        if not msg:
            return {"type": "UserResponse", "message": "What dashboard would you like?"}

        patch_triggers = [
            "change ", "rename ", "remove ", "delete ", "add a ", "add an ",
            "add error", "add latency", "add threshold", "add alert",
            "edit ", "update ", "set ", "make the ", "make latency",
            "make error", "reorder ", "move ", "resize ", "threshold",
            "more prominent", "bigger", "highlight", "promote",
        ]
        if current and any(t in low for t in patch_triggers):
            return {
                "type": "PatchIntent",
                "target_dashboard_id": current,
                "requested_changes": [msg],
                "message_to_user": "Working on those changes.",
            }

        if any(t in low for t in ["metric", "promql", "query", "prometheus"]):
            return {
                "type": "PrometheusIntent",
                "question": msg,
                "message_to_user": "Let me check your Prometheus data.",
            }

        if any(t in low for t in ["dashboard", "chart", "graph", "show me", "monitor", "overview"]):
            title = _extract_title(msg)
            widget_hints = _extract_widget_hints(low)
            metric_hints = _extract_metric_hints(low)
            return {
                "type": "DashboardIntent",
                "summary": title or "New dashboard",
                "requirements": {
                    "title": title or "New dashboard",
                    "goal": msg,
                    "metrics_hints": metric_hints,
                    "widget_hints": widget_hints,
                    "refresh_interval_hint": "30s",
                },
                "clarification_needed": False,
                "message_to_user": "Building that dashboard now.",
            }

        return {
            "type": "UserResponse",
            "message": (
                "Tell me what you'd like to see — for example "
                "'a CPU and memory dashboard for my nodes'."
            ),
        }

    # -- dashboard-spec-agent -----------------------------------------------
    def _dashboard_spec(self, args: dict[str, Any]) -> dict[str, Any]:
        intent = args.get("intent") or {}
        req = intent.get("requirements") or {}
        title = req.get("title") or intent.get("summary") or "New dashboard"
        metric_hints = req.get("metrics_hints") or []
        widget_hints = req.get("widget_hints") or []
        goal = (req.get("goal") or "").lower()
        dashboard_id = _slugify(title) or f"dash-{uuid.uuid4().hex[:8]}"

        # If the user's intent looks like an API observability dashboard,
        # use the rich golden composition.
        if _looks_like_api_observability(goal, metric_hints, title):
            widgets = _api_observability_widgets()
        else:
            widgets = _build_default_widgets(widget_hints, metric_hints)

        spec = {
            "dashboard_id": dashboard_id,
            "title": title,
            "description": req.get("goal") or "",
            "layout": {"columns": 12, "row_height": 40},
            "variables": [],
            "widgets": widgets,
            "refresh_interval": req.get("refresh_interval_hint") or "30s",
        }
        return {"type": "DashboardSpec", "spec": spec}

    # -- patch-agent --------------------------------------------------------
    def _patch(self, args: dict[str, Any]) -> dict[str, Any]:
        intent = args.get("intent") or {}
        dashboard = args.get("dashboard") or {}
        target = intent.get("target_dashboard_id") or dashboard.get("dashboard_id")
        changes: list[str] = intent.get("requested_changes") or []
        ops: list[dict[str, Any]] = []
        text = " ".join(changes).lower()

        existing_widgets = dashboard.get("widgets") or []

        # -- reorder: move critical widgets to the top -------------------
        if any(k in text for k in [
            "move critical", "critical widgets", "critical to the top",
            "reorder", "move to the top", "promote critical",
        ]):
            critical_types = {"gauge", "alert_list"}
            critical_keywords = ("error", "latency", "p95", "p99",
                                  "availability", "up")

            def crit_rank(w: dict) -> int:
                t = w.get("type") or ""
                title = (w.get("title") or "").lower()
                if any(k in title for k in critical_keywords):
                    return 0
                if t in critical_types:
                    return 1
                return 2

            if len(existing_widgets) >= 2:
                ids = sorted(
                    [w.get("id") for w in existing_widgets if w.get("id")],
                    key=lambda wid: crit_rank(
                        next((w for w in existing_widgets if w.get("id") == wid),
                             {})
                    ),
                )
                ops.append({"op": "reorder_widgets", "order": ids})

        # -- make X more prominent: grow and move up --------------------
        if any(k in text for k in ["prominent", "bigger", "make the",
                                     "highlight"]):
            wid = _first_id_in_text(text, existing_widgets)
            if wid:
                ops.append({
                    "op": "update_widget",
                    "widget_id": wid,
                    "fields": {
                        "position": {"x": 0, "y": 0, "w": 12, "h": 6},
                    },
                })

        # -- add threshold at X% ---------------------------------------
        if "threshold" in text:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|s|ms)?", text)
            if m:
                raw = float(m.group(1))
                unit = (m.group(2) or "").lower()
                # Normalize: "2%" -> 0.02; "500ms" -> 0.5; "2s" -> 2.0
                if unit in ("%", "percent"):
                    value = raw / 100.0
                elif unit == "ms":
                    value = raw / 1000.0
                else:
                    value = raw
                wid = _first_id_in_text(text, existing_widgets)
                if wid:
                    existing = next(
                        (w for w in existing_widgets if w.get("id") == wid),
                        None,
                    )
                    thresholds = list((existing or {}).get("thresholds") or [])
                    thresholds.append({
                        "value": value,
                        "color": "#f59e0b",
                        "label": f"warn >{raw}{unit}".strip(),
                    })
                    ops.append({
                        "op": "update_widget",
                        "widget_id": wid,
                        "fields": {"thresholds": thresholds},
                    })

        # -- change CPU chart to memory chart / metric swap ------------
        swap_match = re.search(
            r"change (?:the\s+)?(\w+)\s+chart\s+to\s+(?:a\s+)?(\w+)(?:\s+chart)?",
            text,
        )
        if swap_match:
            from_kw = swap_match.group(1)
            to_kw = swap_match.group(2)
            metric_map = {
                "cpu": ("process_cpu_seconds_total", "rate(process_cpu_seconds_total[5m])", "s/s"),
                "memory": ("process_resident_memory_bytes", "process_resident_memory_bytes", "bytes"),
                "ram": ("process_resident_memory_bytes", "process_resident_memory_bytes", "bytes"),
                "requests": ("http_requests_total", "sum(rate(http_requests_total[5m]))", "req/s"),
                "latency": ("http_request_duration_seconds_bucket",
                             "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
                             "s"),
                "errors": ("http_requests_total",
                            'sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))',
                            "%"),
            }
            wid = None
            for w in existing_widgets:
                title = (w.get("title") or "").lower()
                promql = (w.get("query") or {}).get("promql", "").lower()
                if from_kw in title or from_kw in promql:
                    wid = w.get("id")
                    break
            to = metric_map.get(to_kw)
            if wid and to:
                metric_name, promql, unit = to
                ops.append({
                    "op": "update_widget",
                    "widget_id": wid,
                    "fields": {
                        "title": to_kw.capitalize(),
                        "query": {
                            "source": "prometheus",
                            "promql": promql,
                            "query_type": "range",
                            "range": "1h",
                            "step": "30s",
                        },
                        "encoding": {"unit": unit},
                    },
                })

        # -- classic remove / rename / add / refresh -------------------
        if not ops and any(k in text for k in ["remove", "delete"]):
            wid = _first_id_in_text(text, existing_widgets)
            if wid:
                ops.append({"op": "remove_widget", "widget_id": wid})

        if not ops and "rename" in text and " to " in text:
            new_title = text.split(" to ", 1)[1].strip().strip("'\"")
            if new_title:
                ops.append({"op": "update_dashboard", "fields": {"title": new_title[:120]}})

        if not ops and "add" in text and any(t in text for t in ["chart", "widget", "graph", "stat"]):
            wtype = _guess_widget_type(text)
            wid = f"w-{uuid.uuid4().hex[:6]}"
            ops.append({
                "op": "add_widget",
                "widget": _default_widget(wid, wtype, title=f"New {wtype}", promql=_default_promql_for(wtype)),
            })

        if not ops and "refresh" in text:
            m = re.search(r"(\d{1,3})\s*(s|m|h)", text)
            if m:
                ops.append({
                    "op": "update_dashboard",
                    "fields": {"refresh_interval": f"{m.group(1)}{m.group(2)}"},
                })

        if not ops:
            ops.append({
                "op": "update_dashboard",
                "fields": {"description": (dashboard.get("description") or "") + " (updated)"},
            })

        return {
            "type": "PatchSpec",
            "spec": {
                "patch_id": f"patch-{uuid.uuid4().hex[:8]}",
                "reason": changes[0][:240] if changes else "user requested change",
                "target_dashboard_id": target,
                "created_by": "patch-agent",
                "operations": ops,
            },
        }

    # -- prometheus-agent ---------------------------------------------------
    def _prometheus(self, args: dict[str, Any]) -> dict[str, Any]:
        question = (args.get("question") or "").strip()
        return {
            "type": "PrometheusQueryReport",
            "report": {
                "queries": [],
                "metric_suggestions": [
                    {"metric": m, "reason": "common metric"}
                    for m in ["up", "process_cpu_seconds_total", "node_memory_MemAvailable_bytes"]
                ],
                "notes": f"question echoed: {question[:200]}" if question else "",
            },
        }

    # -- browser-eval-agent -------------------------------------------------
    def _browser_eval(self, args: dict[str, Any]) -> dict[str, Any]:
        report = args.get("report") or {}
        missing = report.get("missing_widgets") or []
        console = report.get("console_errors") or []

        if not report.get("page_loaded", False):
            return {
                "type": "DeveloperTicket",
                "source_agent": "browser-eval-agent",
                "severity": "high",
                "summary": "page failed to load",
                "user_visible_effect": "dashboard page does not render",
                "technical_evidence": {"console_errors": console},
                "requested_action": "investigate frontend /dashboard route and renderer",
                "safety_notes": "",
            }
        if missing and not console:
            return {
                "type": "BugReport",
                "report": {
                    "bug_id": f"bug-{uuid.uuid4().hex[:8]}",
                    "source": "browser_evaluator",
                    "severity": "medium",
                    "summary": f"{len(missing)} widget(s) missing on screen",
                    "evidence": {
                        "console_errors": console,
                        "missing_widgets": missing,
                        "layout_errors": report.get("layout_errors") or [],
                        "prometheus_errors": report.get("prometheus_errors") or [],
                        "screenshot_path": report.get("screenshot_path"),
                    },
                    "suspected_cause": "renderer did not map one or more widget types",
                    "suggested_fix_type": "code",
                },
            }
        if console:
            return {
                "type": "DeveloperTicket",
                "source_agent": "browser-eval-agent",
                "severity": "medium",
                "summary": "console errors while rendering dashboard",
                "user_visible_effect": "dashboard may render incorrectly",
                "technical_evidence": {"console_errors": console[:5]},
                "requested_action": "investigate widget-toolkit components for crashes on empty data",
                "safety_notes": "",
            }
        return {
            "type": "BugReport",
            "report": {
                "bug_id": f"bug-{uuid.uuid4().hex[:8]}",
                "source": "browser_evaluator",
                "severity": "low",
                "summary": "clean render",
                "evidence": {},
                "suspected_cause": "",
                "suggested_fix_type": "unknown",
            },
        }

    # -- review_rendered ----------------------------------------------------
    def _review_rendered(self, args: dict[str, Any]) -> dict[str, Any]:
        """Helper's hot-path review heuristic.

        Decides: approve / patch / escalate based on the evaluation
        report and the current attempt number.
        """
        report = args.get("report") or {}
        spec = args.get("dashboard") or {}
        attempt = int(args.get("attempt") or 1)

        if not report.get("page_loaded", False):
            return {
                "type": "ReviewDecision",
                "decision": "escalate",
                "rationale": "page did not load; beyond Helper patch",
            }

        missing = report.get("missing_widgets") or []
        if missing:
            remaining = [
                w for w in spec.get("widgets") or []
                if w.get("id") not in missing
            ]
            if not remaining:
                return {
                    "type": "ReviewDecision",
                    "decision": "escalate",
                    "rationale": "every widget failed to render",
                }
            ops = [
                {"op": "remove_widget", "widget_id": wid} for wid in missing
            ]
            return {
                "type": "ReviewDecision",
                "decision": "patch",
                "rationale": f"remove {len(missing)} widget(s) that failed to render",
                "patch": {
                    "patch_id": f"rv-{uuid.uuid4().hex[:8]}",
                    "reason": "auto-remove missing widgets",
                    "target_dashboard_id": spec.get("dashboard_id"),
                    "created_by": "helper-review-agent",
                    "operations": ops,
                },
            }

        if (report.get("console_errors") or []) and attempt >= 3:
            return {
                "type": "ReviewDecision",
                "decision": "escalate",
                "rationale": "console errors persist after retries",
            }

        return {
            "type": "ReviewDecision",
            "decision": "approve",
            "rationale": "rendered cleanly",
        }

    # -- rescue_review (Big guy in review mode, no code edits) --------------
    def _rescue_review(self, args: dict[str, Any]) -> dict[str, Any]:
        report = args.get("report") or {}
        console = report.get("console_errors") or []
        missing = report.get("missing_widgets") or []

        if console:
            return {
                "type": "RescueDecision",
                "kind": "ticket",
                "rationale": "console errors indicate a code-level bug",
                "ticket": {
                    "ticket_id": f"tkt-{uuid.uuid4().hex[:8]}",
                    "source_agent": "big-guy-developer-agent",
                    "severity": "high",
                    "summary": "rescue review: console errors",
                    "user_visible_effect": "dashboard renders but has JS errors",
                    "technical_evidence": {"console_errors": console[:5]},
                    "requested_action": "investigate widget-toolkit components",
                    "safety_notes": "",
                },
            }

        if missing:
            return {
                "type": "RescueDecision",
                "kind": "ask_user",
                "rationale": "widgets failed to render after retries",
                "questions": [
                    "The visual I tried to build didn't render cleanly. "
                    "Could you describe the dashboard in more detail and "
                    "say which metrics you'd like to see?"
                ],
            }

        return {
            "type": "RescueDecision",
            "kind": "ask_user",
            "rationale": "need more detail to recover",
            "questions": [
                "I built the dashboard but something seems off. "
                "Can you tell me exactly what you expected to see?"
            ],
        }

    # -- developer_fix (Big guy in developer mode) --------------------------
    def _developer_fix(self, args: dict[str, Any]) -> dict[str, Any]:
        instruction = (
            args.get("instruction") or args.get("ticket_id") or "fix"
        )
        return {
            "type": "DeveloperReport",
            "report_id": f"rpt-{uuid.uuid4().hex[:8]}",
            "ticket_id": args.get("ticket_id"),
            "instruction": str(instruction)[:256],
            "actions_taken": ["mock runtime: no code changes made"],
            "tests_run": [],
            "summary": "mock runtime cannot make code changes",
            "status": "rejected",
        }


# ---------------------------------------------------------------------------
# Helpers used only by the mock runtime
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:48]


def _extract_title(msg: str) -> str:
    m = re.search(r"(?:dashboard|board|page)\s+(?:for|about|called)\s+([^.?!\n]+)", msg, re.I)
    if m:
        return m.group(1).strip()[:80]
    return ""


def _extract_widget_hints(msg: str) -> list[str]:
    hints: list[str] = []
    if any(k in msg for k in ["trend", "time", "graph", "line"]):
        hints.append("line_chart")
    if any(k in msg for k in ["stat", "number", "single value", "kpi", "count"]):
        hints.append("stat_card")
    if any(k in msg for k in ["gauge", "percent"]):
        hints.append("gauge")
    if any(k in msg for k in ["table", "list of"]):
        hints.append("table")
    if "alert" in msg:
        hints.append("alert_list")
    if not hints:
        hints = ["line_chart", "stat_card"]
    return hints


def _extract_metric_hints(msg: str) -> list[str]:
    hints: list[str] = []
    # API / HTTP observability metrics
    if any(k in msg for k in ["http", "request rate", "rps", "traffic"]):
        hints.append("http_requests_total")
    if any(k in msg for k in ["latency", "p95", "p99", "duration"]):
        hints.append("http_request_duration_seconds")
    if "error rate" in msg or "5xx" in msg or "error_rate" in msg:
        hints.append("http_requests_total")
    # CPU
    if "cpu" in msg:
        hints.append("process_cpu_seconds_total")
        hints.append("node_cpu_seconds_total")
    # Memory
    if "memory" in msg or "ram" in msg:
        hints.append("process_resident_memory_bytes")
        hints.append("node_memory_MemAvailable_bytes")
    # Alerts
    if "alert" in msg:
        hints.append("ALERTS")
    # Availability
    if any(k in msg for k in ["up", "availability", "service health", "health"]):
        hints.append("up")
    # Dedup preserving order
    return list(dict.fromkeys(hints))


def _default_promql_for(widget_type: str) -> str:
    return {
        "line_chart": 'rate(http_requests_total[5m])',
        "stat_card": 'sum(up)',
        "gauge": 'avg(up)',
        "table": 'topk(10, http_requests_total)',
        "alert_list": 'ALERTS{alertstate="firing"}',
    }.get(widget_type, "up")


def _default_widget(wid: str, wtype: str, *, title: str, promql: str) -> dict[str, Any]:
    query_type = "instant" if wtype in {"stat_card", "gauge", "alert_list", "table"} else "range"
    query = {
        "source": "prometheus",
        "promql": promql,
        "query_type": query_type,
    }
    if query_type == "range":
        query["range"] = "1h"
        query["step"] = "30s"
    return {
        "id": wid,
        "type": wtype,
        "title": title,
        "description": "",
        "query": query,
        "position": {"x": 0, "y": 0, "w": 6, "h": 6},
        "encoding": {},
        "thresholds": [],
        "options": {},
    }


def _build_default_widgets(widget_hints: list[str], metric_hints: list[str]) -> list[dict[str, Any]]:
    widgets: list[dict[str, Any]] = []
    x = 0
    y = 0
    idx = 0
    wants = list(dict.fromkeys(widget_hints)) or ["line_chart", "stat_card"]
    for wtype_name in wants:
        if wtype_name not in {t.value for t in WidgetType}:
            continue
        metric = metric_hints[idx % len(metric_hints)] if metric_hints else None
        promql = _default_promql_for(wtype_name)
        if metric:
            if wtype_name == "line_chart":
                promql = f"rate({metric}[5m])" if metric.endswith("_total") else metric
            elif wtype_name in {"stat_card", "gauge"}:
                promql = f"avg({metric})"
            elif wtype_name == "table":
                promql = f"topk(10, {metric})"
        w = 6 if wtype_name in {"line_chart", "table"} else 3
        h = 6 if wtype_name in {"line_chart", "table"} else 4
        if x + w > 12:
            x = 0
            y += h
        widget = _default_widget(
            f"w-{idx}-{wtype_name}",
            wtype_name,
            title=(metric or wtype_name).replace("_", " ").title(),
            promql=promql,
        )
        widget["position"] = {"x": x, "y": y, "w": w, "h": h}
        widgets.append(widget)
        x += w
        idx += 1
    return widgets


_TITLE_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on",
    "usage", "rate", "chart", "widget", "panel", "list",
})


def _first_id_in_text(text: str, widgets: list[dict[str, Any]]) -> str | None:
    """Match a widget id given a free-text user instruction.

    Order of precedence:
    1. Exact widget id appears in text.
    2. Exact widget title substring.
    3. Any significant word from the widget title appears in text.
       (e.g. "latency" matches "Request latency p95").
    4. Scored by number of title words present in text — most matches
       wins.

    Returns None when no match is good enough, so callers can skip
    the operation rather than patch the wrong widget.
    """
    text_l = text.lower()

    # 1. id match
    for w in widgets:
        wid = w.get("id") or ""
        if wid and wid.lower() in text_l:
            return wid

    # 2. full-title substring
    for w in widgets:
        t = (w.get("title") or "").lower().strip()
        if t and t in text_l:
            return w.get("id")

    # 3 & 4. significant-word match, highest score wins
    def score(w: dict) -> int:
        t = (w.get("title") or "").lower()
        tokens = [
            tok for tok in re.split(r"[\s/\-]+", t)
            if tok and tok not in _TITLE_STOPWORDS and len(tok) > 2
        ]
        return sum(1 for tok in tokens if tok in text_l)

    ranked = sorted(widgets, key=score, reverse=True)
    if ranked and score(ranked[0]) > 0:
        return ranked[0].get("id")

    # No good match — do NOT guess.
    return None


def _guess_widget_type(text: str) -> str:
    if "stat" in text or "number" in text or "kpi" in text:
        return "stat_card"
    if "gauge" in text or "percent" in text:
        return "gauge"
    if "table" in text:
        return "table"
    if "alert" in text:
        return "alert_list"
    return "line_chart"


# ---------------------------------------------------------------------------
# Golden API-observability composition
# ---------------------------------------------------------------------------

_API_OBS_KEYWORDS = (
    "api", "http request", "http_requests", "request rate", "latency",
    "p95", "p99", "error rate", "service availability", "observability",
    "rps", "http_request_duration",
)


def _looks_like_api_observability(
    goal: str, metric_hints: list[str], title: str
) -> bool:
    blob = " ".join([goal, title, " ".join(metric_hints)]).lower()
    hits = sum(1 for kw in _API_OBS_KEYWORDS if kw in blob)
    return hits >= 2


def _api_observability_widgets() -> list[dict[str, Any]]:
    """Return the 8-widget golden API observability composition.

    Mirrors `backend/app/samples/api_observability.json` so the
    heuristic mock and the Real LLM both converge on a similar
    structure for this scenario.
    """
    return [
        {"id": "w-request-rate", "type": "line_chart",
         "title": "HTTP request rate", "description": "",
         "query": {"source": "prometheus",
                   "promql": "sum(rate(http_requests_total[5m]))",
                   "query_type": "range", "range": "1h", "step": "30s"},
         "position": {"x": 0, "y": 0, "w": 6, "h": 6},
         "encoding": {"unit": "req/s"}, "thresholds": [], "options": {}},
        {"id": "w-error-rate", "type": "line_chart",
         "title": "5xx error rate", "description": "",
         "query": {"source": "prometheus",
                   "promql": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))",
                   "query_type": "range", "range": "1h", "step": "30s"},
         "position": {"x": 6, "y": 0, "w": 6, "h": 6},
         "encoding": {"unit": "%"},
         "thresholds": [
             {"value": 0.01, "color": "#f59e0b", "label": "warn >1%"},
             {"value": 0.05, "color": "#ef4444", "label": "crit >5%"},
         ],
         "options": {}},
        {"id": "w-latency-p95", "type": "line_chart",
         "title": "Request latency p95", "description": "",
         "query": {"source": "prometheus",
                   "promql": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
                   "query_type": "range", "range": "1h", "step": "30s"},
         "position": {"x": 0, "y": 6, "w": 6, "h": 6},
         "encoding": {"unit": "s"},
         "thresholds": [
             {"value": 0.5, "color": "#f59e0b", "label": "warn >500ms"},
             {"value": 1.0, "color": "#ef4444", "label": "crit >1s"},
         ],
         "options": {}},
        {"id": "w-latency-p99", "type": "line_chart",
         "title": "Request latency p99", "description": "",
         "query": {"source": "prometheus",
                   "promql": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
                   "query_type": "range", "range": "1h", "step": "30s"},
         "position": {"x": 6, "y": 6, "w": 6, "h": 6},
         "encoding": {"unit": "s"},
         "thresholds": [
             {"value": 1.0, "color": "#f59e0b", "label": "warn >1s"},
             {"value": 2.0, "color": "#ef4444", "label": "crit >2s"},
         ],
         "options": {}},
        {"id": "w-availability", "type": "gauge",
         "title": "Service availability", "description": "",
         "query": {"source": "prometheus",
                   "promql": "avg(up)",
                   "query_type": "instant"},
         "position": {"x": 0, "y": 12, "w": 3, "h": 4},
         "encoding": {},
         "thresholds": [
             {"value": 0.95, "color": "#f59e0b", "label": "warn <95%"},
             {"value": 0.99, "color": "#34d399", "label": "healthy"},
         ],
         "options": {"min": 0, "max": 1, "decimals": 2}},
        {"id": "w-cpu", "type": "line_chart",
         "title": "CPU usage", "description": "",
         "query": {"source": "prometheus",
                   "promql": "rate(process_cpu_seconds_total[5m])",
                   "query_type": "range", "range": "1h", "step": "30s"},
         "position": {"x": 3, "y": 12, "w": 5, "h": 4},
         "encoding": {"unit": "s/s"},
         "thresholds": [], "options": {}},
        {"id": "w-memory", "type": "line_chart",
         "title": "Resident memory", "description": "",
         "query": {"source": "prometheus",
                   "promql": "process_resident_memory_bytes",
                   "query_type": "range", "range": "1h", "step": "30s"},
         "position": {"x": 8, "y": 12, "w": 4, "h": 4},
         "encoding": {"unit": "bytes"},
         "thresholds": [], "options": {}},
        {"id": "w-alerts", "type": "alert_list",
         "title": "Firing alerts", "description": "",
         "query": {"source": "prometheus",
                   "promql": "ALERTS{alertstate=\"firing\"}",
                   "query_type": "instant"},
         "position": {"x": 0, "y": 16, "w": 12, "h": 4},
         "encoding": {},
         "thresholds": [],
         "options": {"severity_filter": "warning", "row_limit": 20}},
    ]
