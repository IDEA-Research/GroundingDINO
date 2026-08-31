"""helperCode runtime wrapper.

Responsibilities:

- Expose a fixed, allow-listed set of *operations* (not arbitrary
  agent names) to the rest of the backend. Operations are the only
  way to reach an agent.
- Enforce which operations the user-facing flow is allowed to invoke
  vs. developer-only.
- Validate the output contract per operation (must be a known
  response `type`).
- Dispatch to a mock runtime or to a subprocess helperCode CLI
  runtime depending on mode (`bin/helpercode`, this project's own
  CLI, unrelated to the OpenCode CLI product).

Safety model (defense in depth — this file is **not** the primary
enforcement point; see `docs/SECURITY_BOUNDARIES.md`):

- Primary agent permissions live in `opencode.json` and in
  `.opencode/agent/*.md`. We do not pass permission flags on the
  CLI — OpenCode is not assumed to support them.
- The backend operation allow-list prevents arbitrary agent names.
- Pydantic schemas reject unsafe output.
- `/api/developer/*` endpoints are the only channel into Big guy.

Modes (selected by `HELPER_DASHBOARD_HELPERCODE`):

- `mock` (default): always use `MockHelperRuntime`.
- `helpercode`: must use the subprocess adapter. If the CLI is missing,
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
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..prometheus import catalog as prom_catalog
from ..prometheus.client import PrometheusUnavailable
from ..services.saved_dashboard_loader import fuzzy_match_saved
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
    # in `bin/helpercode`. See docs/SECURITY_BOUNDARIES.md.
    "rescue_extend":      {"agent": "big-guy-developer-agent", "command": "rescue-extend"},
    "developer_fix":      {"agent": "big-guy-developer-agent", "command": "developer-fix"},
    # `author_alert_rule` turns an AlertRuleIntent into a SystemAlertRule
    # envelope (system-metric domain, structurally SHADOW — see
    # specs/system_rule_spec.py). JSON-only; the Python validator is the
    # authority on the result.
    "author_alert_rule":  {"agent": "dashboard-spec-agent",    "command": None},
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
    "author_alert_rule",
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
        "AlertRuleIntent",
        # Library recall: "open my saved X" → the orchestrator resolves
        # the entry and loads a clone deterministically (no LLM
        # regeneration of the saved spec).
        "LoadSavedIntent",
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
    "author_alert_rule": frozenset({"SystemAlertRule", "ClarificationRequest"}),
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
    invalid output type, helpercode-mode failures, etc.)."""


# ---------------------------------------------------------------------------
# Subprocess adapter
# ---------------------------------------------------------------------------


class _HelperCodeUnavailable(Exception):
    """Binary missing, not executable, or timed out. Structured error
    for the runtime wrapper to decide whether to raise or fall back."""

    def __init__(self, reason: str, *, kind: str):
        super().__init__(reason)
        self.reason = reason
        self.kind = kind


class _HelperCodeInvalidOutput(Exception):
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
# `HELPER_DASHBOARD_HELPERCODE_CMD` (a JSON array of strings). Only
# `{bin}` and `{agent}` are substituted, and `{agent}` is always one
# of our allow-listed values.
_DEFAULT_CMD_TEMPLATE: list[str] = ["{bin}", "run", "--agent", "{agent}", "--json"]

_DEFAULT_TIMEOUT_S: float = 120.0


_ALLOWED_AGENTS: frozenset[str] = frozenset(OPERATIONS[op]["agent"] for op in OPERATIONS)


def _trace_enabled() -> bool:
    return (os.getenv("HELPER_DASHBOARD_HELPERCODE_TRACE", "").strip().lower()
            in {"1", "true", "yes", "on"})


def _trace(msg: str) -> None:
    if not _trace_enabled():
        return
    line = f"[helper-dashboard][helpercode-trace] {msg}"
    print(line, file=sys.stderr)
    path = (os.getenv("HELPERCODE_TRACE_FILE") or "").strip()
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


class RealHelperCodeRuntime:
    """Subprocess adapter for the helperCode CLI.

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
            raise _HelperCodeUnavailable(
                "helperCode binary not found or not executable",
                kind="binary_missing",
            )

        stdin_payload = json.dumps({
            "operation": operation,
            "agent": agent,
            # Compatibility: some helperCode CLI builds expect `command` to be
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
            raise _HelperCodeUnavailable(str(exc), kind="binary_missing") from exc
        except subprocess.TimeoutExpired as exc:
            raise _HelperCodeUnavailable(
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
                f"helpercode subprocess {kind}: rc={proc.returncode} "
                f"sig={sig_name} stderr={stderr[:200]!r}"
            )
            raise _HelperCodeInvalidOutput(
                f"exit {proc.returncode}"
                + (f" ({sig_name})" if sig_name else ""),
                kind=kind,
                stderr=stderr,
                stdout=stdout,
            )
        if not stdout:
            raise _HelperCodeInvalidOutput(
                "empty stdout", kind="empty_output", stderr=stderr
            )
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise _HelperCodeInvalidOutput(
                f"non-JSON stdout: {exc.msg}",
                kind="non_json",
                stdout=stdout,
                stderr=stderr,
            ) from exc

        if not isinstance(result, dict) or "type" not in result:
            raise _HelperCodeInvalidOutput(
                "response is not a typed JSON object",
                kind="missing_type",
                stdout=stdout,
                stderr=stderr,
            )

        _trace(
            f"invoke:parsed type={result.get('type')!r} runtime_used={result.get('runtime_used')!r}"
        )
        return result


# ---------------------------------------------------------------------------
# DevFix adapter — real OpenCode CLI for `developer_fix` only
# ---------------------------------------------------------------------------
#
# Behind HELPER_DASHBOARD_DEVFIX_RUNTIME=opencode-cli (default stays
# "helpercode"), the ONE developer operation `developer_fix` is routed
# to the real OpenCode CLI (1.18.4) instead of bin/helpercode. The run
# happens inside a detached git worktree (see devfix_worktree.py); on
# an honest `status: "resolved"` report the changes are copied back
# through an independent allow-list, all-or-nothing. Everything else
# (chat ops, rescue paths) stays on the hand-rolled CLI.

from . import devfix_log as _devfix_log  # noqa: E402
from . import devfix_worktree as _devfix_worktree  # noqa: E402


_DEVFIX_AGENT = "big-guy-developer-fix"
_DEVFIX_DEFAULT_BIN = "/home/cluster/.nvm/versions/node/v20.20.2/bin/opencode"
_DEVFIX_DEFAULT_MODEL = "openrouter/anthropic/claude-opus-4.6"
_DEVFIX_DEFAULT_TIMEOUT_S = 600.0
_DEVFIX_DEFAULT_HOME = "/home/cluster"
_DEVFIX_MESSAGE = (
    "Operation developer_fix. Read .devfix/payload.json for the "
    "DeveloperTicket, make the minimal fix within your permissions, run "
    "focused tests, and end your final message with the DeveloperReport "
    "fenced json block."
)


def _devfix_mode() -> str:
    """helpercode (default) | opencode-cli. Unknown values fall back to
    helpercode (fail-safe: never route to the paid CLI by accident)."""
    raw = (os.getenv("HELPER_DASHBOARD_DEVFIX_RUNTIME") or "helpercode").strip().lower()
    if raw not in {"helpercode", "opencode-cli"}:
        _warn(
            f"unknown HELPER_DASHBOARD_DEVFIX_RUNTIME {raw!r}; "
            "using 'helpercode'"
        )
        return "helpercode"
    return raw


def _resolve_devfix_bin() -> str:
    return (
        os.getenv("HELPER_DASHBOARD_DEVFIX_BIN") or _DEVFIX_DEFAULT_BIN
    ).strip()


def _resolve_devfix_model() -> str:
    return (
        os.getenv("HELPER_DASHBOARD_DEVFIX_MODEL") or _DEVFIX_DEFAULT_MODEL
    ).strip()


def _resolve_devfix_timeout() -> float:
    raw = os.getenv("HELPER_DASHBOARD_DEVFIX_TIMEOUT_SECONDS")
    if not raw:
        return _DEVFIX_DEFAULT_TIMEOUT_S
    try:
        v = float(raw)
        return v if v > 0 else _DEVFIX_DEFAULT_TIMEOUT_S
    except ValueError:
        return _DEVFIX_DEFAULT_TIMEOUT_S


def _devfix_env(bin_path: str) -> dict[str, str]:
    """Minimal child environment. Deliberately NOT inheriting
    os.environ — no OPENROUTER_API_KEY (or anything else) leaks; the
    CLI reads auth from $HOME/.local/share/opencode/auth.json."""
    home = (
        os.getenv("HELPER_DASHBOARD_DEVFIX_HOME") or _DEVFIX_DEFAULT_HOME
    ).strip()
    path_override = (os.getenv("HELPER_DASHBOARD_DEVFIX_PATH") or "").strip()
    if path_override:
        path = path_override
    else:
        parts = [
            os.path.dirname(bin_path),          # nvm bin dir (not on default PATH)
            os.path.dirname(sys.executable),    # python3 for test commands
            f"{home}/.local/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        ]
        seen: set[str] = set()
        path = ":".join(
            p for p in parts if p and not (p in seen or seen.add(p))
        )
    return {
        "HOME": home,
        "PATH": path,
        "TERM": "dumb",
        "LANG": "C.UTF-8",
        "NO_COLOR": "1",
    }


def _devfix_parse_events(stdout: str) -> list[dict[str, Any]]:
    """`--format json` emits one JSON object per line; non-JSON notice
    lines may be interleaved — skip anything unparseable."""
    events: list[dict[str, Any]] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _devfix_extract_report_text(events: list[dict[str, Any]]) -> str | None:
    """The last "text" event carries the final assistant message (the
    stream has NO terminal result event)."""
    text: str | None = None
    for ev in events:
        if ev.get("type") != "text":
            continue
        part = ev.get("part")
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text = part["text"]
    return text


_DEVFIX_JSON_FENCE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
_DEVFIX_ANY_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


def _devfix_extract_fenced_json(text: str) -> str:
    """Last ```json fence; fallback: last bare ``` fence; fallback:
    the whole text (caller json.loads-es and classifies)."""
    matches = _DEVFIX_JSON_FENCE.findall(text)
    if not matches:
        matches = _DEVFIX_ANY_FENCE.findall(text)
    return matches[-1].strip() if matches else text.strip()


def _devfix_wrong_agent(events: list[dict[str, Any]]) -> str | None:
    """Belt-and-braces for the silent subagent-fallback trap: if any
    event metadata names a different agent, return that name."""
    for ev in events:
        part = ev.get("part")
        candidates: list[Any] = [ev.get("agent")]
        if isinstance(part, dict):
            candidates.append(part.get("agent"))
            meta = part.get("metadata")
            if isinstance(meta, dict):
                candidates.append(meta.get("agent"))
        for cand in candidates:
            if isinstance(cand, str) and cand and cand != _DEVFIX_AGENT:
                return cand
    return None


# The run log is pure observability: it must NEVER break (or block) the
# actual fix run, so every call is routed through these guards and a
# missing/broken logger degrades to silence.

def _log_event(run_log: "_devfix_log.RunLogger | None", raw: Any) -> None:
    if run_log is None:
        return
    try:
        run_log.event(raw)
    except Exception:  # pragma: no cover - observability only
        pass


def _log_note(run_log: "_devfix_log.RunLogger | None", text: str) -> None:
    if run_log is None:
        return
    try:
        run_log.note(text)
    except Exception:  # pragma: no cover - observability only
        pass


def _log_finish(
    run_log: "_devfix_log.RunLogger | None",
    status: str,
    error_kind: str | None = None,
) -> None:
    if run_log is None:
        return
    try:
        run_log.finish(status, error_kind=error_kind)
    except Exception:  # pragma: no cover - observability only
        pass


class DevFixOpenCodeCLIRuntime:
    """Subprocess adapter for the REAL OpenCode CLI, `developer_fix`
    only.

    Invariants:

    - Only the `developer_fix` operation is accepted.
    - The CLI runs inside a detached git worktree, never the live tree
      (cwd + --dir = <worktree>/helper-dashboard).
    - Agent is pinned to `big-guy-developer-fix` (mode: primary —
      `run --agent` on a subagent silently falls back to build).
    - NEVER passes --auto: headless permission "ask" auto-rejects,
      which is the fail-closed behavior we want.
    - Child env is minimal and explicit — no os.environ inheritance.
    - Changes reach the main tree only via devfix_worktree.copy_back's
      independent all-or-nothing allow-list, and only on an honest
      `status: "resolved"` report.
    - Raises the same exception types as RealHelperCodeRuntime so mode
      dispatch and auto_fix keep working unchanged.
    """

    def invoke(self, operation: str, agent: str, args: dict[str, Any]) -> dict[str, Any]:
        if operation != "developer_fix":
            raise RuntimeError_(
                f"DevFixOpenCodeCLIRuntime only handles 'developer_fix', "
                f"got {operation!r}"
            )
        if agent != OPERATIONS[operation]["agent"]:
            raise RuntimeError_(
                f"operation/agent mismatch: {operation!r} -> "
                f"{OPERATIONS[operation]['agent']!r} but caller passed {agent!r}"
            )

        bin_path = _resolve_devfix_bin()
        if not os.path.isfile(bin_path) or not os.access(bin_path, os.X_OK):
            raise _HelperCodeUnavailable(
                f"OpenCode CLI not found or not executable: {bin_path}",
                kind="binary_missing",
            )

        # Opportunistic cleanup of leftovers from crashed runs.
        try:
            _devfix_worktree.sweep_stale()
        except Exception:  # pragma: no cover - defensive
            pass

        payload = {
            "operation": operation,
            "agent": agent,
            "command": str(OPERATIONS[operation].get("command") or ""),
            "args": args,
        }
        ticket_id = str(args.get("ticket_id") or "manual")

        # Live run log (storage/devfix_runs/): the web UI tails this to
        # show what Big Guy thinks and does. Never load-bearing.
        run_log: _devfix_log.RunLogger | None = None
        try:
            run_log = _devfix_log.start_run(
                ticket_id,
                instruction=str(args.get("instruction") or ""),
                model=_resolve_devfix_model(),
            )
        except Exception:  # pragma: no cover - observability only
            _warn("devfix: run log unavailable; continuing without it")

        try:
            handle = _devfix_worktree.create(ticket_id, payload=payload)
        except _devfix_worktree.WorktreeError as exc:
            _log_note(run_log, f"worktree creation failed: {exc}")
            _log_finish(run_log, "failed", "binary_missing")
            raise _HelperCodeUnavailable(
                f"worktree creation failed: {exc}", kind="binary_missing",
            ) from exc

        _log_note(run_log, f"worktree created: {handle.project_dir}")

        try:
            return self._run_in_worktree(handle, operation, run_log)
        finally:
            try:
                _devfix_worktree.remove(handle)
            except Exception:  # pragma: no cover - defensive
                _warn(f"devfix: failed to remove worktree {handle.worktree_root}")

    # -- internals ---------------------------------------------------

    def _run_in_worktree(
        self,
        handle: "_devfix_worktree.WorktreeHandle",
        operation: str,
        run_log: "_devfix_log.RunLogger | None" = None,
    ) -> dict[str, Any]:
        import signal

        bin_path = _resolve_devfix_bin()
        model = _resolve_devfix_model()
        timeout = _resolve_devfix_timeout()
        project_dir = str(handle.project_dir)

        argv = [
            bin_path, "run",
            "--agent", _DEVFIX_AGENT,
            "--format", "json",
            # --thinking surfaces "reasoning" events in the JSON stream
            # so the live log can show WHAT BIG GUY THINKS, not just
            # what he does.
            "--thinking",
            "-m", model,
            "--dir", project_dir,
            "--print-logs", "--log-level", "WARN",
            _DEVFIX_MESSAGE,
        ]

        t0 = time.monotonic()
        _trace(
            f"devfix:start ticket={handle.ticket_id!r} model={model!r} "
            f"timeout={timeout}s wt={project_dir!r}"
        )

        try:
            proc = subprocess.Popen(  # noqa: S603 - shell=False, fixed argv
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
                cwd=project_dir,
                env=_devfix_env(bin_path),
            )
        except FileNotFoundError as exc:
            _log_finish(run_log, "failed", "binary_missing")
            raise _HelperCodeUnavailable(str(exc), kind="binary_missing") from exc

        # Incremental consumption (replaces communicate()): a reader
        # thread tails stdout line by line so every CLI event reaches
        # the live run log AS IT HAPPENS; a second thread drains stderr
        # so neither pipe buffer can deadlock the child. The main
        # thread keeps the exact timeout contract communicate() gave
        # us (wait + SIGKILL to the process group).
        stdout_lines: list[str] = []
        stderr_chunks: list[str] = []
        events: list[dict[str, Any]] = []

        def _drain_stdout() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                stdout_lines.append(line)
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(obj, dict):
                    events.append(obj)
                    _log_event(run_log, obj)

        def _drain_stderr() -> None:
            assert proc.stderr is not None
            for chunk in proc.stderr:
                stderr_chunks.append(chunk)

        t_out = threading.Thread(
            target=_drain_stdout, name="devfix-stdout", daemon=True,
        )
        t_err = threading.Thread(
            target=_drain_stderr, name="devfix-stderr", daemon=True,
        )
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:  # pragma: no cover - already-dead race
                pass
            try:
                proc.wait(timeout=10)
            except Exception:  # pragma: no cover - defensive
                pass
            t_out.join(timeout=5)
            t_err.join(timeout=5)
            _log_finish(run_log, "failed", "timeout")
            raise _HelperCodeUnavailable(
                f"timeout after {timeout}s", kind="timeout",
            )

        # EOF on both pipes follows process exit; short joins are a
        # belt-and-braces bound, not an expected wait.
        t_out.join(timeout=10)
        t_err.join(timeout=10)
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:  # pragma: no cover - defensive
                pass

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_chunks).strip()
        elapsed = time.monotonic() - t0
        _trace(
            f"devfix:done rc={proc.returncode} elapsed={elapsed:.2f}s "
            f"events={len(events)} stderr_len={len(stderr)}"
        )

        if proc.returncode != 0:
            error_texts = " ".join(
                json.dumps(ev) for ev in events if ev.get("type") == "error"
            )
            blob = f"{stderr} {error_texts}".lower()
            kind = "nonzero_exit"
            if ("credits exhausted" in blob or "http 402" in blob
                    or "payment required" in blob or "status 402" in blob):
                kind = "credits_exhausted"
            elif ("rejected the api key" in blob or "http 401" in blob
                    or "unauthorized" in blob or "invalid api key" in blob):
                kind = "bad_api_key"
            elif ("rate-limited" in blob or "rate limit" in blob
                    or "http 429" in blob or "status 429" in blob):
                kind = "rate_limited"
            _log_finish(run_log, "failed", kind)
            raise _HelperCodeInvalidOutput(
                f"opencode exit {proc.returncode}",
                kind=kind,
                stderr=stderr,
                stdout=stdout,
            )

        text = _devfix_extract_report_text(events)
        if text is None:
            _log_finish(run_log, "failed", "empty_output")
            raise _HelperCodeInvalidOutput(
                "no text event in opencode output",
                kind="empty_output",
                stderr=stderr,
                stdout=stdout,
            )

        wrong_agent = _devfix_wrong_agent(events)
        if wrong_agent is not None:
            _log_finish(run_log, "failed", "missing_type")
            raise _HelperCodeInvalidOutput(
                f"opencode ran agent {wrong_agent!r} instead of "
                f"{_DEVFIX_AGENT!r} — likely silent subagent fallback "
                "(the agent file must be mode: primary)",
                kind="missing_type",
                stderr=stderr,
                stdout=stdout,
            )

        candidate = _devfix_extract_fenced_json(text)
        try:
            report = json.loads(candidate)
        except json.JSONDecodeError as exc:
            _log_finish(run_log, "failed", "non_json")
            raise _HelperCodeInvalidOutput(
                f"no parseable DeveloperReport JSON in final message: {exc.msg}",
                kind="non_json",
                stdout=text,
                stderr=stderr,
            ) from exc

        if not isinstance(report, dict) or "type" not in report:
            _log_finish(run_log, "failed", "missing_type")
            raise _HelperCodeInvalidOutput(
                "DeveloperReport is not a typed JSON object",
                kind="missing_type",
                stdout=text,
                stderr=stderr,
            )

        if report.get("status") == "resolved":
            try:
                applied = _devfix_worktree.copy_back(handle)
            except _devfix_worktree.CopyBackViolation as exc:
                _log_note(run_log, f"copy-back refused: {exc}")
                _log_finish(run_log, "failed", "copyback_violation")
                raise RuntimeError_(str(exc)) from exc
            _trace(f"devfix:copied_back {applied!r}")
            summary = f"copy-back: {len(applied)} files applied"
            if applied:
                summary += " — " + ", ".join(str(p) for p in applied)
            _log_note(run_log, summary)
        # Non-resolved statuses: nothing is copied at all.

        report["runtime_used"] = "opencode"
        report_status = report.get("status")
        if report_status in ("resolved", "rejected", "needs_context"):
            _log_finish(run_log, str(report_status))
        else:
            _log_finish(run_log, "failed", "bad_status")
        return report


def _resolve_cmd(agent: str) -> list[str] | None:
    bin_name = os.getenv("HELPER_DASHBOARD_HELPERCODE_BIN", "./bin/helpercode")
    raw_template = os.getenv("HELPER_DASHBOARD_HELPERCODE_CMD")

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
                f"HELPER_DASHBOARD_HELPERCODE_CMD is invalid ({exc}); "
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
    raw = os.getenv("HELPER_DASHBOARD_HELPERCODE_TIMEOUT_SECONDS")
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


class HelperCodeRuntime:
    """Primary entry point for invoking agents.

    Callers use `invoke_operation(operation, args, *, developer=False)`.
    The legacy `invoke_helper(agent, args)` remains as a thin
    compatibility wrapper for tests and older code paths.
    """

    def __init__(self) -> None:
        self._mock = MockHelperRuntime()
        self._real = RealHelperCodeRuntime()
        self._devfix = DevFixOpenCodeCLIRuntime()

    # --- mode selection ---------------------------------------------
    @staticmethod
    def _mode() -> str:
        return (os.getenv("HELPER_DASHBOARD_HELPERCODE") or "mock").strip().lower()

    def _backend_for(self, operation: str):
        """Real-CLI selector for the non-mock branches: developer_fix
        goes to the real OpenCode CLI iff the operator flipped
        HELPER_DASHBOARD_DEVFIX_RUNTIME=opencode-cli; everything else
        (and the default) stays on bin/helpercode."""
        if operation in DEVELOPER_OPERATIONS and _devfix_mode() == "opencode-cli":
            return self._devfix
        return self._real

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

        elif mode == "helpercode":
            try:
                result = self._backend_for(operation).invoke(operation, agent, args)
            except (_HelperCodeUnavailable, _HelperCodeInvalidOutput) as exc:
                # Strict by design — no silent fallback.
                raise RuntimeError_(
                    f"helpercode runtime failure ({exc.kind}): {exc}"
                ) from exc
            result.setdefault("runtime_used", "helpercode")

        elif mode == "auto":
            try:
                result = self._backend_for(operation).invoke(operation, agent, args)
                result.setdefault("runtime_used", "helpercode")
            except (_HelperCodeUnavailable, _HelperCodeInvalidOutput) as exc:
                reason = f"{exc.kind}:{exc}"
                _warn(f"auto mode: falling back to mock ({reason})")
                result = self._mock.invoke(agent, args, operation=operation)
                # Overwrite any inner value — the marker must be explicit.
                result["runtime_used"] = "mock_fallback"
                result["fallback_reason"] = reason

        else:
            raise RuntimeError_(
                f"unknown HELPER_DASHBOARD_HELPERCODE mode: {mode!r} "
                "(expected 'mock', 'helpercode', or 'auto')"
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
    """Deterministic, in-process stand-in for the real helperCode runtime.

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
        if operation == "author_alert_rule":
            return self._author_alert_rule(args)

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

        # Alert-rule intent: an alert keyword PLUS a threshold-ish signal.
        # Checked before the dashboard branch so "alert me when CPU is
        # above 90%" doesn't fall through to chart-building.
        alert_kw = any(t in low for t in [
            "alert", "notify me", "rule", "警告", "警報", "規則",
        ])
        threshold_kw = any(t in low for t in [
            "when", "if", "above", "below", "over", "under", "exceed",
            ">", "<", "%", "超過", "低於", "高於",
        ])
        if alert_kw and threshold_kw:
            return {
                "type": "AlertRuleIntent",
                "summary": msg[:120],
                "request_text": msg,
                "message_to_user": "Setting up that alert rule now.",
            }

        # Saved-library recall — checked before the prometheus/dashboard
        # branches so "show me my saved CPU dashboard" loads the stored
        # snapshot instead of regenerating a spec, and before the
        # generic fallback so "use my saved widget" stops dead-ending
        # in small talk. Single fuzzy match → LoadSavedIntent (the
        # orchestrator resolves + clones deterministically); zero or
        # several → ClarificationRequest listing the library.
        saved_triggers = [
            "saved", "library", "存", "載入", "叫出", "開啟我的",
            "my saved", "use my", "open my",
        ]
        if any(t in low for t in saved_triggers):
            saved = args.get("saved_dashboards") or []
            if not saved:
                return {
                    "type": "UserResponse",
                    "message": (
                        "Your dashboard library is empty — build a "
                        "dashboard and I'll offer to save it."
                    ),
                }
            matches = fuzzy_match_saved(low, saved)
            if len(matches) == 1:
                m = matches[0]
                name = str(m.get("name") or "")
                return {
                    "type": "LoadSavedIntent",
                    "saved_id": m.get("saved_id"),
                    "name_query": (name or msg)[:96],
                    "message_to_user": (
                        f"Loading {name!r} from your library."
                    ),
                }
            names = ", ".join(
                str(s.get("name") or "") for s in saved[:10]
            )
            question = (
                f"Which saved dashboard would you like? "
                f"Your library has: {names}."
            )
            return {
                "type": "ClarificationRequest",
                "questions": [question],
                "message_to_user": question,
            }

        if any(t in low for t in ["metric", "promql", "query", "prometheus"]):
            return {
                "type": "PrometheusIntent",
                "question": msg,
                "message_to_user": "Let me check your Prometheus data.",
            }

        if any(t in low for t in [
            "dashboard", "chart", "graph", "show me", "monitor", "overview",
            # Clinician phrasing — "幫我盯心率和血氧" must reach the
            # dashboard-spec path, not fall through to small talk.
            "watch", "盯", "監控", "監測", "监控", "监测", "儀表板",
            "仪表板", "看板",
        ]):
            title = _extract_title(msg)
            widget_hints = _extract_widget_hints(low)
            metric_hints, catalog_available = _resolve_metric_hints(msg)
            return {
                "type": "DashboardIntent",
                "summary": title or "New dashboard",
                "requirements": {
                    "title": title or "New dashboard",
                    "goal": msg,
                    "metrics_hints": metric_hints,
                    "widget_hints": widget_hints,
                    "refresh_interval_hint": "30s",
                    # Explicit degradation marker: when the live metric
                    # catalog could not be fetched, the spec generator
                    # says so in the dashboard description instead of
                    # silently defaulting to made-up metrics.
                    "metric_catalog_available": catalog_available,
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

    # -- author_alert_rule (system-metric rule author) ------------------------
    def _author_alert_rule(self, args: dict[str, Any]) -> dict[str, Any]:
        """Deterministic SystemAlertRule author. Mirrors the JSON contract
        in bin/helpercode; the Python validator downstream is authoritative."""
        intent = args.get("intent") or {}
        text = str(
            intent.get("request_text") or intent.get("summary") or ""
        ).lower()

        kind = "cpu_utilization_pct"
        if "disk" in text and ("io" in text or "i/o" in text):
            kind = "disk_io_utilization_pct"
        elif "disk" in text or "space" in text or "磁碟" in text:
            kind = "disk_usage_pct"
        elif "memory" in text or "ram" in text or "記憶體" in text:
            kind = "memory_used_pct"
        elif "load" in text:
            kind = "load1"

        comparator = "<" if any(
            t in text for t in ["below", "under", "低於", "less than"]
        ) else ">"

        threshold = 90.0
        m = re.search(
            r"(?:above|below|over|under|exceeds?|超過|低於|高於|>|<)\s*"
            r"(\d+(?:\.\d+)?)",
            text,
        ) or re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if m:
            threshold = float(m.group(1))

        minutes = 5
        fm = re.search(r"(\d+)\s*(?:minutes?|min\b|分鐘)", text)
        if fm:
            minutes = max(1, min(60, int(fm.group(1))))

        return {
            "type": "SystemAlertRule",
            "spec": {
                "id": f"sys-{kind.replace('_', '-')}-{uuid.uuid4().hex[:6]}",
                "title": f"System alert: {kind} {comparator} {threshold:g}",
                "metric_kind": kind,
                "comparator": comparator,
                "threshold": threshold,
                "for": f"{minutes}m",
                "severity": "warning",
                "mode": "shadow",
                "created_by": "helper-chat",
            },
        }

    # -- dashboard-spec-agent -----------------------------------------------
    def _dashboard_spec(self, args: dict[str, Any]) -> dict[str, Any]:
        intent = args.get("intent") or {}
        req = intent.get("requirements") or {}
        title = req.get("title") or intent.get("summary") or "New dashboard"
        metric_hints = req.get("metrics_hints") or []
        widget_hints = req.get("widget_hints") or []
        goal = (req.get("goal") or "").lower()
        catalog_available = req.get("metric_catalog_available", True)
        dashboard_id = _slugify(title) or f"dash-{uuid.uuid4().hex[:8]}"

        # When the classifier passed no metric hints (or the operation
        # was invoked directly), try resolving them from the goal text
        # against the medical vocabulary + live catalog.
        if not metric_hints and goal:
            metric_hints, catalog_available = _resolve_metric_hints(goal)

        # If the user's intent looks like an API observability dashboard,
        # use the rich golden composition.
        if _looks_like_api_observability(goal, metric_hints, title):
            widgets = _api_observability_widgets()
        elif not metric_hints:
            # No silent defaults: an explicit, clearly-labeled
            # placeholder instead of fabricated http_requests_total
            # charts on a clinical dashboard.
            widgets = [
                _no_metric_placeholder_widget(
                    catalog_available=catalog_available
                )
            ]
        else:
            widgets = _build_default_widgets(widget_hints, metric_hints)

        description = req.get("goal") or ""
        if not catalog_available:
            note = (
                "[metric catalog unavailable — Prometheus could not be "
                "reached at generation time, so no metric hints were "
                "resolved]"
            )
            description = f"{description} {note}".strip()

        spec = {
            "dashboard_id": dashboard_id,
            "title": title,
            "description": description[:1024],
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
        # Real suggestions from the live catalog (medical synonyms +
        # catalog matches) — never a canned list. Catalog failure is
        # reported explicitly, never papered over.
        try:
            detailed = prom_catalog.get_catalog().suggest_detailed(
                question, limit=8
            )
        except PrometheusUnavailable as exc:
            return {
                "type": "PrometheusQueryReport",
                "report": {
                    "queries": [],
                    "metric_suggestions": [],
                    "notes": f"metric catalog unavailable: {exc}",
                },
            }
        suggestions = [
            {
                "metric": s.metric,
                "reason": s.reason if s.present else (
                    f"{s.reason} (not currently present — "
                    "no active stream)"
                ),
            }
            for s in detailed
        ]
        return {
            "type": "PrometheusQueryReport",
            "report": {
                "queries": [],
                "metric_suggestions": suggestions,
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


# System-metric vocabulary (kept tiny). Unlike the medical synonyms,
# every candidate here is VERIFIED against the live catalog before it
# becomes a hint — system metrics that aren't scraped on this box must
# not produce dead widgets.
_SYSTEM_METRIC_SYNONYMS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("http", "request rate", "rps", "traffic"),
     ("http_requests_total",)),
    (("latency", "p95", "p99", "duration"),
     ("http_request_duration_seconds",
      "http_request_duration_seconds_bucket")),
    (("error rate", "5xx", "error_rate"), ("http_requests_total",)),
    (("cpu",), ("process_cpu_seconds_total", "node_cpu_seconds_total")),
    (("memory", "ram"),
     ("process_resident_memory_bytes", "node_memory_MemAvailable_bytes")),
    (("alert",), ("ALERTS",)),
    (("up", "availability", "service health", "health"), ("up",)),
)


def _resolve_metric_hints(msg: str) -> tuple[list[str], bool]:
    """Resolve free text to real metric names via the live catalog.

    Returns ``(hints, catalog_available)``.

    - Medical synonyms (心率/spo2/呼吸器/...) resolve to `patient_*` /
      `ventilator_*` / `cerebral_*` names and are kept even when absent
      from the live catalog — those series vanish whenever no camera
      stream runs, yet they are still the right answer.
    - System synonyms (cpu/memory/latency/http/...) are kept only when
      the metric actually exists in the live catalog.
    - Exact metric names typed by the user are honored when present.
    - Catalog unreachable → ``([], False)``: the caller must surface an
      explicit "catalog unavailable" note, never silent defaults.
    """
    try:
        live = prom_catalog.get_catalog().list_metrics()
    except PrometheusUnavailable:
        return [], False
    live_set = set(live)
    low = msg.lower()

    hints: list[str] = list(prom_catalog.resolve_synonyms(msg))
    for keys, candidates in _SYSTEM_METRIC_SYNONYMS:
        if any(k in low for k in keys):
            hints.extend(m for m in candidates if m in live_set)
    # Exact metric-name mentions (e.g. "watch node_load1").
    for token in re.findall(r"[A-Za-z_:][A-Za-z0-9_:]{2,}", msg):
        if token in live_set:
            hints.append(token)
    return list(dict.fromkeys(hints)), True


def _extract_metric_hints(msg: str) -> list[str]:
    """Catalog-backed hint extraction (names only; see _resolve_metric_hints)."""
    hints, _available = _resolve_metric_hints(msg)
    return hints


def _default_promql_for(widget_type: str) -> str:
    """Metric-less default query for a widget type.

    The old http_requests_total defaults are gone: with no metric
    context, chart-like widgets get an explicit `vector(0)` placeholder
    (callers MUST label it — see `_no_metric_placeholder_widget`)
    instead of a fabricated system metric on a clinical dashboard.
    """
    return {
        "stat_card": 'sum(up)',
        "gauge": 'avg(up)',
        "alert_list": 'ALERTS{alertstate="firing"}',
    }.get(widget_type, "vector(0)")


_NO_METRIC_NOTE = (
    "Placeholder — no metric specified; this intentionally reads 0. "
    "Edit the query to point at a real metric."
)


def _default_widget(
    wid: str, wtype: str, *, title: str, promql: str, description: str = "",
) -> dict[str, Any]:
    query_type = "instant" if wtype in {"stat_card", "gauge", "alert_list", "table"} else "range"
    query = {
        "source": "prometheus",
        "promql": promql,
        "query_type": query_type,
    }
    if query_type == "range":
        query["range"] = "1h"
        query["step"] = "30s"
    if not description and promql == "vector(0)":
        # A vector(0) placeholder must never render unlabeled.
        description = _NO_METRIC_NOTE
    return {
        "id": wid,
        "type": wtype,
        "title": title,
        "description": description[:512],
        "query": query,
        "position": {"x": 0, "y": 0, "w": 6, "h": 6},
        "encoding": {},
        "thresholds": [],
        "options": {},
    }


def _no_metric_placeholder_widget(
    *, catalog_available: bool,
) -> dict[str, Any]:
    """Explicit, clearly-labeled stand-in when zero metrics resolved.

    Never a silent http_requests_total default: the widget says exactly
    why it shows 0 so the clinician knows nothing real is being watched.
    """
    if catalog_available:
        description = (
            "No matching metric was found in the live Prometheus catalog "
            "for this request. This placeholder intentionally reads 0 — "
            "rephrase with a known vital (e.g. 心率 / SpO2 / 呼吸) or a "
            "real metric name."
        )
    else:
        description = (
            "The metric catalog was unavailable (Prometheus unreachable) "
            "while generating this dashboard, so no metric could be "
            "matched. This placeholder intentionally reads 0 — retry "
            "once Prometheus is back."
        )
    return _default_widget(
        "w-no-metric-match",
        "stat_card",
        title="No matching metric",
        promql="vector(0)",
        description=description,
    )


def _promql_for_metric(metric: str, wtype_name: str) -> str:
    """Shape a widget query for one resolved metric.

    Clinical gauges (patient_*/ventilator_*/cerebral_*) MUST aggregate
    `max by (patient_id)`: the `device_model` label is free-text LLM
    output with MULTIPLE variants per physical device, so one patient's
    vital appears as N parallel series — sum would multiply the value
    (observed live: HR 163 rendered as 814 after sum-collapse). `max`
    over duplicates of the same reading is the honest collapse.
    """
    if prom_catalog.is_clinical_metric(metric):
        return f"max by (patient_id) ({metric})"
    if metric.endswith("_total"):
        # Counters keep rate().
        rate = f"rate({metric}[5m])"
        if wtype_name in {"stat_card", "gauge"}:
            return f"sum({rate})"
        if wtype_name == "table":
            return f"topk(10, {rate})"
        return rate
    # Non-counter system metrics keep their existing shapes.
    if wtype_name in {"stat_card", "gauge"}:
        return f"avg({metric})"
    if wtype_name == "table":
        return f"topk(10, {metric})"
    return metric


def _build_default_widgets(widget_hints: list[str], metric_hints: list[str]) -> list[dict[str, Any]]:
    widgets: list[dict[str, Any]] = []
    x = 0
    y = 0
    idx = 0
    wants = [
        w for w in dict.fromkeys(widget_hints)
        if w in {t.value for t in WidgetType}
    ] or ["line_chart", "stat_card"]
    metrics = list(dict.fromkeys(metric_hints))
    # Every resolved metric gets a widget (cycling through the wanted
    # widget types) — "盯心率和血氧" must chart BOTH vitals even when
    # only one widget type was hinted.
    slots = max(len(wants), len(metrics)) if metrics else len(wants)
    for slot in range(slots):
        wtype_name = wants[slot % len(wants)]
        metric = metrics[slot % len(metrics)] if metrics else None
        if metric:
            promql = _promql_for_metric(metric, wtype_name)
        else:
            promql = _default_promql_for(wtype_name)
        w = 6 if wtype_name in {"line_chart", "table"} else 3
        h = 6 if wtype_name in {"line_chart", "table"} else 4
        if x + w > 12:
            x = 0
            y += h
        widget = _default_widget(
            f"w-{idx}-{_slugify(metric) if metric else wtype_name}",
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


# Base metric behind each golden widget — used to annotate widgets whose
# metric is absent from the live catalog (annotate, never fabricate).
_API_OBS_WIDGET_METRIC: dict[str, str] = {
    "w-request-rate": "http_requests_total",
    "w-error-rate": "http_requests_total",
    "w-latency-p95": "http_request_duration_seconds_bucket",
    "w-latency-p99": "http_request_duration_seconds_bucket",
    "w-availability": "up",
    "w-cpu": "process_cpu_seconds_total",
    "w-memory": "process_resident_memory_bytes",
    "w-alerts": "ALERTS",
}


def _api_observability_widgets() -> list[dict[str, Any]]:
    """Return the 8-widget golden API observability composition.

    Mirrors `backend/app/samples/api_observability.json` so the
    heuristic mock and the Real LLM both converge on a similar
    structure for this scenario.

    When the live catalog is reachable, widgets whose base metric does
    not exist on this box are annotated "(metric not present)" — the
    composition keeps its shape, but nobody mistakes a dead panel for
    a monitored one. Catalog unreachable → no annotation (unknown is
    not the same as absent).
    """
    widgets = _api_observability_widget_list()
    try:
        live = set(prom_catalog.get_catalog().list_metrics())
    except PrometheusUnavailable:
        return widgets
    for w in widgets:
        metric = _API_OBS_WIDGET_METRIC.get(w["id"])
        if metric and metric not in live:
            base = w.get("description") or ""
            w["description"] = (
                f"{base} (metric not present)".strip()[:512]
            )
    return widgets


def _api_observability_widget_list() -> list[dict[str, Any]]:
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
