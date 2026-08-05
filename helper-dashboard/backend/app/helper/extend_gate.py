"""Safety gates for the auto-extend pipeline.

The widget toolkit is a **cache of pre-built widgets** so the agent can
skip code-gen for common types — it is NOT a fence on what the agent
is allowed to build. When the user asks for a widget type that isn't
cached, `rescue_extend` runs by default and writes the new widget.

These gates protect that default path against unsafe inputs. Each
layer fails closed: if the check raises or returns False, the
extension is declined and the caller surfaces an honest failure
message.

  1. widget_type rules  denylist + length + regex
  2. User-msg safety    simple prompt-injection signature check
  3. Daily quota        N extends per day per process (default 50)
  4. Audit log          every decision (allowed or refused) is recorded
                         to backend/app/storage/extend_audit/<date>.jsonl

The gates are deliberately conservative on the *safety* axis — false
negatives are fine (user gets a clarify message), false positives are
not (LLM might edit something dangerous). They are NOT a policy
switch: extend runs by default, no opt-in required.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Layer 1: widget_type rules
# ---------------------------------------------------------------------------


# Names we never let through, even if the regex matches and an
# operator approved them. These are token-shaped equivalents of
# "shell injection" — the LLM choosing one of these is a strong
# signal of either a misunderstanding or a manipulation attempt.
_WIDGET_TYPE_DENYLIST: frozenset[str] = frozenset({
    "script", "iframe", "eval", "exec", "system", "shell",
    "rawhtml", "raw_html", "command", "subprocess",
    "import_widget", "remote_widget", "fetch_widget",
    "include", "require", "process",
})

# The Pydantic ExtendRequest already enforces this regex, but the gate
# re-checks because by the time we get here we've crossed an LLM
# boundary and shouldn't trust prior validation.
_WIDGET_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{2,31}$")


def check_widget_type(widget_type: str) -> tuple[bool, str | None]:
    """Returns (allowed, reason_if_not).

    Reasons are user-readable but designed for the audit log, not for
    the end user (who gets a generic "couldn't finish" message).
    """
    if not isinstance(widget_type, str):
        return False, "widget_type is not a string"
    if not _WIDGET_TYPE_RE.match(widget_type):
        return False, f"widget_type {widget_type!r} fails regex"
    if widget_type in _WIDGET_TYPE_DENYLIST:
        return False, f"widget_type {widget_type!r} in denylist"
    # Token-level subwords too — e.g. someone trying "myscriptchart"
    # to dodge the exact-match denylist.
    lowered = widget_type.lower()
    for bad in _WIDGET_TYPE_DENYLIST:
        if bad in lowered.replace("_", ""):
            return False, f"widget_type {widget_type!r} contains denylisted substring {bad!r}"
    return True, None


# ---------------------------------------------------------------------------
# Layer 2: user-message safety
# ---------------------------------------------------------------------------


# Conservative signature list. We are not trying to catch every
# possible jailbreak (that's an unwinnable arms race) — just the
# common patterns that have a high false-positive cost for an extend
# decision. Each pattern is matched case-insensitive on the user's
# original message.
_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.IGNORECASE),
    # Match "you are now [optional words] agent" — the LLM-roleplay
    # injection. Up to 40 chars between "now" and "agent" so multi-word
    # role names ("code editor agent") are caught.
    re.compile(r"you\s+are\s+now\s+(?:\w+\s+){0,5}agent\b",
                re.IGNORECASE),
    re.compile(r"(?:system|developer)\s*prompt", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your|the)\s+(?:prompt|instructions|system)",
                re.IGNORECASE),
    re.compile(r"bypass\s+(?:the\s+|all\s+|any\s+)?(?:safety|validation|allow[- ]?list|check)",
                re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"</?system>", re.IGNORECASE),
    re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
    re.compile(r"rm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bexec\s*\(", re.IGNORECASE),
    re.compile(r"\bos\.system\s*\(", re.IGNORECASE),
    re.compile(r"\bsubprocess\.(?:run|call|Popen)", re.IGNORECASE),
)


def check_user_message_safety(message: str) -> tuple[bool, str | None]:
    """Returns (safe, reason_if_not)."""
    if not isinstance(message, str):
        return True, None  # nothing to check; gate doesn't fail on empty
    for pat in _PROMPT_INJECTION_PATTERNS:
        m = pat.search(message)
        if m:
            return False, f"user message matched injection pattern {pat.pattern!r}"
    return True, None


# ---------------------------------------------------------------------------
# Layer 3: daily quota (per-process; in-memory counter)
# ---------------------------------------------------------------------------


_QUOTA_DEFAULT = 50
_QUOTA_LOCK = threading.Lock()
_QUOTA_STATE: dict[str, int] = {}


def _quota_max() -> int:
    raw = os.getenv("HELPER_DASHBOARD_AUTO_EXTEND_DAILY_QUOTA", str(_QUOTA_DEFAULT))
    try:
        n = int(raw)
    except ValueError:
        return _QUOTA_DEFAULT
    return max(0, min(n, 1000))  # hard upper bound


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def check_quota_available() -> tuple[bool, str | None]:
    """Returns (available, reason_if_not) without consuming."""
    with _QUOTA_LOCK:
        used = _QUOTA_STATE.get(_today_key(), 0)
    cap = _quota_max()
    if used >= cap:
        return False, f"daily quota exhausted: {used}/{cap}"
    return True, None


def consume_quota() -> None:
    """Increment today's counter. Caller must have just passed
    `check_quota_available`. Idempotent only at the day boundary."""
    with _QUOTA_LOCK:
        key = _today_key()
        _QUOTA_STATE[key] = _QUOTA_STATE.get(key, 0) + 1


def reset_quota_for_tests() -> None:
    """Clear the in-memory counter. Tests use this to make assertions
    deterministic; production callers should never need it."""
    with _QUOTA_LOCK:
        _QUOTA_STATE.clear()


# ---------------------------------------------------------------------------
# Layer 4: audit log
# ---------------------------------------------------------------------------


_AUDIT_DIR = (
    Path(__file__).resolve().parent.parent / "storage" / "extend_audit"
)
_AUDIT_LOCK = threading.Lock()


@dataclass
class GateDecision:
    """The outcome of running an ExtendRequest through every gate."""

    allowed: bool
    refusal_reason: str | None = None
    widget_type: str = ""
    user_message_excerpt: str = ""
    layer: str | None = None  # which layer refused, if any


def _audit_path() -> Path:
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    return _AUDIT_DIR / f"{_today_key()}.jsonl"


def write_audit_entry(
    decision: GateDecision,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one structured line to today's audit file. Never raises;
    audit failures must not break the extend flow."""
    entry: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "allowed": decision.allowed,
        "widget_type": decision.widget_type,
        "user_message_excerpt": decision.user_message_excerpt[:240],
        "layer": decision.layer,
        "refusal_reason": decision.refusal_reason,
    }
    if extra:
        # Only allowlisted extra keys, to avoid accidental secrets.
        for k in ("status", "actions_taken", "tests_run", "report_id",
                  "error", "duration_ms"):
            if k in extra:
                entry[k] = extra[k]
    try:
        with _AUDIT_LOCK:
            with _audit_path().open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception:
        # Never block extend on a log write failure.
        pass


# ---------------------------------------------------------------------------
# Composite: run every gate
# ---------------------------------------------------------------------------


def evaluate_gates(
    *, widget_type: str, user_message: str = "",
) -> GateDecision:
    """Run all safety gate layers in order. The first refusal short-
    circuits and records why. Quota is NOT consumed by this function —
    the caller decides when to call `consume_quota`.
    """
    excerpt = (user_message or "")[:240]

    ok, why = check_widget_type(widget_type)
    if not ok:
        d = GateDecision(
            allowed=False, layer="widget_type_rules",
            refusal_reason=why,
            widget_type=widget_type, user_message_excerpt=excerpt,
        )
        write_audit_entry(d)
        return d

    ok, why = check_user_message_safety(user_message)
    if not ok:
        d = GateDecision(
            allowed=False, layer="user_message_safety",
            refusal_reason=why,
            widget_type=widget_type, user_message_excerpt=excerpt,
        )
        write_audit_entry(d)
        return d

    ok, why = check_quota_available()
    if not ok:
        d = GateDecision(
            allowed=False, layer="quota",
            refusal_reason=why,
            widget_type=widget_type, user_message_excerpt=excerpt,
        )
        write_audit_entry(d)
        return d

    d = GateDecision(
        allowed=True,
        widget_type=widget_type,
        user_message_excerpt=excerpt,
    )
    # Don't write the "allowed" audit yet — the runner writes the final
    # outcome (with status / actions_taken) after the LLM finishes.
    return d
