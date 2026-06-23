"""Extend-request parsing.

Specialist agents (`dashboard-spec-agent`, `patch-agent`) emit a
`DeveloperTicket` when the user asks for a widget type that is not in
the toolkit. The ticket's `requested_action` follows a fixed string
pattern that downstream code parses to recognize "this is a request to
extend the widget toolkit, not a generic bug report".

Pattern (case-insensitive on the leading verb, widget_type is strict
snake_case):

    extend widget toolkit with <widget_type>

This module is the single source of truth for that pattern. M5 will
add a denylist of forbidden widget_type values (script, iframe, ...)
and a regex-strictness check; for M1 the parser only extracts the
candidate name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# Lowercase snake_case, 3-32 chars, must start with a letter.
_WIDGET_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{2,31}$")

# The full requested_action pattern. Tolerant of leading/trailing
# whitespace and case on the verb, but strict on the widget_type slot.
_EXTEND_RE = re.compile(
    r"^\s*extend\s+widget\s+toolkit\s+with\s+([A-Za-z0-9_]+)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExtendRequest:
    """A parsed extend-toolkit request derived from a DeveloperTicket."""

    widget_type: str
    source_agent: str
    summary: str
    user_visible_effect: str


def parse_extend_ticket(raw: dict[str, Any] | None) -> ExtendRequest | None:
    """Recognize a DeveloperTicket-shaped dict as an extend-toolkit
    request. Returns the parsed `ExtendRequest` or `None` if the
    `requested_action` does not match the canonical pattern (or the
    candidate widget_type fails the naming rule).

    The function is intentionally pure: it does not check denylists,
    quotas, or env flags — those are the caller's job (added in M5).
    """
    if not isinstance(raw, dict):
        return None
    requested = raw.get("requested_action")
    if not isinstance(requested, str):
        return None

    m = _EXTEND_RE.match(requested)
    if not m:
        return None

    candidate = m.group(1).lower()
    if not _WIDGET_TYPE_RE.match(candidate):
        return None

    return ExtendRequest(
        widget_type=candidate,
        source_agent=str(raw.get("source_agent") or "")[:64],
        summary=str(raw.get("summary") or "")[:256],
        user_visible_effect=str(raw.get("user_visible_effect") or "")[:512],
    )
