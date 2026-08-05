"""Session + helper memory.

Simple in-memory session store keyed by `session_id`, backed by a
rolling append to `docs/USER_SESSION_LOG.md`. The memory file is the
only persistent user-facing history.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DOCS = Path(__file__).resolve().parent.parent.parent.parent / "docs"
_SESSION_LOG = _DOCS / "USER_SESSION_LOG.md"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


class SessionMemory:
    def __init__(self) -> None:
        self._per_session: dict[str, dict[str, Any]] = defaultdict(dict)

    # ------------------------------------------------------------
    def get_current_dashboard_id(self, session_id: str) -> str | None:
        return self._per_session.get(session_id, {}).get("current_dashboard_id")

    def set_current_dashboard_id(self, session_id: str, dashboard_id: str) -> None:
        self._per_session[session_id]["current_dashboard_id"] = dashboard_id

    def record(
        self,
        session_id: str,
        *,
        action: str,
        dashboard_id: str | None,
        summary: str,
        preferences: dict[str, Any] | None = None,
    ) -> None:
        entry = (
            f"### {_now()} — session {session_id}\n\n"
            f"- action: {action}\n"
            f"- dashboard_id: {dashboard_id or ''}\n"
            f"- summary: {summary[:240]}\n"
        )
        if preferences:
            entry += f"- preferences: {preferences}\n"
        entry += "\n"
        try:
            _SESSION_LOG.parent.mkdir(parents=True, exist_ok=True)
            with _SESSION_LOG.open("a", encoding="utf-8") as fh:
                fh.write(entry)
        except OSError:
            # Never break the request over a log write failure.
            pass
