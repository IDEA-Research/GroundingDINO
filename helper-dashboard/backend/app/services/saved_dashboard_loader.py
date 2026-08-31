"""Load a saved dashboard into the working store — the SINGLE load surface.

Both `POST /api/saved_dashboards/{saved_id}/load` (the library button)
and the chat orchestrator's LoadSavedIntent dispatch resolve through
`load_saved_dashboard`, so the clone semantics stay identical on every
path: the SavedDashboard's spec snapshot is deep-cloned with a FRESH
dashboard_id (specs/saved_dashboard.py designed loads to clone — never
reuse the stored id, because dashboard-id reuse by the LLM is common
and a later chat patch would silently apply to a different spec that
happens to share the id), the clone is persisted via
`DashboardStore.save_dashboard` so `/api/dashboard/{id}` and chat
patches resolve against exactly what the user is looking at, and the
entry's `used_count` is bumped.
"""

from __future__ import annotations

import re
import uuid

from ..specs import DashboardSpec, SavedDashboard
from .dashboard_store import DashboardStore
from .saved_dashboard_store import SavedDashboardStore


def fresh_dashboard_id(base: str) -> str:
    """New unique dashboard_id for a clone.

    Follows the codebase id convention (readable slug + short uuid —
    compare ``dash-{uuid4().hex[:8]}`` in the runtimes and the saved
    store's collision suffix): keep the original id as the readable
    base and append a uuid suffix so the clone never collides with the
    original or with earlier clones. The result always matches
    ``[a-zA-Z0-9_-]+`` and stays within DashboardSpec's 64-char limit.
    """
    base = re.sub(r"[^a-zA-Z0-9_\-]+", "-", base or "").strip("-") or "dash"
    return f"{base[:55]}-{uuid.uuid4().hex[:8]}"


def load_saved_dashboard(
    entry: SavedDashboard,
    *,
    dashboards: DashboardStore,
    saved: SavedDashboardStore,
) -> DashboardSpec:
    """Clone ``entry.spec`` under a fresh dashboard_id, persist the
    clone in the working dashboard store, bump the library entry's
    used_count, and return the persisted clone."""
    clone = entry.spec.model_copy(
        deep=True,
        update={"dashboard_id": fresh_dashboard_id(entry.spec.dashboard_id)},
    )
    dashboards.save_dashboard(clone)
    saved.bump_used(entry.saved_id)
    return clone


_CJK_RE = re.compile("[㐀-鿿]")


def fuzzy_match_saved(text: str, summaries: list[dict]) -> list[dict]:
    """Match free text against saved-dashboard summaries (dict shape of
    SavedDashboardSummary: saved_id / name / tags / ...).

    Two tiers, deterministic:

    1. Strong: the entry's full name, saved_id, or any tag appears in
       the text. If any entry matches strongly, only strong matches
       are returned.
    2. Weak: any significant token of the entry's name (>= 3 chars, or
       >= 2 for CJK) appears in the text.

    Returns the matching summary dicts (possibly empty or several —
    callers treat exactly-one as resolvable and anything else as a
    clarification)."""
    low = (text or "").lower()
    strong: list[dict] = []
    weak: list[dict] = []
    for s in summaries:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name") or "").strip().lower()
        saved_id = str(s.get("saved_id") or "").strip().lower()
        tags = [str(t).strip().lower() for t in (s.get("tags") or [])]
        if (
            (name and name in low)
            or (saved_id and saved_id in low)
            or any(t and t in low for t in tags)
        ):
            strong.append(s)
            continue
        tokens = [
            t
            for t in re.split("[^a-z0-9㐀-鿿]+", name)
            if len(t) >= (2 if _CJK_RE.search(t) else 3)
        ]
        if tokens and any(t in low for t in tokens):
            weak.append(s)
    return strong if strong else weak
