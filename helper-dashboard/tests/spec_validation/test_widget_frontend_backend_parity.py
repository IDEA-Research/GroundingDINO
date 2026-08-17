"""Cross-language parity: backend `WidgetType` enum vs the frontend mirror.

Why this file exists: `backend/app/specs/widget_schema_doc.py` generates the
LLM-facing widget schema straight from the Python `WidgetType` enum, so the
*backend* side is guaranteed self-consistent. But `frontend/lib/spec-schema.ts`
is a hand-maintained mirror of that enum (its own header comment says so —
"Adding a new widget type is a Big guy task that must update both sides") and
nothing generates it from Python. The `rescue_extend` six-file atomic
snapshot (`helper/extend_snapshot.py::_allowed_paths_for`) guarantees all six
files are written together or not at all, but never checks that their
*content* agrees — a widget type could validate on the backend and render as
nothing (or fail to typecheck) on the frontend without any existing test
catching it. (The `test_extend_<type>.py` files' docstrings claim to
"validate frontend schema" registration but only ever import from the
backend — they do not.)

These tests close that gap with plain-text extraction (not a real TS
parser) against the small, fixed set of widget-toolkit files. They are
intentionally simple: fragile to a real reformat of spec-schema.ts /
renderer.tsx, but cheap, dependency-free, and run on every `pytest tests/`.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.helper.extend_snapshot import _camel
from app.specs.widget_spec import WidgetType

# tests/spec_validation/test_widget_frontend_backend_parity.py ->
# parents[2] is the helper-dashboard/ root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC_SCHEMA_TS = _REPO_ROOT / "frontend" / "lib" / "spec-schema.ts"
_RENDERER_TSX = _REPO_ROOT / "frontend" / "lib" / "renderer.tsx"
_WIDGET_TOOLKIT_DIR = _REPO_ROOT / "frontend" / "widget-toolkit"

_TS_STRING_RE = re.compile(r'"([a-z][a-z0-9_]*)"')


def _backend_widget_types() -> set[str]:
    return {wt.value for wt in WidgetType}


def _read_frontend_file(path: Path) -> str:
    # frontend/lib/ is gitignored by the parent repo's `lib/` pattern
    # (docs/agent-ops/LESSONS.md, 2026-07-03) — disk, not git, is ground
    # truth. If this assertion ever fires on a fresh clone, that gitignore
    # trap (not a missing file) is almost certainly the real cause.
    assert path.is_file(), (
        f"{path} not found on disk. frontend/lib/ is gitignored by the "
        "parent repo's `lib/` pattern — check the filesystem directly, "
        "never `git log` / `git show`."
    )
    return path.read_text(encoding="utf-8")


def _extract_between(text: str, start_marker: str, end_char: str) -> str:
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_char, start)
    return text[start:end]


# ---------------------------------------------------------------------------
# frontend/lib/spec-schema.ts
# ---------------------------------------------------------------------------


def test_widget_type_union_matches_backend_enum():
    """`export type WidgetType = | "a" | "b" | ...;` must list exactly the
    same values as the backend `WidgetType` enum. A value missing here means
    the frontend cannot even typecheck a spec the backend already accepts;
    an extra value here means the frontend believes in a type the backend
    validator will reject on sight."""
    text = _read_frontend_file(_SPEC_SCHEMA_TS)
    block = _extract_between(text, "export type WidgetType =", ";")
    ts_types = set(_TS_STRING_RE.findall(block))
    backend_types = _backend_widget_types()
    assert ts_types == backend_types, (
        "frontend/lib/spec-schema.ts `WidgetType` union has drifted from "
        "backend/app/specs/widget_spec.py `WidgetType` enum.\n"
        f"  backend has, TS union missing: {sorted(backend_types - ts_types)}\n"
        f"  TS union has, backend missing: {sorted(ts_types - backend_types)}"
    )


def test_widget_types_const_matches_backend_enum():
    """`export const WIDGET_TYPES: WidgetType[] = [...]` is the RUNTIME
    array `renderer.tsx`'s `isSupportedType()` narrows against. It is a
    second, independent literal — not derived from the union type above —
    so it can drift even if the union stays correct."""
    text = _read_frontend_file(_SPEC_SCHEMA_TS)
    block = _extract_between(
        text, "export const WIDGET_TYPES: WidgetType[] = [", "]"
    )
    ts_types = set(_TS_STRING_RE.findall(block))
    backend_types = _backend_widget_types()
    assert ts_types == backend_types, (
        "frontend/lib/spec-schema.ts `WIDGET_TYPES` runtime array has "
        "drifted from backend/app/specs/widget_spec.py `WidgetType` enum.\n"
        f"  backend has, array missing: {sorted(backend_types - ts_types)}\n"
        f"  array has, backend missing: {sorted(ts_types - backend_types)}"
    )


# ---------------------------------------------------------------------------
# frontend/lib/renderer.tsx
# ---------------------------------------------------------------------------


def test_renderer_switch_covers_every_widget_type():
    """`renderWidget()`'s closed switch must have a `case "<type>":` for
    every backend `WidgetType`. Falling through to `default` is deliberate
    defense-in-depth for a type the validator already rejected (per the
    comment right there in renderer.tsx) — but a type the backend VALIDATES
    and the switch has no case for will silently render nothing in
    production, which is a product bug, not defense-in-depth."""
    text = _read_frontend_file(_RENDERER_TSX)
    missing = sorted(
        wt for wt in _backend_widget_types() if f'case "{wt}"' not in text
    )
    assert not missing, (
        f"renderer.tsx has no `case` for: {missing} — these WidgetType "
        "values pass backend validation but always render as null."
    )


# ---------------------------------------------------------------------------
# frontend/widget-toolkit/<Camel>Widget.tsx
# ---------------------------------------------------------------------------


def test_every_widget_type_has_a_toolkit_component():
    """Every backend `WidgetType` must have a matching
    `frontend/widget-toolkit/<Camel>Widget.tsx` file, using the exact same
    snake_case -> CamelCase convention `extend_snapshot._camel` uses to
    decide what `rescue_extend` is allowed to write — imported directly so
    this test can never silently drift from that convention itself."""
    missing = []
    for wt in sorted(_backend_widget_types()):
        component = _WIDGET_TOOLKIT_DIR / f"{_camel(wt)}Widget.tsx"
        if not component.is_file():
            missing.append(f"{wt} -> {component.name}")
    assert not missing, (
        "backend WidgetType values with no frontend widget-toolkit "
        "component:\n  " + "\n  ".join(missing)
    )
