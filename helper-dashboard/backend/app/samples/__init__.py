"""Golden sample DashboardSpec JSON files.

These are reference dashboards — used for documentation, demos, and
Tier 1/2 fixtures. Load with `load_sample(name)`.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..specs import DashboardSpec


_DIR = Path(__file__).resolve().parent


def load_sample_raw(name: str) -> dict:
    """Return the raw JSON dict of a sample, without Pydantic parsing."""
    p = _DIR / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"sample {name!r} not found at {p}")
    return json.loads(p.read_text())


def load_sample(name: str) -> DashboardSpec:
    """Return a validated DashboardSpec for the sample."""
    return DashboardSpec.model_validate(load_sample_raw(name))


def list_samples() -> list[str]:
    return sorted(p.stem for p in _DIR.glob("*.json"))
