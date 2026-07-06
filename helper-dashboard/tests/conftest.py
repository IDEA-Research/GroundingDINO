"""Make `app` importable as `helper_dashboard.app` from the test root."""

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


@pytest.fixture(autouse=True)
def _no_background_auto_fix(monkeypatch):
    """Auto-fix spawns background developer_fix threads and writes real
    audit files under backend/app/storage/auto_fix_audit/. Tests that
    exercise it opt in explicitly (tests/auto_fix sets the env back to
    1 and redirects the audit dir); everything else runs with the kill
    switch on so opencode-mode tests don't fire the pipeline."""
    monkeypatch.setenv("HELPER_DASHBOARD_AUTO_FIX", "0")
