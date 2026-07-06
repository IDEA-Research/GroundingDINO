"""Make `app` importable as `helper_dashboard.app` from the test root."""

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


@pytest.fixture(autouse=True)
def _no_system_rules_loop(monkeypatch, tmp_path):
    """The system-rule wall-clock loop defaults ON in production (it only
    reads Prometheus and records shadow events). Tests must not tick
    against a live Prometheus or write into the real storage dirs, so the
    loop is disabled and the stores are pointed at tmp_path; tests that
    exercise the service opt in explicitly."""
    monkeypatch.setenv("SYSTEM_RULES_ENABLED", "0")
    monkeypatch.setenv("SYSTEM_RULES_DIR", str(tmp_path / "system_rules"))
    monkeypatch.setenv(
        "SYSTEM_ALERT_STATE_DIR", str(tmp_path / "system_alert_state")
    )
    from app.services import system_rules
    system_rules.reset_service_for_tests()
    yield
    system_rules.reset_service_for_tests()


@pytest.fixture(autouse=True)
def _no_background_auto_fix(monkeypatch):
    """Auto-fix spawns background developer_fix threads and writes real
    audit files under backend/app/storage/auto_fix_audit/. Tests that
    exercise it opt in explicitly (tests/auto_fix sets the env back to
    1 and redirects the audit dir); everything else runs with the kill
    switch on so opencode-mode tests don't fire the pipeline."""
    monkeypatch.setenv("HELPER_DASHBOARD_AUTO_FIX", "0")
