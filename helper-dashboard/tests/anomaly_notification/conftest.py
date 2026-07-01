"""Isolate build-audit + alert-state + notification dirs for the INC3 suite.

The dispatcher and notifier emit build-audit records via
`anomaly_build_audit.append()` (resolves its dir from
`ANOMALY_BUILD_AUDIT_DIR`) and the local-file receiver writes under
`storage/anomaly_notifications`. Point them all at a temp dir so this suite
never writes into the real storage tree, and — critically — clear
`ANOMALY_TEST_WEBHOOK_URL` so no test can ever reach a real Discord endpoint.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_notify(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALY_BUILD_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("ANOMALY_ALERT_STATE_DIR", str(tmp_path / "state_default"))
    # Inviolable: a test must NEVER hit the real Discord endpoint. Ensure the
    # env var is absent so build_default_notifier() falls back to local file.
    monkeypatch.delenv("ANOMALY_TEST_WEBHOOK_URL", raising=False)
    yield
