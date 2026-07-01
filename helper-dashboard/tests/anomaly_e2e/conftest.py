"""Isolate every audit / state / notification sink for the INC5 e2e suite.

The e2e wiring exercises the whole loop (evaluator -> state store -> notifier
-> lifecycle audit -> decision_flow widget) plus break-glass, so it touches the
build-audit, the lifecycle-audit, the alert-state store, and the notification
receiver. Point them ALL at a temp dir and — inviolable — clear
``ANOMALY_TEST_WEBHOOK_URL`` so no test can ever reach a real Discord endpoint.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_e2e(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALY_BUILD_AUDIT_DIR", str(tmp_path / "build_audit"))
    monkeypatch.setenv("ANOMALY_LIFECYCLE_AUDIT_DIR", str(tmp_path / "lifecycle"))
    monkeypatch.setenv("ANOMALY_ALERT_STATE_DIR", str(tmp_path / "state_default"))
    # A test must NEVER hit the real Discord endpoint.
    monkeypatch.delenv("ANOMALY_TEST_WEBHOOK_URL", raising=False)
    yield
