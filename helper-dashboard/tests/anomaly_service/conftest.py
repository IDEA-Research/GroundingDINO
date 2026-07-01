"""Isolate the build-audit + alert-state dirs so the service suite does not
write into the real `storage/anomaly_build_audit` / `storage/alert_state`.

The store is always constructed with an explicit `base_dir` in the tests, but
the service also emits build-audit records via `anomaly_build_audit.append()`,
which resolves its dir from `ANOMALY_BUILD_AUDIT_DIR`. Point both at a temp
dir for the duration of the suite.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALY_BUILD_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("ANOMALY_LIFECYCLE_AUDIT_DIR", str(tmp_path / "lifecycle"))
    monkeypatch.setenv("ANOMALY_ALERT_STATE_DIR", str(tmp_path / "state_default"))
    yield
