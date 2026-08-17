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


# Deterministic metric catalog for the whole suite (Tier 0, no network):
# spec generation is catalog-backed now, and tests must not depend on a
# live Prometheus. The fixed list mirrors what the box exposes while a
# camera stream runs (clinical patient_*/ventilator_*/cerebral_* families
# + the common system metrics the heuristics reference). Catalog unit
# tests that need the REAL fetch path restore it via
# `MetricCatalog._real_fetch_names_for_tests`.
_TEST_CATALOG_METRICS = [
    "ALERTS",
    "cerebral_rso2_left_percentage",
    "cerebral_rso2_right_percentage",
    "http_request_duration_seconds",
    "http_request_duration_seconds_bucket",
    "http_requests_total",
    "node_cpu_seconds_total",
    "node_filesystem_avail_bytes",
    "node_load1",
    "node_memory_MemAvailable_bytes",
    "node_memory_MemTotal_bytes",
    "patient_blood_pressure_diastolic_mmhg",
    "patient_blood_pressure_systolic_mmhg",
    "patient_heart_rate_bpm",
    "patient_respiration_rate_per_minute",
    "patient_spo2_percentage",
    "process_cpu_seconds_total",
    "process_resident_memory_bytes",
    "prometheus_build_info",
    "up",
    "ventilator_fio2_percentage",
    "ventilator_peep_cmh2o",
    "ventilator_pinsp_cmh2o",
    "ventilator_pip_cmh2o",
]


@pytest.fixture(autouse=True)
def _fixed_metric_catalog(monkeypatch):
    from app.prometheus import catalog as catalog_mod

    original = catalog_mod.MetricCatalog._fetch_names
    # Keep the real fetch reachable for catalog unit tests.
    monkeypatch.setattr(
        catalog_mod.MetricCatalog,
        "_real_fetch_names_for_tests",
        original,
        raising=False,
    )
    monkeypatch.setattr(
        catalog_mod.MetricCatalog,
        "_fetch_names",
        lambda self: list(_TEST_CATALOG_METRICS),
    )
    catalog_mod.get_catalog().invalidate()
    yield
    catalog_mod.get_catalog().invalidate()


@pytest.fixture(autouse=True)
def _devfix_runs_tmp(monkeypatch, tmp_path):
    """The devfix runtime streams a live run log into
    backend/app/storage/devfix_runs/. Tests must never write into the
    real store, so the whole suite is redirected to tmp_path (the
    devfix_log/API tests read this same dir back)."""
    monkeypatch.setenv(
        "HELPER_DASHBOARD_DEVFIX_RUNS_DIR", str(tmp_path / "devfix_runs")
    )


@pytest.fixture(autouse=True)
def _no_background_auto_fix(monkeypatch):
    """Auto-fix spawns background developer_fix threads and writes real
    audit files under backend/app/storage/auto_fix_audit/. Tests that
    exercise it opt in explicitly (tests/auto_fix sets the env back to
    1 and redirects the audit dir); everything else runs with the kill
    switch on so helpercode-mode tests don't fire the pipeline."""
    monkeypatch.setenv("HELPER_DASHBOARD_AUTO_FIX", "0")
