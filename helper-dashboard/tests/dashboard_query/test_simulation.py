"""Developer simulation (`/api/dev/simulation`) + its effect on the
query, label-values and datasource endpoints.

Gates under test: OFF by default; env HELPER_DASHBOARD_ALLOW_SIMULATION=1
AND a valid X-Developer-Token are both required; every simulated
payload is stamped simulated=true + sim_scenario; scenarios map to the
documented data states; offline -> 502 path, error -> 400 path.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.api import dashboard as dashboard_api
from app.main import app
from app.prometheus.client import PrometheusUnavailable
from app.services import simulation


client = TestClient(app)
TOKEN = "dev-secret-b"
HDR = {"X-Developer-Token": TOKEN}
HR = "max by (patient_id) (patient_heart_rate_bpm)"
SPO2 = "max by (patient_id) (patient_spo2_percentage)"


@pytest.fixture(autouse=True)
def _sim_env(monkeypatch):
    monkeypatch.setenv("HELPER_DASHBOARD_DEV_TOKEN", TOKEN)
    monkeypatch.setenv("HELPER_DASHBOARD_ALLOW_SIMULATION", "1")
    monkeypatch.delenv("HELPER_DASHBOARD_STALE_AFTER_SECONDS", raising=False)
    simulation.reset()
    # The real client must never be touched while simulation is active.
    def _boom(*_a, **_k):
        raise AssertionError("real Prometheus client called during simulation")
    monkeypatch.setattr(dashboard_api._prom, "query", _boom)
    monkeypatch.setattr(dashboard_api._prom, "query_range", _boom)
    monkeypatch.setattr(dashboard_api._prom, "label_values", _boom)
    yield
    simulation.reset()


def _set(scenario: str, headers=HDR):
    return client.post("/api/dev/simulation", json={"scenario": scenario}, headers=headers)


def _query(payload: dict):
    return client.post("/api/dashboard/query", json={"source": "prometheus", **payload})


# ---------------------------------------------------------------------------
# gating
# ---------------------------------------------------------------------------


def test_off_by_default_and_state_shape():
    r = client.get("/api/dev/simulation", headers=HDR)
    assert r.status_code == 200
    body = r.json()
    assert body["active"] is False and body["scenario"] == "off"
    assert body["allowed"] is True
    assert set(body["scenarios"]) == {"off", "normal", "nodata", "stale", "partial", "offline", "error"}


def test_requires_dev_token():
    assert client.get("/api/dev/simulation").status_code == 403
    assert _set("normal", headers={}).status_code == 403
    assert _set("normal", headers={"X-Developer-Token": "wrong"}).status_code == 403
    assert simulation.is_active() is False


def test_requires_env_kill_switch(monkeypatch):
    monkeypatch.delenv("HELPER_DASHBOARD_ALLOW_SIMULATION", raising=False)
    r = _set("normal")
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "simulation_disabled"
    assert simulation.is_active() is False
    # "off" is always accepted.
    assert _set("off").status_code == 200


def test_env_gate_wins_over_selected_state(monkeypatch):
    assert _set("normal").status_code == 200
    assert simulation.is_active() is True
    monkeypatch.delenv("HELPER_DASHBOARD_ALLOW_SIMULATION", raising=False)
    assert simulation.is_active() is False
    assert simulation.scenario() == "off"


def test_unknown_scenario_is_422():
    assert _set("chaos").status_code == 422


def test_trailing_slash_forms_registered():
    assert client.get("/api/dev/simulation/", headers=HDR).status_code == 200
    assert client.post("/api/dev/simulation/", json={"scenario": "off"}, headers=HDR).status_code == 200


def test_datasource_status_is_public_and_shows_simulation(monkeypatch):
    _set("normal")
    r = client.get("/api/datasource/status")  # no header needed
    assert r.status_code == 200
    body = r.json()
    assert body["simulation"] == {"active": True, "scenario": "normal"}
    assert body["prometheus"]["reachable"] is True
    assert body["exporter"]["job"] == "nicu_jetson" and body["exporter"]["health"] == "up"
    assert "patient_heart_rate_bpm" in body["clinical_metrics_present"]
    assert body["stale"] is False and body["last_sample_ts"] is not None


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


def test_normal_two_patients_plausible_vitals_flagged_simulated():
    _set("normal")
    r = _query({"promql": HR, "query_type": "instant", "step": "30s"})
    assert r.status_code == 200
    body = r.json()
    assert body["simulated"] is True and body["sim_scenario"] == "normal"
    assert body["data_status"] == "ok" and body["n_series"] == 2
    ids = sorted(row["metric"]["patient_id"] for row in body["table"])
    assert ids == ["P-01", "P-02"]
    for row in body["table"]:
        assert 120 <= row["value"] <= 160
    assert body["freshness"]["stale"] is False
    assert body["freshness"]["sample_ts"] is not None

    r = _query({"promql": SPO2, "query_type": "instant"})
    for row in r.json()["table"]:
        assert 92 <= row["value"] <= 99

    r = _query({"promql": "max by (patient_id) (cerebral_rso2_left_percentage)", "query_type": "instant"})
    for row in r.json()["table"]:
        assert 60 <= row["value"] <= 80


def test_normal_is_deterministic():
    _set("normal")
    payload = {"promql": HR, "query_type": "instant", "end": 1_700_000_000}
    a = _query(payload).json()["table"]
    b = _query(payload).json()["table"]
    assert a == b


def test_normal_range_and_patient_filter():
    _set("normal")
    r = _query({"promql": HR, "query_type": "range", "range": "5m", "step": "30s",
                "filters": {"patient_id": "P-02"}})
    body = r.json()
    assert body["simulated"] is True
    assert body["n_series"] == 1
    assert body["series_labels"] == [{"patient_id": "P-02"}]
    assert len([p for p in body["points"] if p["v"] is not None]) >= 10
    assert all(p["v"] is None or 120 <= p["v"] <= 160 for p in body["points"])


def test_nodata_is_empty_everywhere():
    _set("nodata")
    body = _query({"promql": HR, "query_type": "instant"}).json()
    assert body["data_status"] == "empty" and body["simulated"] is True
    assert body["sim_scenario"] == "nodata"
    assert body["freshness"]["sample_ts"] is None
    lv = client.get("/api/dashboard/label-values", params={"label": "patient_id"}).json()
    assert lv["values"] == [] and lv["simulated"] is True
    ds = client.get("/api/datasource/status").json()
    assert ds["clinical_metrics_present"] == []
    assert ds["exporter"]["health"] == "up"  # exporter up, nothing scraped


def test_stale_keeps_data_but_flags_freshness():
    _set("stale")
    now = int(time.time())
    body = _query({"promql": HR, "query_type": "instant", "step": "30s"}).json()
    assert body["data_status"] == "ok"
    f = body["freshness"]
    assert f["stale"] is True
    assert abs((now - f["sample_ts"]) - simulation.STALE_AGE_SECONDS) <= 60
    assert f["age_seconds"] > f["budget_seconds"]
    ds = client.get("/api/datasource/status").json()
    assert ds["stale"] is True and ds["age_seconds"] >= 800


def test_stale_range_stops_before_the_right_edge():
    _set("stale")
    body = _query({"promql": HR, "query_type": "range", "range": "1h", "step": "30s"}).json()
    ts = [p["t"] for p in body["points"] if p["v"] is not None]
    assert ts and body["end"] - ts[-1] >= simulation.STALE_AGE_SECONDS - 60


def test_partial_every_other_cell_is_null_in_queries_table():
    _set("partial")
    body = _query({
        "query_type": "instant",
        "queries": [
            {"key": "hr", "promql": HR, "label": "心率", "unit": "bpm"},
            {"key": "spo2", "promql": SPO2, "label": "血氧", "unit": "%"},
        ],
    }).json()
    assert body["simulated"] is True and body["sim_scenario"] == "partial"
    rows = {row["labels"]["patient_id"]: row["values"] for row in body["table"]}
    assert set(rows) == {"P-01", "P-02"}
    assert rows["P-01"]["hr"] is not None and rows["P-01"]["spo2"] is None
    assert rows["P-02"]["hr"] is None and rows["P-02"]["spo2"] is not None
    assert body["data_status"] == "ok"


def test_partial_range_has_a_null_gap():
    _set("partial")
    body = _query({"promql": HR, "query_type": "range", "range": "1h", "step": "30s"}).json()
    assert any(p["v"] is None for p in body["points"])


def test_offline_is_the_502_path():
    _set("offline")
    r = _query({"promql": HR, "query_type": "instant"})
    assert r.status_code == 502
    assert "Prometheus unavailable" in r.json()["detail"]
    assert client.get("/api/dashboard/label-values", params={"label": "patient_id"}).status_code == 502
    ds = client.get("/api/datasource/status").json()
    assert ds["prometheus"]["reachable"] is False
    assert ds["exporter"]["health"] == "unknown"
    assert ds["simulation"] == {"active": True, "scenario": "offline"}


def test_error_is_the_400_path():
    _set("error")
    r = _query({"promql": HR, "query_type": "instant"})
    assert r.status_code == 400
    assert "simulated" in r.json()["detail"]
    r = _query({"promql": HR, "query_type": "range", "range": "5m", "step": "30s"})
    assert r.status_code == 400


def test_label_values_under_simulation():
    _set("normal")
    r = client.get("/api/dashboard/label-values", params={"label": "patient_id"})
    assert r.json()["values"] == ["P-01", "P-02"]
    r = client.get("/api/dashboard/label-values", params={"label": "device_model"})
    assert r.json()["values"] == [simulation.DEVICE_MODEL]


def test_off_returns_to_real_client(monkeypatch):
    _set("normal")
    assert _set("off").json()["active"] is False

    def _fake_query(promql, time_s=None, **_kw):
        if "timestamp(" in promql:
            raise PrometheusUnavailable("no probe")
        return {"source": "prometheus", "data": {"result": [{"metric": {}, "value": [1, "7"]}]}}

    monkeypatch.setattr(dashboard_api._prom, "query", _fake_query)
    body = _query({"promql": "up", "query_type": "instant"}).json()
    assert body["simulated"] is False and body["latest"] == 7.0


def test_simulated_client_unit_shapes():
    """Direct unit checks on the PromQL reader behind the scenarios."""
    c = simulation.SimulatedPrometheus("normal", now=1_700_000_000)
    v = c.query("max(patient_heart_rate_bpm)", time_s=1_700_000_000)
    assert len(v["data"]["result"]) == 1 and v["data"]["result"][0]["metric"] == {}
    v = c.query('patient_heart_rate_bpm{patient_id="P-01"}')
    assert [s["metric"]["patient_id"] for s in v["data"]["result"]] == ["P-01"]
    ts = c.query("max(timestamp(patient_heart_rate_bpm))", time_s=1_700_000_000)
    assert float(ts["data"]["result"][0]["value"][1]) == 1_700_000_000
    m = c.query_range("max by (patient_id) (patient_spo2_percentage)", range_s=300, step_s=30, end=1_700_000_000)
    assert len(m["data"]["result"]) == 2 and len(m["data"]["result"][0]["values"]) == 11
    assert c.targets()[0]["labels"]["job"] == "nicu_jetson"
    with pytest.raises(PrometheusUnavailable):
        simulation.SimulatedPrometheus("offline").query("up")
    with pytest.raises(simulation.SimulatedQueryError):
        simulation.SimulatedPrometheus("error").query("up")
