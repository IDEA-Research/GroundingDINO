"""Additive query-contract tests (DESIGN 2026-09-07 §4).

Covers: label filters applied BEFORE the mean-collapse, multi-metric
`queries` pivot with null cells, freshness/stale from `max(timestamp())`,
gap nulls in range points, and the pinned legacy fields (data_status,
end, range_seconds) surviving alongside the new ones. The Prometheus
client is stubbed — no network.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.api import dashboard as dashboard_api
from app.api.dashboard import _to_widget_data, staleness_budget_seconds
from app.main import app
from app.prometheus.client import PrometheusUnavailable


client = TestClient(app)


def _vector(rows):
    return {
        "source": "prometheus",
        "data": {
            "resultType": "vector",
            "result": [
                {"metric": labels, "value": [1000, str(v)]} for labels, v in rows
            ],
        },
    }


def _no_probe(*_a, **_k):
    # Freshness probes are best-effort: an unreachable probe must never
    # fail the query, and sample_ts stays null.
    raise PrometheusUnavailable("probe refused")


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


def test_filter_applies_before_mean_collapse(monkeypatch):
    def _fake_query(promql, time_s=None, **_kw):
        if "timestamp(" in promql:
            raise PrometheusUnavailable("no probe")
        return _vector([
            ({"patient_id": "P-01", "device_model": "A"}, 140),
            ({"patient_id": "P-02", "device_model": "A"}, 160),
        ])

    monkeypatch.setattr(dashboard_api._prom, "query", _fake_query)
    r = client.post("/api/dashboard/query", json={
        "source": "prometheus", "promql": "max by (patient_id) (patient_heart_rate_bpm)",
        "query_type": "instant", "filters": {"patient_id": "P-02"},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["latest"] == 160.0  # only P-02, not the 150 mean
    assert body["n_series"] == 1
    assert body["data_status"] == "ok"
    assert body["filters_applied"] == {"patient_id": "P-02"}
    assert body["series_labels"] == [{"patient_id": "P-02", "device_model": "A"}]
    assert body["simulated"] is False and body["sim_scenario"] is None


def test_filter_drops_series_lacking_the_label(monkeypatch):
    def _fake_query(promql, time_s=None, **_kw):
        if "timestamp(" in promql:
            raise PrometheusUnavailable("no probe")
        return _vector([({}, 0.5)])  # avg(...) — no patient label at all

    monkeypatch.setattr(dashboard_api._prom, "query", _fake_query)
    r = client.post("/api/dashboard/query", json={
        "source": "prometheus", "promql": "avg(up)", "query_type": "instant",
        "filters": {"patient_id": "P-01"},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["data_status"] == "empty"
    assert body["latest"] is None and body["table"] == []


def test_empty_filter_values_are_ignored(monkeypatch):
    def _fake_query(promql, time_s=None, **_kw):
        if "timestamp(" in promql:
            raise PrometheusUnavailable("no probe")
        return _vector([({"patient_id": "P-01"}, 1)])

    monkeypatch.setattr(dashboard_api._prom, "query", _fake_query)
    r = client.post("/api/dashboard/query", json={
        "source": "prometheus", "promql": "up", "query_type": "instant",
        "filters": {"patient_id": ""},
    })
    assert r.status_code == 200
    assert r.json()["filters_applied"] == {}
    assert r.json()["n_series"] == 1


def test_unknown_filter_key_is_rejected():
    r = client.post("/api/dashboard/query", json={
        "source": "prometheus", "promql": "up", "query_type": "instant",
        "filters": {"job": "x"},
    })
    assert r.status_code == 422


def test_range_filter_collapses_only_selected_patient(monkeypatch):
    def _fake_range(promql, *, range_s, step_s, end=None, **_kw):
        return {
            "source": "prometheus",
            "data": {"resultType": "matrix", "result": [
                {"metric": {"patient_id": "P-01"}, "values": [[990, "100"], [1020, "100"]]},
                {"metric": {"patient_id": "P-02"}, "values": [[990, "200"], [1020, "200"]]},
            ]},
        }

    monkeypatch.setattr(dashboard_api._prom, "query_range", _fake_range)
    monkeypatch.setattr(dashboard_api._prom, "query", _no_probe)
    r = client.post("/api/dashboard/query", json={
        "source": "prometheus", "promql": "max by (patient_id) (patient_heart_rate_bpm)",
        "query_type": "range", "range": "5m", "step": "30s", "end": 1020,
        "filters": {"patient_id": "P-01"},
    })
    body = r.json()
    assert [p["v"] for p in body["points"]] == [100.0, 100.0]
    assert body["n_series"] == 1
    assert body["step_seconds"] == 30
    assert body["end"] == 1020 and body["range_seconds"] == 300


# ---------------------------------------------------------------------------
# queries pivot
# ---------------------------------------------------------------------------


def test_queries_pivot_joins_by_patient_with_null_cells(monkeypatch):
    calls: list[str] = []

    def _fake_query(promql, time_s=None, **_kw):
        calls.append(promql)
        if "timestamp(" in promql:
            # per-cell sample time probe: max by (patient_id) (timestamp(m))
            if "patient_heart_rate_bpm" in promql:
                return _vector([({"patient_id": "P-01"}, 5000), ({"patient_id": "P-02"}, 4990)])
            return _vector([({"patient_id": "P-01"}, 5010)])
        if "patient_heart_rate_bpm" in promql:
            return _vector([
                ({"patient_id": "P-01", "device_model": "M"}, 140),
                ({"patient_id": "P-02", "device_model": "M"}, 150),
            ])
        if "patient_spo2_percentage" in promql:
            return _vector([({"patient_id": "P-01", "device_model": "M"}, 96)])
        raise AssertionError(promql)

    monkeypatch.setattr(dashboard_api._prom, "query", _fake_query)
    r = client.post("/api/dashboard/query", json={
        "source": "prometheus", "query_type": "instant", "end": 5010, "step": "30s",
        "queries": [
            {"key": "hr", "promql": "max by (patient_id) (patient_heart_rate_bpm)", "label": "心率", "unit": "bpm"},
            {"key": "spo2", "promql": "max by (patient_id) (patient_spo2_percentage)", "label": "血氧", "unit": "%"},
        ],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == [
        {"key": "hr", "label": "心率", "unit": "bpm"},
        {"key": "spo2", "label": "血氧", "unit": "%"},
    ]
    assert body["join_label"] == "patient_id"
    rows = body["table"]
    assert [row["labels"]["patient_id"] for row in rows] == ["P-01", "P-02"]
    assert rows[0]["values"] == {"hr": 140.0, "spo2": 96.0}
    assert rows[1]["values"] == {"hr": 150.0, "spo2": None}  # null, never 0
    assert rows[0]["ts"] == {"hr": 5000, "spo2": 5010}
    assert rows[1]["ts"] == {"hr": 4990, "spo2": None}
    # legacy row consumers still see metric/value
    assert rows[0]["metric"]["patient_id"] == "P-01" and rows[0]["value"] == 140.0
    assert body["data_status"] == "ok"
    assert body["n_series"] == 2
    assert body["points"] == [] and body["latest"] is None
    # Every column evaluated at the SAME step-aligned end.
    assert all(c is not None for c in calls)
    # Freshness = max over the raw metric probes (spo2 newest at 5010).
    assert body["freshness"]["sample_ts"] == 5010
    # pinned legacy fields survive (5010 is already step-aligned for 30s)
    assert body["end"] == 5010 and body["range_seconds"] == 3600


def test_queries_table_all_null_is_empty(monkeypatch):
    def _fake_query(promql, time_s=None, **_kw):
        return _vector([])

    monkeypatch.setattr(dashboard_api._prom, "query", _fake_query)
    r = client.post("/api/dashboard/query", json={
        "source": "prometheus", "query_type": "instant",
        "queries": [{"key": "hr", "promql": "max by (patient_id) (patient_heart_rate_bpm)"}],
    })
    body = r.json()
    assert body["data_status"] == "empty"
    assert body["table"] == []
    assert body["columns"][0]["label"] == "hr"  # label defaults to key


def test_queries_limit_is_six():
    r = client.post("/api/dashboard/query", json={
        "source": "prometheus", "query_type": "instant",
        "queries": [{"key": f"k{i}", "promql": "up"} for i in range(7)],
    })
    assert r.status_code == 422


def test_promql_required_without_queries():
    r = client.post("/api/dashboard/query", json={"source": "prometheus", "query_type": "instant"})
    assert r.status_code == 400
    assert "promql" in r.json()["detail"]


# ---------------------------------------------------------------------------
# freshness / stale
# ---------------------------------------------------------------------------


def _freshness_client(monkeypatch, sample_ts: int):
    def _fake_query(promql, time_s=None, **_kw):
        if promql.startswith("max(timestamp("):
            return _vector([({}, sample_ts)])
        return _vector([({"patient_id": "P-01"}, 140)])

    monkeypatch.setattr(dashboard_api._prom, "query", _fake_query)


def test_freshness_stale_when_sample_older_than_budget(monkeypatch):
    monkeypatch.delenv("HELPER_DASHBOARD_STALE_AFTER_SECONDS", raising=False)
    now = int(time.time())
    _freshness_client(monkeypatch, now - 900)
    r = client.post("/api/dashboard/query", json={
        "source": "prometheus", "promql": "max by (patient_id) (patient_heart_rate_bpm)",
        "query_type": "instant", "step": "30s",
    })
    f = r.json()["freshness"]
    assert f["sample_ts"] == now - 900
    assert 890 <= f["age_seconds"] <= 910
    assert f["budget_seconds"] == 120
    assert f["stale"] is True
    assert r.json()["data_status"] == "ok"  # stale is a FLAG, data_status stays ok


def test_freshness_fresh_within_budget(monkeypatch):
    now = int(time.time())
    _freshness_client(monkeypatch, now - 10)
    r = client.post("/api/dashboard/query", json={
        "source": "prometheus", "promql": "patient_heart_rate_bpm", "query_type": "instant",
    })
    f = r.json()["freshness"]
    assert f["stale"] is False and f["age_seconds"] <= 15


def test_freshness_budget_env_and_two_step_floor(monkeypatch):
    monkeypatch.setenv("HELPER_DASHBOARD_STALE_AFTER_SECONDS", "1000")
    assert staleness_budget_seconds(30) == 1000
    monkeypatch.delenv("HELPER_DASHBOARD_STALE_AFTER_SECONDS", raising=False)
    assert staleness_budget_seconds(30) == 120
    assert staleness_budget_seconds(300) == 600  # never below 2 x step
    monkeypatch.setenv("HELPER_DASHBOARD_STALE_AFTER_SECONDS", "bogus")
    assert staleness_budget_seconds(None) == 120


def test_freshness_probe_uses_catalog_metric_with_filter_selector(monkeypatch):
    probes: list[str] = []

    def _fake_query(promql, time_s=None, **_kw):
        if promql.startswith("max(timestamp("):
            probes.append(promql)
            return _vector([({}, 1)])
        return _vector([({"patient_id": "P-02"}, 1)])

    monkeypatch.setattr(dashboard_api._prom, "query", _fake_query)
    client.post("/api/dashboard/query", json={
        "source": "prometheus",
        "promql": "max by (patient_id) (patient_spo2_percentage)",
        "query_type": "instant", "filters": {"patient_id": "P-02"},
    })
    assert probes == ['max(timestamp(patient_spo2_percentage{patient_id="P-02"}))']


def test_freshness_null_when_no_catalog_metric_in_promql(monkeypatch):
    def _fake_query(promql, time_s=None, **_kw):
        assert "timestamp(" not in promql  # nothing to probe
        return _vector([({}, 1)])

    monkeypatch.setattr(dashboard_api._prom, "query", _fake_query)
    r = client.post("/api/dashboard/query", json={
        "source": "prometheus", "promql": "vector(1)", "query_type": "instant",
    })
    f = r.json()["freshness"]
    assert f == {"sample_ts": None, "age_seconds": None, "stale": False, "budget_seconds": 120}


def test_failed_probe_never_fails_the_query(monkeypatch):
    def _fake_query(promql, time_s=None, **_kw):
        if "timestamp(" in promql:
            raise PrometheusUnavailable("probe down")
        return _vector([({}, 1)])

    monkeypatch.setattr(dashboard_api._prom, "query", _fake_query)
    r = client.post("/api/dashboard/query", json={
        "source": "prometheus", "promql": "up", "query_type": "instant",
    })
    assert r.status_code == 200
    assert r.json()["freshness"]["sample_ts"] is None


# ---------------------------------------------------------------------------
# gap nulls
# ---------------------------------------------------------------------------


def test_range_gap_inserts_null_point_after_last_sample():
    resp = {
        "source": "prometheus",
        "data": {"resultType": "matrix", "result": [
            {"metric": {}, "values": [[1000, "1"], [1030, "2"], [1200, "3"]]},
        ]},
    }
    out = _to_widget_data(resp, "range", step_s=30)
    assert out["points"] == [
        {"t": 1000, "v": 1.0},
        {"t": 1030, "v": 2.0},
        {"t": 1060, "v": None},  # gap > 2 x step -> explicit null
        {"t": 1200, "v": 3.0},
    ]
    assert out["latest"] == 3.0  # latest ignores the null
    assert out["data_status"] == "ok"


def test_range_exact_two_step_gap_is_not_a_gap():
    resp = {
        "source": "prometheus",
        "data": {"resultType": "matrix", "result": [
            {"metric": {}, "values": [[1000, "1"], [1060, "2"]]},
        ]},
    }
    out = _to_widget_data(resp, "range", step_s=30)
    assert [p["v"] for p in out["points"]] == [1.0, 2.0]


def test_range_gap_nulls_via_endpoint(monkeypatch):
    def _fake_range(promql, *, range_s, step_s, end=None, **_kw):
        return {"source": "prometheus", "data": {"result": [
            {"metric": {}, "values": [[900, "1"], [1200, "2"]]},
        ]}}

    monkeypatch.setattr(dashboard_api._prom, "query_range", _fake_range)
    monkeypatch.setattr(dashboard_api._prom, "query", _no_probe)
    r = client.post("/api/dashboard/query", json={
        "source": "prometheus", "promql": "up", "query_type": "range",
        "range": "5m", "step": "30s", "end": 1200,
    })
    pts = r.json()["points"]
    assert pts[1] == {"t": 930, "v": None}
    assert r.json()["data_status"] == "ok"


# ---------------------------------------------------------------------------
# label values
# ---------------------------------------------------------------------------


def test_label_values_allow_list_and_time_scope(monkeypatch):
    seen = {}

    def _fake_lv(label, *, start=None, end=None, match=None):
        seen.update(label=label, start=start, end=end, match=match)
        return ["P-02", "P-01", ""]

    monkeypatch.setattr(dashboard_api._prom, "label_values", _fake_lv)
    r = client.get("/api/dashboard/label-values", params={"label": "patient_id", "hours": 6})
    assert r.status_code == 200
    body = r.json()
    assert body["values"] == ["P-02", "P-01"]  # empties dropped, order preserved
    assert body["label"] == "patient_id" and body["hours"] == 6
    assert seen["end"] - seen["start"] == 6 * 3600
    assert any("patient" in m for m in seen["match"])

    r = client.get("/api/dashboard/label-values", params={"label": "job"})
    assert r.status_code == 400
    r = client.get("/api/dashboard/label-values", params={"label": "patient_id", "hours": 0})
    assert r.status_code == 422


def test_label_values_502_when_prometheus_down(monkeypatch):
    def _raise(*_a, **_k):
        raise PrometheusUnavailable("label values query failed: refused")

    monkeypatch.setattr(dashboard_api._prom, "label_values", _raise)
    r = client.get("/api/dashboard/label-values", params={"label": "device_model"})
    assert r.status_code == 502
    assert "Prometheus unavailable" in r.json()["detail"]


def test_label_values_route_does_not_shadow_dashboard_lookup():
    # `/label-values` is a literal route registered before `/{dashboard_id}`;
    # an unknown dashboard id must still 404 (not be parsed as a label).
    r = client.get("/api/dashboard/no-such-dashboard-xyz")
    assert r.status_code == 404


@pytest.mark.parametrize("bad", ["bogus", "10x", ""])
def test_parse_duration_rejects_garbage(bad):
    from app.api.dashboard import _parse_duration

    assert _parse_duration(bad) is None
