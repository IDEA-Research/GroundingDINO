"""Tests for the widget-query endpoint (POST /api/dashboard/query).

No mock data, ever, on this path. The key invariants are: (1) a widget
pinned to source != "prometheus" is rejected with HTTP 400 and never
reaches the live client; (2) an unreachable Prometheus is HTTP 502 with
the underlying reason — never fabricated data; (3) every success carries
an explicit `data_status` ("ok" with series, "empty" with zero
series/points) so the frontend can render a loud NO-DATA state; (4)
Prometheus vector/matrix results collapse into the renderer's WidgetData
shape. No test here touches a live Prometheus — the client is stubbed.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import dashboard as dashboard_api
from app.api.dashboard import _parse_duration, _to_widget_data
from app.main import app
from app.prometheus.client import PrometheusUnavailable


client = TestClient(app)


def test_parse_duration():
    assert _parse_duration("1h") == 3600
    assert _parse_duration("30s") == 30
    assert _parse_duration("5m") == 300
    assert _parse_duration("500ms") == 1  # clamped to >= 1s
    assert _parse_duration(None) is None
    assert _parse_duration("bogus") is None


def test_instant_transform_averages_series_and_keeps_table():
    # MEAN, not sum: clinical gauges arrive as N parallel series (one per
    # free-text device_model variant). Sum once rendered HR 163 as 814.
    resp = {
        "source": "prometheus",
        "data": {
            "resultType": "vector",
            "result": [
                {"metric": {"mode": "user"}, "value": [1000, "0.5"]},
                {"metric": {"mode": "system"}, "value": [1000, "0.25"]},
            ],
        },
    }
    out = _to_widget_data(resp, "instant")
    assert out["source"] == "prometheus"
    assert out["data_status"] == "ok"
    assert out["points"] == []
    assert out["latest"] == 0.375  # averaged across series, never summed
    assert out["n_series"] == 2  # frontend annotates "2 series (averaged)"
    assert len(out["table"]) == 2  # per-series rows preserved


def test_instant_single_series_keeps_exact_value():
    resp = {
        "source": "prometheus",
        "data": {
            "resultType": "vector",
            "result": [{"metric": {}, "value": [1000, "163"]}],
        },
    }
    out = _to_widget_data(resp, "instant")
    assert out["latest"] == 163.0
    assert out["n_series"] == 1


def test_range_transform_averages_across_series_per_timestamp():
    resp = {
        "source": "prometheus",
        "data": {
            "resultType": "matrix",
            "result": [
                {"metric": {"cpu": "0"}, "values": [[1000, "1.0"], [1030, "2.0"]]},
                {"metric": {"cpu": "1"}, "values": [[1000, "0.5"], [1030, "0.5"]]},
            ],
        },
    }
    out = _to_widget_data(resp, "range")
    assert out["source"] == "prometheus"
    assert out["data_status"] == "ok"
    assert out["points"] == [{"t": 1000, "v": 0.75}, {"t": 1030, "v": 1.25}]
    assert out["latest"] == 1.25
    assert out["n_series"] == 2


def test_range_transform_divides_by_series_present_at_each_timestamp():
    # A series that starts mid-window must not drag down earlier points.
    resp = {
        "source": "prometheus",
        "data": {
            "resultType": "matrix",
            "result": [
                {"metric": {"d": "a"}, "values": [[1000, "100"], [1030, "100"]]},
                {"metric": {"d": "b"}, "values": [[1030, "200"]]},
            ],
        },
    }
    out = _to_widget_data(resp, "range")
    assert out["points"] == [{"t": 1000, "v": 100.0}, {"t": 1030, "v": 150.0}]
    assert out["n_series"] == 2


def test_zero_series_is_data_status_empty():
    # A successful query with no series must say so EXPLICITLY — silent
    # emptiness once hid an 11-day stale outage.
    resp = {"source": "prometheus", "data": {"result": []}}
    out_i = _to_widget_data(resp, "instant")
    assert out_i["data_status"] == "empty"
    assert out_i["latest"] is None and out_i["table"] == []
    assert out_i["n_series"] == 0
    out_r = _to_widget_data(resp, "range")
    assert out_r["data_status"] == "empty"
    assert out_r["points"] == [] and out_r["latest"] is None
    assert out_r["n_series"] == 0


def test_mock_source_is_rejected_with_400(monkeypatch):
    called = {"n": 0}

    def _boom(*_a, **_k):  # pragma: no cover - must never run
        called["n"] += 1
        raise AssertionError("live client was queried for a mock-pinned widget")

    monkeypatch.setattr(dashboard_api._prom, "query", _boom)
    monkeypatch.setattr(dashboard_api._prom, "query_range", _boom)

    r = client.post(
        "/api/dashboard/query",
        json={"source": "mock", "promql": "anything", "query_type": "instant"},
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "mock data source is not allowed"
    assert called["n"] == 0


def test_prometheus_down_is_502_with_reason(monkeypatch):
    def _raise(*_a, **_k):
        raise PrometheusUnavailable("instant query failed: connection refused")

    monkeypatch.setattr(dashboard_api._prom, "query", _raise)

    r = client.post(
        "/api/dashboard/query",
        json={"source": "prometheus", "promql": "up", "query_type": "instant"},
    )
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert "Prometheus unavailable" in detail
    assert "connection refused" in detail


def test_prometheus_down_is_502_for_range_too(monkeypatch):
    def _raise(*_a, **_k):
        raise PrometheusUnavailable("range query failed: timed out")

    monkeypatch.setattr(dashboard_api._prom, "query_range", _raise)

    r = client.post(
        "/api/dashboard/query",
        json={
            "source": "prometheus",
            "promql": "up",
            "query_type": "range",
            "range": "1h",
            "step": "30s",
        },
    )
    assert r.status_code == 502
    assert "timed out" in r.json()["detail"]


def test_prometheus_query_transformed_via_endpoint(monkeypatch):
    def _fake_query(promql):
        return {
            "source": "prometheus",
            "data": {
                "resultType": "vector",
                "result": [{"metric": {}, "value": [1000, "0.42"]}],
            },
        }

    monkeypatch.setattr(dashboard_api._prom, "query", _fake_query)

    r = client.post(
        "/api/dashboard/query",
        json={"source": "prometheus", "promql": "up", "query_type": "instant"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "prometheus"
    assert body["data_status"] == "ok"
    assert body["latest"] == 0.42
    assert body["n_series"] == 1


def test_empty_result_via_endpoint_reports_empty(monkeypatch):
    def _fake_query(promql):
        return {"source": "prometheus", "data": {"resultType": "vector", "result": []}}

    monkeypatch.setattr(dashboard_api._prom, "query", _fake_query)

    r = client.post(
        "/api/dashboard/query",
        json={
            "source": "prometheus",
            "promql": "nonexistent_metric_xyz",
            "query_type": "instant",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data_status"] == "empty"
    assert body["latest"] is None and body["table"] == []
