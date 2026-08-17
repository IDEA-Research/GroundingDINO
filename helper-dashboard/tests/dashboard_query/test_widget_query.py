"""Tests for the widget-query endpoint (POST /api/dashboard/query).

This endpoint is a new consumer of prometheus/client.py, so the key
invariants are: (1) the `source` flag is passed through unscrubbed, so a
mock fallback is never presented as real; (2) a widget pinned to
source != "prometheus" never reaches the live client; (3) Prometheus
vector/matrix results collapse into the renderer's WidgetData shape.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import dashboard as dashboard_api
from app.api.dashboard import _parse_duration, _to_widget_data
from app.main import app


client = TestClient(app)


def test_parse_duration():
    assert _parse_duration("1h") == 3600
    assert _parse_duration("30s") == 30
    assert _parse_duration("5m") == 300
    assert _parse_duration("500ms") == 1  # clamped to >= 1s
    assert _parse_duration(None) is None
    assert _parse_duration("bogus") is None


def test_instant_transform_sums_series_and_keeps_table():
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
    assert out["points"] == []
    assert out["latest"] == 0.75  # summed across series
    assert len(out["table"]) == 2  # per-series rows preserved


def test_range_transform_sums_across_series_per_timestamp():
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
    assert out["points"] == [{"t": 1000, "v": 1.5}, {"t": 1030, "v": 2.5}]
    assert out["latest"] == 2.5


def test_source_flag_passed_through_unscrubbed():
    # A mock fallback from the client must surface as source="mock".
    resp = {"source": "mock", "data": {"result": []}}
    assert _to_widget_data(resp, "instant")["source"] == "mock"
    assert _to_widget_data(resp, "range")["source"] == "mock"


def test_mock_pinned_widget_never_hits_live_client(monkeypatch):
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
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "mock"
    assert body["points"] == [] and body["latest"] is None and body["table"] == []
    assert called["n"] == 0


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
    assert body["latest"] == 0.42
