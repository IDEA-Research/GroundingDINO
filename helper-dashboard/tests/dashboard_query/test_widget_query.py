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
    def _fake_query(promql, **_kw):
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
    def _fake_query(promql, **_kw):
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


# ---------------------------------------------------------------------------
# Shared refresh contract: optional `end` on the request, floored to the
# widget's step; response echoes `end` + `range_seconds` so charts pin
# their x-domain to the exact window the backend used.
# ---------------------------------------------------------------------------


def test_range_end_is_floored_to_step_and_echoed(monkeypatch):
    seen = {}

    def _fake_range(promql, *, range_s, step_s, end=None, **_kw):
        seen.update(promql=promql, range_s=range_s, step_s=step_s, end=end)
        return {"source": "prometheus", "data": {"result": []}}

    monkeypatch.setattr(dashboard_api._prom, "query_range", _fake_range)

    r = client.post(
        "/api/dashboard/query",
        json={
            "source": "prometheus",
            "promql": "up",
            "query_type": "range",
            "range": "1h",
            "step": "30s",
            "end": 1017,  # not step-aligned: floor(1017 / 30) * 30 = 990
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert seen["end"] == 990  # backend floored the shared tick to the step
    assert seen["range_s"] == 3600 and seen["step_s"] == 30
    assert body["end"] == 990  # echoed: the end ACTUALLY used
    assert body["range_seconds"] == 3600
    # Wire contract unchanged alongside the new fields.
    assert body["data_status"] == "empty"
    assert body["n_series"] == 0


def test_instant_query_pins_time_to_the_same_aligned_end(monkeypatch):
    """stat_card/gauge instant queries must evaluate at the same
    step-aligned instant as chart right edges, not a per-request now."""
    seen = {}

    def _fake_query(promql, time_s=None, **_kw):
        seen["time_s"] = time_s
        return {
            "source": "prometheus",
            "data": {"result": [{"metric": {}, "value": [990, "1"]}]},
        }

    monkeypatch.setattr(dashboard_api._prom, "query", _fake_query)

    r = client.post(
        "/api/dashboard/query",
        json={
            "source": "prometheus",
            "promql": "up",
            "query_type": "instant",
            "step": "30s",
            "end": 1017,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert seen["time_s"] == 990  # same flooring as the range path
    assert body["end"] == 990
    assert body["range_seconds"] == 3600  # default window still reported


def test_omitted_end_defaults_to_aligned_now(monkeypatch):
    import time as _time

    seen = {}

    def _fake_range(promql, *, range_s, step_s, end=None, **_kw):
        seen["end"] = end
        return {"source": "prometheus", "data": {"result": []}}

    monkeypatch.setattr(dashboard_api._prom, "query_range", _fake_range)

    before = int(_time.time())
    r = client.post(
        "/api/dashboard/query",
        json={
            "source": "prometheus",
            "promql": "up",
            "query_type": "range",
            "range": "5m",
            "step": "30s",
        },
    )
    after = int(_time.time())
    assert r.status_code == 200
    body = r.json()
    assert seen["end"] is not None
    assert seen["end"] % 30 == 0  # aligned even without a client end
    assert (before - 30) <= seen["end"] <= after
    assert body["end"] == seen["end"]
    assert body["range_seconds"] == 300


def test_client_query_range_floors_end_defensively(monkeypatch):
    """PrometheusClient itself floors `end` to the step (defense in
    depth), so even a caller that skips the API alignment cannot send
    an unaligned window."""
    from app.prometheus.client import PrometheusClient

    captured = {}
    c = PrometheusClient(base_url="http://example.invalid")

    def _fake_get(path):
        captured["path"] = path
        return {"data": {"result": []}}

    monkeypatch.setattr(c, "_get", _fake_get)

    c.query_range("up", range_s=600, step_s=30, end=1017)
    assert "end=990" in captured["path"]
    assert "start=390" in captured["path"]  # 990 - 600
    assert "step=30" in captured["path"]

    c.query("up", time_s=990)
    assert "time=990" in captured["path"]
