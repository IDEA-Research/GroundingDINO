"""Prometheus client tests — mock fallback behavior."""

from app.prometheus.client import PrometheusClient


def test_mock_fallback_instant(monkeypatch):
    # Force an unreachable base URL so the client falls back to mock.
    c = PrometheusClient(base_url="http://127.0.0.1:1")
    data = c.query("up")
    assert data["source"] == "mock"
    assert data["data"]["resultType"] == "vector"


def test_mock_fallback_range():
    c = PrometheusClient(base_url="http://127.0.0.1:1")
    data = c.query_range("up", range_s=120, step_s=30)
    assert data["source"] == "mock"
    assert data["data"]["resultType"] == "matrix"
    values = data["data"]["result"][0]["values"]
    assert 2 <= len(values) <= 5  # 120/30 = 4


def test_probe_unreachable():
    c = PrometheusClient(base_url="http://127.0.0.1:1")
    reachable, returns_data = c.probe("up")
    assert reachable is False
    assert returns_data is None
