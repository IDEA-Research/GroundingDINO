"""Prometheus client tests — honest failure, no mock fallback ever.

The client must NEVER fabricate data: an unreachable Prometheus raises
`PrometheusUnavailable` (carrying the underlying reason) instead of
returning a seeded sine wave. `probe` keeps its tri-state contract.
"""

import pytest

import app.prometheus.client as client_mod
from app.prometheus.client import PrometheusClient, PrometheusUnavailable


def test_query_raises_when_unreachable():
    c = PrometheusClient(base_url="http://127.0.0.1:1")
    with pytest.raises(PrometheusUnavailable):
        c.query("up")


def test_query_range_raises_when_unreachable():
    c = PrometheusClient(base_url="http://127.0.0.1:1")
    with pytest.raises(PrometheusUnavailable):
        c.query_range("up", range_s=120, step_s=30)


def test_unavailable_error_carries_the_underlying_reason():
    c = PrometheusClient(base_url="http://127.0.0.1:1")
    with pytest.raises(PrometheusUnavailable) as excinfo:
        c.query("up")
    # The message must explain WHY (connection refused / unreachable...),
    # so the API layer can surface an actionable 502 detail.
    assert str(excinfo.value).strip()
    assert excinfo.value.__cause__ is not None


def test_mock_generator_is_gone():
    # The silent sine-wave fallback must not exist in any form.
    assert not hasattr(client_mod, "_mock_response")


def test_probe_unreachable():
    c = PrometheusClient(base_url="http://127.0.0.1:1")
    reachable, returns_data = c.probe("up")
    assert reachable is False
    assert returns_data is None
