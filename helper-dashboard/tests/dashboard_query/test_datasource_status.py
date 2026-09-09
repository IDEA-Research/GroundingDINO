"""GET /api/datasource/status — public, coarse, honest.

The Prometheus client is stubbed; each test pins one hop of the
"is my data still coming in?" answer (reachability, exporter target
health, clinical metrics present, newest patient_* sample).
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.api import datasource as datasource_api
from app.main import app
from app.prometheus.client import PrometheusUnavailable


client = TestClient(app)


def _stub(monkeypatch, *, reachable=True, targets=None, sample_ts=None, targets_error=None):
    monkeypatch.setattr(
        datasource_api._prom, "probe", lambda promql: (reachable, True if reachable else None)
    )

    def _targets():
        if targets_error:
            raise PrometheusUnavailable(targets_error)
        return targets or []

    def _query(promql, time_s=None):
        assert promql.startswith("max(timestamp(")
        if sample_ts is None:
            return {"source": "prometheus", "data": {"result": []}}
        return {"source": "prometheus", "data": {"result": [{"metric": {}, "value": [1, str(sample_ts)]}]}}

    monkeypatch.setattr(datasource_api._prom, "targets", _targets)
    monkeypatch.setattr(datasource_api._prom, "query", _query)


def test_status_shape_when_everything_is_up(monkeypatch):
    now = int(time.time())
    _stub(
        monkeypatch,
        targets=[{"labels": {"job": "nicu_jetson", "instance": "x:8000"}, "health": "up",
                  "lastError": "", "lastScrape": "2026-09-08T00:00:00Z"}],
        sample_ts=now - 20,
    )
    r = client.get("/api/datasource/status")
    assert r.status_code == 200
    body = r.json()
    assert body["backend_ok"] is True
    assert body["prometheus"]["reachable"] is True
    assert body["prometheus"]["url_host_only"] in ("localhost", "127.0.0.1") or "://" not in body["prometheus"]["url_host_only"]
    assert body["exporter"]["job"] == "nicu_jetson" and body["exporter"]["health"] == "up"
    assert body["exporter"]["last_error_short"] is None
    # Test catalog fixture pins the clinical families.
    assert "patient_heart_rate_bpm" in body["clinical_metrics_present"]
    assert all(m.startswith(("patient_", "ventilator_", "cerebral_")) for m in body["clinical_metrics_present"])
    assert body["last_sample_ts"] == now - 20
    assert 15 <= body["age_seconds"] <= 30
    assert body["stale"] is False
    assert body["freshness_budget_seconds"] == 120
    assert body["simulation"] == {"active": False, "scenario": None}
    # Trailing-slash form registered too.
    assert client.get("/api/datasource/status/").status_code == 200


def test_exporter_down_with_short_error(monkeypatch):
    long_err = "Get http://x:8000/metrics: dial tcp: connection refused " * 5
    _stub(
        monkeypatch,
        targets=[{"labels": {"job": "nicu_jetson"}, "health": "down", "lastError": long_err}],
        sample_ts=int(time.time()) - 5000,
    )
    body = client.get("/api/datasource/status").json()
    assert body["exporter"]["health"] == "down"
    assert len(body["exporter"]["last_error_short"]) <= 120
    assert body["stale"] is True and body["age_seconds"] >= 4900


def test_job_absent_is_unknown(monkeypatch):
    _stub(monkeypatch, targets=[{"labels": {"job": "prometheus"}, "health": "up"}], sample_ts=None)
    body = client.get("/api/datasource/status").json()
    assert body["exporter"]["health"] == "unknown"
    assert body["last_sample_ts"] is None and body["age_seconds"] is None
    assert body["stale"] is False


def test_prometheus_unreachable(monkeypatch):
    _stub(monkeypatch, reachable=False)
    body = client.get("/api/datasource/status").json()
    assert body["backend_ok"] is True
    assert body["prometheus"]["reachable"] is False
    assert body["exporter"]["health"] == "unknown"
    assert body["clinical_metrics_present"] == []
    assert body["last_sample_ts"] is None


def test_targets_endpoint_failure_is_unknown_not_500(monkeypatch):
    _stub(monkeypatch, targets_error="targets query failed: 500", sample_ts=None)
    r = client.get("/api/datasource/status")
    assert r.status_code == 200
    assert r.json()["exporter"]["health"] == "unknown"
    assert "targets query failed" in r.json()["exporter"]["last_error_short"]


def test_client_targets_and_label_values_paths(monkeypatch):
    from app.prometheus.client import PrometheusClient

    c = PrometheusClient(base_url="http://example.invalid")
    captured = {}

    def _fake_get(path):
        captured["path"] = path
        if path.startswith("/api/v1/targets"):
            return {"data": {"activeTargets": [{"labels": {"job": "nicu_jetson"}, "health": "up"}, "junk"]}}
        return {"data": ["b", "a"]}

    monkeypatch.setattr(c, "_get", _fake_get)
    assert c.targets() == [{"labels": {"job": "nicu_jetson"}, "health": "up"}]
    assert captured["path"] == "/api/v1/targets?state=active"
    assert c.label_values("patient_id", start=10, end=20, match=['{__name__=~"patient_.*"}']) == ["a", "b"]
    assert captured["path"].startswith("/api/v1/label/patient_id/values?start=10&end=20&match%5B%5D=")
