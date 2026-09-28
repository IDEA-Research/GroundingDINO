"""WP-6: FastAPI shell with a fake engine (no Mongo): auth, trusted host, errors, rendering, CLI parser."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from nicu_qe.__main__ import build_parser
from nicu_qe.api import create_app, render
from nicu_qe.config import Settings, UnsafeConfigError
from nicu_qe.engine import QueryError, UnknownMetric
from nicu_qe.models import Point, Series

UTC = timezone.utc
T = datetime(2026, 9, 10, 1, 0, tzinfo=UTC)


class FakeEngine:
    def __init__(self):
        self.calls = []

    def health(self):
        return 503, {"status": "error", "db": "error: ServerSelectionTimeoutError"}

    def catalog(self):
        return {"metrics": [{"name": "hr", "last_observed_at": T}]}

    def series(self, metric, from_, to, **kw):
        self.calls.append(("series", metric, from_, to, kw))
        if metric == "nope":
            raise UnknownMetric("unknown metric 'nope'")
        if metric == "boom":
            raise RuntimeError("secret internal detail /etc/passwd")
        if metric == "bad":
            raise QueryError("to must be after from")
        return Series(metric=metric, unit="bpm", step=None, agg=None, from_=T, to=T, points=[Point(t=T, v=1.0)])

    def latest(self, metric, **kw):
        self.calls.append(("latest", metric, kw))
        return {"metric": metric, "data_status": "empty"}

    def stats(self, metric, from_, to, **kw):
        self.calls.append(("stats", metric, kw))
        return {"rows": [], "data_status": "empty"}

    def coverage(self, from_, to, **kw):
        return {"rows": [], "data_status": "empty"}

    def sessions(self, from_, to, **kw):
        return {"rows": [], "data_status": "empty"}

    def unmapped_keys(self, **kw):
        return {"rows": [{"key": "x", "sample_values": [{"type": "string", "len": 5, "masked": "ab***"}]}]}

    @staticmethod
    def aliases():
        return {"metric_aliases": {"version": "v"}}


def client(token="s3cret", **env):
    e = {"QE_DB": "medical_monitor_qe_test", "QE_API_TOKEN": token, **env}
    st = Settings.from_env(e)
    fe = FakeEngine()
    app = create_app(st, engine=fe, start_etl=False, allowed_hosts=["testserver", "127.0.0.1", "localhost"])
    return TestClient(app, raise_server_exceptions=False), fe


def test_token_required_on_v1_not_health():
    c, _ = client()
    r = c.get("/v1/catalog")
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
    assert c.get("/v1/catalog", headers={"X-QE-Token": "wrong"}).status_code == 401
    ok = c.get("/v1/catalog", headers={"X-QE-Token": "s3cret"})
    assert ok.status_code == 200 and ok.json()["metrics"][0]["last_observed_at"] == "2026-09-10T09:00:00+08:00"
    h = c.get("/health")
    assert h.status_code == 503 and h.json()["status"] == "error"


def test_no_token_mode_only_on_loopback():
    c, _ = client(token="", QE_ALLOW_NO_TOKEN="1", QE_BIND="127.0.0.1:8021")
    assert c.get("/v1/catalog").status_code == 200
    with pytest.raises(UnsafeConfigError):
        client(token="")
    with pytest.raises(UnsafeConfigError):
        client(token="", QE_ALLOW_NO_TOKEN="1", QE_BIND="0.0.0.0:8021")


def test_trusted_host_rejects_foreign_host():
    st = Settings.from_env({"QE_DB": "medical_monitor_qe_test", "QE_API_TOKEN": "t"})
    c = TestClient(create_app(st, engine=FakeEngine(), start_etl=False))  # default allow-list only
    r = c.get("/health", headers={"host": "evil.example.com"})
    assert r.status_code == 400
    r2 = c.get("/health", headers={"host": "127.0.0.1:8020"})
    assert r2.status_code == 503


H = {"X-QE-Token": "s3cret"}


def test_error_mapping_without_stack():
    c, _ = client()
    r = c.get("/v1/series", params={"metric": "nope", "from": "2026-09-10", "to": "2026-09-11"}, headers=H)
    assert r.status_code == 404 and r.json() == {"error": {"code": "unknown_metric", "message": "unknown metric 'nope'"}}
    r = c.get("/v1/series", params={"metric": "bad", "from": "2026-09-10", "to": "2026-09-11"}, headers=H)
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"
    r = c.get("/v1/series", params={"metric": "boom", "from": "2026-09-10", "to": "2026-09-11"}, headers=H)
    assert r.status_code == 500
    assert "passwd" not in r.text and "Traceback" not in r.text
    r = c.get("/v1/series", params={"metric": "hr"}, headers=H)  # missing from/to
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"
    r = c.get("/v1/series", params={"metric": "hr", "from": "a", "to": "b", "limit": "x"}, headers=H)
    assert r.status_code == 400


def test_series_params_and_epoch():
    c, fe = client()
    r = c.get("/v1/series", params={"metric": "hr", "from": "2026-09-10", "to": "2026-09-11", "t_format": "epoch",
                                    "session_id": "abc", "exclude_quality": ""}, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["points"][0]["t"] == T.timestamp() and body["data_status"] == "ok"
    kw = fe.calls[-1][4]
    assert kw["session_id"] == "abc" and kw["exclude_quality"] == "" and kw["include_analysis_time"] is False
    r = c.get("/v1/series", params={"metric": "hr", "from": "x", "to": "y", "t_format": "unix"}, headers=H)
    assert r.status_code == 400


def test_other_endpoints_empty_200():
    c, _ = client()
    assert c.get("/v1/latest", params={"metric": "hr"}, headers=H).json()["data_status"] == "empty"
    for path, params in (("/v1/stats", {"metric": "hr", "from": "2026-09-01", "to": "2026-09-30"}),
                         ("/v1/coverage", {"from": "2026-09-01", "to": "2026-09-30"}),
                         ("/v1/sessions", {})):
        r = c.get(path, params=params, headers=H)
        assert r.status_code == 200 and r.json()["data_status"] == "empty"
    u = c.get("/v1/unmapped-keys", headers=H).json()
    assert u["rows"][0]["sample_values"][0]["masked"] == "ab***"
    assert c.get("/v1/aliases", headers=H).json()["metric_aliases"]["version"] == "v"


def test_only_get_allowed():
    c, _ = client()
    assert c.post("/v1/catalog", headers=H).status_code == 405


def test_render():
    out = render({"t": T, "x": [T], "n": None}, "iso", "Asia/Taipei")
    assert out == {"t": "2026-09-10T09:00:00+08:00", "x": ["2026-09-10T09:00:00+08:00"], "n": None}
    assert render(datetime(2026, 9, 10, 1), "epoch", "Asia/Taipei") == T.timestamp()  # naive = UTC


@pytest.mark.parametrize("argv", [
    ["serve"], ["serve", "--pidfile", "/tmp/x.pid"], ["etl", "--once"], ["etl", "--full"], ["etl", "--loop"],
    ["etl", "--reconcile", "--summary"], ["indexes", "ensure"], ["indexes", "check"], ["indexes", "drop", "--yes"],
    ["check"],
])
def test_cli_parser(argv):
    assert build_parser().parse_args(argv).fn


def test_cli_parser_rejects_conflicts():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["etl", "--once", "--full"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["indexes", "rebuild"])


def test_render_offset_cache_matches_zoneinfo_across_dst():
    from datetime import timedelta

    from nicu_qe.tz import to_display

    start = datetime(2026, 3, 8, 5, 0, tzinfo=UTC)  # US DST starts 2026-03-08 07:00 UTC
    ts = [start + timedelta(minutes=7 * i) for i in range(60)]
    out = render(ts, "iso", "America/New_York")
    assert out == [to_display(t, "America/New_York").isoformat() for t in ts]
    assert render(ts, "iso", "Asia/Taipei") == [to_display(t, "Asia/Taipei").isoformat() for t in ts]


class _ErrEngine(FakeEngine):
    def health(self):
        return 200, {"status": "degraded", "db": "ok", "qe_db": "ok",
                     "etl": {"last_ok": False,
                             "last_error": "full: ServerSelectionTimeoutError: 10.0.0.5:27017 timed out, secret"}}


def _err_client(token="s3cret", **env):
    st = Settings.from_env({"QE_DB": "medical_monitor_qe_test", "QE_API_TOKEN": token, **env})
    app = create_app(st, engine=_ErrEngine(), start_etl=False, allowed_hosts=["testserver"])
    return TestClient(app, raise_server_exceptions=False)


def test_health_last_error_class_only_without_token():
    c = _err_client()
    for hdr in ({}, {"X-QE-Token": "wrong"}):
        r = c.get("/health", headers=hdr)
        assert r.status_code == 200
        assert r.json()["etl"]["last_error"] == "ServerSelectionTimeoutError"
        assert "10.0.0.5" not in r.text
    full = c.get("/health", headers={"X-QE-Token": "s3cret"}).json()["etl"]["last_error"]
    assert full.startswith("full: ServerSelectionTimeoutError: 10.0.0.5")


def test_health_last_error_no_token_mode_is_redacted():
    c = _err_client(token="", QE_ALLOW_NO_TOKEN="1", QE_BIND="127.0.0.1:8021")
    assert c.get("/health").json()["etl"]["last_error"] == "ServerSelectionTimeoutError"


def test_error_class_only_formats():
    from nicu_qe.api import error_class_only
    assert error_class_only(None) is None
    assert error_class_only("incremental: AutoReconnect: host x") == "AutoReconnect"
    assert error_class_only("OperationFailure: bad auth user=y") == "OperationFailure"
    assert error_class_only("lease held by another process") == "LeaseHeld"
    assert error_class_only("weird free text") == "Error"
