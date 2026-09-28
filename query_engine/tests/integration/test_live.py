"""Integration tests on orin11 (live source read-only, derived writes only to medical_monitor_qe_test).

Baselines are computed here from the source DB itself (DESIGN 12.7 / REVIEW H4), not from old reports.
"""

from datetime import datetime, timedelta, timezone

import pytest

from nicu_qe.etl import ALL_METRIC, OBS, ROLLUP, SESSIONS
from nicu_qe.tz import iter_buckets, parse_step, wallclock_to_utc

UTC = timezone.utc


def _busy_range(engine, days=1):
    top = list(engine.qe[ROLLUP].aggregate([{"$match": {"metric": "hr"}},
                                            {"$group": {"_id": "$day", "n": {"$sum": "$count"}}},
                                            {"$sort": {"n": -1}}, {"$limit": 1}]))
    d = datetime.strptime(top[0]["_id"], "%Y-%m-%d")
    f = wallclock_to_utc(d, engine.settings.display_tz)
    return f, f + timedelta(days=days)


def test_observation_count_matches_source(mongo, etl):
    etl.run_once()
    n_src = mongo.source["screen_analysis"].count_documents({})
    n_obs = mongo.qe[OBS].count_documents({})
    assert n_obs == n_src


def test_second_once_is_a_noop(etl):
    etl.run_once()
    r = etl.run_once()
    assert r["changes"] == 0, r


def test_exactly_one_primary_per_group(mongo):
    bad = list(mongo.qe[OBS].aggregate([
        {"$group": {"_id": {"f": "$frame_result_id", "s": "$screen_key"},
                    "p": {"$sum": {"$cond": ["$primary", 1, 0]}}}},
        {"$match": {"p": {"$ne": 1}}}, {"$limit": 5}]))
    assert bad == []


def test_manual_example_2025_11_05(mongo):
    """findings example: ..._2025.11.05_12-03-31 + 600 s -> 2025-11-05T04:13:31Z."""
    v = mongo.source["video_analysis"].find_one({"video_name": {"$regex": r"2025\.11\.05_12-03-31"}}, {"_id": 1})
    if v is None:
        pytest.skip("example video not present")
    f = mongo.source["frame_results"].find_one({"video_analysis_id": v["_id"], "minute": 600}, {"_id": 1})
    assert f is not None
    obs = list(mongo.qe[OBS].find({"frame_result_id": f["_id"]}, {"observed_at": 1, "time_basis": 1}))
    assert obs
    for o in obs:
        assert o["observed_at"] == datetime(2025, 11, 5, 4, 13, 31, tzinfo=UTC)
        assert o["time_basis"] == "derived_from_filename"


def test_stream_capture_equals_source_analyzed_minus_tz(mongo):
    sample = list(mongo.qe[OBS].aggregate([
        {"$match": {"source": "stream", "time_basis": {"$in": ["capture", "capture_legacy"]},
                    "flags": {"$ne": "reanalyzed"}}},
        {"$sample": {"size": 200}}, {"$project": {"src_id": 1, "observed_at": 1}}]))
    assert sample
    ids = [s["src_id"] for s in sample]
    src = {d["_id"]: d["analyzed_at"] for d in
           mongo.source["screen_analysis"].find({"_id": {"$in": ids}}, {"analyzed_at": 1})}
    for s in sample:
        raw = src[s["src_id"]]  # naive Taipei wall clock labelled UTC
        assert s["observed_at"] == (raw - timedelta(hours=8)).replace(tzinfo=UTC)


def test_values_match_raw_source(mongo):
    """300 random monitor observations without corrections: m.hr / m.spo2 equal the raw medical_values."""
    sample = list(mongo.qe[OBS].aggregate([
        {"$match": {"device_family": "monitor", "is_corrected": False, "mk": "hr"}},
        {"$sample": {"size": 300}}, {"$project": {"src_id": 1, "m": 1}}]))
    ids = [s["src_id"] for s in sample]
    src = {d["_id"]: d.get("medical_values") or {} for d in
           mongo.source["screen_analysis"].find({"_id": {"$in": ids}}, {"medical_values": 1})}
    checked = 0
    for s in sample:
        mv = src[s["src_id"]]
        raw_hr = mv.get("heart_rate")
        if isinstance(raw_hr, (int, float)) and not isinstance(raw_hr, bool):
            assert s["m"]["hr"] == float(raw_hr)
            checked += 1
        raw_sp = mv.get("spo2")
        if isinstance(raw_sp, (int, float)) and not isinstance(raw_sp, bool) and "spo2" in s["m"]:
            assert s["m"]["spo2"] == float(raw_sp)
    assert checked > 100


def test_rollup_consistent_with_observations(mongo):
    total = sum(r["count"] for r in mongo.qe[ROLLUP].find({"metric": ALL_METRIC}, {"count": 1}))
    n = mongo.qe[OBS].count_documents({"primary": True, "observed_at": {"$type": "date"}})
    assert total == n
    hr_roll = sum(r["count"] for r in mongo.qe[ROLLUP].find({"metric": "hr"}, {"count": 1}))
    assert hr_roll == mongo.qe[OBS].count_documents({"primary": True, "mk": "hr", "observed_at": {"$type": "date"}})


def test_no_secret_fields_in_derived_db(mongo):
    for coll in (OBS, SESSIONS, "unmapped_keys", "meta", ROLLUP):
        for f in ("video_path", "rtsp_url", "api_key", "screen_image_path", "raw_analysis"):
            assert mongo.qe[coll].count_documents({f: {"$exists": True}}) == 0, (coll, f)
    assert mongo.qe[SESSIONS].count_documents({"video_name_masked": {"$regex": "rtsp|uploads|/"}}) == 0


def test_indexes_check_passes(engine):
    from nicu_qe import indexes

    ok, results = indexes.check(engine)
    assert ok, [r for r in results if not r["ok"]]


@pytest.mark.parametrize("step", ["7m", "90m", "1h", "1d", "2d"])
def test_mongo_datetrunc_matches_python_buckets(engine, step):
    f, t = _busy_range(engine, days=6)
    s = engine.series("hr", f, t, step=step, fill="none")
    grid = set(iter_buckets(f, t, parse_step(step), engine.settings.display_tz))
    assert s.points, "expected data in the busiest range"
    assert {p.t for p in s.points} <= grid


def test_engine_end_to_end(engine):
    f, t = _busy_range(engine)
    raw = engine.series("hr", f, t, limit=10000)
    assert raw.points and all(f <= p.t < t for p in raw.points)
    assert all(p.time_basis != "analysis" for p in raw.points)
    assert [p.t for p in raw.points] == sorted(p.t for p in raw.points)
    b = engine.series("hr", f, t, step="5m")
    assert sum(p.n for p in b.points) == len(raw.points)
    st = engine.stats("hr", f, t, group_by="session")
    assert sum(r["count"] for r in st["rows"]) == len([p for p in raw.points if p.v is not None])
    for r in st["rows"]:
        assert r["min"] <= r["p05"] <= r["p50"] <= r["p95"] <= r["max"]
    lat = engine.latest("hr")
    assert lat["data_status"] == "ok" and lat["t"] is not None
    cov = engine.coverage(t - timedelta(days=399), t)
    assert cov["source_table"] == "daily_rollup" and cov["rows"]
    ses = engine.sessions(f, t)
    assert ses["rows"]
    for r in ses["rows"]:
        assert r["first_observed_at"] < t and r["last_observed_at"] >= f
    cat = engine.catalog()
    assert any(m["name"] == "hr" and m["n_obs_total"] > 0 for m in cat["metrics"])
    for r in engine.unmapped_keys()["rows"]:
        for sv in r["sample_values"]:
            assert set(sv) == {"type", "len", "masked"}


def test_coverage_rollup_equals_observation_path(engine):
    f, t = _busy_range(engine, days=3)
    a = engine.coverage(f, t)  # day-aligned -> rollup
    b = engine.coverage(f, t - timedelta(milliseconds=1))  # not aligned -> observations
    assert a["source_table"] == "daily_rollup" and b["source_table"] == "observations"
    assert [(r["bucket"], r["count"], r["metrics"].get("hr")) for r in a["rows"]] == \
           [(r["bucket"], r["count"], r["metrics"].get("hr")) for r in b["rows"]]


def test_llm_model_overrides_primary(engine, mongo):
    d = mongo.qe[OBS].find_one({"primary": False, "mk": "hr", "time_basis": {"$ne": "analysis"}},
                               {"observed_at": 1, "llm_model": 1, "src_id": 1})
    if d is None:
        pytest.skip("no non-primary hr observation")
    t0 = d["observed_at"]
    s = engine.series("hr", t0, t0 + timedelta(seconds=1), llm_model=d["llm_model"], exclude_quality="")
    assert str(d["src_id"]) in {p.src_id for p in s.points}
    s2 = engine.series("hr", t0, t0 + timedelta(seconds=1), exclude_quality="")
    assert str(d["src_id"]) not in {p.src_id for p in s2.points}


def test_reconcile_repairs_tampered_derived_doc(mongo, etl):
    """Stand-in for 'changed in place': the derived doc diverges, reconcile restores it."""
    d = mongo.qe[OBS].find_one({"mk": "hr", "primary": True}, {"src_id": 1, "m": 1, "h": 1})
    mongo.qe[OBS].update_one({"_id": d["_id"]}, {"$set": {"m.hr": -1.0, "h": "tampered"}})
    r = etl.reconcile()
    assert r["updated"] >= 1
    back = mongo.qe[OBS].find_one({"_id": d["_id"]}, {"m": 1, "h": 1})
    assert back["m"]["hr"] == d["m"]["hr"] and back["h"] == d["h"]


def test_http_api_with_token(settings, engine):
    from fastapi.testclient import TestClient

    from nicu_qe.api import create_app

    st = settings.__class__(**{**settings.__dict__, "api_token": "itest-token"})
    app = create_app(st, engine=engine, start_etl=False, allowed_hosts=["testserver"])
    c = TestClient(app)
    assert c.get("/v1/catalog").status_code == 401
    h = {"X-QE-Token": "itest-token"}
    f, t = _busy_range(engine)
    r = c.get("/v1/series", params={"metric": "hr", "from": f.isoformat(), "to": t.isoformat(), "step": "5m",
                                    "t_format": "epoch"}, headers=h)
    assert r.status_code == 200 and r.json()["data_status"] == "ok"
    assert isinstance(r.json()["points"][0]["t"], float)
    bad = c.get("/v1/series", params={"metric": "nope", "from": "2026-01-01", "to": "2026-01-02"}, headers=h)
    assert bad.status_code == 404
    e = c.get("/v1/series", params={"metric": "hr", "from": "2030-01-01", "to": "2030-01-02"}, headers=h)
    assert e.status_code == 200 and e.json()["data_status"] == "empty"
    assert c.get("/health").status_code == 200
