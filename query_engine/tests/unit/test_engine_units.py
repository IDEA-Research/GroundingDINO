"""WP-5 pure logic: parameter validation, match/pipeline builders, buckets and fill, percentiles, masking."""

from datetime import date, datetime, timedelta, timezone

import pytest

from nicu_qe.config import Settings
from nicu_qe.engine import (
    QueryEngine,
    QueryError,
    TooManyValues,
    UnknownMetric,
    clean_str,
    mask_sample,
    median,
    parse_exclude_quality,
    percentile,
    suggest_step,
)
from nicu_qe.tz import bucket_floor, bucket_unit, iter_buckets, parse_step, parse_time_param

UTC = timezone.utc
TPE = "Asia/Taipei"


def engine(**env):
    return QueryEngine(Settings.from_env({"QE_DB": "medical_monitor_qe_test", **env}), mongo=object())


# ------------------------------------------------------------------ percentiles
def test_percentile_linear_interpolation():
    v = [1.0, 2.0, 3.0, 4.0]
    assert percentile(v, 0.5) == 2.5
    assert percentile(v, 0.0) == 1.0 and percentile(v, 1.0) == 4.0
    assert percentile(v, 0.05) == pytest.approx(1.15)  # numpy.percentile(v, 5) == 1.15
    assert percentile(v, 0.95) == pytest.approx(3.85)
    assert percentile([], 0.5) is None
    assert percentile([7.0], 0.95) == 7.0
    with pytest.raises(ValueError):
        percentile(v, 1.5)


def test_median():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 3, 2]) == 2.5
    assert median([]) is None


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("bad", ["$where", " $gt", "a\x00b", "x" * 201])
def test_clean_str_rejects(bad):
    with pytest.raises(QueryError):
        clean_str("session_id", bad)


def test_clean_str_types():
    assert clean_str("x", None) is None
    assert clean_str("x", "  ") is None
    assert clean_str("x", " abc ") == "abc"
    with pytest.raises(QueryError):
        clean_str("x", {"$ne": 1})
    with pytest.raises(QueryError):
        clean_str("x", ["a"])


def test_filters_whitelist_and_enums():
    e = engine()
    assert e.filters({"session_id": "s1", "source": "stream", "device_family": "unknown"}) == {
        "session_id": "s1", "source": "stream", "device_family": "unknown"}
    for bad in ({"foo": "x"}, {"source": "rtsp"}, {"device_family": "philips"}, {"camera": "$ne"},
                {"session_id": {"$ne": None}}):
        with pytest.raises(QueryError):
            e.filters(bad)


def test_metric_must_be_in_catalog():
    e = engine()
    assert e.metric_spec("hr").unit == "bpm"
    for bad in ("nope", "m.hr", "$hr", "HR", "", None, "hr.x"):
        with pytest.raises(UnknownMetric):
            e.metric_spec(bad)
    assert UnknownMetric.status == 404 and QueryError.status == 400


def test_exclude_quality():
    assert parse_exclude_quality(None) == ["out_of_range", "non_numeric", "user_marked_bad"]
    assert parse_exclude_quality("") == []
    assert parse_exclude_quality("out_of_range,corrected") == ["corrected", "out_of_range"]
    with pytest.raises(QueryError):
        parse_exclude_quality("$gt")
    with pytest.raises(QueryError):
        parse_exclude_quality({"a": 1})


def test_time_range_date_only_and_half_open():
    e = engine()
    f, t = e.time_range("2026-09-10", "2026-09-10")
    assert f == datetime(2026, 9, 9, 16, tzinfo=UTC)
    assert t == datetime(2026, 9, 10, 16, tzinfo=UTC)  # to = next day 00:00 +08:00
    f, t = e.time_range("2026-09-10T08:00:00", "2026-09-10T09:00:00+08:00")
    assert f == datetime(2026, 9, 10, 0, tzinfo=UTC) and t == datetime(2026, 9, 10, 1, tzinfo=UTC)
    # '+' turned into a space by URL decoding
    f, _ = e.time_range("2026-09-10T08:00:00 08:00", "2026-09-11")
    assert f == datetime(2026, 9, 10, 0, tzinfo=UTC)
    with pytest.raises(QueryError):
        e.time_range("2026-09-10", "2026-09-09")
    with pytest.raises(QueryError):
        e.time_range("2025-01-01", "2026-09-10")  # > 400 days
    with pytest.raises(QueryError):
        e.time_range(None, "2026-09-10")
    with pytest.raises(QueryError):
        e.time_range("2026-13-01", "2026-09-10")


def test_parse_time_param_variants():
    assert parse_time_param(date(2026, 1, 2), TPE, is_end=False) == datetime(2026, 1, 1, 16, tzinfo=UTC)
    assert parse_time_param(datetime(2026, 1, 2, 8), TPE, is_end=True) == datetime(2026, 1, 2, 0, tzinfo=UTC)
    aware = datetime(2026, 1, 2, 8, tzinfo=UTC)
    assert parse_time_param(aware, TPE, is_end=False) == aware
    assert parse_time_param("2026-01-02T00:00:00Z", TPE, is_end=False) == datetime(2026, 1, 2, tzinfo=UTC)


def test_build_match_shapes():
    e = engine()
    f, t = datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 2, tzinfo=UTC)
    m = e.build_match("hr", f, t, {"session_id": "s"}, False, ["out_of_range"])
    assert m == {"mk": "hr", "observed_at": {"$gte": f, "$lt": t}, "primary": True,
                 "time_basis": {"$ne": "analysis"}, "session_id": "s", "q.hr": {"$nin": ["out_of_range"]}}
    m2 = e.build_match("hr", f, None, {"llm_model": "gpt-4o"}, True, [])
    assert "primary" not in m2 and m2["llm_model"] == "gpt-4o" and "time_basis" not in m2
    assert m2["observed_at"] == {"$gte": f}
    assert "q.hr" not in m2


# ------------------------------------------------------------------ buckets
def test_bucket_unit():
    assert bucket_unit(timedelta(minutes=5)) == ("minute", 5)
    assert bucket_unit(timedelta(seconds=90)) == ("second", 90)
    assert bucket_unit(timedelta(hours=1)) == ("hour", 1)
    assert bucket_unit(timedelta(days=2)) == ("day", 2)
    assert bucket_unit(timedelta(minutes=90)) == ("minute", 90)


def test_bucket_floor_local_alignment():
    t = datetime(2026, 9, 10, 1, 7, 30, tzinfo=UTC)  # 09:07:30 +08:00
    assert bucket_floor(t, parse_step("5m"), TPE) == datetime(2026, 9, 10, 1, 5, tzinfo=UTC)
    # 1d buckets start at local (display tz) midnight = 16:00 UTC the day before
    assert bucket_floor(t, parse_step("1d"), TPE) == datetime(2026, 9, 9, 16, tzinfo=UTC)
    assert bucket_floor(t, parse_step("1h"), TPE) == datetime(2026, 9, 10, 1, tzinfo=UTC)


def test_iter_buckets_half_open():
    f = datetime(2026, 9, 10, 0, tzinfo=UTC)
    t = datetime(2026, 9, 10, 1, tzinfo=UTC)
    b = list(iter_buckets(f, t, parse_step("15m"), TPE))
    assert len(b) == 4 and b[0] == f and b[-1] == datetime(2026, 9, 10, 0, 45, tzinfo=UTC)
    b2 = list(iter_buckets(f + timedelta(minutes=7), t, parse_step("15m"), TPE))
    assert b2[0] == f  # first bucket is floor(from)


def test_suggest_step():
    assert suggest_step(timedelta(days=1), 10000) == "60s"
    assert suggest_step(timedelta(days=30), 10000) == "5m"
    assert suggest_step(timedelta(days=400), 1000) == "12h"


# ------------------------------------------------------------------ masking
def test_mask_sample():
    assert mask_sample("possibly peep") == {"type": "string", "len": 13, "masked": "po" + "*" * 10}
    assert mask_sample(12) == {"type": "int", "len": 2, "masked": "12"}
    assert mask_sample(None) == {"type": "null", "len": 0, "masked": ""}
    m = mask_sample({"name": "王小明"})
    assert m["type"] == "object" and "王小明" not in m["masked"]
    assert mask_sample([1, 2])["type"] == "array"


# ------------------------------------------------------------------ series pipeline with a fake collection
class FakeObs:
    def __init__(self, buckets, count=0):
        self.buckets = buckets
        self.count = count
        self.pipelines = []
        self.finds = []

    def aggregate(self, pipeline, **kw):
        assert "maxTimeMS" in kw
        self.pipelines.append(pipeline)
        if "$facet" in pipeline[-1]:
            return iter([{"tb": [{"_id": "capture", "n": 3}], "qc": [{"_id": "ok", "n": 3}], "ss": [{"_id": "s1"}]}])
        return iter(self.buckets)

    def find_one(self, *a, **kw):
        assert "max_time_ms" in kw
        return None

    def count_documents(self, flt, **kw):
        return self.count

    def find(self, flt, proj=None, **kw):
        self.finds.append((flt, kw))
        return iter([])


class FakeDB(dict):
    pass


class FakeMongo:
    def __init__(self, obs):
        self.qe = FakeDB(observations=obs)


def fake_engine(obs, **env):
    return QueryEngine(Settings.from_env({"QE_DB": "medical_monitor_qe_test", **env}), mongo=FakeMongo(obs))


def test_series_bucket_fill_null_and_pipeline():
    b1 = datetime(2026, 9, 10, 0, 5, tzinfo=UTC)
    obs = FakeObs([{"_id": b1, "n": 3, "v": 150.0, "min": 140, "max": 160}])
    e = fake_engine(obs)
    s = e.series("hr", "2026-09-10T08:00:00", "2026-09-10T08:20:00", step="5m")
    assert [p.t for p in s.points] == [datetime(2026, 9, 10, 0, m, tzinfo=UTC) for m in (0, 5, 10, 15)]
    assert [p.v for p in s.points] == [None, 150.0, None, None]
    assert [p.n for p in s.points] == [0, 3, 0, 0]
    grp = obs.pipelines[0][-2]["$group"]
    assert grp["_id"]["$dateTrunc"] == {"date": "$observed_at", "unit": "minute", "binSize": 5, "timezone": TPE}
    match = obs.pipelines[0][0]["$match"]
    assert match["observed_at"] == {"$gte": datetime(2026, 9, 10, 0, tzinfo=UTC),
                                    "$lt": datetime(2026, 9, 10, 0, 20, tzinfo=UTC)}
    assert s.data_status == "ok" and s.time_basis_counts == {"capture": 3} and s.sessions == ["s1"]
    d = s.to_dict()
    assert d["fill"] == "null" and d["n_points"] == 4 and d["points"][0] == {
        "t": s.points[0].t, "v": None, "n": 0, "min": None, "max": None}


def test_series_fill_none_and_empty_status():
    obs = FakeObs([])
    s = fake_engine(obs).series("hr", "2026-09-10", "2026-09-10", step="1h", fill="none")
    assert s.points == [] and s.data_status == "empty"
    s2 = fake_engine(FakeObs([])).series("hr", "2026-09-10", "2026-09-10", step="1h")
    assert len(s2.points) == 24 and s2.data_status == "empty"


def test_series_truncation_suggests_step():
    obs = FakeObs([])
    s = fake_engine(obs, QE_MAX_POINTS="10").series("hr", "2026-09-10", "2026-09-10", step="1h")
    assert s.truncated and len(s.points) == 10 and s.suggested_step == "3h"


def test_series_median_limit():
    obs = FakeObs([], count=11)
    with pytest.raises(TooManyValues):
        fake_engine(obs, QE_MAX_STATS_VALUES="10").series("hr", "2026-09-10", "2026-09-10", step="1h", agg="median")


def test_series_median_python_side():
    b = datetime(2026, 9, 9, 16, tzinfo=UTC)
    obs = FakeObs([{"_id": b, "n": 4, "vals": [1.0, 9.0, 3.0, 4.0], "min": 1, "max": 9}], count=4)
    s = fake_engine(obs).series("hr", "2026-09-10", "2026-09-10", step="1d", agg="median")
    assert s.points[0].v == 3.5
    assert "vals" in obs.pipelines[0][-2]["$group"]


def test_series_bad_params():
    e = fake_engine(FakeObs([]))
    with pytest.raises(QueryError):
        e.series("hr", "2026-09-10", "2026-09-10", step="5x")
    with pytest.raises(QueryError):
        e.series("hr", "2026-09-10", "2026-09-10", step="5m", agg="sum")
    with pytest.raises(QueryError):
        e.series("hr", "2026-09-10", "2026-09-10", fill="zero")
    with pytest.raises(QueryError):
        e.series("hr", "2026-09-10", "2026-09-10", limit=0)
    with pytest.raises(UnknownMetric):
        e.series("xyz", "2026-09-10", "2026-09-10")


def test_series_raw_limit_and_sort():
    obs = FakeObs([])
    fake_engine(obs).series("spo2", "2026-09-10", "2026-09-10", limit=5)
    flt, kw = obs.finds[0]
    assert kw["sort"] == [("observed_at", 1)] and kw["limit"] == 6 and "max_time_ms" in kw
    assert flt["mk"] == "spo2" and flt["primary"] is True


def test_optional_time_range_bad_value_is_400():
    with pytest.raises(QueryError):
        engine().time_range("garbage", None, required=False)
