"""WP-4 pure logic: read-only wrapper, DB-name guard, primary groups, observation building, filters."""

from datetime import datetime, timedelta, timezone

import pytest
from bson import ObjectId

from nicu_qe.aliases import get_registry
from nicu_qe.config import Settings, UnsafeConfigError, guard_db_names
from nicu_qe.etl import (
    EtlContext,
    analyzed_filter,
    build_frame_observations,
    correction_filter,
    doc_hash,
    id_filter,
    mask_video_name,
)
from nicu_qe.mongo import Mongo, ReadOnlyCollection, ReadOnlySource, ReadOnlyViolation
from nicu_qe.normalize import assign_primary, choose_primary

UTC = timezone.utc


# ------------------------------------------------------------------ read-only wrapper
class RecordingColl:
    name = "screen_analysis"

    def __init__(self):
        self.calls = []

    def __getattr__(self, item):
        def f(*a, **kw):
            self.calls.append((item, a, kw))
            return []
        return f


class RecordingDB:
    name = "medical_monitor_db"

    def __init__(self):
        self.colls = {}

    def __getitem__(self, n):
        return self.colls.setdefault(n, RecordingColl())


WRITE_METHODS = ["insert_one", "insert_many", "update_one", "update_many", "replace_one", "delete_one",
                 "delete_many", "bulk_write", "drop", "create_index", "create_indexes", "drop_index",
                 "drop_indexes", "find_one_and_update", "find_one_and_replace", "find_one_and_delete", "rename",
                 "watch", "with_options", "database"]


@pytest.mark.parametrize("meth", WRITE_METHODS)
def test_readonly_collection_has_no_write_methods(meth):
    ro = ReadOnlyCollection(RecordingColl(), 1000)
    assert not hasattr(ro, meth)


def test_readonly_source_has_no_escape_hatch():
    src = ReadOnlySource(RecordingDB(), 1000)
    for attr in ("client", "command", "drop_collection", "create_collection", "get_collection"):
        assert not hasattr(src, attr)
    with pytest.raises(ReadOnlyViolation):
        src.foo = 1
    with pytest.raises(ReadOnlyViolation):
        src["x"].name = "y"


def test_readonly_calls_carry_max_time():
    db = RecordingDB()
    src = ReadOnlySource(db, 1234)
    src["screen_analysis"].find({"a": 1}, {"b": 1}, max_time_ms=99999)
    src["screen_analysis"].aggregate([{"$match": {}}])
    src["screen_analysis"].count_documents({})
    src["screen_analysis"].distinct("k")
    src["screen_analysis"].estimated_document_count()
    src["screen_analysis"].find_one({})
    calls = db["screen_analysis"].calls
    assert [c[0] for c in calls] == ["find", "aggregate", "count_documents", "distinct",
                                     "estimated_document_count", "find_one"]
    assert calls[0][2]["max_time_ms"] == 1234  # caller cannot raise the limit
    assert calls[1][2]["maxTimeMS"] == 1234
    assert calls[2][2]["maxTimeMS"] == 1234
    assert calls[5][2]["max_time_ms"] == 1234


@pytest.mark.parametrize("stage", [{"$out": "x"}, {"$merge": {"into": "x"}}])
def test_readonly_aggregate_refuses_write_stages(stage):
    src = ReadOnlySource(RecordingDB(), 1000)
    with pytest.raises(ReadOnlyViolation):
        src["screen_analysis"].aggregate([{"$match": {}}, stage])


# ------------------------------------------------------------------ QE_DB guard
@pytest.mark.parametrize("qe,src", [
    ("medical_monitor_db", "medical_monitor_db"),
    ("medical_monitor_qe_prod", "medical_monitor_db"),
    ("admin", "medical_monitor_db"),
    ("medical_monitor_qe ", "medical_monitor_db"),
    ("medical_monitor_qe", "medical_monitor_qe"),
    ("", "medical_monitor_db"),
])
def test_guard_db_names_rejects(qe, src):
    with pytest.raises(UnsafeConfigError):
        guard_db_names(qe, src)


def test_guard_db_names_accepts():
    guard_db_names("medical_monitor_qe", "medical_monitor_db")
    guard_db_names("medical_monitor_qe_test", "medical_monitor_db")


def test_settings_refuse_source_db_as_qe_db():
    with pytest.raises(UnsafeConfigError):
        Settings.from_env({"QE_DB": "medical_monitor_db"})
    with pytest.raises(UnsafeConfigError):
        Settings.from_env({"QE_DB": "medical_monitor_qe", "QE_SOURCE_DB": "medical_monitor_qe"})


def test_mongo_handle_rechecks_guard():
    st = Settings.from_env({"QE_DB": "medical_monitor_qe_test"})
    bad = st.__class__(**{**st.__dict__, "db": "medical_monitor_db"})
    with pytest.raises(UnsafeConfigError):
        Mongo(bad, source_client={}, qe_client={})


def test_validate_for_serve():
    Settings.from_env({"QE_API_TOKEN": "t"}).validate_for_serve()
    Settings.from_env({"QE_ALLOW_NO_TOKEN": "1", "QE_BIND": "127.0.0.1:8021"}).validate_for_serve()
    with pytest.raises(UnsafeConfigError):
        Settings.from_env({}).validate_for_serve()
    with pytest.raises(UnsafeConfigError):
        Settings.from_env({"QE_ALLOW_NO_TOKEN": "1", "QE_BIND": "0.0.0.0:8021"}).validate_for_serve()
    with pytest.raises(ValueError):
        Settings.from_env({"QE_ALLOW_NO_TOKEN": "maybe"})


def test_new_settings_defaults():
    s = Settings.from_env({"QE_SOURCE_TZ": "Asia/Tokyo"})
    assert s.filename_tz == "Asia/Tokyo"  # defaults to QE_SOURCE_TZ
    assert s.stream_reliable_from.isoformat() == "2026-06-01"
    assert s.etl_reconcile_s == 21600 and s.max_stats_values == 200000
    assert "redacted" in s.edited_markers
    assert Settings.from_env({"QE_FILENAME_TZ": "UTC"}).filename_tz == "UTC"


# ------------------------------------------------------------------ primary groups (H8)
def test_assign_primary_one_per_group():
    f1 = ObjectId()
    a = datetime(2026, 1, 1, tzinfo=UTC)
    c = [
        {"_id": ObjectId(), "frame_result_id": f1, "original_screen_number": 1, "success": True, "analyzed_at": a},
        {"_id": ObjectId(), "frame_result_id": f1, "original_screen_number": 1, "success": True,
         "analyzed_at": a + timedelta(seconds=1)},
        {"_id": ObjectId(), "frame_result_id": f1, "original_screen_number": 2, "success": False, "analyzed_at": a},
        {"_id": ObjectId(), "frame_result_id": f1, "original_screen_number": 2, "success": True, "analyzed_at": a,
         "is_corrected": False},
    ]
    assert assign_primary(c) == [False, True, False, True]


def test_assign_primary_tiebreak_by_larger_id_and_fallback_to_screen_number():
    f1 = ObjectId()
    a = datetime(2026, 1, 1, tzinfo=UTC)
    o1, o2 = ObjectId(), ObjectId()
    c = [
        {"_id": o1, "frame_result_id": f1, "original_screen_number": None, "screen_number": 3, "success": True, "analyzed_at": a},
        {"_id": o2, "frame_result_id": f1, "screen_number": 3, "success": True, "analyzed_at": a},
        {"_id": ObjectId(), "frame_result_id": f1, "screen_number": 4, "success": True, "analyzed_at": a},
    ]
    assert assign_primary(c) == [False, True, True]
    # v1 behaviour without ids is unchanged (first wins on ties)
    assert choose_primary([{"success": True}, {"success": True}]) == 0


def test_assign_primary_corrected_beats_newer():
    f1 = ObjectId()
    a = datetime(2026, 1, 1, tzinfo=UTC)
    c = [
        {"_id": ObjectId(), "frame_result_id": f1, "original_screen_number": 1, "success": True,
         "analyzed_at": a, "is_corrected": True},
        {"_id": ObjectId(), "frame_result_id": f1, "original_screen_number": 1, "success": True,
         "analyzed_at": a + timedelta(days=1)},
    ]
    assert assign_primary(c) == [True, False]


# ------------------------------------------------------------------ observation building
def _ctx(now=datetime(2026, 9, 28, tzinfo=UTC)):
    return EtlContext(Settings.from_env({"QE_DB": "medical_monitor_qe_test"}), get_registry(), now)


def _video(source="stream", name="101", start=None, edited=False):
    return {"session_id": "sess", "source": source, "camera": "camera-1", "patient_id": None, "task_name": None,
            "video_start_utc": start, "edited": edited}


def test_stream_frame_sibling_restore_and_primary():
    fid, vid = ObjectId(), ObjectId()
    cap = datetime(2026, 9, 21, 14, 58, 0)  # source wall clock (naive)
    frame = {"_id": fid, "video_analysis_id": vid, "processed_at": cap + timedelta(seconds=20), "minute": 66.2}
    s_ok = {"_id": ObjectId(), "frame_result_id": fid, "original_screen_number": 1, "screen_number": 1,
            "detected_model": "Philips mp30", "medical_values": {"heart_rate": 150, "spo2": 95}, "success": True,
            "analyzed_at": cap, "llm_model": "gpt-4o"}
    s_re = {"_id": ObjectId(), "frame_result_id": fid, "original_screen_number": 2, "screen_number": 2,
            "detected_model": "Philips mp30", "medical_values": {"heart_rate": 151}, "success": True,
            "analyzed_at": cap + timedelta(days=2), "llm_model": "gpt-4o", "has_glare": True}
    docs = build_frame_observations([s_ok, s_re], frame, _video(), _ctx())
    d_ok, d_re = docs
    assert d_ok["observed_at"] == datetime(2026, 9, 21, 6, 58, tzinfo=UTC)
    assert (d_ok["time_basis"], d_ok["time_confidence"]) == ("capture", "high")
    assert d_re["observed_at"] == d_ok["observed_at"]  # restored from the sibling screen
    assert "reanalyzed" in d_re["flags"] and "time_from_sibling" in d_re["flags"]
    assert d_re["q"]["hr"] == ["user_marked_bad"] and "user_marked_bad" in d_re["flags"]
    assert d_ok["mk"] == ["hr", "spo2"] and d_ok["m"]["hr"] == 150.0
    assert d_ok["primary"] and d_re["primary"]  # different screens -> two groups
    assert d_ok["processed_at_utc"] == datetime(2026, 9, 21, 6, 58, 20, tzinfo=UTC)
    assert d_ok["analyzed_at_utc"] == datetime(2026, 9, 21, 6, 58, tzinfo=UTC)
    assert all(len(d["h"]) == 40 for d in docs)
    for forbidden in ("video_path", "rtsp_url", "screen_image_path", "raw_analysis", "api_key"):
        assert all(forbidden not in d for d in docs)


def test_stream_reanalyzed_without_sibling_uses_processed():
    fid = ObjectId()
    cap = datetime(2026, 9, 21, 14, 58, 0)
    frame = {"_id": fid, "processed_at": cap + timedelta(seconds=20), "minute": 0}
    s = {"_id": ObjectId(), "frame_result_id": fid, "original_screen_number": 1, "detected_model": "unknown",
         "medical_values": {}, "success": True, "analyzed_at": cap + timedelta(days=1)}
    d = build_frame_observations([s], frame, _video(), _ctx())[0]
    assert d["time_basis"] == "capture_approx" and d["observed_at"] == datetime(2026, 9, 21, 6, 58, 20, tzinfo=UTC)
    assert "empty" in d["flags"] and d["mk"] == []


def test_stream_legacy_before_reliable_from():
    fid = ObjectId()
    cap = datetime(2026, 2, 6, 10, 0, 0)
    frame = {"_id": fid, "processed_at": cap + timedelta(seconds=20), "minute": 0}
    s = {"_id": ObjectId(), "frame_result_id": fid, "original_screen_number": 1, "detected_model": "Philips mp30",
         "medical_values": {"heart_rate": 140}, "success": True, "analyzed_at": cap}
    d = build_frame_observations([s], frame, _video(), _ctx())[0]
    assert (d["time_basis"], d["time_confidence"]) == ("capture_legacy", "medium")


def test_manual_frame_filename_time_and_duplicates():
    from nicu_qe.tz import wallclock_to_utc

    fid = ObjectId()
    start = wallclock_to_utc(datetime(2025, 11, 5, 12, 3, 31), "Asia/Taipei")
    frame = {"_id": fid, "processed_at": datetime(2026, 9, 23, 21, 0), "minute": 600,
             "original_image_path": "video_screen_analysis/x/frames/frame_0600s.jpg"}
    base = {"frame_result_id": fid, "original_screen_number": 7, "screen_number": 4,
            "detected_model": "Philips mp30", "success": True}
    s1 = dict(base, _id=ObjectId(), medical_values={"heart_rate": 168}, analyzed_at=datetime(2026, 9, 23, 21, 11),
              llm_model="gpt-4o")
    s2 = dict(base, _id=ObjectId(), medical_values={"heart_rate": 170}, analyzed_at=datetime(2026, 9, 23, 21, 12),
              llm_model="google/gemini-3.1-flash-lite")
    docs = build_frame_observations([s1, s2], frame, _video("manual", start=start, edited=True), _ctx())
    assert docs[0]["observed_at"] == datetime(2025, 11, 5, 4, 13, 31, tzinfo=UTC)
    assert docs[0]["time_basis"] == "derived_from_filename" and docs[0]["time_confidence"] == "low"
    assert "edited_video" in docs[0]["flags"]
    assert [d["primary"] for d in docs] == [False, True]  # same screen, newer analysis wins


def test_manual_offset_mismatch_flag():
    fid = ObjectId()
    start = datetime(2025, 11, 5, 4, 3, 31, tzinfo=UTC)
    frame = {"_id": fid, "minute": 660, "original_image_path": "frames/frame_0600s.jpg"}
    s = {"_id": ObjectId(), "frame_result_id": fid, "original_screen_number": 1, "detected_model": "Philips mp30",
         "medical_values": {"heart_rate": 100}, "success": True, "analyzed_at": datetime(2026, 1, 1)}
    d = build_frame_observations([s], frame, _video("manual", start=start), _ctx())[0]
    assert "offset_mismatch" in d["flags"] and "offset_mismatch" in d["q"]["hr"]


def test_orphan_screen_without_frame_or_video():
    s = {"_id": ObjectId(), "frame_result_id": ObjectId(), "original_screen_number": 1,
         "detected_model": "Philips mp30", "medical_values": {"heart_rate": 100}, "success": True,
         "analyzed_at": datetime(2026, 1, 1, 12)}
    d = build_frame_observations([s], None, None, _ctx())[0]
    assert d["source"] == "manual" and d["time_basis"] == "analysis" and d["session_id"] is None


def test_future_time_flag_on_values():
    fid = ObjectId()
    frame = {"_id": fid, "minute": 0}
    s = {"_id": ObjectId(), "frame_result_id": fid, "original_screen_number": 1, "detected_model": "Philips mp30",
         "medical_values": {"heart_rate": 100}, "success": True, "analyzed_at": datetime(2026, 9, 1)}
    start = datetime(2027, 1, 1, tzinfo=UTC)
    d = build_frame_observations([s], frame, _video("manual", start=start), _ctx())[0]
    assert d["time_basis"] == "analysis" and "future_time" in d["flags"] and "future_time" in d["q"]["hr"]


def test_doc_hash_ignores_etl_at():
    a = {"src_id": 1, "m": {"hr": 1.0}, "etl_at": datetime(2026, 1, 1)}
    b = {"src_id": 1, "m": {"hr": 1.0}, "etl_at": datetime(2027, 1, 1), "h": "x"}
    assert doc_hash(a) == doc_hash(b)
    assert doc_hash(a) != doc_hash({"src_id": 1, "m": {"hr": 2.0}})


def test_mask_video_name():
    assert mask_video_name("240f3d04-aaaa_2025.11.05_12-03-31_redacted") == "2025.11.05_12-03-31"
    assert mask_video_name("101") is None
    assert mask_video_name(None) is None


# ------------------------------------------------------------------ incremental filters (H1, M2, M5)
def test_incremental_filters():
    wm = ObjectId.from_datetime(datetime(2026, 9, 23, 5, 11, 30, tzinfo=UTC))
    f = id_filter(wm)
    assert f["_id"]["$gt"].generation_time == datetime(2026, 9, 23, 5, 6, 30, tzinfo=UTC)
    a = analyzed_filter(datetime(2026, 9, 23, 13, 11, 30, tzinfo=UTC))  # tz label is dropped
    assert a == {"analyzed_at": {"$gt": datetime(2026, 9, 23, 13, 1, 30)}}
    c = correction_filter({"last_corrected_at": datetime(2026, 9, 1), "quality_marked_at": None})
    assert c == {"$or": [{"last_corrected_at": {"$gt": datetime(2026, 9, 1)}},
                         {"quality_marked_at": {"$type": "date"}},
                         {"model_corrected_at": {"$type": "date"}}]}


# ------------------------------------------------------------------ incremental selection end-to-end (fake source)
def _match(doc, flt):
    for k, cond in flt.items():
        if k == "$or":
            if not any(_match(doc, c) for c in cond):
                return False
            continue
        v = doc.get(k)
        if isinstance(cond, dict):
            for op, arg in cond.items():
                if op == "$gt" and not (v is not None and v > arg):
                    return False
                if op == "$in" and v not in arg:
                    return False
                if op == "$type" and not isinstance(v, datetime):
                    return False
        elif v != cond:
            return False
    return True


class FakeSrcColl:
    def __init__(self, docs):
        self.docs = docs

    def find(self, flt=None, projection=None, **kw):
        return [d for d in self.docs if _match(d, flt or {})]


class FakeSrc(dict):
    def __getitem__(self, name):
        return FakeSrcColl(dict.get(self, name, []))


class Counter0:
    def estimated_document_count(self):
        return 0


def test_run_once_selects_inplace_reanalysis_new_ids_corrections_and_video_changes(monkeypatch):
    from nicu_qe.etl import Etl, load_videos

    t0 = datetime(2026, 9, 20, 10, 0)  # source coordinates
    wm_id = ObjectId.from_datetime(datetime(2026, 9, 20, 2, 0, tzinfo=UTC))
    old_id = ObjectId.from_datetime(datetime(2026, 9, 1, tzinfo=UTC))
    new_id = ObjectId.from_datetime(datetime(2026, 9, 20, 3, 0, tzinfo=UTC))
    late_id = ObjectId.from_datetime(datetime(2026, 9, 20, 1, 57, tzinfo=UTC))  # committed late (M2)
    f_re, f_new, f_late, f_corr, f_vid, f_quiet = (ObjectId() for _ in range(6))
    v_changed, v_same = ObjectId(), ObjectId()
    screens = [
        {"_id": old_id, "frame_result_id": f_re, "analyzed_at": t0 + timedelta(hours=1)},  # re-analysed in place
        {"_id": new_id, "frame_result_id": f_new, "analyzed_at": t0 - timedelta(days=3)},
        {"_id": late_id, "frame_result_id": f_late, "analyzed_at": t0 - timedelta(days=5)},
        {"_id": ObjectId.from_datetime(datetime(2026, 8, 1, tzinfo=UTC)), "frame_result_id": f_corr,
         "analyzed_at": t0 - timedelta(days=30), "last_corrected_at": t0 + timedelta(minutes=1)},
        {"_id": ObjectId.from_datetime(datetime(2026, 8, 2, tzinfo=UTC)), "frame_result_id": f_quiet,
         "analyzed_at": t0 - timedelta(days=30), "last_corrected_at": t0 - timedelta(days=1)},
    ]
    videos = [{"_id": v_changed, "session_id": "s1", "video_name": "x_2025.11.05_12-03-31", "source": "manual"},
              {"_id": v_same, "session_id": "s2", "video_name": "y", "source": "manual"}]
    frames = [{"_id": f_vid, "video_analysis_id": v_changed}, {"_id": f_quiet, "video_analysis_id": v_same}]
    src = FakeSrc(screen_analysis=screens, video_analysis=videos, frame_results=frames)

    class FakeMongo:
        source = src

    st = Settings.from_env({"QE_DB": "medical_monitor_qe_test"})
    hashes = {str(k): v["hash"] for k, v in load_videos(FakeMongo, st).items()}
    hashes[str(v_changed)] = "old-hash"
    meta = {"watermark_id": wm_id, "watermark_analyzed_src": t0, "video_hashes": hashes,
            "correction_watermarks_src": {"last_corrected_at": t0}, "last_correction_rescan_at": None,
            "last_reconcile_at": datetime.now(UTC)}
    etl = Etl.__new__(Etl)
    etl.mongo, etl.st, etl.registry, etl.src, etl.ms, etl.owner = FakeMongo, st, get_registry(), src, 1000, "t"
    etl.qe = {"observations": Counter0(), "sessions": Counter0()}
    seen = {}
    monkeypatch.setattr(etl, "meta", lambda: meta, raising=False)
    monkeypatch.setattr(etl, "acquire_lease", lambda ttl: True, raising=False)
    monkeypatch.setattr(etl, "release_lease", lambda: None, raising=False)
    monkeypatch.setattr(etl, "_set_meta", lambda f: seen.setdefault("meta", f), raising=False)

    def fake_process(fids, videos, ctx, stats=None):
        seen["fids"] = set(fids)
        return {"inserted": 0, "updated": 1, "deleted": 0, "unchanged": 0, "frames": 0, "days": set(),
                "sessions": set()}

    monkeypatch.setattr(etl, "process_frames", fake_process, raising=False)
    for name in ("rebuild_rollup", "rebuild_sessions", "rebuild_unmapped"):
        monkeypatch.setattr(etl, name, lambda *a, **k: 0, raising=False)
    r = etl.run_once()
    assert seen["fids"] == {f_re, f_new, f_late, f_corr, f_vid}  # f_quiet untouched
    assert r["reasons"]["video_changed"] == 1 and r["reasons"]["correction"] == 1
    m = seen["meta"]
    assert m["watermark_id"] == new_id
    assert m["watermark_analyzed_src"] == t0 + timedelta(hours=1)
    assert m["correction_watermarks_src"]["last_corrected_at"] == t0 + timedelta(minutes=1)
    assert m["video_hashes"][str(v_changed)] != "old-hash"
