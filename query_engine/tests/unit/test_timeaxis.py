"""observed_at rules (DESIGN section 5) with the findings-mongo examples."""

from datetime import datetime, timedelta, timezone

import pytest

from nicu_qe.timeaxis import (
    frame_offset_from_path,
    infer_source,
    offset_mismatch,
    parse_video_name_time,
    resolve,
    resolve_observed_at,
)
from nicu_qe.tz import source_wallclock_to_utc, to_display

TPE = "Asia/Taipei"


def test_manual_example_from_findings():
    name = "240f3d04-aaaa-bbbb-cccc-dddddddddddd_2025.11.05_12-03-31_redacted"
    start_local = parse_video_name_time(name)
    assert start_local == datetime(2025, 11, 5, 12, 3, 31)
    start_utc = source_wallclock_to_utc(start_local, TPE)
    analyzed = source_wallclock_to_utc(datetime(2026, 9, 22, 10, 0), TPE)
    obs, basis, conf = resolve_observed_at("manual", analyzed, analyzed, start_utc, 600)
    assert obs == datetime(2025, 11, 5, 4, 13, 31, tzinfo=timezone.utc)
    assert to_display(obs, TPE).isoformat() == "2025-11-05T12:13:31+08:00"
    assert (basis, conf) == ("derived_from_filename", "medium")


def test_stream_capture():
    analyzed = source_wallclock_to_utc(datetime(2026, 9, 21, 14, 58, 0), TPE)
    processed = source_wallclock_to_utc(datetime(2026, 9, 21, 14, 58, 20), TPE)
    obs, basis, conf = resolve_observed_at("stream", analyzed, processed, None, 66.24)
    assert obs.isoformat() == "2026-09-21T06:58:00+00:00"
    assert (basis, conf) == ("capture", "high")


def test_stream_reanalyzed_uses_processed_at():
    processed = source_wallclock_to_utc(datetime(2026, 9, 21, 14, 58, 20), TPE)
    analyzed = processed + timedelta(days=3)
    obs, basis, conf = resolve_observed_at("stream", analyzed, processed, None, 66.24)
    assert obs == processed
    assert (basis, conf) == ("capture_approx", "medium")


def test_stream_drift_boundary():
    # v2 (REVIEW M1): asymmetric rule, re-analysed only when analyzed_at > processed_at + 5 s.
    # A long LLM backlog (processed_at far AFTER analyzed_at) keeps the capture time.
    a = datetime(2026, 9, 21, 6, 58, tzinfo=timezone.utc)
    assert resolve_observed_at("stream", a, a + timedelta(minutes=5), None, None)[1] == "capture"
    assert resolve_observed_at("stream", a, a + timedelta(minutes=30), None, None)[1] == "capture"
    assert resolve_observed_at("stream", a, a - timedelta(seconds=5), None, None)[1] == "capture"
    assert resolve_observed_at("stream", a, a - timedelta(seconds=6), None, None)[1] == "capture_approx"
    assert resolve_observed_at("stream", a, None, None, None)[1] == "capture"


def test_stream_reanalyzed_restores_from_sibling():
    processed = datetime(2026, 9, 21, 6, 58, 20, tzinfo=timezone.utc)
    sibling = datetime(2026, 9, 21, 6, 58, 0, tzinfo=timezone.utc)
    tr = resolve("stream", processed + timedelta(days=2), processed, None, None, sibling_capture_utc=sibling)
    assert (tr.observed_at, tr.time_basis, tr.time_confidence) == (sibling, "capture", "high")
    assert "reanalyzed" in tr.flags and "time_from_sibling" in tr.flags
    tr2 = resolve("stream", processed + timedelta(days=2), processed, None, None)
    assert (tr2.observed_at, tr2.time_basis) == (processed, "capture_approx")
    assert tr2.flags == ("reanalyzed",)


def test_stream_legacy_before_reliable_from():
    cut = datetime(2026, 5, 31, 16, 0, tzinfo=timezone.utc)  # 2026-06-01 00:00 +08:00
    early = datetime(2026, 2, 1, 1, 0, tzinfo=timezone.utc)
    late = datetime(2026, 9, 1, 1, 0, tzinfo=timezone.utc)
    assert resolve("stream", early, early, None, None, stream_reliable_from_utc=cut).time_basis == "capture_legacy"
    assert resolve("stream", early, early, None, None, stream_reliable_from_utc=cut).time_confidence == "medium"
    assert resolve("stream", late, late, None, None, stream_reliable_from_utc=cut).time_basis == "capture"


def test_edited_video_lowers_confidence():
    start = datetime(2025, 11, 5, 4, 3, 31, tzinfo=timezone.utc)
    tr = resolve("manual", None, None, start, 600, edited=True)
    assert (tr.time_basis, tr.time_confidence, tr.flags) == ("derived_from_filename", "low", ("edited_video",))


def test_edited_video_flag_on_analysis_basis():
    analyzed = datetime(2026, 9, 1, 1, 0, tzinfo=timezone.utc)
    from nicu_qe.config import DEFAULT_EDITED_MARKERS
    from nicu_qe.timeaxis import is_edited_video
    edited = is_edited_video("patient_遮蔽.mp4", DEFAULT_EDITED_MARKERS.split(","))
    assert edited is True
    tr = resolve("manual", analyzed, None, None, None, edited=edited)
    assert (tr.time_basis, tr.time_confidence) == ("analysis", "low")
    assert "edited_video" in tr.flags
    tr2 = resolve("stream", analyzed, analyzed, None, None, edited=True)
    assert (tr2.time_basis, tr2.time_confidence) == ("capture", "low") and "edited_video" in tr2.flags
    assert "edited_video" not in resolve("manual", analyzed, None, None, None).flags


def test_future_time_falls_back_to_analysis():
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    analyzed = datetime(2026, 9, 27, tzinfo=timezone.utc)
    start = datetime(2026, 10, 5, tzinfo=timezone.utc)
    tr = resolve("manual", analyzed, None, start, 0, now_utc=now)
    assert (tr.observed_at, tr.time_basis, tr.time_confidence) == (analyzed, "analysis", "low")
    assert "future_time" in tr.flags
    ok = resolve("manual", analyzed, None, now + timedelta(hours=20), 0, now_utc=now)
    assert ok.time_basis == "derived_from_filename"


@pytest.mark.parametrize("name,expected", [
    ("x_12025.11.05_12-03-31", None),                              # digit before the year: no boundary
    ("x_2025.11.05_12-03-311", None),                              # digit after the seconds
    ("x_2025.11.05_12-03-31_2026.01.01_00-00-00", datetime(2025, 11, 5, 12, 3, 31)),  # first group wins
    ("x_2025.11.05_25-61-61", None),                               # invalid time -> None, no exception
    ("x_2025.02.30_10-00-00_2025.03.01_10-00-00", None),           # first group invalid -> None (no guessing)
])
def test_parse_video_name_time_v2(name, expected):
    assert parse_video_name_time(name, now_year=2026) == expected


@pytest.mark.parametrize("name,expected", [
    ("uuid_2025.11.05_12-03-31_redacted", True),
    ("uuid_2025.11.05_12-03-31_遮蔽", True),
    ("uuid_2025.11.05_12-03-31_cut2", True),
    ("uuid_2025.11.05_12-03-31_circuit", False),
    ("uuid_2025.11.05_12-03-31", False),
    (None, False),
])
def test_is_edited_video(name, expected):
    from nicu_qe.config import DEFAULT_EDITED_MARKERS
    from nicu_qe.timeaxis import is_edited_video
    assert is_edited_video(name, DEFAULT_EDITED_MARKERS.split(",")) is expected


@pytest.mark.parametrize("name,expected", [
    ("uuid_2025.11.05_12-03-31_redacted", datetime(2025, 11, 5, 12, 3, 31)),
    ("uuid_2026.03.19_13-44-45", datetime(2026, 3, 19, 13, 44, 45)),
    ("2025.11.13 06-33-25", datetime(2025, 11, 13, 6, 33, 25)),
    ("2025.11.13 06-33-25.mp4", datetime(2025, 11, 13, 6, 33, 25)),
    ("med20250812-3", None),
    ("遮蔽", None),
    ("移除螢幕", None),
    ("uuid_1999.01.01_00-00-00", None),       # year before 2015
    ("uuid_2025.13.01_00-00-00", None),       # invalid month
    ("uuid_2099.01.01_00-00-00", None),       # beyond this year + 1
    (None, None),
])
def test_parse_video_name_time(name, expected):
    assert parse_video_name_time(name) == expected


def test_year_window_uses_now_year():
    assert parse_video_name_time("x_2027.01.01_00-00-00", now_year=2026) == datetime(2027, 1, 1)
    assert parse_video_name_time("x_2028.01.01_00-00-00", now_year=2026) is None


@pytest.mark.parametrize("name", ["med20250812-3", "遮蔽", "移除螢幕"])
def test_unparseable_names_fall_back_to_analysis(name):
    analyzed = datetime(2026, 9, 22, 2, 0, tzinfo=timezone.utc)
    start = parse_video_name_time(name)
    assert start is None
    obs, basis, conf = resolve_observed_at("manual", analyzed, analyzed, start, 600)
    assert (obs, basis, conf) == (analyzed, "analysis", "low")


def test_dataset_follows_manual_rule():
    start = datetime(2025, 11, 13, 6, 33, 25, tzinfo=timezone.utc)
    obs, basis, _ = resolve_observed_at("dataset", None, None, start, 60)
    assert obs == start + timedelta(seconds=60) and basis == "derived_from_filename"


def test_resolve_dataclass():
    tr = resolve("manual", None, None, None, None)
    assert (tr.observed_at, tr.time_basis, tr.time_confidence) == (None, "analysis", "low")


@pytest.mark.parametrize("source,path,expected", [
    (None, "RTSP:rtsp://u:p@192.168.0.100:554/Streaming/Channels/101", "stream"),
    (None, "uploads/abc_2025.11.05_12-03-31.mp4", "manual"),
    (None, "/somewhere/else.mp4", "manual"),
    (None, None, "manual"),
    ("", "RTSP:rtsp://x", "stream"),
    ("stream", "uploads/x.mp4", "stream"),
    ("dataset", None, "dataset"),
])
def test_infer_source(source, path, expected):
    assert infer_source(source, path) == expected


@pytest.mark.parametrize("path,expected", [
    ("video_screen_analysis/<uuid>/frames/frame_0600s.jpg", 600),
    ("frames/frame_00425_15249s.jpg", 15249),
    ("stream_frame_ab12cd34_00001_0066s.jpg", 66),
    ("video_screen_analysis/<uuid>/screens/frame_0600s-screen_crop_7.jpg", 600),
    ("frames/frame.jpg", None),
    (None, None),
])
def test_frame_offset_from_path(path, expected):
    assert frame_offset_from_path(path) == expected


def test_offset_mismatch():
    assert not offset_mismatch(600, "frame_0600s.jpg")
    assert not offset_mismatch(15249.69, "frame_00425_15249s.jpg")
    assert offset_mismatch(660, "frame_0600s.jpg")
    assert not offset_mismatch(600, None)
