"""Timezone helpers and Settings.from_env."""

from datetime import datetime, timedelta, timezone

import pytest

from nicu_qe.config import Settings
from nicu_qe.tz import parse_duration, parse_iso, parse_step, source_wallclock_to_utc, to_display

TPE = "Asia/Taipei"


def test_source_wallclock_naive():
    utc = source_wallclock_to_utc(datetime(2026, 9, 21, 14, 58, 0), TPE)
    assert utc.isoformat() == "2026-09-21T06:58:00+00:00"


def test_source_wallclock_aware_utc_label():
    # pymongo tz_aware=True returns the same value labelled UTC
    utc = source_wallclock_to_utc(datetime(2026, 9, 21, 14, 58, 0, tzinfo=timezone.utc), TPE)
    assert utc.isoformat() == "2026-09-21T06:58:00+00:00"


def test_source_wallclock_uses_zoneinfo_not_constant():
    # a zone with DST proves the offset is not hard-coded
    winter = source_wallclock_to_utc(datetime(2026, 1, 15, 12, 0), "Europe/Berlin")
    summer = source_wallclock_to_utc(datetime(2026, 7, 15, 12, 0), "Europe/Berlin")
    assert winter.hour == 11 and summer.hour == 10


def test_to_display():
    d = to_display(datetime(2026, 9, 21, 6, 58, tzinfo=timezone.utc), TPE)
    assert d.isoformat() == "2026-09-21T14:58:00+08:00"
    assert to_display(datetime(2026, 9, 21, 6, 58), TPE).isoformat() == "2026-09-21T14:58:00+08:00"


def test_parse_iso():
    assert parse_iso("2026-09-21T14:58:00", TPE).isoformat() == "2026-09-21T06:58:00+00:00"
    assert parse_iso("2026-09-21T14:58:00+08:00", "UTC").isoformat() == "2026-09-21T06:58:00+00:00"
    assert parse_iso("2026-09-21T06:58:00Z", TPE).isoformat() == "2026-09-21T06:58:00+00:00"
    assert parse_iso("2026-09-21", TPE).isoformat() == "2026-09-20T16:00:00+00:00"
    with pytest.raises(ValueError):
        parse_iso("yesterday", TPE)
    with pytest.raises(ValueError):
        parse_iso("", TPE)


@pytest.mark.parametrize("s,td", [
    ("30s", timedelta(seconds=30)), ("5m", timedelta(minutes=5)), ("1h", timedelta(hours=1)),
    ("1d", timedelta(days=1)), ("15M", timedelta(minutes=15)),
])
def test_parse_step(s, td):
    assert parse_step(s) == td


@pytest.mark.parametrize("bad", ["0m", "5", "m", "5w", "1h30m", "-5m", ""])
def test_parse_step_invalid(bad):
    with pytest.raises(ValueError):
        parse_step(bad)


def test_parse_duration():
    assert parse_duration("1h30m") == timedelta(minutes=90)
    assert parse_duration("7d") == timedelta(days=7)
    assert parse_duration("1d 12h") == timedelta(hours=36)
    for bad in ("", "abc", "5", "1h x", "0s"):
        with pytest.raises(ValueError):
            parse_duration(bad)


def test_settings_defaults():
    s = Settings.from_env({})
    assert s.source_uri == "mongodb://127.0.0.1:27017"
    assert s.source_db == "medical_monitor_db"
    assert s.db == "medical_monitor_qe"
    assert (s.bind_host, s.bind_port, s.bind) == ("127.0.0.1", 8020, "127.0.0.1:8020")
    assert s.source_tz == TPE and s.display_tz == TPE
    assert (s.etl_interval_s, s.etl_correction_rescan_s) == (60, 600)
    assert (s.max_range_days, s.max_points, s.mongo_max_time_ms) == (400, 10000, 10000)
    assert s.api_token is None


def test_settings_overrides_and_validation():
    s = Settings.from_env({"QE_DB": "medical_monitor_qe_test", "QE_BIND": "0.0.0.0:9000", "QE_API_TOKEN": "t"})
    assert s.db == "medical_monitor_qe_test" and s.bind_port == 9000 and s.api_token == "t"
    with pytest.raises(ValueError):
        Settings.from_env({"QE_SOURCE_TZ": "Mars/Base"})
    with pytest.raises(ValueError):
        Settings.from_env({"QE_BIND": "8020"})
    with pytest.raises(ValueError):
        Settings.from_env({"QE_MAX_POINTS": "0"})


@pytest.mark.parametrize("name", ["medical_monitor_qe_test_x", "xmedical_monitor_qe", "medical_monitor_qe\n",
                                  "medical_monitor_qe_test\n"])
def test_db_name_fullmatch_rejects(name):
    from nicu_qe.config import UnsafeConfigError, guard_db_names
    with pytest.raises(UnsafeConfigError):
        guard_db_names(name, "medical_monitor_db")


def test_db_name_fullmatch_accepts():
    from nicu_qe.config import guard_db_names
    guard_db_names("medical_monitor_qe", "medical_monitor_db")
    guard_db_names("medical_monitor_qe_test", "medical_monitor_db")
