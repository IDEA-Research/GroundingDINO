"""Timezone helpers: source wall-clock repair, display conversion, ISO/step/duration parsing."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Union
from zoneinfo import ZoneInfo

TzLike = Union[str, ZoneInfo, timezone]

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_STEP_RE = re.compile(r"^\s*(\d+)\s*([smhd])\s*$", re.IGNORECASE)
_DURATION_PART_RE = re.compile(r"(\d+)\s*([smhd])", re.IGNORECASE)


def _zone(tz: TzLike):
    """Return a tzinfo for an IANA name or pass through an existing tzinfo."""
    if isinstance(tz, str):
        return ZoneInfo(tz)
    return tz


def source_wallclock_to_utc(dt: datetime, source_tz: TzLike) -> datetime:
    """Reinterpret a source Date (local wall clock mislabelled as UTC) as source_tz and return true UTC."""
    if dt.tzinfo is not None:
        # The label says UTC (pymongo tz_aware=True); normalise to that label, then drop it.
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    local = dt.replace(tzinfo=_zone(source_tz))
    return local.astimezone(timezone.utc)


def wallclock_to_utc(dt_naive: datetime, tz: TzLike) -> datetime:
    """Interpret a naive wall-clock datetime in tz and return an aware UTC datetime."""
    if dt_naive.tzinfo is not None:
        return dt_naive.astimezone(timezone.utc)
    return dt_naive.replace(tzinfo=_zone(tz)).astimezone(timezone.utc)


def to_display(dt_utc: datetime, display_tz: TzLike) -> datetime:
    """Convert a UTC datetime (naive values are treated as UTC) to display_tz."""
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(_zone(display_tz))


def parse_iso(s: str, default_tz: TzLike) -> datetime:
    """Parse an ISO-8601 string into aware UTC; strings without an offset are read as default_tz."""
    if not isinstance(s, str) or not s.strip():
        raise ValueError("empty time string")
    text = s.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 time: {s!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_zone(default_tz))
    return dt.astimezone(timezone.utc)


def parse_step(s: str) -> timedelta:
    """Parse a single-unit step such as '30s', '5m', '1h', '1d' into a positive timedelta."""
    if not isinstance(s, str):
        raise ValueError(f"step must be a string, got {type(s).__name__}")
    m = _STEP_RE.match(s)
    if not m:
        raise ValueError(f"invalid step {s!r}; expected <int><s|m|h|d>")
    n = int(m.group(1))
    if n <= 0:
        raise ValueError(f"step must be > 0: {s!r}")
    return timedelta(seconds=n * _UNIT_SECONDS[m.group(2).lower()])


def parse_duration(s: str) -> timedelta:
    """Parse a duration made of one or more <int><s|m|h|d> parts, e.g. '90m' or '1d12h'."""
    if not isinstance(s, str) or not s.strip():
        raise ValueError("empty duration")
    text = s.strip()
    pos = 0
    total = 0
    for m in _DURATION_PART_RE.finditer(text):
        if text[pos:m.start()].strip():
            raise ValueError(f"invalid duration {s!r}")
        total += int(m.group(1)) * _UNIT_SECONDS[m.group(2).lower()]
        pos = m.end()
    if pos == 0 or text[pos:].strip():
        raise ValueError(f"invalid duration {s!r}; expected e.g. '15m', '1h30m', '7d'")
    if total <= 0:
        raise ValueError(f"duration must be > 0: {s!r}")
    return timedelta(seconds=total)


# ------------------------------------------------------------------ query parameters (WP-5)
_DATE_ONLY_RE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")
# "2026-09-10T08:00:00 08:00": a '+' offset that became a space through URL decoding
_SPACE_OFFSET_RE = re.compile(r"^(.*\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?) (\d{2}:?\d{2})$")
_REFERENCE_LOCAL = datetime(2000, 1, 1)


def parse_time_param(value, display_tz: TzLike, *, is_end: bool) -> datetime:
    """Parse a from/to parameter into aware UTC.

    Accepts an aware/naive datetime, a date, an ISO-8601 string, or a date-only string.
    Naive values are read in display_tz. A date-only value means that day's 00:00 (display tz)
    for `from`, and the NEXT day's 00:00 for `to` (API v2 #4, half-open [from, to)).
    """
    from datetime import date as _date

    if isinstance(value, datetime):
        return wallclock_to_utc(value, display_tz)
    if isinstance(value, _date):
        d = value + timedelta(days=1) if is_end else value
        return wallclock_to_utc(datetime(d.year, d.month, d.day), display_tz)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("time parameter must be a non-empty string")
    m = _DATE_ONLY_RE.match(value)
    if m:
        try:
            d = _date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError as exc:
            raise ValueError(f"invalid date: {value!r}") from exc
        return parse_time_param(d, display_tz, is_end=is_end)
    text = value.strip()
    sm = _SPACE_OFFSET_RE.match(text)
    if sm:
        text = f"{sm.group(1)}+{sm.group(2)}"
    return parse_iso(text, display_tz)


def bucket_unit(step: timedelta):
    """Return (mongo $dateTrunc unit, binSize) for a step: the largest of day/hour/minute/second dividing it."""
    secs = int(step.total_seconds())
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)):
        if secs % size == 0:
            return unit, secs // size
    raise ValueError("step must be a whole number of seconds")


def bucket_floor(t_utc: datetime, step: timedelta, tz: TzLike) -> datetime:
    """Python twin of $dateTrunc(binSize, timezone): bins counted in local wall time from 2000-01-01 00:00."""
    zone = _zone(tz)
    local = to_display(t_utc, zone).replace(tzinfo=None)
    secs = int(step.total_seconds())
    n = int((local - _REFERENCE_LOCAL).total_seconds() // secs)
    start_local = _REFERENCE_LOCAL + timedelta(seconds=n * secs)
    return start_local.replace(tzinfo=zone).astimezone(timezone.utc)


def iter_buckets(from_utc: datetime, to_utc: datetime, step: timedelta, tz: TzLike):
    """Yield bucket starts (UTC) covering [from, to): the first is floor(from), then +step in local time."""
    zone = _zone(tz)
    cur = bucket_floor(from_utc, step, zone)
    while cur < to_utc:
        yield cur
        local = to_display(cur, zone).replace(tzinfo=None) + step
        cur = local.replace(tzinfo=zone).astimezone(timezone.utc)
