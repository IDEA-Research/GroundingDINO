from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ParsedTimeWindow:
    start: datetime
    end: datetime

    @property
    def range_seconds(self) -> int:
        return int((self.end - self.start).total_seconds())


_TIME_RE = r"\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?"
_TIME_VALUE_RE = re.compile(
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<period>a\.?m\.?|p\.?m\.?)?",
    re.IGNORECASE,
)
_WINDOW_RE = re.compile(
    rf"(?P<start>{_TIME_RE})\s*(?:to|until|through|and|-|~|到|至)\s*(?P<end>{_TIME_RE})",
    re.IGNORECASE,
)


def parse_prompt_time_window(prompt: str, *, now: datetime | None = None) -> ParsedTimeWindow | None:
    match = _WINDOW_RE.search(prompt)
    if match is None:
        return None

    timezone = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Taipei"))
    local_now = now.astimezone(timezone) if now is not None else datetime.now(timezone)
    start_time = _parse_time(match.group("start"), inherited_period=None)
    end_period = _period_of(match.group("end")) or _period_of(match.group("start"))
    end_time = _parse_time(match.group("end"), inherited_period=end_period)

    start = datetime.combine(local_now.date(), start_time, tzinfo=timezone)
    end = datetime.combine(local_now.date(), end_time, tzinfo=timezone)
    if end <= start:
        if start.hour < 12 and end.hour <= 12:
            end += timedelta(hours=12)
        else:
            end += timedelta(days=1)

    if end <= start:
        return None
    return ParsedTimeWindow(start=start, end=end)


def _parse_time(value: str, *, inherited_period: str | None) -> time:
    match = _TIME_VALUE_RE.search(value)
    if match is None:
        raise ValueError(f"Invalid time value: {value}")

    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    period = _normalize_period(match.group("period") or inherited_period)
    if period == "pm" and hour < 12:
        hour += 12
    if period == "am" and hour == 12:
        hour = 0
    return time(hour=hour, minute=minute)


def _period_of(value: str) -> str | None:
    match = _TIME_VALUE_RE.search(value)
    if match is None:
        return None
    return _normalize_period(match.group("period"))


def _normalize_period(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.lower().replace(".", "")
    if lowered in {"am", "a"}:
        return "am"
    if lowered in {"pm", "p"}:
        return "pm"
    return None
