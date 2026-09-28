"""Service settings read from QE_* environment variables (API_CONTRACT section 0)."""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

# DESIGN 12.4 / API v2 #10: the derived DB name must match this and differ from the source DB.
QE_DB_NAME_RE = re.compile(r"^medical_monitor_qe(_test)?$")
DEFAULT_EDITED_MARKERS = "redacted,遮蔽,移除,edited,cut"


class UnsafeConfigError(ValueError):
    """Raised when the configuration could make the service write to the wrong database."""


def guard_db_names(qe_db: str, source_db: str) -> None:
    """Refuse any QE_DB that is not medical_monitor_qe(_test) or that equals QE_SOURCE_DB."""
    if not isinstance(qe_db, str) or not QE_DB_NAME_RE.fullmatch(qe_db):
        raise UnsafeConfigError(
            f"QE_DB={qe_db!r} rejected: must match {QE_DB_NAME_RE.pattern} (derived DB only)"
        )
    if qe_db == source_db:
        raise UnsafeConfigError(f"QE_DB must differ from QE_SOURCE_DB (both {qe_db!r})")


def is_loopback_host(host: str) -> bool:
    """True for localhost / 127.0.0.0/8 / ::1."""
    h = host.strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False

DEFAULTS = {
    "QE_SOURCE_URI": "mongodb://127.0.0.1:27017",
    "QE_SOURCE_DB": "medical_monitor_db",
    "QE_DB": "medical_monitor_qe",
    "QE_BIND": "127.0.0.1:8020",
    "QE_SOURCE_TZ": "Asia/Taipei",
    "QE_DISPLAY_TZ": "Asia/Taipei",
    "QE_ETL_INTERVAL_S": "60",
    "QE_ETL_CORRECTION_RESCAN_S": "600",
    "QE_MAX_RANGE_DAYS": "400",
    "QE_MAX_POINTS": "10000",
    "QE_MONGO_MAX_TIME_MS": "10000",
    "QE_API_TOKEN": "",
    "QE_ALLOW_NO_TOKEN": "0",
    "QE_FILENAME_TZ": "",  # empty -> same as QE_SOURCE_TZ
    "QE_STREAM_RELIABLE_FROM": "2026-06-01",
    "QE_ETL_RECONCILE_S": "21600",
    "QE_MAX_STATS_VALUES": "200000",
    "QE_EDITED_MARKERS": DEFAULT_EDITED_MARKERS,
}


def _parse_bind(value: str) -> Tuple[str, int]:
    """Split 'host:port' into (host, port)."""
    host, sep, port = value.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise ValueError(f"QE_BIND must look like host:port, got {value!r}")
    port_i = int(port)
    if not 0 < port_i < 65536:
        raise ValueError(f"QE_BIND port out of range: {port_i}")
    return host, port_i


def _positive_int(name: str, value: str) -> int:
    """Parse a strictly positive integer setting."""
    try:
        v = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if v <= 0:
        raise ValueError(f"{name} must be > 0, got {v}")
    return v


@dataclass(frozen=True)
class Settings:
    """Immutable service configuration."""

    source_uri: str
    source_db: str
    db: str
    bind_host: str
    bind_port: int
    source_tz: str
    display_tz: str
    etl_interval_s: int
    etl_correction_rescan_s: int
    max_range_days: int
    max_points: int
    mongo_max_time_ms: int
    api_token: Optional[str]
    allow_no_token: bool = False
    filename_tz: str = "Asia/Taipei"
    stream_reliable_from: date = date(2026, 6, 1)
    etl_reconcile_s: int = 21600
    max_stats_values: int = 200000
    edited_markers: Tuple[str, ...] = tuple(DEFAULT_EDITED_MARKERS.split(","))

    def validate_for_serve(self) -> None:
        """HTTP-serving checks: a token is mandatory unless QE_ALLOW_NO_TOKEN=1 on a loopback bind."""
        if self.api_token:
            return
        if not self.allow_no_token:
            raise UnsafeConfigError("QE_API_TOKEN is required (set QE_ALLOW_NO_TOKEN=1 only for loopback testing)")
        if not is_loopback_host(self.bind_host):
            raise UnsafeConfigError(
                f"QE_ALLOW_NO_TOKEN=1 is only allowed with a loopback QE_BIND, got {self.bind_host!r}"
            )

    @property
    def bind(self) -> str:
        """Return the bind address as 'host:port'."""
        return f"{self.bind_host}:{self.bind_port}"

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "Settings":
        """Build Settings from environment variables, applying defaults and validation."""
        env = os.environ if environ is None else environ

        def get(name: str) -> str:
            raw = env.get(name)
            return DEFAULTS[name] if raw is None or raw.strip() == "" else raw.strip()

        host, port = _parse_bind(get("QE_BIND"))
        source_tz = get("QE_SOURCE_TZ")
        display_tz = get("QE_DISPLAY_TZ")
        for name, tz in (("QE_SOURCE_TZ", source_tz), ("QE_DISPLAY_TZ", display_tz)):
            try:
                ZoneInfo(tz)
            except Exception as exc:  # ZoneInfoNotFoundError / ValueError
                raise ValueError(f"{name} is not a valid IANA timezone: {tz!r}") from exc
        token = get("QE_API_TOKEN")
        filename_tz = get("QE_FILENAME_TZ") if env.get("QE_FILENAME_TZ", "").strip() else source_tz
        try:
            ZoneInfo(filename_tz)
        except Exception as exc:
            raise ValueError(f"QE_FILENAME_TZ is not a valid IANA timezone: {filename_tz!r}") from exc
        try:
            reliable_from = date.fromisoformat(get("QE_STREAM_RELIABLE_FROM"))
        except ValueError as exc:
            raise ValueError("QE_STREAM_RELIABLE_FROM must be YYYY-MM-DD") from exc
        allow_raw = get("QE_ALLOW_NO_TOKEN").lower()
        if allow_raw not in ("0", "1", "true", "false", "yes", "no"):
            raise ValueError(f"QE_ALLOW_NO_TOKEN must be 0/1, got {allow_raw!r}")
        markers = tuple(m.strip().lower() for m in get("QE_EDITED_MARKERS").split(",") if m.strip())
        source_db = get("QE_SOURCE_DB")
        qe_db = get("QE_DB")
        guard_db_names(qe_db, source_db)
        return cls(
            source_uri=get("QE_SOURCE_URI"),
            source_db=source_db,
            db=qe_db,
            bind_host=host,
            bind_port=port,
            source_tz=source_tz,
            display_tz=display_tz,
            etl_interval_s=_positive_int("QE_ETL_INTERVAL_S", get("QE_ETL_INTERVAL_S")),
            etl_correction_rescan_s=_positive_int(
                "QE_ETL_CORRECTION_RESCAN_S", get("QE_ETL_CORRECTION_RESCAN_S")
            ),
            max_range_days=_positive_int("QE_MAX_RANGE_DAYS", get("QE_MAX_RANGE_DAYS")),
            max_points=_positive_int("QE_MAX_POINTS", get("QE_MAX_POINTS")),
            mongo_max_time_ms=_positive_int("QE_MONGO_MAX_TIME_MS", get("QE_MONGO_MAX_TIME_MS")),
            api_token=token or None,
            allow_no_token=allow_raw in ("1", "true", "yes"),
            filename_tz=filename_tz,
            stream_reliable_from=reliable_from,
            etl_reconcile_s=_positive_int("QE_ETL_RECONCILE_S", get("QE_ETL_RECONCILE_S")),
            max_stats_values=_positive_int("QE_MAX_STATS_VALUES", get("QE_MAX_STATS_VALUES")),
            edited_markers=markers,
        )
