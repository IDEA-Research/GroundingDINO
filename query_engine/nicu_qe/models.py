"""Dataclasses shared by the normalization, time-axis and (later) ETL/query layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:  # type-only; this layer never talks to Mongo
    from bson import ObjectId  # provided by pymongo

    IdLike = Union[ObjectId, str]
else:
    IdLike = Any

# time_basis values (API_CONTRACT section 1)
TIME_BASIS_CAPTURE = "capture"
TIME_BASIS_CAPTURE_APPROX = "capture_approx"
TIME_BASIS_FILENAME = "derived_from_filename"
TIME_BASIS_ANALYSIS = "analysis"
TIME_BASIS_CAPTURE_LEGACY = "capture_legacy"  # contract v2: stream before QE_STREAM_RELIABLE_FROM

TIME_BASES = (
    TIME_BASIS_CAPTURE,
    TIME_BASIS_CAPTURE_APPROX,
    TIME_BASIS_CAPTURE_LEGACY,
    TIME_BASIS_FILENAME,
    TIME_BASIS_ANALYSIS,
)

CONFIDENCE_BY_BASIS = {
    TIME_BASIS_CAPTURE: "high",
    TIME_BASIS_CAPTURE_APPROX: "medium",
    TIME_BASIS_CAPTURE_LEGACY: "medium",
    TIME_BASIS_FILENAME: "medium",
    TIME_BASIS_ANALYSIS: "low",
}

# per-value quality flags (API_CONTRACT section 1)
Q_OK = "ok"
Q_OUT_OF_RANGE = "out_of_range"
Q_NON_NUMERIC = "non_numeric"
Q_PARSED_COMPOSITE = "parsed_composite"
Q_UNIT_NORMALIZED = "unit_normalized"
Q_CORRECTED = "corrected"
Q_ALIAS_CONFLICT = "alias_conflict"  # not in contract v1; see README "Decisions"
Q_USER_MARKED_BAD = "user_marked_bad"  # contract v2: has_glare / has_occlusion set by a user
Q_AMBIGUOUS = "ambiguous"  # contract v2: numeric ie_ratio (may only be the E part)
Q_UNIT_STRIPPED = "unit_stripped"  # contract v2: "12%", "45 mmHg"
Q_FUTURE_TIME = "future_time"  # contract v2: observed_at > now + 1 day
Q_OFFSET_MISMATCH = "offset_mismatch"  # contract v2

QUALITY_FLAGS = (
    Q_OK,
    Q_OUT_OF_RANGE,
    Q_NON_NUMERIC,
    Q_PARSED_COMPOSITE,
    Q_UNIT_NORMALIZED,
    Q_CORRECTED,
    Q_ALIAS_CONFLICT,
    Q_USER_MARKED_BAD,
    Q_AMBIGUOUS,
    Q_UNIT_STRIPPED,
    Q_FUTURE_TIME,
    Q_OFFSET_MISMATCH,
)
DEFAULT_EXCLUDE_QUALITY = (Q_OUT_OF_RANGE, Q_NON_NUMERIC, Q_USER_MARKED_BAD)

# document-level flags
FLAG_EMPTY = "empty"
FLAG_INVALID_MEDICAL_VALUES = "invalid_medical_values"
FLAG_UNMAPPED_CORRECTION = "unmapped_correction"
FLAG_OFFSET_MISMATCH = "offset_mismatch"
FLAG_EDITED_VIDEO = "edited_video"
FLAG_FUTURE_TIME = "future_time"
FLAG_REANALYZED = "reanalyzed"  # stream analyzed_at overwritten (analyzed_at > processed_at + 5 s)
FLAG_TIME_FROM_SIBLING = "time_from_sibling"  # observed_at restored from a sibling screen's analyzed_at
FLAG_USER_MARKED_BAD = "user_marked_bad"


@dataclass
class NormalizedValues:
    """Result of normalizing one screen_analysis medical_values (+ corrections) object."""

    m: Dict[str, float] = field(default_factory=dict)
    q: Dict[str, List[str]] = field(default_factory=dict)
    raw: Dict[str, str] = field(default_factory=dict)
    unmapped: Dict[str, Any] = field(default_factory=dict)
    raw_keys: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def add_q(self, canon: str, flag: str) -> None:
        """Append a quality flag for a canonical metric (no duplicates, drops 'ok' once a real flag exists)."""
        lst = self.q.setdefault(canon, [])
        if flag != Q_OK and Q_OK in lst:
            lst.remove(Q_OK)
        if flag == Q_OK and any(f != Q_OK for f in lst):
            return
        if flag not in lst:
            lst.append(flag)

    def add_flag(self, flag: str) -> None:
        """Append a document-level flag once."""
        if flag not in self.flags:
            self.flags.append(flag)


@dataclass(frozen=True)
class TimeResolution:
    """observed_at plus how it was derived."""

    observed_at: Optional[datetime]
    time_basis: str
    time_confidence: str
    flags: tuple = ()


@dataclass
class Observation:
    """One medical_monitor_qe.observations document (API_CONTRACT section 5)."""

    src_id: IdLike
    observed_at: Optional[datetime]
    analyzed_at_utc: Optional[datetime]
    processed_at_utc: Optional[datetime]
    time_basis: str
    time_confidence: str
    source: str
    session_id: Optional[str] = None
    camera: Optional[str] = None
    patient_id: Optional[str] = None
    task_name: Optional[str] = None
    device_family: str = "unknown"
    device_model: str = "unknown"
    detected_model_raw: Optional[str] = None
    llm_model: Optional[str] = None
    primary: bool = False
    success: Optional[bool] = None
    m: Dict[str, float] = field(default_factory=dict)
    mk: List[str] = field(default_factory=list)
    q: Dict[str, List[str]] = field(default_factory=dict)
    raw: Dict[str, str] = field(default_factory=dict)
    raw_keys: List[str] = field(default_factory=list)
    unmapped: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    video_analysis_id: Optional[IdLike] = None
    frame_result_id: Optional[IdLike] = None
    frame_offset_s: Optional[float] = None
    etl_version: Optional[str] = None
    etl_at: Optional[datetime] = None

    def to_doc(self) -> Dict[str, Any]:
        """Return a plain dict suitable for insertion into the derived DB."""
        return asdict(self)


# ---------------------------------------------------------------- query results (WP-5)
@dataclass
class Point:
    """One series point. Raw mode fills the per-observation fields; bucketed mode fills n/min/max."""

    t: datetime
    v: Optional[float]
    n: Optional[int] = None
    min: Optional[float] = None
    max: Optional[float] = None
    session_id: Optional[str] = None
    camera: Optional[str] = None
    device_family: Optional[str] = None
    device_model: Optional[str] = None
    time_basis: Optional[str] = None
    quality: Optional[List[str]] = None
    llm_model: Optional[str] = None
    src_id: Optional[str] = None

    def to_dict(self, raw: bool) -> Dict[str, Any]:
        """Return the contract shape: raw points carry provenance, bucketed points carry n/min/max."""
        if raw:
            return {
                "t": self.t, "v": self.v, "session_id": self.session_id, "camera": self.camera,
                "device_family": self.device_family, "device_model": self.device_model,
                "time_basis": self.time_basis, "quality": self.quality, "llm_model": self.llm_model,
                "src_id": self.src_id,
            }
        return {"t": self.t, "v": self.v, "n": self.n, "min": self.min, "max": self.max}


@dataclass
class Series:
    """Result of QueryEngine.series (API_CONTRACT section 3 /v1/series + v2)."""

    metric: str
    unit: str
    step: Optional[str]
    agg: Optional[str]
    from_: datetime
    to: datetime
    points: List[Point] = field(default_factory=list)
    truncated: bool = False
    suggested_step: Optional[str] = None
    time_basis_counts: Dict[str, int] = field(default_factory=dict)
    quality_counts: Dict[str, int] = field(default_factory=dict)
    sessions: List[str] = field(default_factory=list)
    latest: Optional[Dict[str, Any]] = None
    fill: Optional[str] = None
    data_status: str = "ok"

    @property
    def n_points(self) -> int:
        """Number of returned points (filled empty buckets included)."""
        return len(self.points)

    def to_dict(self) -> Dict[str, Any]:
        """Return the contract JSON shape (datetimes left as aware UTC for the caller to format)."""
        raw = self.step is None
        return {
            "metric": self.metric, "unit": self.unit, "step": self.step, "agg": None if raw else self.agg,
            "fill": None if raw else self.fill, "from": self.from_, "to": self.to,
            "points": [p.to_dict(raw) for p in self.points], "n_points": self.n_points,
            "truncated": self.truncated, "suggested_step": self.suggested_step,
            "time_basis_counts": self.time_basis_counts, "quality_counts": self.quality_counts,
            "sessions": self.sessions, "latest": self.latest, "data_status": self.data_status,
        }
