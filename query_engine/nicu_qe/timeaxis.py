"""Time axis: recover observed_at from stream capture time or manual-upload file names (DESIGN section 5)."""

from __future__ import annotations

import posixpath
import re
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple

from .models import (
    CONFIDENCE_BY_BASIS,
    FLAG_EDITED_VIDEO,
    FLAG_FUTURE_TIME,
    FLAG_REANALYZED,
    FLAG_TIME_FROM_SIBLING,
    TIME_BASIS_ANALYSIS,
    TIME_BASIS_CAPTURE,
    TIME_BASIS_CAPTURE_APPROX,
    TIME_BASIS_CAPTURE_LEGACY,
    TIME_BASIS_FILENAME,
    TimeResolution,
)

# L1: digit boundaries on both sides; only the FIRST date-time group in a name is used.
VIDEO_NAME_TIME_RE = re.compile(r"(?<!\d)(\d{4})\.(\d{2})\.(\d{2})[ _](\d{2})-(\d{2})-(\d{2})(?!\d)")
FRAME_OFFSET_RE = re.compile(r"_(\d+)s(?=[._\-]|$)")
MIN_YEAR = 2015
# M1: a stream doc counts as re-analysed (analyzed_at overwritten) when analyzed_at > processed_at + 5 s.
REANALYZE_TOLERANCE = timedelta(seconds=5)
STREAM_MAX_DRIFT = REANALYZE_TOLERANCE  # kept for backward compatibility of the name
FUTURE_TOLERANCE = timedelta(days=1)
OFFSET_MISMATCH_TOLERANCE_S = 2.0

SOURCE_STREAM = "stream"
SOURCE_MANUAL = "manual"
SOURCE_DATASET = "dataset"


def parse_video_name_time(video_name: Optional[str], now_year: Optional[int] = None) -> Optional[datetime]:
    """Return the naive wall-clock time of the first date-time group in a video name, or None."""
    if not isinstance(video_name, str):
        return None
    max_year = (now_year if now_year is not None else datetime.now().year) + 1
    m = VIDEO_NAME_TIME_RE.search(video_name)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(g) for g in m.groups())
    if not MIN_YEAR <= y <= max_year:
        return None
    try:
        return datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None


def is_edited_video(video_name: Optional[str], markers: Sequence[str]) -> bool:
    """True if the video name carries a post-production marker (M15). ASCII markers need non-letter edges."""
    if not isinstance(video_name, str) or not markers:
        return False
    low = video_name.lower()
    for mk in markers:
        mk = mk.strip().lower()
        if not mk:
            continue
        if mk.isascii():
            if re.search(rf"(?<![a-z]){re.escape(mk)}(?![a-z])", low):
                return True
        elif mk in low:
            return True
    return False


def infer_source(source: Optional[str], video_path: Optional[str]) -> str:
    """Return the source; when missing, infer from video_path ('RTSP:' -> stream, 'uploads/' -> manual, else manual)."""
    if isinstance(source, str) and source.strip():
        return source.strip().lower()
    if isinstance(video_path, str):
        vp = video_path.strip()
        if vp.upper().startswith("RTSP:"):
            return SOURCE_STREAM
        if vp.replace("\\", "/").startswith("uploads/"):
            return SOURCE_MANUAL
    return SOURCE_MANUAL


def frame_offset_from_path(original_image_path: Optional[str]) -> Optional[int]:
    """Extract the '<N>s' frame offset (seconds) from a frame image path, e.g. frame_0600s.jpg -> 600."""
    if not isinstance(original_image_path, str) or not original_image_path:
        return None
    base = posixpath.basename(original_image_path.replace("\\", "/"))
    matches = FRAME_OFFSET_RE.findall(base)
    return int(matches[-1]) if matches else None


def offset_mismatch(frame_offset_s: Optional[float], original_image_path: Optional[str]) -> bool:
    """True when the path-embedded offset differs from frame_results.minute by more than 2 s."""
    from_path = frame_offset_from_path(original_image_path)
    if from_path is None or frame_offset_s is None:
        return False
    return abs(float(frame_offset_s) - from_path) > OFFSET_MISMATCH_TOLERANCE_S


def _resolve(
    source: str,
    analyzed_at_utc: Optional[datetime],
    processed_at_utc: Optional[datetime],
    video_start_utc: Optional[datetime],
    frame_offset_s: Optional[float],
    sibling_capture_utc: Optional[datetime] = None,
    stream_reliable_from_utc: Optional[datetime] = None,
    now_utc: Optional[datetime] = None,
    edited: bool = False,
) -> TimeResolution:
    """Core of the DESIGN 5 + 12.2 rules; all datetimes are true UTC."""
    flags: List[str] = []
    observed: Optional[datetime]
    if source == SOURCE_STREAM:
        if analyzed_at_utc is not None and (
            processed_at_utc is None or analyzed_at_utc <= processed_at_utc + REANALYZE_TOLERANCE
        ):
            observed, basis = analyzed_at_utc, TIME_BASIS_CAPTURE
        elif analyzed_at_utc is not None:  # overwritten by a re-analysis
            flags.append(FLAG_REANALYZED)
            if sibling_capture_utc is not None:
                observed, basis = sibling_capture_utc, TIME_BASIS_CAPTURE
                flags.append(FLAG_TIME_FROM_SIBLING)
            else:
                observed, basis = processed_at_utc, TIME_BASIS_CAPTURE_APPROX
        elif processed_at_utc is not None:
            observed, basis = processed_at_utc, TIME_BASIS_CAPTURE_APPROX
        else:
            observed, basis = None, TIME_BASIS_ANALYSIS
        if (
            basis == TIME_BASIS_CAPTURE
            and stream_reliable_from_utc is not None
            and observed is not None
            and observed < stream_reliable_from_utc
        ):
            basis = TIME_BASIS_CAPTURE_LEGACY
    elif video_start_utc is not None and frame_offset_s is not None:
        # manual, dataset and any other value follow the file-name rule
        observed, basis = video_start_utc + timedelta(seconds=float(frame_offset_s)), TIME_BASIS_FILENAME
    else:
        observed, basis = analyzed_at_utc, TIME_BASIS_ANALYSIS
    confidence = CONFIDENCE_BY_BASIS[basis]
    if edited:  # any basis (API_CONTRACT 6 #5 v2.1): post-edited file name -> low + edited_video
        confidence = "low"
        flags.append(FLAG_EDITED_VIDEO)
    if now_utc is not None and observed is not None and observed > now_utc + FUTURE_TOLERANCE:
        observed, basis, confidence = analyzed_at_utc, TIME_BASIS_ANALYSIS, CONFIDENCE_BY_BASIS[TIME_BASIS_ANALYSIS]
        flags.append(FLAG_FUTURE_TIME)
    return TimeResolution(observed, basis, confidence, tuple(flags))


def resolve_observed_at(
    source: str,
    analyzed_at_utc: Optional[datetime],
    processed_at_utc: Optional[datetime],
    video_start_utc: Optional[datetime],
    frame_offset_s: Optional[float],
    **kwargs,
) -> Tuple[Optional[datetime], str, str]:
    """Apply the time-axis table; returns (observed_at, basis, confidence). kwargs as in resolve()."""
    tr = _resolve(source, analyzed_at_utc, processed_at_utc, video_start_utc, frame_offset_s, **kwargs)
    return tr.observed_at, tr.time_basis, tr.time_confidence


def resolve(
    source: str,
    analyzed_at_utc: Optional[datetime],
    processed_at_utc: Optional[datetime],
    video_start_utc: Optional[datetime],
    frame_offset_s: Optional[float],
    *,
    sibling_capture_utc: Optional[datetime] = None,
    stream_reliable_from_utc: Optional[datetime] = None,
    now_utc: Optional[datetime] = None,
    edited: bool = False,
) -> TimeResolution:
    """Resolve observed_at with the v2 rules (sibling restore, capture_legacy, edited_video, future_time)."""
    return _resolve(
        source, analyzed_at_utc, processed_at_utc, video_start_utc, frame_offset_s,
        sibling_capture_utc, stream_reliable_from_utc, now_utc, edited,
    )
