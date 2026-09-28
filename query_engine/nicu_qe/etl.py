"""ETL: medical_monitor_db (read-only) -> medical_monitor_qe observations / daily_rollup / sessions / unmapped_keys.

DESIGN 12.1 / API 5 + 6. Unit of work is a frame (frame_result_id): whenever any screen of a frame
may have changed, every screen of that frame is re-read and rebuilt, so primary selection (per
(frame_result_id, original_screen_number ?? screen_number)) and sibling time restoration are always
computed on the complete group. Writes are diffed by a content hash so a no-op run writes nothing.

Incremental sources of change (union, every run):
  (a) _id > watermark_id - 5 min            (M2: concurrent writers commit out of _id order)
  (b) analyzed_at > watermark_analyzed - 10 min (H1: re-analysis overwrites analyzed_at in place;
      compared in source coordinates, i.e. wall clock labelled UTC)
  (c) video_analysis hash of video_name/source/camera_name/patient_id/task_name/video_path kind
  (d) every QE_ETL_CORRECTION_RESCAN_S: last_corrected_at / quality_marked_at / model_corrected_at
      greater than the maxima seen so far (no source index -> COLLSCAN, acknowledged in DESIGN 12.1)
  (e) every QE_ETL_RECONCILE_S: full in-place pass + orphan deletion (L9: sibling promotion follows
      from recomputing the group).
Watermarks are the largest SOURCE values seen, never the local clock (M5).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import statistics
import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .aliases import AliasRegistry, get_registry
from .config import Settings
from .models import (
    CONFIDENCE_BY_BASIS,
    FLAG_FUTURE_TIME,
    FLAG_OFFSET_MISMATCH,
    FLAG_USER_MARKED_BAD,
    Q_FUTURE_TIME,
    Q_OFFSET_MISMATCH,
    Q_USER_MARKED_BAD,
    TIME_BASIS_FILENAME,
)
from .mongo import Mongo, chunked
from .normalize import assign_primary, normalize_medical_values
from .timeaxis import (
    VIDEO_NAME_TIME_RE,
    REANALYZE_TOLERANCE,
    infer_source,
    is_edited_video,
    offset_mismatch,
    parse_video_name_time,
    resolve,
)
from .tz import source_wallclock_to_utc, wallclock_to_utc

log = logging.getLogger("nicu_qe.etl")

ETL_VERSION = "wp4-2026-09-28.1"
OBS = "observations"
OBS_REBUILD = "observations_rebuild"
ROLLUP = "daily_rollup"
SESSIONS = "sessions"
UNMAPPED = "unmapped_keys"
META = "meta"
META_ETL_ID = "etl"
LEASE_ID = "etl_lease"
ALL_METRIC = "__all__"  # daily_rollup pseudo-metric: one row per observation (document count)

BATCH = 500
ID_LOOKBACK = timedelta(minutes=5)
ANALYZED_LOOKBACK = timedelta(minutes=10)

SCREEN_PROJECTION = {
    "_id": 1, "frame_result_id": 1, "screen_number": 1, "original_screen_number": 1,
    "detected_model": 1, "corrected_model": 1, "medical_values": 1, "corrected_medical_values": 1,
    "is_corrected": 1, "success": 1, "analyzed_at": 1, "llm_model": 1, "has_glare": 1,
    "has_occlusion": 1, "last_corrected_at": 1, "quality_marked_at": 1, "model_corrected_at": 1,
}
FRAME_PROJECTION = {"_id": 1, "minute": 1, "video_analysis_id": 1, "processed_at": 1, "original_image_path": 1}
# video_path is read ONLY to classify its prefix (it contains RTSP credentials); it is never stored.
VIDEO_PROJECTION = {
    "_id": 1, "session_id": 1, "source": 1, "camera_name": 1, "patient_id": 1, "task_name": 1,
    "video_name": 1, "video_path": 1, "total_frames": 1,
}
CORRECTION_FIELDS = ("last_corrected_at", "quality_marked_at", "model_corrected_at")


def id_filter(watermark_id: Any) -> Dict[str, Any]:
    """(a) _id > watermark_id - 5 min (ObjectId time is true UTC)."""
    from bson import ObjectId

    return {"_id": {"$gt": ObjectId.from_datetime(watermark_id.generation_time - ID_LOOKBACK)}}


def analyzed_filter(watermark_analyzed_src: datetime) -> Dict[str, Any]:
    """(b) analyzed_at > watermark - 10 min, in SOURCE coordinates (naive wall clock labelled UTC)."""
    return {"analyzed_at": {"$gt": _naive(watermark_analyzed_src) - ANALYZED_LOOKBACK}}


def correction_filter(watermarks: Mapping[str, Any]) -> Dict[str, Any]:
    """(d) any of the three correction timestamps greater than the largest value seen so far."""
    ors = []
    for f in CORRECTION_FIELDS:
        w = _naive(watermarks.get(f)) if watermarks else None
        ors.append({f: {"$gt": w}} if w is not None else {f: {"$type": "date"}})
    return {"$or": ors}


class LeaseHeld(RuntimeError):
    """Another process holds the ETL lease."""


def _utcnow() -> datetime:
    """Aware UTC now."""
    return datetime.now(timezone.utc)


def _naive(dt: Optional[datetime]) -> Optional[datetime]:
    """Drop tzinfo (source coordinates are naive wall clock values)."""
    if dt is None:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def _json_default(o: Any) -> Any:
    """json.dumps fallback for ObjectId / datetime."""
    if isinstance(o, datetime):
        if o.tzinfo is not None:
            o = o.astimezone(timezone.utc)
        return o.isoformat()
    return str(o)


def doc_hash(doc: Mapping[str, Any]) -> str:
    """Content hash of an observation, ignoring etl_at / h."""
    body = {k: v for k, v in doc.items() if k not in ("etl_at", "h", "_id")}
    return hashlib.sha1(json.dumps(body, sort_keys=True, default=_json_default, ensure_ascii=False).encode("utf-8")).hexdigest()


def _video_path_kind(video_path: Any) -> str:
    """Classify video_path without keeping it: rtsp / uploads / other / none."""
    if not isinstance(video_path, str) or not video_path.strip():
        return "none"
    vp = video_path.strip()
    if vp.upper().startswith("RTSP:"):
        return "rtsp"
    if vp.replace("\\", "/").startswith("uploads/"):
        return "uploads"
    return "other"


def mask_video_name(video_name: Any) -> Optional[str]:
    """Return only the date-time fragment of a video name (e.g. 2025.11.05_12-03-31), else None."""
    if not isinstance(video_name, str):
        return None
    m = VIDEO_NAME_TIME_RE.search(video_name)
    return m.group(0) if m else None


class EtlContext:
    """Per-run constants for building observations."""

    def __init__(self, settings: Settings, registry: AliasRegistry, now_utc: datetime) -> None:
        """Capture settings, registry and a fixed 'now' for one run."""
        self.settings = settings
        self.registry = registry
        self.now_utc = now_utc
        self.etl_at = now_utc
        d: date = settings.stream_reliable_from
        self.reliable_from_utc = wallclock_to_utc(datetime(d.year, d.month, d.day), settings.display_tz)
        self.aliases_version = registry.version


def load_videos(mongo: Mongo, settings: Settings) -> Dict[Any, Dict[str, Any]]:
    """Read all video_analysis docs (small table) into {video_id: info}; credentials never kept."""
    out: Dict[Any, Dict[str, Any]] = {}
    for v in mongo.source["video_analysis"].find({}, VIDEO_PROJECTION):
        kind = _video_path_kind(v.get("video_path"))
        pseudo_path = {"rtsp": "RTSP:", "uploads": "uploads/"}.get(kind)
        source = infer_source(v.get("source"), pseudo_path)
        name = v.get("video_name")
        start_local = parse_video_name_time(name)
        info = {
            "session_id": v.get("session_id"),
            "source": source,
            "source_raw": v.get("source"),
            "camera": v.get("camera_name"),
            "patient_id": v.get("patient_id"),
            "task_name": v.get("task_name"),
            "video_name_masked": mask_video_name(name),
            "video_start_local": start_local.isoformat() if start_local else None,
            "video_start_utc": wallclock_to_utc(start_local, settings.filename_tz) if start_local else None,
            "edited": is_edited_video(name, settings.edited_markers),
            "total_frames": v.get("total_frames"),
        }
        hsrc = {
            "video_name": name, "source": v.get("source"), "camera_name": v.get("camera_name"),
            "patient_id": v.get("patient_id"), "task_name": v.get("task_name"), "kind": kind,
            "session_id": v.get("session_id"), "total_frames": v.get("total_frames"),
        }
        info["hash"] = hashlib.sha1(json.dumps(hsrc, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()
        out[v["_id"]] = info
    return out


def _sibling_capture(screens: Sequence[Mapping[str, Any]], processed_raw: Optional[datetime]) -> Optional[datetime]:
    """Earliest analyzed_at (raw) among screens of one frame that were NOT overwritten by a re-analysis."""
    if processed_raw is None:
        return None
    limit = processed_raw + REANALYZE_TOLERANCE
    vals = [s.get("analyzed_at") for s in screens if isinstance(s.get("analyzed_at"), datetime)]
    good = [v for v in vals if v <= limit]
    return min(good) if good else None


def build_frame_observations(
    screens: Sequence[Mapping[str, Any]],
    frame: Optional[Mapping[str, Any]],
    video: Optional[Mapping[str, Any]],
    ctx: EtlContext,
) -> List[Dict[str, Any]]:
    """Build the observation docs for all screens of ONE frame, including primary flags."""
    st = ctx.settings
    frame = frame or {}
    video = video or {}
    processed_raw = frame.get("processed_at") if isinstance(frame.get("processed_at"), datetime) else None
    processed_utc = source_wallclock_to_utc(processed_raw, st.source_tz) if processed_raw else None
    sib_raw = _sibling_capture(screens, processed_raw)
    sib_utc = source_wallclock_to_utc(sib_raw, st.source_tz) if sib_raw else None
    minute = frame.get("minute")
    frame_offset_s = float(minute) if isinstance(minute, (int, float)) and not isinstance(minute, bool) else None
    mismatch = offset_mismatch(frame_offset_s, frame.get("original_image_path"))
    source = video.get("source") or infer_source(None, None)

    docs: List[Dict[str, Any]] = []
    for s in screens:
        analyzed_raw = s.get("analyzed_at") if isinstance(s.get("analyzed_at"), datetime) else None
        analyzed_utc = source_wallclock_to_utc(analyzed_raw, st.source_tz) if analyzed_raw else None
        family, model = ctx.registry.classify_device(s.get("detected_model"), s.get("corrected_model"))
        nv = normalize_medical_values(s.get("medical_values"), s.get("corrected_medical_values"), family, ctx.registry)
        tr = resolve(
            source, analyzed_utc, processed_utc, video.get("video_start_utc"), frame_offset_s,
            sibling_capture_utc=sib_utc, stream_reliable_from_utc=ctx.reliable_from_utc,
            now_utc=ctx.now_utc, edited=bool(video.get("edited")),
        )
        flags = list(nv.flags)
        for f in tr.flags:
            if f not in flags:
                flags.append(f)
        per_value: List[str] = []
        if s.get("has_glare") is True or s.get("has_occlusion") is True:
            per_value.append(Q_USER_MARKED_BAD)
            flags.append(FLAG_USER_MARKED_BAD)
        if mismatch and tr.time_basis == TIME_BASIS_FILENAME:
            per_value.append(Q_OFFSET_MISMATCH)
            flags.append(FLAG_OFFSET_MISMATCH)
        if FLAG_FUTURE_TIME in tr.flags:
            per_value.append(Q_FUTURE_TIME)
        for canon in list(nv.q):
            for f in per_value:
                nv.add_q(canon, f)
        osn = s.get("original_screen_number")
        docs.append({
            "src_id": s["_id"],
            "frame_result_id": s.get("frame_result_id"),
            "video_analysis_id": frame.get("video_analysis_id"),
            "screen_number": s.get("screen_number"),
            "original_screen_number": osn,
            "screen_key": osn if osn is not None else s.get("screen_number"),
            "observed_at": tr.observed_at,
            "analyzed_at_utc": analyzed_utc,
            "processed_at_utc": processed_utc,
            "time_basis": tr.time_basis,
            "time_confidence": tr.time_confidence,
            "source": source,
            "session_id": video.get("session_id"),
            "camera": video.get("camera"),
            "patient_id": video.get("patient_id"),
            "task_name": video.get("task_name"),
            "device_family": family,
            "device_model": model,
            "detected_model_raw": s.get("detected_model"),
            "corrected_model": s.get("corrected_model"),
            "llm_model": s.get("llm_model"),
            "success": s.get("success"),
            "is_corrected": bool(s.get("is_corrected")),
            "primary": False,
            "m": nv.m,
            "mk": sorted(nv.m),
            "q": nv.q,
            "raw": nv.raw,
            "raw_keys": nv.raw_keys,
            "unmapped": nv.unmapped,
            "flags": flags,
            "frame_offset_s": frame_offset_s,
            "aliases_version": ctx.aliases_version,
            "etl_version": ETL_VERSION,
        })
    ranks = [
        {"is_corrected": d["is_corrected"], "success": d["success"], "analyzed_at": d["analyzed_at_utc"],
         "src_id": d["src_id"], "frame_result_id": d["frame_result_id"],
         "original_screen_number": d["original_screen_number"], "screen_number": d["screen_number"]}
        for d in docs
    ]
    for d, p in zip(docs, assign_primary(ranks)):
        d["primary"] = p
        d["h"] = doc_hash(d)
        d["etl_at"] = ctx.etl_at
    return docs


class Etl:
    """ETL runner. All writes go to the guarded QE DB; the source is only read."""

    def __init__(self, mongo: Mongo, settings: Settings, registry: Optional[AliasRegistry] = None,
                 owner: Optional[str] = None) -> None:
        """Bind to Mongo handles and settings."""
        self.mongo = mongo
        self.st = settings
        self.registry = registry or get_registry()
        self.src = mongo.source
        self.qe = mongo.qe
        self.ms = settings.mongo_max_time_ms
        self.owner = owner or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------ lease
    def acquire_lease(self, ttl_s: int) -> bool:
        """Take or renew the meta etl_lease (TTL-indexed expires_at). False if held by someone else."""
        from pymongo.errors import DuplicateKeyError

        now = _utcnow()
        try:
            self.qe[META].find_one_and_update(
                {"_id": LEASE_ID, "$or": [{"owner": self.owner}, {"expires_at": {"$lt": now}}]},
                {"$set": {"owner": self.owner, "expires_at": now + timedelta(seconds=ttl_s), "renewed_at": now}},
                upsert=True,
            )
            return True
        except DuplicateKeyError:
            return False

    def release_lease(self) -> None:
        """Release the lease if we hold it."""
        try:
            self.qe[META].delete_one({"_id": LEASE_ID, "owner": self.owner})
        except Exception as exc:  # pragma: no cover - best effort
            log.warning("lease release failed: %s", exc)

    def lease_info(self) -> Optional[Dict[str, Any]]:
        """Current lease document."""
        return self.qe[META].find_one({"_id": LEASE_ID}, max_time_ms=self.ms)

    # ------------------------------------------------------------------ meta
    def meta(self) -> Dict[str, Any]:
        """The meta etl document (empty dict if none)."""
        return self.qe[META].find_one({"_id": META_ETL_ID}, max_time_ms=self.ms) or {}

    def _set_meta(self, fields: Mapping[str, Any]) -> None:
        """$set fields on the meta etl document."""
        self.qe[META].update_one({"_id": META_ETL_ID}, {"$set": dict(fields)}, upsert=True)

    # ------------------------------------------------------------------ source snapshots
    def _source_watermarks(self) -> Dict[str, Any]:
        """Largest source values right now: _id, analyzed_at and the three correction timestamps."""
        s = self.src["screen_analysis"]
        top = list(s.find({}, {"_id": 1}, sort=[("_id", -1)], limit=1))
        top_a = list(s.find({"analyzed_at": {"$type": "date"}}, {"analyzed_at": 1}, sort=[("analyzed_at", -1)], limit=1))
        group: Dict[str, Any] = {"_id": None}
        for f in CORRECTION_FIELDS:
            group[f] = {"$max": f"${f}"}
        corr = list(s.aggregate([{"$match": {"$or": [{f: {"$type": "date"}} for f in CORRECTION_FIELDS]}},
                                 {"$group": group}]))
        c = corr[0] if corr else {}
        return {
            "watermark_id": top[0]["_id"] if top else None,
            "watermark_analyzed_src": top_a[0]["analyzed_at"] if top_a else None,
            "correction_watermarks_src": {f: c.get(f) for f in CORRECTION_FIELDS},
        }

    # ------------------------------------------------------------------ core frame processing
    def _fetch_frames(self, fids: Sequence[Any]) -> Tuple[Dict[Any, List[Dict[str, Any]]], Dict[Any, Dict[str, Any]]]:
        """Read every screen of the given frames plus the frame docs."""
        screens: Dict[Any, List[Dict[str, Any]]] = {}
        for s in self.src["screen_analysis"].find({"frame_result_id": {"$in": list(fids)}}, SCREEN_PROJECTION):
            screens.setdefault(s.get("frame_result_id"), []).append(s)
        frames = {f["_id"]: f for f in self.src["frame_results"].find({"_id": {"$in": list(fids)}}, FRAME_PROJECTION)}
        return screens, frames

    def _build(self, screens: Mapping[Any, List[Dict[str, Any]]], frames: Mapping[Any, Dict[str, Any]],
               videos: Mapping[Any, Dict[str, Any]], ctx: EtlContext) -> List[Dict[str, Any]]:
        """Build observation docs for a set of frames."""
        out: List[Dict[str, Any]] = []
        for fid, group in screens.items():
            frame = frames.get(fid)
            video = videos.get(frame.get("video_analysis_id")) if frame else None
            out.extend(build_frame_observations(group, frame, video, ctx))
        return out

    def _apply(self, fids: Sequence[Any], docs: List[Dict[str, Any]], coll: str, stats: Dict[str, Any]) -> None:
        """Diff docs against the QE collection for these frames and write only changes."""
        from pymongo import DeleteMany, ReplaceOne

        target = self.qe[coll]
        existing = {
            e["src_id"]: e
            for e in target.find({"frame_result_id": {"$in": list(fids)}},
                                 {"src_id": 1, "h": 1, "observed_at": 1, "session_id": 1}, max_time_ms=self.ms)
        }
        ops: List[Any] = []
        new_ids = set()
        for d in docs:
            new_ids.add(d["src_id"])
            old = existing.get(d["src_id"])
            if old is not None and old.get("h") == d["h"]:
                stats["unchanged"] += 1
                continue
            ops.append(ReplaceOne({"src_id": d["src_id"]}, d, upsert=True))
            stats["updated" if old is not None else "inserted"] += 1
            self._touch(stats, d.get("observed_at"), d.get("session_id"))
            if old is not None:
                self._touch(stats, old.get("observed_at"), old.get("session_id"))
        gone = [sid for sid in existing if sid not in new_ids]
        for sid in gone:
            old = existing[sid]
            self._touch(stats, old.get("observed_at"), old.get("session_id"))
        if gone:
            ops.append(DeleteMany({"src_id": {"$in": gone}}))
            stats["deleted"] += len(gone)
        for chunk in chunked(ops, BATCH):
            target.bulk_write(chunk, ordered=False)

    def _touch(self, stats: Dict[str, Any], observed_at: Optional[datetime], session_id: Any) -> None:
        """Record an affected display-tz day and session for aggregate maintenance."""
        if isinstance(observed_at, datetime):
            stats["days"].add(_display_day(observed_at, self.st.display_tz))
        stats["sessions"].add(session_id)

    def process_frames(self, fids: Iterable[Any], videos: Mapping[Any, Dict[str, Any]], ctx: EtlContext,
                       stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Rebuild all observations of the given frames in place (diffed)."""
        stats = stats if stats is not None else _new_stats()
        for chunk in chunked(sorted(set(f for f in fids if f is not None)), BATCH):
            screens, frames = self._fetch_frames(chunk)
            docs = self._build(screens, frames, videos, ctx)
            self._apply(chunk, docs, OBS, stats)
            stats["frames"] += len(chunk)
        return stats

    def _iter_all_frames(self, upto_id: Any = None):
        """Stream every screen ordered by frame_result_id (index frame_result_id_1), grouped per BATCH frames."""
        flt: Dict[str, Any] = {}
        cur = self.src["screen_analysis"].find(flt, SCREEN_PROJECTION, sort=[("frame_result_id", 1)], batch_size=2000)
        group: Dict[Any, List[Dict[str, Any]]] = {}
        last = object()
        for s in cur:
            fid = s.get("frame_result_id")
            if fid != last and len(group) >= BATCH:
                yield group
                group = {}
            group.setdefault(fid, []).append(s)
            last = fid
        if group:
            yield group

    # ------------------------------------------------------------------ aggregates
    def rebuild_rollup(self, days: Optional[Set[str]] = None, coll: str = OBS) -> int:
        """Recompute daily_rollup, fully (days=None, atomic $out) or for the given display-tz days."""
        tz = self.st.display_tz
        match: Dict[str, Any] = {"primary": True, "observed_at": {"$type": "date"}}
        if days is not None:
            if not days:
                return 0
            ranges = [_day_range(d, tz) for d in sorted(days)]
            match["$or"] = [{"observed_at": {"$gte": a, "$lt": b}} for a, b in ranges]
        pipeline: List[Dict[str, Any]] = [
            {"$match": match},
            {"$project": {
                "day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$observed_at", "timezone": tz}},
                "source": 1, "session_id": 1, "camera": 1, "device_family": 1, "time_basis": 1, "observed_at": 1,
                "keys": {"$concatArrays": [[ALL_METRIC], {"$ifNull": ["$mk", []]}]},
            }},
            {"$unwind": "$keys"},
            {"$group": {
                "_id": {"day": "$day", "source": "$source", "session_id": "$session_id", "camera": "$camera",
                        "device_family": "$device_family", "time_basis": "$time_basis", "metric": "$keys"},
                "count": {"$sum": 1}, "first_t": {"$min": "$observed_at"}, "last_t": {"$max": "$observed_at"},
            }},
            {"$project": {
                "_id": 0, "day": "$_id.day", "source": "$_id.source", "session_id": "$_id.session_id",
                "camera": "$_id.camera", "device_family": "$_id.device_family", "time_basis": "$_id.time_basis",
                "metric": "$_id.metric", "count": 1, "first_t": 1, "last_t": 1,
            }},
        ]
        if days is None:
            pipeline.append({"$out": ROLLUP})
            list(self.qe[coll].aggregate(pipeline, maxTimeMS=max(self.ms, 120000), allowDiskUse=True))
            return self.qe[ROLLUP].estimated_document_count()
        rows = list(self.qe[coll].aggregate(pipeline, maxTimeMS=self.ms, allowDiskUse=True))
        self.qe[ROLLUP].delete_many({"day": {"$in": sorted(days)}})
        for chunk in chunked(rows, BATCH):
            self.qe[ROLLUP].insert_many(chunk, ordered=False)
        return len(rows)

    def rebuild_sessions(self, videos: Mapping[Any, Dict[str, Any]], session_ids: Optional[Set[Any]] = None) -> int:
        """Recompute the sessions collection (all, or the given session ids)."""
        from pymongo import ReplaceOne

        match: Dict[str, Any] = {"primary": True}
        if session_ids is not None:
            if not session_ids:
                return 0
            match["session_id"] = {"$in": list(session_ids)}
        per: Dict[Any, Dict[str, Any]] = {}
        for r in self.qe[OBS].aggregate([
            {"$match": match},
            {"$group": {"_id": {"s": "$session_id", "b": "$time_basis"}, "n": {"$sum": 1},
                        "first": {"$min": "$observed_at"}, "last": {"$max": "$observed_at"},
                        "families": {"$addToSet": "$device_family"},
                        "edited": {"$max": {"$in": ["edited_video", {"$ifNull": ["$flags", []]}]}}}},
        ], maxTimeMS=self.ms, allowDiskUse=True):
            sid = r["_id"]["s"]
            p = per.setdefault(sid, {"n": 0, "first": None, "last": None, "families": set(), "bases": {}, "edited": False})
            p["n"] += r["n"]
            p["bases"][r["_id"]["b"]] = p["bases"].get(r["_id"]["b"], 0) + r["n"]
            p["families"].update(x for x in r["families"] if x)
            p["edited"] = p["edited"] or bool(r.get("edited"))
            for k, fn in (("first", min), ("last", max)):
                if r[k] is not None:
                    p[k] = r[k] if p[k] is None else fn(p[k], r[k])
        rmatch: Dict[str, Any] = {"metric": {"$ne": ALL_METRIC}}
        if session_ids is not None:
            rmatch["session_id"] = {"$in": list(session_ids)}
        metrics: Dict[Any, List[str]] = {
            r["_id"]: sorted(r["m"]) for r in self.qe[ROLLUP].aggregate(
                [{"$match": rmatch}, {"$group": {"_id": "$session_id", "m": {"$addToSet": "$metric"}}}],
                maxTimeMS=self.ms)
        }
        by_session: Dict[Any, List[Dict[str, Any]]] = {}
        for vid, v in videos.items():
            by_session.setdefault(v.get("session_id"), []).append(v)
        wanted = set(per) | set(by_session)
        if session_ids is not None:
            wanted &= set(session_ids)
        ops = []
        for sid in wanted:
            if sid is None:
                continue
            vs = by_session.get(sid, [])
            v0 = vs[-1] if vs else {}
            p = per.get(sid, {"n": 0, "first": None, "last": None, "families": set(), "bases": {}, "edited": False})
            basis = max(p["bases"], key=p["bases"].get) if p["bases"] else None
            conf = CONFIDENCE_BY_BASIS.get(basis) if basis else None
            if p["edited"]:  # any basis, consistent with timeaxis._resolve
                conf = "low"
            ops.append(ReplaceOne({"session_id": sid}, {
                "session_id": sid, "source": v0.get("source"), "camera": v0.get("camera"),
                "patient_id": v0.get("patient_id"), "task_name": v0.get("task_name"),
                "video_name_masked": v0.get("video_name_masked"), "video_start_local": v0.get("video_start_local"),
                "n_videos": len(vs), "n_frames": sum(int(x.get("total_frames") or 0) for x in vs),
                "first_observed_at": p["first"], "last_observed_at": p["last"], "n_observations": p["n"],
                "device_families": sorted(p["families"]), "metrics": metrics.get(sid, []),
                "time_basis": basis, "time_basis_counts": p["bases"], "time_confidence": conf,
            }, upsert=True))
        for chunk in chunked(ops, BATCH):
            self.qe[SESSIONS].bulk_write(chunk, ordered=False)
        if session_ids is None:
            self.qe[SESSIONS].delete_many({"session_id": {"$nin": [s for s in wanted if s is not None]}})
        return len(ops)

    def rebuild_unmapped(self) -> int:
        """Recompute unmapped_keys from observations (atomic $out)."""
        pipeline = [
            {"$match": {"unmapped": {"$exists": True, "$ne": {}}}},
            {"$project": {"device_family": 1, "t": {"$ifNull": ["$observed_at", "$analyzed_at_utc"]},
                          "kv": {"$objectToArray": "$unmapped"}}},
            {"$unwind": "$kv"},
            {"$group": {"_id": {"f": "$device_family", "k": "$kv.k", "ty": {"$type": "$kv.v"}},
                        "count": {"$sum": 1}, "samples": {"$firstN": {"input": "$kv.v", "n": 3}},
                        "last_seen": {"$max": "$t"}}},
            {"$group": {"_id": {"f": "$_id.f", "k": "$_id.k"}, "count": {"$sum": "$count"},
                        "value_types": {"$push": {"k": "$_id.ty", "v": "$count"}},
                        "samples": {"$push": "$samples"}, "last_seen": {"$max": "$last_seen"}}},
            {"$project": {"_id": 0, "device_family": "$_id.f", "key": "$_id.k", "count": 1,
                          "value_types": {"$arrayToObject": "$value_types"},
                          "sample_values": {"$slice": [{"$reduce": {"input": "$samples", "initialValue": [],
                                                                     "in": {"$concatArrays": ["$$value", "$$this"]}}}, 3]},
                          "last_seen": 1}},
            {"$out": UNMAPPED},
        ]
        list(self.qe[OBS].aggregate(pipeline, maxTimeMS=max(self.ms, 60000), allowDiskUse=True))
        return self.qe[UNMAPPED].estimated_document_count()

    # ------------------------------------------------------------------ full rebuild
    def full(self, lease_ttl_s: int = 900) -> Dict[str, Any]:
        """--full: build observations_rebuild from scratch, then renameCollection(dropTarget=True)."""
        from . import indexes

        if not self.acquire_lease(lease_ttl_s):
            raise LeaseHeld(f"ETL lease held by {self.lease_info()}")
        t0 = time.monotonic()
        now = _utcnow()
        try:
            wm = self._source_watermarks()
            ctx = EtlContext(self.st, self.registry, now)
            videos = load_videos(self.mongo, self.st)
            self.qe[OBS_REBUILD].drop()
            indexes.ensure(self.qe, only=[OBS], rename={OBS: OBS_REBUILD})
            stats = _new_stats()
            seen: Set[Any] = set()
            offsets: List[float] = []
            for group in self._iter_all_frames():
                fids = list(group)
                frames = {f["_id"]: f for f in self.src["frame_results"].find({"_id": {"$in": fids}}, FRAME_PROJECTION)}
                docs = self._build(group, frames, videos, ctx)
                for chunk in chunked(docs, BATCH):
                    self.qe[OBS_REBUILD].insert_many(chunk, ordered=False)
                stats["inserted"] += len(docs)
                stats["frames"] += len(fids)
                for scr in group.values():
                    for s in scr:
                        seen.add(s["_id"])
                        a = s.get("analyzed_at")
                        if isinstance(a, datetime) and len(offsets) < 50000:
                            gen = s["_id"].generation_time.replace(tzinfo=None)
                            offsets.append((a - gen).total_seconds() / 3600.0)
            self.mongo.rename_into(OBS_REBUILD, OBS)
            indexes.ensure(self.qe)  # idempotent; covers rollup/sessions/unmapped/meta too
            n_rollup = self.rebuild_rollup(None)
            n_sessions = self.rebuild_sessions(videos, None)
            n_unmapped = self.rebuild_unmapped()
            indexes.ensure(self.qe)  # $out recreated collections keep indexes, but be explicit
            elapsed = time.monotonic() - t0
            meta = {
                **wm,
                "video_hashes": {str(k): v["hash"] for k, v in videos.items()},
                "last_run_at": now, "last_full_at": now, "last_reconcile_at": now,
                "last_correction_rescan_at": now, "last_ok": True, "last_error": None,
                "last_mode": "full", "last_changes": stats["inserted"], "last_elapsed_s": round(elapsed, 3),
                "observations": self.qe[OBS].estimated_document_count(),
                "sessions": n_sessions, "rollup_rows": n_rollup, "unmapped_keys": n_unmapped,
                "aliases_version": self.registry.version, "etl_version": ETL_VERSION,
                "source_tz_offset_hours_detected": round(statistics.median(offsets), 2) if offsets else None,
            }
            self._set_meta(meta)
            return {"mode": "full", "elapsed_s": round(elapsed, 3), "observations": meta["observations"],
                    "screens_seen": len(seen), "frames": stats["frames"], "rollup_rows": n_rollup,
                    "sessions": n_sessions, "unmapped_keys": n_unmapped,
                    "source_tz_offset_hours_detected": meta["source_tz_offset_hours_detected"]}
        except Exception as exc:
            self._record_error("full", exc)
            raise
        finally:
            self.release_lease()

    # ------------------------------------------------------------------ incremental
    def run_once(self, lease_ttl_s: int = 900, hold_lease: bool = False) -> Dict[str, Any]:
        """One incremental pass (a)-(e). Falls back to full() when no watermark exists yet."""
        meta = self.meta()
        if not meta.get("watermark_id"):
            return self.full(lease_ttl_s)
        if not self.acquire_lease(lease_ttl_s):
            raise LeaseHeld(f"ETL lease held by {self.lease_info()}")
        t0 = time.monotonic()
        now = _utcnow()
        try:
            ctx = EtlContext(self.st, self.registry, now)
            videos = load_videos(self.mongo, self.st)
            old_hashes = meta.get("video_hashes") or {}
            changed_vids = [vid for vid, v in videos.items() if old_hashes.get(str(vid)) != v["hash"]]
            fids: Set[Any] = set()
            reasons = {"video_changed": 0, "new_id": 0, "analyzed_at": 0, "correction": 0}
            if changed_vids:
                for chunk in chunked(changed_vids, BATCH):
                    for f in self.src["frame_results"].find({"video_analysis_id": {"$in": chunk}}, {"_id": 1}):
                        fids.add(f["_id"])
                reasons["video_changed"] = len(changed_vids)

            wm_id = meta["watermark_id"]
            max_id, max_an = wm_id, _naive(meta.get("watermark_analyzed_src"))
            for s in self.src["screen_analysis"].find(id_filter(wm_id), {"frame_result_id": 1, "analyzed_at": 1}):
                fids.add(s.get("frame_result_id"))
                reasons["new_id"] += 1
                max_id = max(max_id, s["_id"])
                a = s.get("analyzed_at")
                if isinstance(a, datetime) and (max_an is None or a > max_an):
                    max_an = a
            if max_an is not None:
                an_lo = _naive(meta.get("watermark_analyzed_src")) or max_an
                for s in self.src["screen_analysis"].find(analyzed_filter(an_lo), {"frame_result_id": 1, "analyzed_at": 1}):
                    fids.add(s.get("frame_result_id"))
                    reasons["analyzed_at"] += 1
                    a = s.get("analyzed_at")
                    if isinstance(a, datetime) and a > max_an:
                        max_an = a

            # never let one bad future analyzed_at push watermark (b) past real re-analyses
            if max_an is not None:
                from zoneinfo import ZoneInfo

                cap = datetime.now(ZoneInfo(self.st.source_tz)).replace(tzinfo=None) + timedelta(hours=1)
                max_an = min(max_an, cap)
            new_meta: Dict[str, Any] = {}
            last_rescan = meta.get("last_correction_rescan_at")
            if last_rescan is None or (now - last_rescan).total_seconds() >= self.st.etl_correction_rescan_s:
                cw = {k: _naive(v) for k, v in (meta.get("correction_watermarks_src") or {}).items()}
                proj = {"frame_result_id": 1, **{f: 1 for f in CORRECTION_FIELDS}}
                for s in self.src["screen_analysis"].find(correction_filter(cw), proj):  # COLLSCAN (no source index)
                    fids.add(s.get("frame_result_id"))
                    reasons["correction"] += 1
                    for f in CORRECTION_FIELDS:
                        v = s.get(f)
                        if isinstance(v, datetime) and (cw.get(f) is None or v > cw[f]):
                            cw[f] = v
                new_meta["correction_watermarks_src"] = cw
                new_meta["last_correction_rescan_at"] = now

            stats = self.process_frames(fids, videos, ctx)
            reconciled = False
            last_rec = meta.get("last_reconcile_at")
            if last_rec is None or (now - last_rec).total_seconds() >= self.st.etl_reconcile_s:
                self._reconcile(videos, ctx, stats)
                new_meta["last_reconcile_at"] = now
                reconciled = True

            changes = stats["inserted"] + stats["updated"] + stats["deleted"]
            if reconciled:
                self.rebuild_rollup(None)
                self.rebuild_sessions(videos, None)
                self.rebuild_unmapped()
            elif changes or changed_vids:
                self.rebuild_rollup(stats["days"])
                sess = set(stats["sessions"]) | {videos[v].get("session_id") for v in changed_vids}
                self.rebuild_sessions(videos, {s for s in sess if s is not None})
                if changes:
                    self.rebuild_unmapped()
            elapsed = time.monotonic() - t0
            new_meta.update({
                "watermark_id": max_id, "watermark_analyzed_src": max_an,
                "video_hashes": {str(k): v["hash"] for k, v in videos.items()},
                "last_run_at": now, "last_ok": True, "last_error": None, "last_mode": "once",
                "last_changes": changes, "last_elapsed_s": round(elapsed, 3),
                "observations": self.qe[OBS].estimated_document_count(),
                "sessions": self.qe[SESSIONS].estimated_document_count(),
                "aliases_version": self.registry.version, "etl_version": ETL_VERSION,
            })
            self._set_meta(new_meta)
            return {"mode": "once", "elapsed_s": round(elapsed, 3), "frames_considered": len(fids),
                    "inserted": stats["inserted"], "updated": stats["updated"], "deleted": stats["deleted"],
                    "unchanged": stats["unchanged"], "changes": changes, "reasons": reasons,
                    "reconciled": reconciled, "days_touched": len(stats["days"])}
        except Exception as exc:
            self._record_error("once", exc)
            raise
        finally:
            if not hold_lease:
                self.release_lease()

    def _reconcile(self, videos: Mapping[Any, Dict[str, Any]], ctx: EtlContext, stats: Dict[str, Any]) -> None:
        """Full in-place diff pass plus orphan deletion (source doc gone -> observation deleted)."""
        seen: Set[Any] = set()
        for group in self._iter_all_frames():
            fids = list(group)
            frames = {f["_id"]: f for f in self.src["frame_results"].find({"_id": {"$in": fids}}, FRAME_PROJECTION)}
            docs = self._build(group, frames, videos, ctx)
            self._apply(fids, docs, OBS, stats)
            stats["frames"] += len(fids)
            for d in docs:
                seen.add(d["src_id"])
        orphans = [o["src_id"] for o in self.qe[OBS].find({}, {"src_id": 1, "observed_at": 1, "session_id": 1},
                                                            max_time_ms=max(self.ms, 60000))
                   if o["src_id"] not in seen]
        for chunk in chunked(orphans, BATCH):
            self.qe[OBS].delete_many({"src_id": {"$in": chunk}})
        stats["deleted"] += len(orphans)
        stats["orphans"] = len(orphans)

    def reconcile(self, lease_ttl_s: int = 900) -> Dict[str, Any]:
        """Stand-alone reconcile (CLI)."""
        if not self.acquire_lease(lease_ttl_s):
            raise LeaseHeld(f"ETL lease held by {self.lease_info()}")
        try:
            now = _utcnow()
            ctx = EtlContext(self.st, self.registry, now)
            videos = load_videos(self.mongo, self.st)
            stats = _new_stats()
            self._reconcile(videos, ctx, stats)
            self.rebuild_rollup(None)
            self.rebuild_sessions(videos, None)
            self.rebuild_unmapped()
            self._set_meta({"last_reconcile_at": now})
            return {k: (len(v) if isinstance(v, set) else v) for k, v in stats.items()}
        finally:
            self.release_lease()

    def _record_error(self, mode: str, exc: Exception) -> None:
        """Store a short error in meta (no stack trace)."""
        try:
            self._set_meta({"last_ok": False, "last_error": f"{mode}: {type(exc).__name__}: {str(exc)[:300]}",
                            "last_error_at": _utcnow()})
        except Exception:
            pass

    # ------------------------------------------------------------------ reporting
    def summary(self) -> Dict[str, Any]:
        """Distribution counters used by the verification report."""
        o = self.qe[OBS]

        def grp(field: str, match: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
            p: List[Dict[str, Any]] = [{"$match": match}] if match else []
            p.append({"$group": {"_id": f"${field}", "n": {"$sum": 1}}})
            return {str(r["_id"]): r["n"] for r in o.aggregate(p, maxTimeMS=self.ms)}

        q = {str(r["_id"]): r["n"] for r in o.aggregate([
            {"$project": {"qa": {"$objectToArray": {"$ifNull": ["$q", {}]}}}}, {"$unwind": "$qa"},
            {"$unwind": "$qa.v"}, {"$group": {"_id": "$qa.v", "n": {"$sum": 1}}}], maxTimeMS=self.ms)}
        flags = {str(r["_id"]): r["n"] for r in o.aggregate([
            {"$unwind": "$flags"}, {"$group": {"_id": "$flags", "n": {"$sum": 1}}}], maxTimeMS=self.ms)}
        dup_primary = list(o.aggregate([
            {"$match": {"primary": True}},
            {"$group": {"_id": {"f": "$frame_result_id", "s": "$screen_key"}, "n": {"$sum": 1}}},
            {"$match": {"n": {"$gt": 1}}}, {"$count": "groups"}], maxTimeMS=self.ms))
        no_primary = list(o.aggregate([
            {"$group": {"_id": {"f": "$frame_result_id", "s": "$screen_key"}, "p": {"$sum": {"$cond": ["$primary", 1, 0]}}}},
            {"$match": {"p": 0}}, {"$count": "groups"}], maxTimeMS=self.ms))
        return {
            "observations": o.count_documents({}, maxTimeMS=self.ms),
            "time_basis": grp("time_basis"),
            "time_confidence": grp("time_confidence"),
            "source": grp("source"),
            "device_family": grp("device_family"),
            "primary": grp("primary"),
            "quality": q,
            "flags": flags,
            "groups_with_multiple_primary": dup_primary[0]["groups"] if dup_primary else 0,
            "groups_without_primary": no_primary[0]["groups"] if no_primary else 0,
        }


def _new_stats() -> Dict[str, Any]:
    """Fresh counters."""
    return {"inserted": 0, "updated": 0, "deleted": 0, "unchanged": 0, "frames": 0, "days": set(), "sessions": set()}


def _display_day(t: datetime, tz: str) -> str:
    """YYYY-MM-DD of an aware/naive-UTC datetime in the display tz."""
    from zoneinfo import ZoneInfo

    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(ZoneInfo(tz)).strftime("%Y-%m-%d")


def _day_range(day: str, tz: str) -> Tuple[datetime, datetime]:
    """[start, end) UTC of a display-tz day."""
    d = date.fromisoformat(day)
    a = wallclock_to_utc(datetime(d.year, d.month, d.day), tz)
    n = d + timedelta(days=1)
    return a, wallclock_to_utc(datetime(n.year, n.month, n.day), tz)


class EtlLoop(threading.Thread):
    """Background ETL inside the API process (single worker). Never raises; retries on Mongo errors."""

    def __init__(self, etl: Etl, interval_s: int, on_ready=None) -> None:
        """Create a daemon thread; call start() then stop()."""
        super().__init__(name="nicu_qe-etl", daemon=True)
        self.etl = etl
        self.interval_s = interval_s
        self._stop_evt = threading.Event()
        self.indexes_ready = False
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self.holds_lease = False

    def stop(self, timeout: float = 30.0) -> None:
        """Signal stop, wait, release the lease."""
        self._stop_evt.set()
        self.join(timeout)

    def run(self) -> None:
        """Loop: ensure indexes (once), take/renew lease, run_once, sleep."""
        from . import indexes

        ttl = max(300, 3 * self.interval_s)
        while not self._stop_evt.is_set():
            try:
                if not self.indexes_ready:
                    indexes.ensure(self.etl.qe)
                    self.indexes_ready = True
                if self.etl.acquire_lease(ttl):
                    self.holds_lease = True
                    self.last_result = self.etl.run_once(lease_ttl_s=ttl, hold_lease=True)
                    self.last_error = None
                else:
                    self.holds_lease = False
                    self.last_error = "lease held by another process"
            except Exception as exc:  # keep running: Mongo may come back
                self.last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                log.warning("ETL iteration failed: %s", self.last_error)
            self._stop_evt.wait(self.interval_s)
        if self.holds_lease:
            self.etl.release_lease()
