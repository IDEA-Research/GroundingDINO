"""Declarative index list for medical_monitor_qe (DESIGN 12.3) with ensure / drop / check.

Only the derived DB is ever touched; the source DB gets no index from this module.

`check` (REVIEW M12): every query shape is explained with executionStats on a fixed non-empty range
(the densest week of the most common metric, found from daily_rollup) and must
  - use one of the expected index names (IXSCAN), never COLLSCAN;
  - have no blocking SORT stage;
  - for find shapes: FETCH docsExamined <= 1.5 x (docs passing the FETCH filter) + 100;
  - for aggregation shapes: the $cursor / query stage must be an IXSCAN on an expected index
    (nReturned of an aggregation is the group count, so it is not used as a ratio).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .etl import ALL_METRIC, META, OBS, ROLLUP, SESSIONS, UNMAPPED


@dataclass(frozen=True)
class IndexSpec:
    """One index: collection, fixed name, key pattern and options."""

    collection: str
    name: str
    keys: Tuple[Tuple[str, int], ...]
    unique: bool = False
    expire_after_seconds: Optional[int] = None

    def options(self) -> Dict[str, Any]:
        """create_index keyword options."""
        o: Dict[str, Any] = {"name": self.name}
        if self.unique:
            o["unique"] = True
        if self.expire_after_seconds is not None:
            o["expireAfterSeconds"] = self.expire_after_seconds
        return o


INDEXES: Tuple[IndexSpec, ...] = (
    IndexSpec(OBS, "src_id_unique", (("src_id", 1),), unique=True),
    IndexSpec(OBS, "mk_observed_at", (("mk", 1), ("observed_at", 1))),
    IndexSpec(OBS, "session_observed_at", (("session_id", 1), ("observed_at", 1))),
    IndexSpec(OBS, "camera_observed_at", (("camera", 1), ("observed_at", 1))),
    IndexSpec(OBS, "family_observed_at", (("device_family", 1), ("observed_at", 1))),
    IndexSpec(OBS, "source_observed_at", (("source", 1), ("observed_at", 1))),
    IndexSpec(OBS, "observed_at", (("observed_at", 1),)),
    IndexSpec(OBS, "frame_result_id", (("frame_result_id", 1),)),
    IndexSpec(ROLLUP, "day_metric", (("day", 1), ("metric", 1))),
    IndexSpec(ROLLUP, "session_day", (("session_id", 1), ("day", 1))),
    IndexSpec(SESSIONS, "session_id_unique", (("session_id", 1),), unique=True),
    IndexSpec(SESSIONS, "source_camera", (("source", 1), ("camera", 1))),
    # interval-overlap query (last >= from AND first < to): both bounds are evaluated on index keys,
    # so FETCH only touches matching sessions (added in WP-4; not in DESIGN 12.3's list).
    IndexSpec(SESSIONS, "last_first_observed_at", (("last_observed_at", 1), ("first_observed_at", 1))),
    IndexSpec(UNMAPPED, "family_key_unique", (("device_family", 1), ("key", 1)), unique=True),
    IndexSpec(META, "expires_at_ttl", (("expires_at", 1),), expire_after_seconds=0),
)


# indexes we created earlier and that the contract has since replaced (API v2 #15); ensure() drops them
RETIRED: Tuple[Tuple[str, str], ...] = ((SESSIONS, "first_observed_at"),)


class IndexConflict(RuntimeError):
    """An existing index with our name has a different definition (or our keys under another name)."""


def _norm_keys(key: Any) -> Tuple[Tuple[str, int], ...]:
    """Key pattern from list_indexes as a tuple of (field, direction)."""
    return tuple((k, int(v)) for k, v in dict(key).items())


def ensure(db: Any, only: Optional[Sequence[str]] = None, rename: Optional[Mapping[str, str]] = None) -> List[str]:
    """Create missing indexes with fixed names; idempotent. Returns 'created'/'ok' lines."""
    rename = rename or {}
    out: List[str] = []
    for spec in INDEXES:
        if only is not None and spec.collection not in only:
            continue
        coll_name = rename.get(spec.collection, spec.collection)
        coll = db[coll_name]
        existing = {ix["name"]: ix for ix in coll.list_indexes()} if coll_name in db.list_collection_names() else {}
        if spec.name in existing:
            ix = existing[spec.name]
            same = (_norm_keys(ix["key"]) == spec.keys and bool(ix.get("unique")) == spec.unique
                    and ix.get("expireAfterSeconds") == spec.expire_after_seconds)
            if not same:
                raise IndexConflict(f"{coll_name}.{spec.name} exists with a different definition: {dict(ix)}")
            out.append(f"ok       {coll_name}.{spec.name}")
            continue
        for ix in existing.values():
            if _norm_keys(ix["key"]) == spec.keys and ix["name"] != spec.name:
                raise IndexConflict(f"{coll_name}: key {spec.keys} already indexed as {ix['name']!r}")
        coll.create_index(list(spec.keys), **spec.options())
        out.append(f"created  {coll_name}.{spec.name}")
    if only is None:
        names = set(db.list_collection_names())
        for coll_name, ix_name in RETIRED:
            if coll_name in names and ix_name in {ix["name"] for ix in db[coll_name].list_indexes()}:
                db[coll_name].drop_index(ix_name)
                out.append(f"retired  {coll_name}.{ix_name}")
    return out


def drop(db: Any) -> List[str]:
    """Drop only the indexes declared here (never _id_, never collections)."""
    out: List[str] = []
    names = set(db.list_collection_names())
    for spec in INDEXES:
        if spec.collection not in names:
            continue
        coll = db[spec.collection]
        if spec.name in {ix["name"] for ix in coll.list_indexes()}:
            coll.drop_index(spec.name)
            out.append(f"dropped  {spec.collection}.{spec.name}")
    return out


# ---------------------------------------------------------------------------- explain analysis
def _walk(plan: Any) -> Iterable[Dict[str, Any]]:
    """Yield every stage dict of a classic or SBE plan tree."""
    if isinstance(plan, Mapping):
        if "stage" in plan:
            yield plan
        for k in ("inputStage", "queryPlan", "outerStage", "innerStage", "thenStage", "elseStage"):
            if k in plan:
                yield from _walk(plan[k])
        for k in ("inputStages",):
            for s in plan.get(k, []) or []:
                yield from _walk(s)
    elif isinstance(plan, list):
        for s in plan:
            yield from _walk(s)


def _find_key(obj: Any, key: str) -> Optional[Any]:
    """First value for `key` anywhere in a nested explain document."""
    if isinstance(obj, Mapping):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def analyze_explain(explain: Mapping[str, Any], expected: Sequence[str], kind: str) -> Dict[str, Any]:
    """Apply the M12 criteria to an explain(executionStats) result."""
    qp = _find_key(explain, "queryPlanner") or {}
    winning = qp.get("winningPlan", {})
    stages = list(_walk(winning))
    names = [s.get("stage") for s in stages]
    idx = sorted({s.get("indexName") for s in stages if s.get("indexName")})
    problems: List[str] = []
    if "COLLSCAN" in names:
        problems.append("COLLSCAN")
    if "SORT" in names:
        problems.append("blocking SORT")
    if not idx:
        problems.append("no IXSCAN")
    elif not set(idx) & set(expected):
        problems.append(f"index {idx} not in expected {list(expected)}")
    es = _find_key(explain, "executionStats") or {}
    docs = es.get("totalDocsExamined")
    n_ret = es.get("nReturned")
    fetch_docs = fetch_ret = None
    for s in _walk(es.get("executionStages", {})):
        if s.get("stage") == "FETCH":
            fetch_docs, fetch_ret = s.get("docsExamined"), s.get("nReturned")
            break
    if kind == "find" and fetch_docs is not None and fetch_ret is not None:
        if fetch_docs > 1.5 * fetch_ret + 100:
            problems.append(f"FETCH docsExamined {fetch_docs} > 1.5*{fetch_ret}+100")
    return {"ok": not problems, "problems": problems, "indexes": idx, "stages": names,
            "totalDocsExamined": docs, "nReturned": n_ret, "fetch_docsExamined": fetch_docs,
            "fetch_nReturned": fetch_ret, "executionTimeMillis": es.get("executionTimeMillis")}


def explain_find(db: Any, coll: str, flt: Mapping[str, Any], sort: Optional[Sequence[Tuple[str, int]]] = None,
                 limit: Optional[int] = None, projection: Optional[Mapping[str, Any]] = None,
                 max_time_ms: int = 10000) -> Dict[str, Any]:
    """explain(executionStats) of a find."""
    cmd: Dict[str, Any] = {"find": coll, "filter": dict(flt), "maxTimeMS": max_time_ms}
    if sort:
        cmd["sort"] = {k: v for k, v in sort}
    if limit:
        cmd["limit"] = limit
    if projection:
        cmd["projection"] = dict(projection)
    return db.command({"explain": cmd, "verbosity": "executionStats"})


def explain_aggregate(db: Any, coll: str, pipeline: Sequence[Mapping[str, Any]], max_time_ms: int = 10000) -> Dict[str, Any]:
    """explain(executionStats) of an aggregation."""
    return db.command({"explain": {"aggregate": coll, "pipeline": list(pipeline), "cursor": {},
                                   "maxTimeMS": max_time_ms}, "verbosity": "executionStats"})


@dataclass
class Shape:
    """One query shape for `indexes check`."""

    name: str
    coll: str
    kind: str
    expected: Sequence[str]
    filter: Dict[str, Any] = field(default_factory=dict)
    sort: Optional[List[Tuple[str, int]]] = None
    limit: Optional[int] = None
    pipeline: Optional[List[Dict[str, Any]]] = None


def densest_week(db: Any, metric: str, tz: str) -> Optional[Tuple[datetime, datetime]]:
    """[start, end) UTC of the 7-day window around the busiest day of `metric` (from daily_rollup)."""
    from .tz import wallclock_to_utc

    top = list(db[ROLLUP].aggregate([
        {"$match": {"metric": metric}}, {"$group": {"_id": "$day", "n": {"$sum": "$count"}}},
        {"$sort": {"n": -1, "_id": 1}}, {"$limit": 1}]))
    if not top:
        return None
    d = datetime.strptime(top[0]["_id"], "%Y-%m-%d")
    start = wallclock_to_utc(d - timedelta(days=3), tz)
    return start, start + timedelta(days=7)


def build_shapes(engine: Any) -> List[Shape]:
    """Query shapes produced by the real QueryEngine match builder (so check = what the API runs)."""
    db = engine.qe
    tz = engine.settings.display_tz
    top = list(db[ROLLUP].aggregate([{"$match": {"metric": {"$ne": ALL_METRIC}}},
                                     {"$group": {"_id": "$metric", "n": {"$sum": "$count"}}},
                                     {"$sort": {"n": -1}}, {"$limit": 1}]))
    metric = top[0]["_id"] if top else "hr"
    week = densest_week(db, metric, tz)
    if week is None:
        raise RuntimeError("daily_rollup is empty: run `etl --full` before `indexes check`")
    f, t = week
    excl = ["out_of_range", "non_numeric", "user_marked_bad"]
    base = engine.build_match(metric, f, t, {}, False, excl)
    one_session = db[OBS].find_one({"mk": metric, "observed_at": {"$gte": f, "$lt": t}, "primary": True},
                                   {"session_id": 1, "device_family": 1})
    sess = (one_session or {}).get("session_id") or "none"
    fam = (one_session or {}).get("device_family") or "monitor"
    by_session = engine.build_match(metric, f, t, {"session_id": sess}, False, excl)
    by_family = engine.build_match(metric, f, t, {"device_family": fam}, False, excl)
    latest = engine.build_match(metric, None, None, {}, False, excl)
    field_ = f"$m.{metric}"
    bucket = [{"$match": base}, {"$group": {"_id": {"$dateTrunc": {"date": "$observed_at", "unit": "minute",
                                                                     "binSize": 5, "timezone": tz}},
                                            "n": {"$sum": 1}, "v": {"$avg": field_}}}]
    from .tz import to_display

    d0 = to_display(t - timedelta(days=400), tz).strftime("%Y-%m-%d")
    d1 = to_display(t, tz).strftime("%Y-%m-%d")
    any_frame = (db[OBS].find_one({}, {"frame_result_id": 1}) or {}).get("frame_result_id")
    any_src = (db[OBS].find_one({}, {"src_id": 1}) or {}).get("src_id")
    return [
        Shape("series_raw", OBS, "find", ["mk_observed_at"], base, [("observed_at", 1)], 10001),
        Shape("latest", OBS, "find", ["mk_observed_at"], latest, [("observed_at", -1)], 1),
        Shape("series_raw_by_session", OBS, "find", ["mk_observed_at", "session_observed_at"], by_session,
              [("observed_at", 1)], 10001),
        Shape("series_raw_by_family", OBS, "find", ["mk_observed_at", "family_observed_at"], by_family,
              [("observed_at", 1)], 10001),
        Shape("stats_find", OBS, "find", ["mk_observed_at"], base),
        Shape("series_bucket_5m", OBS, "aggregate", ["mk_observed_at"], pipeline=bucket),
        Shape("coverage_rollup_400d", ROLLUP, "aggregate", ["day_metric"], pipeline=[
            {"$match": {"day": {"$gte": d0, "$lt": d1}, "metric": {"$exists": True},
                        "time_basis": {"$ne": "analysis"}}},
            {"$group": {"_id": {"d": "$day", "m": "$metric"}, "n": {"$sum": "$count"}}}]),
        Shape("sessions_range", SESSIONS, "find", ["last_first_observed_at"],
              engine.sessions_match(f, t, {}), None, 5000),
        Shape("etl_frame_lookup", OBS, "find", ["frame_result_id"], {"frame_result_id": {"$in": [any_frame]}}),
        Shape("etl_src_lookup", OBS, "find", ["src_id_unique"], {"src_id": any_src}),
    ]


def check(engine: Any) -> Tuple[bool, List[Dict[str, Any]]]:
    """Run every shape; returns (all_ok, results)."""
    db = engine.qe
    results = []
    all_ok = True
    for s in build_shapes(engine):
        if s.kind == "find":
            ex = explain_find(db, s.coll, s.filter, s.sort, s.limit, max_time_ms=engine.ms)
        else:
            ex = explain_aggregate(db, s.coll, s.pipeline or [], max_time_ms=engine.ms)
        r = analyze_explain(ex, s.expected, s.kind)
        r["shape"] = s.name
        all_ok = all_ok and r["ok"]
        results.append(r)
    return all_ok, results
