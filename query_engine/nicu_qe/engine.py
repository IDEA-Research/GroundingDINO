"""QueryEngine: read-only queries over medical_monitor_qe (API_CONTRACT sections 3-4 + v2 section 6).

All public methods are synchronous (pymongo); FastAPI calls them through asyncio.to_thread.
Times: parameters accept aware/naive datetimes, dates, ISO strings or date-only strings (naive =
QE_DISPLAY_TZ); ranges are half-open [from, to). Returned datetimes are aware UTC (library) and the
HTTP layer renders them in the display tz (v2 #12).

Injection surface (REVIEW L13): metric must be a catalog name before it is used in a field path;
every filter is a typed string from a fixed whitelist, `$`-prefixed values are refused, enums are
checked, nothing is json-decoded, no server-side JS.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .aliases import (
    DEVICE_FAMILIES_PATH,
    METRIC_ALIASES_PATH,
    METRIC_FAMILIES,
    UNKNOWN_FAMILY,
    AliasRegistry,
    MetricSpec,
    get_registry,
)
from .config import Settings
from .etl import ALL_METRIC, META, META_ETL_ID, OBS, ROLLUP, SESSIONS, UNMAPPED
from .models import DEFAULT_EXCLUDE_QUALITY, QUALITY_FLAGS, TIME_BASIS_ANALYSIS, Point, Series
from .tz import bucket_floor, bucket_unit, iter_buckets, parse_duration, parse_step, parse_time_param, to_display

SOURCES = ("stream", "manual", "dataset")
FAMILIES = tuple(METRIC_FAMILIES) + (UNKNOWN_FAMILY,)
AGGS = ("avg", "min", "max", "last", "count", "median")
GROUP_BY = ("session", "camera", "device_family", "day", "hour", "none")
COVERAGE_GROUPS = ("day", "hour")
FILLS = ("null", "none")
FILTER_KEYS = ("session_id", "camera", "source", "device_family", "llm_model", "patient_id")
SUGGESTED_STEPS = ("60s", "5m", "15m", "1h", "3h", "6h", "12h", "1d", "7d")
_METRIC_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
MAX_SESSIONS_LISTED = 200


class QueryError(Exception):
    """Client error (HTTP 400)."""

    status = 400
    code = "bad_request"


class UnknownMetric(QueryError):
    """Metric not in the catalog (HTTP 404)."""

    status = 404
    code = "unknown_metric"


class TooManyValues(QueryError):
    """Median over too many values (HTTP 400)."""

    code = "too_many_values"


# ---------------------------------------------------------------------------- pure helpers
def clean_str(name: str, value: Any, max_len: int = 200) -> Optional[str]:
    """Validate a typed string filter: str only, no leading '$', no NUL, bounded length. Empty -> None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise QueryError(f"{name} must be a string")
    s = value.strip()
    if not s:
        return None
    if s.startswith("$") or "\x00" in s:
        raise QueryError(f"{name}: value not allowed")
    if len(s) > max_len:
        raise QueryError(f"{name}: too long")
    return s


def percentile(sorted_vals: Sequence[float], p: float) -> Optional[float]:
    """Linear-interpolation percentile (numpy 'linear' / Hyndman-Fan type 7) of an ascending list."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be within [0, 1]")
    k = (n - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo))


def median(vals: Iterable[float]) -> Optional[float]:
    """Median of numeric values (None when empty)."""
    s = sorted(v for v in vals if v is not None)
    return percentile(s, 0.5)


def suggest_step(span: timedelta, max_points: int) -> str:
    """Smallest standard step whose bucket count fits max_points."""
    for s in SUGGESTED_STEPS:
        if span / parse_step(s) <= max_points:
            return s
    days = math.ceil(span.total_seconds() / 86400 / max_points)
    return f"{max(days, 1)}d"


def mask_sample(value: Any) -> Dict[str, Any]:
    """v2 #9: an unmapped sample is reported as type + length + first 2 characters only."""
    if value is None:
        typ, text = "null", ""
    elif isinstance(value, bool):
        typ, text = "bool", str(value)
    elif isinstance(value, int):
        typ, text = "int", str(value)
    elif isinstance(value, float):
        typ, text = "double", repr(value)
    elif isinstance(value, str):
        typ, text = "string", value
    elif isinstance(value, Mapping):
        typ, text = "object", json.dumps(value, ensure_ascii=False, default=str)
    elif isinstance(value, (list, tuple)):
        typ, text = "array", json.dumps(list(value), ensure_ascii=False, default=str)
    else:
        typ, text = type(value).__name__, str(value)
    n = len(text)
    masked = text[:2] + "*" * min(max(n - 2, 0), 10)
    return {"type": typ, "len": n, "masked": masked}


def parse_exclude_quality(value: Any) -> List[str]:
    """None -> default list; '' -> []; 'a,b' or list -> validated list."""
    if value is None:
        return list(DEFAULT_EXCLUDE_QUALITY)
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",") if x.strip()]
    elif isinstance(value, (list, tuple)):
        items = []
        for x in value:
            if not isinstance(x, str):
                raise QueryError("exclude_quality entries must be strings")
            items.append(x.strip())
    else:
        raise QueryError("exclude_quality must be a comma-separated string")
    bad = [x for x in items if x not in QUALITY_FLAGS]
    if bad:
        raise QueryError(f"unknown quality flag(s): {bad}; allowed {list(QUALITY_FLAGS)}")
    return sorted(set(items))


def _aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize a datetime read back from Mongo to aware UTC."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class QueryEngine:
    """Library entry point (API_CONTRACT section 4)."""

    def __init__(self, settings: Settings, mongo: Any = None, registry: Optional[AliasRegistry] = None) -> None:
        """Bind settings; the Mongo handle is created lazily (no connection until the first query)."""
        self.settings = settings
        self.registry = registry or get_registry()
        self._mongo = mongo

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "QueryEngine":
        """Build from QE_* environment variables."""
        return cls(Settings.from_env(environ))

    @property
    def mongo(self):
        """Lazily created Mongo handle (guarded QE DB name)."""
        if self._mongo is None:
            from .mongo import Mongo

            self._mongo = Mongo(self.settings)
        return self._mongo

    @property
    def qe(self):
        """The derived database handle."""
        return self.mongo.qe

    @property
    def ms(self) -> int:
        """maxTimeMS for every query."""
        return self.settings.mongo_max_time_ms

    # ------------------------------------------------------------------ validation
    def metric_spec(self, metric: Any) -> MetricSpec:
        """Return the catalog entry; unknown -> UnknownMetric (404)."""
        if not isinstance(metric, str) or not _METRIC_NAME_RE.match(metric.strip()):
            raise UnknownMetric(f"unknown metric {metric!r}")
        spec = self.registry.metrics.get(metric.strip())
        if spec is None:
            raise UnknownMetric(f"unknown metric {metric!r}")
        return spec

    def time_range(self, from_: Any, to: Any, required: bool = True) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Parse [from, to) with the date-only rule; enforce order and QE_MAX_RANGE_DAYS."""
        tz = self.settings.display_tz
        if from_ is None or to is None or (isinstance(from_, str) and not from_.strip()) or (isinstance(to, str) and not to.strip()):
            if required:
                raise QueryError("from and to are required")
            try:
                f = parse_time_param(from_, tz, is_end=False) if from_ not in (None, "") else None
                t = parse_time_param(to, tz, is_end=True) if to not in (None, "") else None
            except ValueError as exc:
                raise QueryError(str(exc)) from exc
            if f and t and t <= f:
                raise QueryError("to must be after from")
            return f, t
        try:
            f = parse_time_param(from_, tz, is_end=False)
            t = parse_time_param(to, tz, is_end=True)
        except ValueError as exc:
            raise QueryError(str(exc)) from exc
        if t <= f:
            raise QueryError("to must be after from")
        if t - f > timedelta(days=self.settings.max_range_days):
            raise QueryError(f"range exceeds QE_MAX_RANGE_DAYS={self.settings.max_range_days}")
        return f, t

    def filters(self, filters: Mapping[str, Any]) -> Dict[str, str]:
        """Whitelist + validate filter kwargs (never expanded from raw query params)."""
        out: Dict[str, str] = {}
        for k, v in filters.items():
            if k not in FILTER_KEYS:
                raise QueryError(f"unknown filter {k!r}")
            s = clean_str(k, v)
            if s is None:
                continue
            if k == "source" and s not in SOURCES:
                raise QueryError(f"source must be one of {SOURCES}")
            if k == "device_family" and s not in FAMILIES:
                raise QueryError(f"device_family must be one of {FAMILIES}")
            out[k] = s
        return out

    def build_match(self, metric: Optional[str], f: Optional[datetime], t: Optional[datetime],
                    flt: Mapping[str, str], include_analysis_time: bool, exclude: Sequence[str]) -> Dict[str, Any]:
        """Observation filter shared by series/latest/stats (mk index first, then observed_at)."""
        m: Dict[str, Any] = {}
        if metric:
            m["mk"] = metric
        if f is not None or t is not None:
            rng: Dict[str, Any] = {}
            if f is not None:
                rng["$gte"] = f
            if t is not None:
                rng["$lt"] = t
            m["observed_at"] = rng
        if flt.get("llm_model"):
            m["llm_model"] = flt["llm_model"]  # explicit model overrides primary selection
        else:
            m["primary"] = True
        if not include_analysis_time:
            m["time_basis"] = {"$ne": TIME_BASIS_ANALYSIS}
        for k in ("session_id", "camera", "source", "device_family", "patient_id"):
            if flt.get(k):
                m[k] = flt[k]
        if metric and exclude:
            m[f"q.{metric}"] = {"$nin": list(exclude)}
        return m

    # ------------------------------------------------------------------ series
    def series(self, metric: str, from_: Any, to: Any, step: Optional[str] = None, agg: str = "avg",
               fill: Optional[str] = None, limit: Optional[int] = None, include_analysis_time: bool = False,
               exclude_quality: Any = None, **filters: Any) -> Series:
        """Raw or bucketed series of one canonical metric."""
        spec = self.metric_spec(metric)
        metric = spec.name
        f, t = self.time_range(from_, to)
        flt = self.filters(filters)
        excl = parse_exclude_quality(exclude_quality)
        if agg not in AGGS:
            raise QueryError(f"agg must be one of {AGGS}")
        fill_v = fill if fill is not None else "null"
        if fill_v not in FILLS:
            raise QueryError(f"fill must be one of {FILLS}")
        max_points = self.settings.max_points
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise QueryError("limit must be a positive integer")
            max_points = min(limit, max_points)
        match = self.build_match(metric, f, t, flt, include_analysis_time, excl)
        field = f"$m.{metric}"
        obs = self.qe[OBS]
        res = Series(metric=metric, unit=spec.unit, step=step or None, agg=agg if step else None,
                     from_=f, to=t, fill=fill_v if step else None)

        if not step:
            cur = obs.find(match, {"_id": 0, "observed_at": 1, f"m.{metric}": 1, f"q.{metric}": 1, "session_id": 1,
                                   "camera": 1, "device_family": 1, "device_model": 1, "time_basis": 1,
                                   "llm_model": 1, "src_id": 1},
                           sort=[("observed_at", 1)], limit=max_points + 1, max_time_ms=self.ms)
            for d in cur:
                res.points.append(Point(
                    t=_aware_utc(d["observed_at"]), v=d.get("m", {}).get(metric), session_id=d.get("session_id"),
                    camera=d.get("camera"), device_family=d.get("device_family"), device_model=d.get("device_model"),
                    time_basis=d.get("time_basis"), quality=(d.get("q") or {}).get(metric), llm_model=d.get("llm_model"),
                    src_id=str(d.get("src_id")),
                ))
            if len(res.points) > max_points:
                res.points = res.points[:max_points]
                res.truncated = True
                res.suggested_step = suggest_step(t - f, self.settings.max_points)
        else:
            try:
                step_td = parse_step(step)
            except ValueError as exc:
                raise QueryError(str(exc)) from exc
            unit, bin_size = bucket_unit(step_td)
            if agg == "median":
                n_vals = obs.count_documents(match, maxTimeMS=self.ms)
                if n_vals > self.settings.max_stats_values:
                    raise TooManyValues(
                        f"median over {n_vals} values exceeds QE_MAX_STATS_VALUES={self.settings.max_stats_values}; "
                        f"use agg=avg, a shorter range, or a larger step with agg=avg"
                    )
            group: Dict[str, Any] = {
                "_id": {"$dateTrunc": {"date": "$observed_at", "unit": unit, "binSize": bin_size,
                                       "timezone": self.settings.display_tz}},
                "n": {"$sum": 1}, "min": {"$min": field}, "max": {"$max": field},
            }
            if agg == "avg":
                group["v"] = {"$avg": field}
            elif agg == "last":
                group["v"] = {"$last": field}
            elif agg == "median":
                group["vals"] = {"$push": field}
            pipeline: List[Dict[str, Any]] = [{"$match": match}]
            if agg == "last":
                pipeline.append({"$sort": {"observed_at": 1}})
            pipeline += [{"$group": group}, {"$sort": {"_id": 1}}]
            got: Dict[datetime, Point] = {}
            for r in obs.aggregate(pipeline, maxTimeMS=self.ms, allowDiskUse=True):
                b = _aware_utc(r["_id"])
                if agg == "median":
                    v = median(r.get("vals") or [])
                elif agg in ("min", "max"):
                    v = r.get(agg)
                elif agg == "count":
                    v = float(r["n"])
                else:
                    v = r.get("v")
                got[b] = Point(t=b, v=v, n=r["n"], min=r.get("min"), max=r.get("max"))
            if fill_v == "null":
                n_buckets = math.ceil((t - bucket_floor(f, step_td, self.settings.display_tz)) / step_td)
                if n_buckets > max_points * 4 and n_buckets > 100000:
                    raise QueryError(f"step {step} gives {n_buckets} buckets; suggested step "
                                     f"{suggest_step(t - f, self.settings.max_points)}")
                pts: Dict[datetime, Point] = {}
                for b in iter_buckets(f, t, step_td, self.settings.display_tz):
                    pts[b] = got.get(b) or Point(t=b, v=None, n=0)
                for b, p in got.items():  # never lose data if Mongo's alignment differed
                    pts.setdefault(b, p)
                res.points = [pts[k] for k in sorted(pts)]
            else:
                res.points = [got[k] for k in sorted(got)]
            if len(res.points) > max_points:
                res.points = res.points[:max_points]
                res.truncated = True
                res.suggested_step = suggest_step(t - f, self.settings.max_points)

        # side information in one pass: basis counts, quality counts (before exclusion), sessions
        base = {k: v for k, v in match.items() if k != f"q.{metric}"}
        qfilter = {f"q.{metric}": match[f"q.{metric}"]} if f"q.{metric}" in match else {}
        facet = list(obs.aggregate([
            {"$match": base},
            {"$facet": {
                "tb": [{"$match": qfilter}, {"$group": {"_id": "$time_basis", "n": {"$sum": 1}}}],
                "qc": [{"$project": {"qq": {"$ifNull": [f"$q.{metric}", []]}}}, {"$unwind": "$qq"},
                       {"$group": {"_id": "$qq", "n": {"$sum": 1}}}],
                "ss": [{"$match": qfilter}, {"$group": {"_id": "$session_id"}}, {"$sort": {"_id": 1}},
                       {"$limit": MAX_SESSIONS_LISTED}],
            }},
        ], maxTimeMS=self.ms, allowDiskUse=True))
        fc = facet[0] if facet else {"tb": [], "qc": [], "ss": []}
        res.time_basis_counts = {r["_id"]: r["n"] for r in fc["tb"] if r["_id"] is not None}
        res.quality_counts = {r["_id"]: r["n"] for r in fc["qc"] if r["_id"] is not None}
        res.sessions = [r["_id"] for r in fc["ss"] if r["_id"] is not None]
        if step is None and not res.truncated:
            if res.points:  # raw and complete: the last point is the latest value in range
                res.latest = {"t": res.points[-1].t, "v": res.points[-1].v}
        else:
            last = obs.find_one(match, {"observed_at": 1, f"m.{metric}": 1}, sort=[("observed_at", -1)],
                                max_time_ms=self.ms)
            if last and last.get("observed_at") is not None:
                res.latest = {"t": _aware_utc(last["observed_at"]), "v": last.get("m", {}).get(metric)}
        has_data = any(p.v is not None for p in res.points)
        res.data_status = "ok" if has_data else "empty"
        return res

    # ------------------------------------------------------------------ latest
    def latest(self, metric: str, max_age: Optional[str] = None, include_analysis_time: bool = False,
               exclude_quality: Any = None, **filters: Any) -> Dict[str, Any]:
        """Newest value of a metric (optionally not older than max_age)."""
        spec = self.metric_spec(metric)
        metric = spec.name
        flt = self.filters(filters)
        excl = parse_exclude_quality(exclude_quality)
        now = datetime.now(timezone.utc)
        f = None
        if max_age:
            try:
                f = now - parse_duration(max_age)
            except ValueError as exc:
                raise QueryError(str(exc)) from exc
        match = self.build_match(metric, f, None, flt, include_analysis_time, excl)
        d = self.qe[OBS].find_one(match, {"observed_at": 1, f"m.{metric}": 1, f"q.{metric}": 1, "session_id": 1,
                                          "camera": 1, "device_family": 1, "time_basis": 1, "llm_model": 1},
                                  sort=[("observed_at", -1)], max_time_ms=self.ms)
        if not d or d.get("observed_at") is None:
            return {"metric": metric, "unit": spec.unit, "data_status": "empty"}
        t = _aware_utc(d["observed_at"])
        return {
            "metric": metric, "unit": spec.unit, "t": t, "v": d.get("m", {}).get(metric),
            "age_seconds": round((now - t).total_seconds(), 3), "session_id": d.get("session_id"),
            "camera": d.get("camera"), "device_family": d.get("device_family"), "time_basis": d.get("time_basis"),
            "quality": (d.get("q") or {}).get(metric), "llm_model": d.get("llm_model"), "data_status": "ok",
        }

    # ------------------------------------------------------------------ stats
    def stats(self, metric: str, from_: Any, to: Any, group_by: str = "none", include_analysis_time: bool = False,
              exclude_quality: Any = None, **filters: Any) -> Dict[str, Any]:
        """Grouped statistics; percentiles computed in Python (Mongo 6.0 has no $percentile)."""
        spec = self.metric_spec(metric)
        metric = spec.name
        if group_by not in GROUP_BY:
            raise QueryError(f"group_by must be one of {GROUP_BY}")
        f, t = self.time_range(from_, to)
        flt = self.filters(filters)
        excl = parse_exclude_quality(exclude_quality)
        match = self.build_match(metric, f, t, flt, include_analysis_time, excl)
        obs = self.qe[OBS]
        tz = self.settings.display_tz
        n = obs.count_documents(match, maxTimeMS=self.ms)
        note = None
        rows: List[Dict[str, Any]] = []
        if n <= self.settings.max_stats_values:
            groups: Dict[Any, Dict[str, Any]] = {}
            cur = obs.find(match, {"observed_at": 1, f"m.{metric}": 1, "session_id": 1, "camera": 1,
                                   "device_family": 1}, max_time_ms=self.ms, batch_size=5000)
            for d in cur:
                v = (d.get("m") or {}).get(metric)
                if v is None:
                    continue
                ts = _aware_utc(d.get("observed_at"))
                key = self._group_key(group_by, d, ts, tz)
                g = groups.setdefault(key, {"vals": [], "first": ts, "last": ts})
                g["vals"].append(float(v))
                if ts is not None:
                    g["first"] = ts if g["first"] is None else min(g["first"], ts)
                    g["last"] = ts if g["last"] is None else max(g["last"], ts)
            for key in sorted(groups, key=lambda k: (k is None, str(k))):
                g = groups[key]
                vals = sorted(g["vals"])
                rows.append({
                    "key": key, "count": len(vals), "mean": sum(vals) / len(vals), "min": vals[0], "max": vals[-1],
                    "p05": percentile(vals, 0.05), "p50": percentile(vals, 0.5), "p95": percentile(vals, 0.95),
                    "first_t": g["first"], "last_t": g["last"],
                })
        else:
            note = (f"{n} values exceed QE_MAX_STATS_VALUES={self.settings.max_stats_values}; "
                    "percentiles omitted (p05/p50/p95 = null)")
            key_expr = self._group_expr(group_by, tz)
            for r in obs.aggregate([
                {"$match": match},
                {"$group": {"_id": key_expr, "count": {"$sum": 1}, "mean": {"$avg": f"$m.{metric}"},
                            "min": {"$min": f"$m.{metric}"}, "max": {"$max": f"$m.{metric}"},
                            "first_t": {"$min": "$observed_at"}, "last_t": {"$max": "$observed_at"}}},
                {"$sort": {"_id": 1}},
            ], maxTimeMS=self.ms, allowDiskUse=True):
                rows.append({"key": r["_id"], "count": r["count"], "mean": r["mean"], "min": r["min"], "max": r["max"],
                             "p05": None, "p50": None, "p95": None, "first_t": _aware_utc(r["first_t"]),
                             "last_t": _aware_utc(r["last_t"])})
        return {
            "metric": metric, "unit": spec.unit, "group_by": group_by, "from": f, "to": t,
            "time_axis": "observed_at", "dedupe": "llm_model" if flt.get("llm_model") else "primary",
            "percentile_method": "linear interpolation (type 7)", "note": note, "rows": rows,
            "data_status": "ok" if rows else "empty",
        }

    @staticmethod
    def _group_key(group_by: str, d: Mapping[str, Any], ts: Optional[datetime], tz: str) -> Any:
        """Python grouping key for stats rows."""
        if group_by == "none":
            return "all"
        if group_by == "session":
            return d.get("session_id")
        if group_by in ("camera", "device_family"):
            return d.get(group_by)
        if ts is None:
            return None
        local = to_display(ts, tz)
        return local.strftime("%Y-%m-%d") if group_by == "day" else local.strftime("%Y-%m-%dT%H:00")

    @staticmethod
    def _group_expr(group_by: str, tz: str) -> Any:
        """Mongo grouping expression twin of _group_key."""
        if group_by == "none":
            return "all"
        if group_by == "session":
            return "$session_id"
        if group_by in ("camera", "device_family"):
            return f"${group_by}"
        fmt = "%Y-%m-%d" if group_by == "day" else "%Y-%m-%dT%H:00"
        return {"$dateToString": {"format": fmt, "date": "$observed_at", "timezone": tz}}

    # ------------------------------------------------------------------ coverage
    def coverage(self, from_: Any, to: Any, group: str = "day", include_analysis_time: bool = False,
                 metric: Optional[str] = None, **filters: Any) -> Dict[str, Any]:
        """Which days/hours have data. Day buckets on whole display days come from daily_rollup."""
        if group not in COVERAGE_GROUPS:
            raise QueryError(f"group must be one of {COVERAGE_GROUPS}")
        f, t = self.time_range(from_, to)
        flt = self.filters(filters)
        if flt.get("llm_model") or flt.get("patient_id"):
            raise QueryError("coverage supports session_id, camera, source, device_family and metric filters")
        mname = self.metric_spec(metric).name if metric else None
        tz = self.settings.display_tz
        aligned = bucket_floor(f, timedelta(days=1), tz) == f and bucket_floor(t, timedelta(days=1), tz) == t
        if group == "day" and aligned:
            rows = self._coverage_rollup(f, t, flt, include_analysis_time, mname, tz)
            src = "daily_rollup"
        else:
            rows = self._coverage_obs(f, t, flt, include_analysis_time, mname, tz, group)
            src = "observations"
        return {"group": group, "from": f, "to": t, "metric": mname, "source_table": src, "rows": rows,
                "data_status": "ok" if rows else "empty"}

    def _coverage_rollup(self, f: datetime, t: datetime, flt: Mapping[str, str], include_analysis: bool,
                         metric: Optional[str], tz: str) -> List[Dict[str, Any]]:
        """Coverage from daily_rollup (index day_1_metric_1)."""
        d0 = to_display(f, tz).strftime("%Y-%m-%d")
        d1 = to_display(t, tz).strftime("%Y-%m-%d")
        match: Dict[str, Any] = {"day": {"$gte": d0, "$lt": d1}}
        match["metric"] = metric if metric else {"$exists": True}
        for k in ("session_id", "camera", "source", "device_family"):
            if flt.get(k):
                match[k] = flt[k]
        if not include_analysis:
            match["time_basis"] = {"$ne": TIME_BASIS_ANALYSIS}
        base_metric = metric or ALL_METRIC
        out: Dict[str, Dict[str, Any]] = {}
        for r in self.qe[ROLLUP].aggregate([
            {"$match": match},
            {"$group": {"_id": {"d": "$day", "m": "$metric", "b": "$time_basis"}, "n": {"$sum": "$count"},
                        "s": {"$addToSet": "$session_id"}}},
        ], maxTimeMS=self.ms, allowDiskUse=True):
            day, m, b = r["_id"]["d"], r["_id"]["m"], r["_id"]["b"]
            row = out.setdefault(day, {"bucket": day, "count": 0, "sessions": set(), "metrics": {}, "time_basis": {}})
            if m == base_metric:
                row["count"] += r["n"]
                row["sessions"].update(x for x in r["s"] if x is not None)
                row["time_basis"][b] = row["time_basis"].get(b, 0) + r["n"]
            if m != ALL_METRIC:
                row["metrics"][m] = row["metrics"].get(m, 0) + r["n"]
        return self._finish_coverage(out)

    def _coverage_obs(self, f: datetime, t: datetime, flt: Mapping[str, str], include_analysis: bool,
                      metric: Optional[str], tz: str, group: str) -> List[Dict[str, Any]]:
        """Coverage computed from observations (hour buckets or non-aligned ranges)."""
        match = self.build_match(metric, f, t, flt, include_analysis, [])
        fmt = "%Y-%m-%d" if group == "day" else "%Y-%m-%dT%H:00"
        bexpr = {"$dateToString": {"format": fmt, "date": "$observed_at", "timezone": tz}}
        out: Dict[str, Dict[str, Any]] = {}
        facet = list(self.qe[OBS].aggregate([
            {"$match": match},
            {"$project": {"b": bexpr, "time_basis": 1, "session_id": 1, "mk": 1}},
            {"$facet": {
                "docs": [{"$group": {"_id": {"b": "$b", "tb": "$time_basis"}, "n": {"$sum": 1},
                                     "s": {"$addToSet": "$session_id"}}}],
                "metrics": [{"$unwind": "$mk"}, {"$group": {"_id": {"b": "$b", "m": "$mk"}, "n": {"$sum": 1}}}],
            }},
        ], maxTimeMS=self.ms, allowDiskUse=True))
        fc = facet[0] if facet else {"docs": [], "metrics": []}
        for r in fc["docs"]:
            b = r["_id"]["b"]
            row = out.setdefault(b, {"bucket": b, "count": 0, "sessions": set(), "metrics": {}, "time_basis": {}})
            row["count"] += r["n"]
            row["sessions"].update(x for x in r["s"] if x is not None)
            row["time_basis"][r["_id"]["tb"]] = row["time_basis"].get(r["_id"]["tb"], 0) + r["n"]
        for r in fc["metrics"]:
            b, m = r["_id"]["b"], r["_id"]["m"]
            if metric and m != metric:
                continue
            row = out.setdefault(b, {"bucket": b, "count": 0, "sessions": set(), "metrics": {}, "time_basis": {}})
            row["metrics"][m] = r["n"]
        return self._finish_coverage(out)

    @staticmethod
    def _finish_coverage(out: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort buckets and turn session sets into counts."""
        rows = []
        for b in sorted(out):
            r = out[b]
            if r["count"] == 0 and not r["metrics"]:
                continue
            r["sessions"] = len(r["sessions"])
            rows.append(r)
        return rows

    # ------------------------------------------------------------------ sessions
    def sessions(self, from_: Any = None, to: Any = None, **filters: Any) -> Dict[str, Any]:
        """Sessions whose observed interval intersects [from, to)."""
        f, t = self.time_range(from_, to, required=False)
        allowed = {k: v for k, v in filters.items()}
        for k in allowed:
            if k not in ("source", "camera", "patient_id", "session_id"):
                raise QueryError(f"unknown filter {k!r}")
        flt = self.filters(allowed)
        match = self.sessions_match(f, t, flt)
        rows = []
        for d in self.qe[SESSIONS].find(match, {"_id": 0}, limit=5000, max_time_ms=self.ms):
            d["first_observed_at"] = _aware_utc(d.get("first_observed_at"))
            d["last_observed_at"] = _aware_utc(d.get("last_observed_at"))
            rows.append(d)
        # sorted here (small result) so the overlap query needs no blocking SORT stage
        far = datetime.max.replace(tzinfo=timezone.utc)
        rows.sort(key=lambda r: (r["first_observed_at"] or far, str(r.get("session_id"))))
        return {"from": f, "to": t, "rows": rows, "data_status": "ok" if rows else "empty"}

    @staticmethod
    def sessions_match(f: Optional[datetime], t: Optional[datetime], flt: Mapping[str, str]) -> Dict[str, Any]:
        """Interval-overlap filter: last_observed_at >= from AND first_observed_at < to."""
        match: Dict[str, Any] = dict(flt)
        if f is not None:
            match["last_observed_at"] = {"$gte": f}
        if t is not None:
            match["first_observed_at"] = {"$lt": t}
        return match

    # ------------------------------------------------------------------ catalog / unmapped / aliases
    def catalog(self) -> Dict[str, Any]:
        """Canonical metrics with 30-day counts and last observation, plus device families (from daily_rollup)."""
        tz = self.settings.display_tz
        since = (to_display(datetime.now(timezone.utc), tz).date() - timedelta(days=30)).isoformat()
        roll = self.qe[ROLLUP]
        not_analysis = {"time_basis": {"$ne": TIME_BASIS_ANALYSIS}}
        n30 = {r["_id"]: r["n"] for r in roll.aggregate([
            {"$match": {"day": {"$gte": since}, **not_analysis}},
            {"$group": {"_id": "$metric", "n": {"$sum": "$count"}}}], maxTimeMS=self.ms)}
        last = {r["_id"]: r["t"] for r in roll.aggregate([
            {"$match": not_analysis}, {"$group": {"_id": "$metric", "t": {"$max": "$last_t"}, "n": {"$sum": "$count"}}}],
            maxTimeMS=self.ms)}
        total = {r["_id"]: r["n"] for r in roll.aggregate([
            {"$match": not_analysis}, {"$group": {"_id": "$metric", "n": {"$sum": "$count"}}}], maxTimeMS=self.ms)}
        fams = {r["_id"]: r["n"] for r in roll.aggregate([
            {"$match": {"metric": ALL_METRIC}}, {"$group": {"_id": "$device_family", "n": {"$sum": "$count"}}}],
            maxTimeMS=self.ms)}
        metrics = []
        for name, s in self.registry.metrics.items():
            metrics.append({
                "name": name, "family": s.family, "unit": s.unit, "range": list(s.range) if s.range else None,
                "aliases_count": len(s.aliases), "n_obs_30d": n30.get(name, 0), "n_obs_total": total.get(name, 0),
                "last_observed_at": _aware_utc(last.get(name)),
            })
        return {"metrics": metrics, "device_families": [{"name": k, "n_observations": fams.get(k, 0)} for k in FAMILIES],
                "aliases_version": self.registry.version, "counts_exclude_time_basis": [TIME_BASIS_ANALYSIS]}

    def unmapped_keys(self, device_family: Optional[str] = None, since: Any = None, limit: int = 500) -> Dict[str, Any]:
        """Keys that did not map to a canonical metric; samples masked (v2 #9)."""
        flt = self.filters({"device_family": device_family} if device_family else {})
        match: Dict[str, Any] = dict(flt)
        if since not in (None, ""):
            try:
                match["last_seen"] = {"$gte": parse_time_param(since, self.settings.display_tz, is_end=False)}
            except ValueError as exc:
                raise QueryError(str(exc)) from exc
        rows = []
        for d in self.qe[UNMAPPED].find(match, {"_id": 0}, sort=[("count", -1), ("key", 1)], limit=int(limit),
                                        max_time_ms=self.ms):
            rows.append({
                "key": d.get("key"), "device_family": d.get("device_family"), "count": d.get("count"),
                "value_types": d.get("value_types") or {},
                "sample_values": [mask_sample(v) for v in (d.get("sample_values") or [])],
                "last_seen": _aware_utc(d.get("last_seen")),
            })
        return {"rows": rows, "data_status": "ok" if rows else "empty"}

    @staticmethod
    def aliases() -> Dict[str, Any]:
        """Raw metric_aliases.json and device_families.json."""
        with open(METRIC_ALIASES_PATH, encoding="utf-8") as fh:
            m = json.load(fh)
        with open(DEVICE_FAMILIES_PATH, encoding="utf-8") as fh:
            d = json.load(fh)
        return {"metric_aliases": m, "device_families": d}

    # ------------------------------------------------------------------ health
    def health(self) -> Tuple[int, Dict[str, Any]]:
        """(http_status, body). 503 when a database is unreachable; 'degraded' when ETL is failing."""
        from . import __version__

        body: Dict[str, Any] = {"status": "ok", "db": "ok", "qe_db": "ok", "etl": None,
                                "source_tz": self.settings.source_tz, "display_tz": self.settings.display_tz,
                                "source_tz_offset_hours_detected": None, "aliases_version": self.registry.version,
                                "version": __version__}
        try:
            self.mongo.ping_source()
        except Exception as exc:
            body["db"] = f"error: {type(exc).__name__}"
        try:
            self.mongo.ping_qe()
            meta = self.qe[META].find_one({"_id": META_ETL_ID}, max_time_ms=self.ms) or {}
        except Exception as exc:
            body["qe_db"] = f"error: {type(exc).__name__}"
            meta = None
        if body["db"] != "ok" or body["qe_db"] != "ok":
            body["status"] = "error"
            return 503, body
        now = datetime.now(timezone.utc)
        last = _aware_utc(meta.get("last_run_at")) if meta else None
        body["etl"] = {
            "last_run_at": last, "last_ok": meta.get("last_ok") if meta else None,
            "last_mode": meta.get("last_mode") if meta else None,
            "watermark_id": str(meta["watermark_id"]) if meta and meta.get("watermark_id") else None,
            "lag_seconds": round((now - last).total_seconds(), 1) if last else None,
            "observations": meta.get("observations") if meta else None,
            "sessions": meta.get("sessions") if meta else None,
            "last_error": meta.get("last_error") if meta else None,
        }
        body["source_tz_offset_hours_detected"] = meta.get("source_tz_offset_hours_detected") if meta else None
        if not meta or not meta.get("last_ok"):
            body["status"] = "degraded"
        elif last and (now - last).total_seconds() > max(600, 5 * self.settings.etl_interval_s):
            body["status"] = "degraded"
        return 200, body
