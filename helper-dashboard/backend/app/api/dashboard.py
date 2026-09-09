"""Dashboard API — read/list stored DashboardSpec JSON files + widget data."""

from __future__ import annotations

import os
import re
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from ..prometheus import catalog as prom_catalog
from ..prometheus.client import PrometheusClient, PrometheusUnavailable
from ..services import simulation
from ..services.dashboard_store import DashboardStore
from ..services.simulation import SimulatedQueryError
from ..services.spec_validator import SpecValidator, SpecValidationError
from ..specs import DashboardSpec
from ..specs.widget_spec import MAX_TABLE_QUERIES, _QUERY_KEY_PATTERN


router = APIRouter()
_store = DashboardStore()
_validator = SpecValidator()
_prom = PrometheusClient()

# Labels the patient/device selector may ask values for. Anything else
# is refused: the label name goes into a Prometheus URL path and this
# endpoint must never become a generic label browser.
LABEL_VALUE_ALLOWED: frozenset[str] = frozenset({"patient_id", "device_model"})

# Staleness budget: a sample older than this is "資料過期". Env-configured
# (never a hardcoded exporter window — that figure is not confirmed in
# the repo) and never below 2 x step, so a slow-stepping chart is not
# flagged stale between two perfectly regular samples.
_STALE_ENV = "HELPER_DASHBOARD_STALE_AFTER_SECONDS"
DEFAULT_STALE_AFTER_SECONDS = 120


def staleness_budget_seconds(step_s: int | None = None) -> int:
    raw = os.getenv(_STALE_ENV, "").strip()
    try:
        budget = int(raw) if raw else DEFAULT_STALE_AFTER_SECONDS
    except ValueError:
        budget = DEFAULT_STALE_AFTER_SECONDS
    if budget <= 0:
        budget = DEFAULT_STALE_AFTER_SECONDS
    if step_s:
        budget = max(budget, 2 * int(step_s))
    return budget


def _client():
    """The live client, or the deterministic simulated one when a
    developer switched a scenario on (never on by default)."""
    if simulation.is_active():
        return simulation.client()
    return _prom


# Both forms registered explicitly: redirect_slashes is disabled on
# the FastAPI app so the browser never follows a cross-origin 307.
@router.get("")
@router.get("/")
def list_dashboards() -> dict:
    return {"dashboards": _store.list_dashboard_ids()}


# ------------------------------------------------------------
# Label values — feeds the 病患/裝置 selector. Registered BEFORE the
# `/{dashboard_id}` route so the literal path wins.
# ------------------------------------------------------------
@router.get("/label-values")
def label_values(
    label: str = Query(..., min_length=1, max_length=64),
    hours: int = Query(24, ge=1, le=168),
) -> dict:
    """Distinct values of an allow-listed label over the last `hours`.

    Scoped to the clinical metric families so the selector never lists
    a `patient_id` that only ever appeared on some unrelated series.
    """
    if label not in LABEL_VALUE_ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"label must be one of {sorted(LABEL_VALUE_ALLOWED)}",
        )
    end = int(time.time())
    start = end - hours * 3600
    prom = _client()
    try:
        values = prom.label_values(
            label,
            start=start,
            end=end,
            match=['{__name__=~"(patient|ventilator|cerebral)_.*"}'],
        )
    except PrometheusUnavailable as exc:
        raise HTTPException(
            status_code=502, detail=f"Prometheus unavailable: {exc}"
        ) from exc
    except SimulatedQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "label": label,
        "hours": hours,
        "values": [v for v in values if v],
        "simulated": simulation.is_active(),
        "sim_scenario": simulation.scenario() if simulation.is_active() else None,
    }


@router.get("/{dashboard_id}")
def get_dashboard(dashboard_id: str) -> dict:
    spec = _store.load_dashboard(dashboard_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="dashboard not found")
    # by_alias=True so decision_flow edges serialize their `from` key (aliased
    # from the `from_` field), which the frontend FlowEdge type expects.
    return {"spec": spec.model_dump(mode="json", by_alias=True)}


@router.post("/validate")
def validate_dashboard(payload: dict) -> dict:
    """Revalidate an in-memory spec without saving it.

    Used by the frontend inspector to check a spec before asking Helper
    to patch it further.
    """
    try:
        spec = _validator.validate_dashboard(payload)
    except SpecValidationError as exc:
        return {"ok": False, "errors": exc.errors}
    # by_alias=True: decision_flow edges must echo back as `from` (canonical),
    # matching get_dashboard above — a client re-rendering the echoed spec
    # would otherwise silently lose every branch edge.
    return {"ok": True, "spec": spec.model_dump(mode="json", by_alias=True)}


# ------------------------------------------------------------
# Widget data — execute a widget's PromQL against live Prometheus.
# ------------------------------------------------------------
class QueryFilters(BaseModel):
    """Label equality filters (the 病患/裝置 selector). Applied in Python
    to the result series BEFORE the mean-collapse — never spliced into
    PromQL, so a selector value can't inject into the query."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str | None = Field(default=None, max_length=128)
    device_model: str | None = Field(default=None, max_length=128)

    def active(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.patient_id is not None and self.patient_id.strip():
            out["patient_id"] = self.patient_id.strip()
        if self.device_model is not None and self.device_model.strip():
            out["device_model"] = self.device_model.strip()
        return out


class QueryColumn(BaseModel):
    """One column of a multi-metric table (`queries`)."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., pattern=_QUERY_KEY_PATTERN)
    promql: str = Field(..., min_length=1, max_length=512)
    label: str | None = Field(default=None, max_length=64)
    unit: str | None = Field(default=None, max_length=32)


class WidgetQueryRequest(BaseModel):
    """A single widget's QuerySpec, sent by the frontend renderer."""

    source: str = "prometheus"
    # Optional only when `queries` (multi-metric table) is present.
    promql: str | None = Field(default=None, max_length=512)
    query_type: str = "instant"
    range: str | None = None
    step: str | None = None
    # Shared refresh tick (epoch seconds). The frontend computes ONE
    # end per refresh tick and sends it to every widget query so all
    # widgets share the same right edge; the backend floors it to this
    # widget's step. None = "now".
    end: int | None = None
    # Additive (DESIGN §4): selector filters, multi-metric table
    # columns joined by `join_label` (default patient_id).
    filters: QueryFilters | None = None
    queries: list[QueryColumn] | None = Field(default=None, max_length=MAX_TABLE_QUERIES)
    join_label: str = Field(default="patient_id", pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")


@router.post("/query")
def query_widget(req: WidgetQueryRequest) -> dict:
    """Run one widget's PromQL and return series data for the renderer.

    No mock data, ever, on this path (fail closed):

    - ``source != "prometheus"`` → HTTP 400. Mock data on a clinical
      dashboard is dangerous; decision_flow specs already reject it at
      validation and this endpoint refuses it at runtime too.
    - Prometheus unreachable → HTTP 502 with the underlying reason, so
      the frontend renders a loud error state instead of fabricated data.
    - Success carries an explicit ``data_status``: ``"ok"`` when series
      are present, ``"empty"`` when the query succeeded but returned no
      data — the frontend renders an unmistakable NO-DATA state (silent
      emptiness has already hidden an 11-day stale outage once).
    - Success also echoes ``end`` (the step-aligned epoch second the
      query actually evaluated at) and ``range_seconds`` so charts pin
      their x-domain to the exact window the backend used.

    Additive contract (DESIGN §4): ``freshness`` (real sample time via
    ``max(timestamp(<metric>))`` + stale flag against the env budget),
    ``step_seconds``, ``filters_applied``, ``series_labels``,
    ``simulated``/``sim_scenario`` (developer simulation only — a
    simulated payload is always labelled), and for ``queries`` tables
    ``columns`` + pivoted ``table`` rows with per-cell ``null`` gaps.
    Range ``points`` carry ``v: null`` entries where consecutive samples
    are more than 2 x step apart so charts draw gaps, never bridges.
    """
    if req.source != "prometheus":
        raise HTTPException(
            status_code=400, detail="mock data source is not allowed"
        )
    if not req.queries and not (req.promql or "").strip():
        raise HTTPException(
            status_code=400, detail="promql (or queries) is required"
        )

    range_s = _parse_duration(req.range) or 3600
    step_s = _parse_duration(req.step) or 30
    # One step-aligned end per request. When the frontend sends the
    # shared per-tick `end`, every widget evaluates against the same
    # instant: range windows are [end-range, end] and instant queries
    # (stat_card/gauge) are pinned to that same end, so stat values
    # and chart right edges agree.
    end = int(req.end) if req.end is not None else int(time.time())
    end -= end % step_s

    filters = req.filters.active() if req.filters else {}
    prom = _client()
    sim_active = simulation.is_active()
    promqls: list[str] = []

    try:
        if req.queries:
            promqls = [c.promql for c in req.queries]
            out = _pivot_queries(
                prom, req.queries, join_label=req.join_label,
                end=end, filters=filters,
            )
        elif req.query_type == "range":
            promqls = [req.promql or ""]
            resp = prom.query_range(
                req.promql, range_s=range_s, step_s=step_s, end=end
            )
            out = _to_widget_data(resp, "range", step_s=step_s, filters=filters)
        else:
            promqls = [req.promql or ""]
            resp = prom.query(req.promql, time_s=end)
            out = _to_widget_data(resp, "instant", filters=filters)
    except PrometheusUnavailable as exc:
        raise HTTPException(
            status_code=502, detail=f"Prometheus unavailable: {exc}"
        ) from exc
    except SimulatedQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    budget = staleness_budget_seconds(step_s)
    # Echo the step-aligned end actually used plus the window length so
    # charts can pin their x-domain to [end - range_seconds, end].
    out["end"] = end
    out["range_seconds"] = range_s
    out["step_seconds"] = step_s
    out["filters_applied"] = filters
    out["freshness"] = _freshness(
        prom, promqls, end=end, filters=filters, budget=budget,
    )
    out["simulated"] = sim_active
    out["sim_scenario"] = simulation.scenario() if sim_active else None
    return out


# ------------------------------------------------------------
# Transforms
# ------------------------------------------------------------
def _series_passes(labels: dict[str, Any], filters: dict[str, str]) -> bool:
    """Label equality; a series LACKING a filtered label is dropped
    (an unlabeled `avg(...)` cannot be attributed to a patient)."""
    for k, v in filters.items():
        if str(labels.get(k, "\x00missing")) != v:
            return False
    return True


def _label_set(labels: dict[str, Any]) -> dict[str, str]:
    return {
        k: str(labels[k]) for k in ("patient_id", "device_model") if k in labels
    }


def _remember_labels(seen: list[dict[str, str]], labels: dict[str, Any]) -> None:
    ls = _label_set(labels)
    if ls and ls not in seen:
        seen.append(ls)


def _to_widget_data(
    resp: dict[str, Any],
    query_type: str,
    *,
    step_s: int | None = None,
    filters: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Transform a Prometheus response into the renderer's WidgetData shape.

    Multi-series results are collapsed into a single series by MEAN, and
    ``n_series`` reports how many series were averaged so the frontend
    can annotate. Mean, never sum: clinical gauges arrive as N parallel
    series (free-text ``device_model`` label variants for one physical
    device), and sum-collapse once rendered a heart rate of 163 bpm as
    814 on a live dashboard. The raw per-series rows are preserved in
    ``table`` for instant queries.

    ``data_status`` makes emptiness explicit: ``"ok"`` when at least one
    usable sample exists, ``"empty"`` when the query succeeded but returned
    no data. The frontend must render "empty" as a loud NO-DATA state.

    ``filters`` (label equality) are applied BEFORE the collapse so a
    patient selection averages only that patient's device variants.
    ``step_s`` enables gap detection on range results: a ``v: null``
    point is inserted after any sample whose successor is more than
    2 x step away.
    """
    source = str(resp.get("source", "prometheus"))
    result = ((resp.get("data") or {}).get("result")) or []
    filters = filters or {}
    series_labels: list[dict[str, str]] = []

    if query_type != "range":
        table: list[dict[str, Any]] = []
        total = 0.0
        for series in result:
            labels = series.get("metric", {}) or {}
            if not _series_passes(labels, filters):
                continue
            pair = series.get("value")
            if not pair or len(pair) < 2:
                continue
            try:
                v = float(pair[1])
            except (TypeError, ValueError):
                continue
            total += v
            _remember_labels(series_labels, labels)
            table.append({"metric": labels, "value": v})
        n_series = len(table)
        return {
            "source": source,
            "data_status": "ok" if n_series else "empty",
            "points": [],
            "latest": (total / n_series) if n_series else None,
            "table": table,
            "n_series": n_series,
            "series_labels": series_labels,
        }

    # range → matrix: average aligned samples across series into one
    # line. Each timestamp divides by the number of series that actually
    # have a sample there (series can start/stop mid-window).
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    n_series = 0
    for series in result:
        labels = series.get("metric", {}) or {}
        if not _series_passes(labels, filters):
            continue
        contributed = False
        for pair in series.get("values") or []:
            if not pair or len(pair) < 2:
                continue
            try:
                ts = int(float(pair[0]))
                v = float(pair[1])
            except (TypeError, ValueError):
                continue
            contributed = True
            sums[ts] = sums.get(ts, 0.0) + v
            counts[ts] = counts.get(ts, 0) + 1
        if contributed:
            n_series += 1
            _remember_labels(series_labels, labels)
    points: list[dict[str, Any]] = []
    prev_ts: int | None = None
    for ts in sorted(sums):
        if step_s and prev_ts is not None and (ts - prev_ts) > 2 * step_s:
            # Gap: emit an explicit null one step after the last sample
            # so the chart breaks the line instead of bridging it.
            points.append({"t": prev_ts + step_s, "v": None})
        points.append({"t": ts, "v": sums[ts] / counts[ts]})
        prev_ts = ts
    real = [p for p in points if p["v"] is not None]
    return {
        "source": source,
        "data_status": "ok" if real else "empty",
        "points": points,
        "latest": real[-1]["v"] if real else None,
        "table": [],
        "n_series": n_series,
        "series_labels": series_labels,
    }


def _pivot_queries(
    prom: Any,
    columns: list[QueryColumn],
    *,
    join_label: str,
    end: int,
    filters: dict[str, str],
) -> dict[str, Any]:
    """Run one instant query per column at the same `end` and join the
    per-series rows by `join_label` (patient_id). A cell with no sample
    is ``null`` — never 0 — and carries its own sample timestamp so the
    table can show 量測時間 per value."""
    rows: dict[str | None, dict[str, Any]] = {}
    series_labels: list[dict[str, str]] = []
    known = _known_metric_names()
    for col in columns:
        resp = prom.query(col.promql, time_s=end)
        result = ((resp.get("data") or {}).get("result")) or []
        for series in result:
            labels = series.get("metric", {}) or {}
            if not _series_passes(labels, filters):
                continue
            pair = series.get("value")
            if not pair or len(pair) < 2:
                continue
            try:
                v = float(pair[1])
            except (TypeError, ValueError):
                continue
            jv = labels.get(join_label)
            jv = str(jv) if jv is not None else None
            row = rows.setdefault(jv, {
                "labels": ({join_label: jv} if jv is not None else {}),
                "values": {},
                "ts": {},
            })
            if row["values"].get(col.key) is not None:
                continue  # first series per (row, column) wins
            row["values"][col.key] = v
            if "device_model" in labels and "device_model" not in row["labels"]:
                row["labels"]["device_model"] = str(labels["device_model"])
            _remember_labels(series_labels, labels)
        # Per-cell sample time: max over the raw metrics behind this
        # column, grouped by the join label (best effort — a probe that
        # fails simply leaves ts null, it never fails the table).
        for metric in prom_catalog.extract_metric_names(col.promql, known):
            probe = (
                f"max by ({join_label}) "
                f"(timestamp({_selector(metric, filters)}))"
            )
            try:
                ts_resp = prom.query(probe, time_s=end)
            except (PrometheusUnavailable, SimulatedQueryError):
                continue
            for series in ((ts_resp.get("data") or {}).get("result")) or []:
                labels = series.get("metric", {}) or {}
                jv = labels.get(join_label)
                jv = str(jv) if jv is not None else None
                if jv not in rows:
                    continue
                ts_val = _pair_value(series.get("value"))
                if ts_val is None:
                    continue
                cur = rows[jv]["ts"].get(col.key)
                rows[jv]["ts"][col.key] = (
                    int(ts_val) if cur is None else max(cur, int(ts_val))
                )
    for row in rows.values():
        for col in columns:
            row["values"].setdefault(col.key, None)
            row["ts"].setdefault(col.key, None)
    ordered = sorted(rows.items(), key=lambda kv: (kv[0] is None, kv[0] or ""))
    table: list[dict[str, Any]] = []
    for _, row in ordered:
        first = next(
            (row["values"][c.key] for c in columns if row["values"][c.key] is not None),
            None,
        )
        # `metric`/`value` kept alongside the pivot shape so legacy row
        # consumers (value + labels) keep working; `value` may be null.
        table.append({**row, "metric": dict(row["labels"]), "value": first})
    has_value = any(
        v is not None for row in table for v in row["values"].values()
    )
    return {
        "source": "prometheus",
        "data_status": "ok" if has_value else "empty",
        "points": [],
        "latest": None,
        "table": table,
        "n_series": len(table),
        "series_labels": series_labels,
        "columns": [
            {"key": c.key, "label": c.label or c.key, "unit": c.unit}
            for c in columns
        ],
        "join_label": join_label,
    }


# ------------------------------------------------------------
# Freshness
# ------------------------------------------------------------
_LABEL_VALUE_ESCAPE = str.maketrans({"\\": "\\\\", '"': '\\"', "\n": "\\n"})


def _selector(metric: str, filters: dict[str, str]) -> str:
    if not filters:
        return metric
    matchers = ",".join(
        f'{k}="{v.translate(_LABEL_VALUE_ESCAPE)}"' for k, v in sorted(filters.items())
    )
    return f"{metric}{{{matchers}}}"


def _pair_value(pair: Any) -> float | None:
    if not pair or len(pair) < 2:
        return None
    try:
        v = float(pair[1])
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def _known_metric_names() -> list[str]:
    """Catalog-matched names for metric extraction. Under simulation the
    simulated catalog; otherwise the live one (empty when unreachable —
    freshness then honestly reports null rather than guessing)."""
    if simulation.is_active():
        return simulation.metric_names()
    try:
        return list(prom_catalog.get_catalog().list_metrics())
    except PrometheusUnavailable:
        return []


def _freshness(
    prom: Any,
    promqls: list[str],
    *,
    end: int,
    filters: dict[str, str],
    budget: int,
    now: int | None = None,
) -> dict[str, Any]:
    """`sample_ts` = max over `max(timestamp(<metric>))` for every
    catalog-matched raw metric in the promqls (instant, at `end`, with
    the selector filters applied so a frozen patient is not masked by
    a live one). null when no raw metric could be identified."""
    known = _known_metric_names()
    metrics: list[str] = []
    for p in promqls:
        for m in prom_catalog.extract_metric_names(p or "", known):
            if m not in metrics:
                metrics.append(m)
    sample_ts: int | None = None
    for metric in metrics:
        probe = f"max(timestamp({_selector(metric, filters)}))"
        try:
            resp = prom.query(probe, time_s=end)
        except (PrometheusUnavailable, SimulatedQueryError):
            continue
        except Exception:  # pragma: no cover - probe must never fail the query
            continue
        for series in ((resp.get("data") or {}).get("result")) or []:
            v = _pair_value(series.get("value"))
            if v is None:
                continue
            sample_ts = int(v) if sample_ts is None else max(sample_ts, int(v))
    now_i = int(now) if now is not None else int(time.time())
    age = max(0, now_i - sample_ts) if sample_ts is not None else None
    return {
        "sample_ts": sample_ts,
        "age_seconds": age,
        "stale": bool(age is not None and age > budget),
        "budget_seconds": budget,
    }


_DURATION_RE = re.compile(r"^(\d+)(ms|s|m|h)$")


def _parse_duration(s: str | None) -> int | None:
    """Parse a Prometheus-style duration ('1h', '30s', '5m', '500ms') to seconds."""
    if not s:
        return None
    m = _DURATION_RE.match(s.strip())
    if not m:
        return None
    value, unit = int(m.group(1)), m.group(2)
    if unit == "ms":
        return max(1, round(value / 1000))
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    return value * 3600
