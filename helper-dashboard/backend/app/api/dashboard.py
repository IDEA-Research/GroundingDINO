"""Dashboard API — read/list stored DashboardSpec JSON files."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..prometheus.client import PrometheusClient, PrometheusUnavailable
from ..services.dashboard_store import DashboardStore
from ..services.spec_validator import SpecValidator, SpecValidationError
from ..specs import DashboardSpec


router = APIRouter()
_store = DashboardStore()
_validator = SpecValidator()
_prom = PrometheusClient()


# Both forms registered explicitly: redirect_slashes is disabled on
# the FastAPI app so the browser never follows a cross-origin 307.
@router.get("")
@router.get("/")
def list_dashboards() -> dict:
    return {"dashboards": _store.list_dashboard_ids()}


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
class WidgetQueryRequest(BaseModel):
    """A single widget's QuerySpec, sent by the frontend renderer."""

    source: str = "prometheus"
    promql: str
    query_type: str = "instant"
    range: str | None = None
    step: str | None = None
    # Shared refresh tick (epoch seconds). The frontend computes ONE
    # end per refresh tick and sends it to every widget query so all
    # widgets share the same right edge; the backend floors it to this
    # widget's step. None = "now".
    end: int | None = None


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
    """
    if req.source != "prometheus":
        raise HTTPException(
            status_code=400, detail="mock data source is not allowed"
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

    try:
        if req.query_type == "range":
            resp = _prom.query_range(
                req.promql, range_s=range_s, step_s=step_s, end=end
            )
        else:
            resp = _prom.query(req.promql, time_s=end)
    except PrometheusUnavailable as exc:
        raise HTTPException(
            status_code=502, detail=f"Prometheus unavailable: {exc}"
        ) from exc

    out = _to_widget_data(resp, req.query_type)
    # Echo the step-aligned end actually used plus the window length so
    # charts can pin their x-domain to [end - range_seconds, end].
    out["end"] = end
    out["range_seconds"] = range_s
    return out


def _to_widget_data(resp: dict[str, Any], query_type: str) -> dict[str, Any]:
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
    """
    source = str(resp.get("source", "prometheus"))
    result = ((resp.get("data") or {}).get("result")) or []

    if query_type != "range":
        table: list[dict[str, Any]] = []
        total = 0.0
        for series in result:
            pair = series.get("value")
            if not pair or len(pair) < 2:
                continue
            try:
                v = float(pair[1])
            except (TypeError, ValueError):
                continue
            total += v
            table.append({"metric": series.get("metric", {}), "value": v})
        n_series = len(table)
        return {
            "source": source,
            "data_status": "ok" if n_series else "empty",
            "points": [],
            "latest": (total / n_series) if n_series else None,
            "table": table,
            "n_series": n_series,
        }

    # range → matrix: average aligned samples across series into one
    # line. Each timestamp divides by the number of series that actually
    # have a sample there (series can start/stop mid-window).
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    n_series = 0
    for series in result:
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
    points = [
        {"t": ts, "v": sums[ts] / counts[ts]} for ts in sorted(sums)
    ]
    return {
        "source": source,
        "data_status": "ok" if points else "empty",
        "points": points,
        "latest": points[-1]["v"] if points else None,
        "table": [],
        "n_series": n_series,
    }


def _parse_duration(s: str | None) -> int | None:
    """Parse a Prometheus-style duration ('1h', '30s', '5m', '500ms') to seconds."""
    if not s:
        return None
    text = s.strip()
    unit = ""
    for suffix in ("ms", "s", "m", "h"):
        if text.endswith(suffix):
            unit = suffix
            num = text[: -len(suffix)]
            break
    else:
        return None
    try:
        value = int(num)
    except ValueError:
        return None
    if unit == "ms":
        return max(1, round(value / 1000))
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    return value * 3600
