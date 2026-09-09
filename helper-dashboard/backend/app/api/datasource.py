"""Data-source status (`GET /api/datasource/status`) — public, coarse.

Answers the user's "is my data still coming in?" separately from
"is the backend up?": Prometheus reachability, the clinical exporter
target's health, which clinical metrics exist, the newest patient_*
sample time, and whether developer simulation is active.

Exposes booleans, timestamps and the Prometheus HOST only — never the
full URL, credentials, or raw error bodies (a short, truncated last
error string is the one diagnostic detail, for the 技術細節 disclosure).
"""

from __future__ import annotations

import time
from typing import Any
from urllib import parse as urllib_parse

from fastapi import APIRouter

from ..prometheus import catalog as prom_catalog
from ..prometheus.client import PROM_URL, PrometheusClient, PrometheusUnavailable
from ..services import simulation
from .dashboard import staleness_budget_seconds

router = APIRouter()
_prom = PrometheusClient()

EXPORTER_JOB = "nicu_jetson"
_LAST_ERROR_MAX = 120


def _host_only(url: str) -> str | None:
    try:
        return urllib_parse.urlsplit(url).hostname
    except ValueError:  # pragma: no cover - defensive
        return None


def _short(text: Any) -> str | None:
    s = str(text or "").strip()
    if not s:
        return None
    return s if len(s) <= _LAST_ERROR_MAX else s[: _LAST_ERROR_MAX - 1] + "…"


def _exporter_status(prom: Any) -> dict[str, Any]:
    try:
        targets = prom.targets()
    except PrometheusUnavailable as exc:
        return {"job": EXPORTER_JOB, "health": "unknown", "last_error_short": _short(exc)}
    health = "unknown"
    last_error: str | None = None
    last_scrape: str | None = None
    for t in targets:
        labels = t.get("labels") or {}
        if str(labels.get("job", "")) != EXPORTER_JOB:
            continue
        h = str(t.get("health") or "unknown").lower()
        # Several instances under one job: any "up" counts as up, but the
        # first error seen is kept so the developer can read it.
        if h == "up":
            health = "up"
        elif health != "up":
            health = "down" if h == "down" else health
        if last_error is None and t.get("lastError"):
            last_error = _short(t.get("lastError"))
        if last_scrape is None and t.get("lastScrape"):
            last_scrape = str(t.get("lastScrape"))
    return {
        "job": EXPORTER_JOB,
        "health": health,
        "last_error_short": last_error,
        "last_scrape": last_scrape,
    }


def _last_patient_sample_ts(prom: Any) -> int | None:
    try:
        resp = prom.query('max(timestamp({__name__=~"patient_.*"}))')
    except (PrometheusUnavailable, simulation.SimulatedQueryError):
        return None
    for series in ((resp.get("data") or {}).get("result")) or []:
        pair = series.get("value")
        if not pair or len(pair) < 2:
            continue
        try:
            v = float(pair[1])
        except (TypeError, ValueError):
            continue
        if v == v:
            return int(v)
    return None


def _clinical_metrics_present(prom_sim_active: bool) -> list[str]:
    if prom_sim_active:
        return [m for m in simulation.metric_names() if prom_catalog.is_clinical_metric(m)]
    try:
        return prom_catalog.get_catalog().clinical_metrics()
    except PrometheusUnavailable:
        return []


@router.get("/status")
@router.get("/status/")
def datasource_status() -> dict:
    now = int(time.time())
    sim_active = simulation.is_active()
    prom: Any = simulation.client() if sim_active else _prom
    reachable, _ = prom.probe("up")
    exporter = _exporter_status(prom) if reachable else {
        "job": EXPORTER_JOB, "health": "unknown", "last_error_short": None,
    }
    metrics = _clinical_metrics_present(sim_active) if reachable else []
    last_ts = _last_patient_sample_ts(prom) if reachable else None
    age = max(0, now - last_ts) if last_ts is not None else None
    budget = staleness_budget_seconds()
    return {
        "backend_ok": True,
        "checked_at": now,
        "prometheus": {
            "reachable": bool(reachable),
            "url_host_only": "simulation" if sim_active else _host_only(PROM_URL),
        },
        "exporter": exporter,
        "clinical_metrics_present": metrics,
        "last_sample_ts": last_ts,
        "age_seconds": age,
        "stale": bool(age is not None and age > budget),
        "freshness_budget_seconds": budget,
        "simulation": {
            "active": sim_active,
            "scenario": simulation.scenario() if sim_active else None,
        },
    }
