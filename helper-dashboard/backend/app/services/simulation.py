"""Developer-only data simulation for the dashboard query path.

Lets a developer (or the Playwright verification run) walk the UI
through every data state — 正常 / 尚無資料 / 資料過期 / 部分缺值 /
來源中斷 / 查詢失敗 — WITHOUT touching the real Prometheus and without
ever showing synthetic numbers as real:

- OFF by default, always. It only switches on when BOTH the env gate
  `HELPER_DASHBOARD_ALLOW_SIMULATION=1` is set AND a developer (dev
  token, see `api/dev_simulation.py`) selected a scenario. State lives
  in this process only (never persisted); a backend restart = off.
- Every simulated query response is stamped `simulated: true` +
  `sim_scenario` by the query endpoint, and `GET /api/datasource/status`
  reports the active scenario, so the frontend can show the 模擬資料
  banner and per-card 模擬 badge (spec 七: 模擬資料必須明確標示).
- Data is deterministic: two patients `P-01` / `P-02` with plausible
  neonatal vitals produced by the same gentle-wander generator as
  `prometheus/neonatal_sim.py` (reused, not re-invented), clamped to its
  neonatal reference ranges.

The simulation is exposed as a drop-in for `PrometheusClient` (same
`query` / `query_range` / `label_values` / `targets` signatures) so the
query endpoint runs simulated payloads through EXACTLY the same
transform, filter, pivot and freshness code as real data — the only
difference is where the vector/matrix came from.
"""

from __future__ import annotations

import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from ..prometheus.client import PrometheusUnavailable
from ..prometheus.neonatal_sim import NEONATAL_RANGES, _sine_wander


ENV_ALLOW = "HELPER_DASHBOARD_ALLOW_SIMULATION"

SCENARIOS: tuple[str, ...] = (
    "off", "normal", "nodata", "stale", "partial", "offline", "error",
)

PATIENTS: tuple[str, ...] = ("P-01", "P-02")
DEVICE_MODEL = "SIM-MONITOR"
EXPORTER_JOB = "nicu_jetson"
# How old the frozen samples are in the `stale` scenario (DESIGN §4).
STALE_AGE_SECONDS = 900

# Per-patient steady-state centres (neonatal; NOT clinical thresholds —
# these are just plausible resting values for a synthetic baby).
_CENTRES: dict[str, dict[str, float]] = {
    "patient_heart_rate_bpm": {"P-01": 140.0, "P-02": 152.0},
    "patient_spo2_percentage": {"P-01": 96.0, "P-02": 94.0},
    "patient_respiration_rate_per_minute": {"P-01": 44.0, "P-02": 50.0},
    "patient_blood_pressure_systolic_mmhg": {"P-01": 62.0, "P-02": 66.0},
    "patient_blood_pressure_diastolic_mmhg": {"P-01": 38.0, "P-02": 41.0},
    "cerebral_rso2_left_percentage": {"P-01": 72.0, "P-02": 68.0},
    "cerebral_rso2_right_percentage": {"P-01": 71.0, "P-02": 70.0},
    "ventilator_fio2_percentage": {"P-01": 30.0, "P-02": 35.0},
    "ventilator_peep_cmh2o": {"P-01": 5.0, "P-02": 6.0},
    "ventilator_pip_cmh2o": {"P-01": 18.0, "P-02": 20.0},
}
# Clamp ranges reused from the neonatal sim where it defines them.
_CLAMP: dict[str, tuple[float, float]] = {
    "patient_heart_rate_bpm": NEONATAL_RANGES["hr"],
    "patient_spo2_percentage": (92.0, 99.0),
    "cerebral_rso2_left_percentage": (60.0, 80.0),
    "cerebral_rso2_right_percentage": (60.0, 80.0),
}
# Column order used by the `partial` scenario's every-other pattern.
CLINICAL_METRICS: tuple[str, ...] = tuple(_CENTRES.keys())


class SimulatedQueryError(Exception):
    """Raised by the `error` scenario: the query endpoint maps it to the
    same HTTP 400 path a bad query takes."""


class SimulationDisabled(Exception):
    """Selecting a scenario while the env gate is off."""


# ---------------------------------------------------------------------------
# In-process state
# ---------------------------------------------------------------------------

_STATE: dict[str, Any] = {"scenario": "off", "since": None}


def allowed() -> bool:
    return os.getenv(ENV_ALLOW, "").strip() == "1"


def scenario() -> str:
    """The active scenario, or "off" (also when the env gate is off,
    even if a scenario was selected earlier — the gate always wins)."""
    if not allowed():
        return "off"
    return str(_STATE.get("scenario") or "off")


def is_active() -> bool:
    return scenario() != "off"


def state() -> dict[str, Any]:
    return {
        "allowed": allowed(),
        "active": is_active(),
        "scenario": scenario(),
        "since": _STATE.get("since") if is_active() else None,
        "scenarios": list(SCENARIOS),
    }


def set_scenario(name: str) -> dict[str, Any]:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}")
    if name != "off" and not allowed():
        raise SimulationDisabled(
            f"{ENV_ALLOW}=1 is required to enable simulation"
        )
    _STATE["scenario"] = name
    _STATE["since"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds")
        if name != "off" else None
    )
    return state()


def reset() -> None:
    """Tests only: back to off."""
    _STATE["scenario"] = "off"
    _STATE["since"] = None


# ---------------------------------------------------------------------------
# Deterministic values
# ---------------------------------------------------------------------------

def metric_names(*, sc: str | None = None) -> list[str]:
    """Metric names the simulated Prometheus 'has'. `nodata` has no
    clinical series at all (exporter up, nothing scraped)."""
    sc = sc or scenario()
    if sc in ("nodata", "off"):
        return ["up"]
    return ["up", *CLINICAL_METRICS]


def sim_value(metric: str, patient: str, t: float) -> float:
    """Plausible reading for (metric, patient) at epoch `t` — pure."""
    centres = _CENTRES.get(metric)
    if centres is None:
        # Unknown / non-clinical metric: a neutral wander so arbitrary
        # dashboards still draw something visibly synthetic.
        return round(_sine_wander(50.0, t, amp=5.0, period_s=600.0), 3)
    centre = centres.get(patient, next(iter(centres.values())))
    # Same shape as NeonatalSim.sample_at (1.5% amplitude, 10-min period),
    # phase-shifted per patient so the two lines never overlap.
    phase = 0.0 if patient == PATIENTS[0] else 150.0
    value = _sine_wander(centre, t + phase, amp=centre * 0.015, period_s=600.0)
    lo, hi = _CLAMP.get(metric, (-math.inf, math.inf))
    return round(min(hi, max(lo, value)), 3)


def _partial_present(metric: str, patient: str) -> bool:
    """`partial`: every other (patient, column) cell is missing."""
    j = CLINICAL_METRICS.index(metric) if metric in CLINICAL_METRICS else 0
    i = PATIENTS.index(patient) if patient in PATIENTS else 0
    return (i + j) % 2 == 0


# ---------------------------------------------------------------------------
# Minimal PromQL reading (enough for the shapes the dashboard emits)
# ---------------------------------------------------------------------------

_BY_RE = re.compile(r"\bby\s*\(([^)]*)\)", re.IGNORECASE)
_MATCHER_RE = re.compile(r"\{([^}]*)\}")
_LABEL_EQ_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\"((?:[^\"\\]|\\.)*)\"")
_TIMESTAMP_RE = re.compile(r"\btimestamp\s*\(", re.IGNORECASE)
_IDENT_RE = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*")


def _metrics_in(promql: str, known: list[str]) -> list[str]:
    known_set = set(known)
    out: list[str] = []
    for m in _IDENT_RE.finditer(promql):
        name = m.group(0)
        if name in known_set and name not in out:
            out.append(name)
    return out


def _by_labels(promql: str) -> set[str] | None:
    m = _BY_RE.search(promql)
    if not m:
        return None
    return {p.strip() for p in m.group(1).split(",") if p.strip()}


def _equality_matchers(promql: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for block in _MATCHER_RE.findall(promql):
        for k, v in _LABEL_EQ_RE.findall(block):
            out[k] = v.replace('\\"', '"').replace("\\\\", "\\")
    return out


class SimulatedPrometheus:
    """`PrometheusClient` look-alike backed by the deterministic generator.

    Only the query shapes the dashboard actually emits are interpreted:
    a raw/aggregated clinical metric selector (optionally `by (patient_id)`
    and with `{patient_id="..."}` equality matchers), `timestamp(<m>)`
    freshness probes, and anything else (treated as one unlabeled series).
    """

    def __init__(self, sc: str | None = None, *, now: float | None = None) -> None:
        self._sc = sc
        self._now = now

    @property
    def scenario(self) -> str:
        return self._sc or scenario()

    def _clock(self) -> float:
        return self._now if self._now is not None else time.time()

    # -- guards -----------------------------------------------------
    def _guard(self) -> None:
        sc = self.scenario
        if sc == "offline":
            raise PrometheusUnavailable(
                "simulated: Prometheus offline (scenario=offline)"
            )
        if sc == "error":
            raise SimulatedQueryError(
                "simulated query error (scenario=error)"
            )

    def _last_sample_ts(self) -> float:
        now = self._clock()
        if self.scenario == "stale":
            return now - STALE_AGE_SECONDS
        return now

    # -- series enumeration ----------------------------------------
    def _series_for(self, promql: str) -> list[tuple[str | None, str | None, dict[str, str]]]:
        """(metric, patient, labels) for every series the promql selects."""
        sc = self.scenario
        known = metric_names(sc=sc)
        metrics = _metrics_in(promql, known)
        matchers = _equality_matchers(promql)
        by = _by_labels(promql)
        out: list[tuple[str | None, str | None, dict[str, str]]] = []
        if sc == "nodata":
            return out
        if not metrics or metrics == ["up"]:
            if matchers.get("patient_id") or matchers.get("device_model"):
                return out
            labels: dict[str, str] = {}
            if metrics == ["up"]:
                labels = {"job": EXPORTER_JOB, "instance": "sim:0"}
                if by is not None:
                    labels = {k: v for k, v in labels.items() if k in by}
            out.append((metrics[0] if metrics else None, None, labels))
            return out
        for metric in metrics:
            if metric == "up":
                continue
            for patient in PATIENTS:
                if matchers.get("patient_id") not in (None, patient):
                    continue
                if matchers.get("device_model") not in (None, DEVICE_MODEL):
                    continue
                if sc == "partial" and not _partial_present(metric, patient):
                    continue
                labels = {"patient_id": patient, "device_model": DEVICE_MODEL}
                if by is not None:
                    labels = {k: v for k, v in labels.items() if k in by}
                out.append((metric, patient, labels))
        return out

    # -- PrometheusClient surface ----------------------------------
    def probe(self, promql: str) -> tuple[bool, bool | None]:
        try:
            data = self.query(promql)
        except PrometheusUnavailable:
            return False, None
        except SimulatedQueryError:
            return True, False
        return True, bool(data.get("data", {}).get("result"))

    def query(self, promql: str, time_s: int | None = None) -> dict[str, Any]:
        self._guard()
        end = float(time_s) if time_s is not None else self._clock()
        sample_ts = min(end, self._last_sample_ts())
        is_ts = bool(_TIMESTAMP_RE.search(promql))
        result: list[dict[str, Any]] = []
        for metric, patient, labels in self._series_for(promql):
            if is_ts:
                value = sample_ts
            elif metric is None:
                value = sim_value("", "", sample_ts)
            else:
                value = sim_value(metric, patient or PATIENTS[0], sample_ts)
            result.append({"metric": labels, "value": [end, str(value)]})
        # `max(...)` without `by` collapses to one unlabeled series.
        if result and _BY_RE.search(promql) is None and re.search(
            r"\b(max|min|avg|sum)\s*\(", promql, re.IGNORECASE
        ):
            agg = max(float(r["value"][1]) for r in result)
            result = [{"metric": {}, "value": [end, str(agg)]}]
        return {
            "status": "success",
            "source": "prometheus",
            "data": {"resultType": "vector", "result": result},
        }

    def query_range(
        self,
        promql: str,
        range_s: int = 3600,
        step_s: int = 30,
        end: int | None = None,
    ) -> dict[str, Any]:
        self._guard()
        end_i = int(end) if end is not None else int(self._clock())
        if step_s > 0:
            end_i -= end_i % step_s
        start = end_i - int(range_s)
        last = self._last_sample_ts()
        sc = self.scenario
        result: list[dict[str, Any]] = []
        for metric, patient, labels in self._series_for(promql):
            values: list[list[Any]] = []
            t = start
            while t <= end_i:
                if t > last:
                    break  # stale: samples stop STALE_AGE_SECONDS ago
                frac = (t - start) / float(range_s or 1)
                if sc == "partial" and 0.4 < frac < 0.6:
                    t += step_s
                    continue  # a visible gap in the middle of the window
                if metric is None:
                    v = sim_value("", "", t)
                else:
                    v = sim_value(metric, patient or PATIENTS[0], t)
                values.append([t, str(v)])
                t += step_s
            if values:
                result.append({"metric": labels, "values": values})
        return {
            "status": "success",
            "source": "prometheus",
            "data": {"resultType": "matrix", "result": result},
        }

    def label_values(
        self,
        label: str,
        *,
        start: int | None = None,
        end: int | None = None,
        match: list[str] | None = None,
    ) -> list[str]:
        self._guard()
        if self.scenario == "nodata":
            return []
        if label == "patient_id":
            return list(PATIENTS)
        if label == "device_model":
            return [DEVICE_MODEL]
        return []

    def targets(self) -> list[dict[str, Any]]:
        if self.scenario == "offline":
            raise PrometheusUnavailable(
                "simulated: Prometheus offline (scenario=offline)"
            )
        return [{
            "labels": {"job": EXPORTER_JOB, "instance": "sim:0"},
            "health": "up",
            "lastError": "",
            "lastScrape": datetime.fromtimestamp(
                self._last_sample_ts(), tz=timezone.utc
            ).isoformat(timespec="seconds"),
        }]


def client(*, now: float | None = None) -> SimulatedPrometheus:
    """The simulated client for the CURRENT scenario."""
    return SimulatedPrometheus(now=now)
