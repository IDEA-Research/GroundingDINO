"""Data providers — turn a live observation into (value, baseline, DataStatus).

The evaluator service must not know HOW an observation is fetched, only that
it gets back the triple the core needs:

    value      the freshest scalar sample of the rule's metric (or None),
    baseline   avg_over_time(metric[24h]) (or None if unavailable),
    status     DataStatus(reachable, returns_data, source, sample_ts,
               history_coverage_s)

Two providers ship here:

  - `SimDataProvider` wraps the synthetic `NeonatalSim` so the whole loop can
    run end-to-end automatically and the integration test can step it with an
    injected clock. It is DETERMINISTIC.

  - `PrometheusDataProvider` reads the real `prometheus/client.py`. Crucially
    it READS the `source` flag on the response (`"prometheus"` vs `"mock"`) —
    the client currently WRITES that flag but nobody reads it, which is the
    silent-mock-fallback hazard. Reading it here is what lets the core FAIL
    CLOSED on fake data: a `source=="mock"` response yields
    `DataStatus.source="mock"`, never a scrubbed-to-prometheus value.

Neither provider makes a clinical decision — that is the core's job. They
only report provenance + freshness honestly.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..prometheus.client import PrometheusClient
from ..prometheus.neonatal_sim import NeonatalSim, Scenario
from .anomaly_core import DataStatus


class Observation:
    """Immutable-ish result handed to the evaluator service each tick."""

    __slots__ = ("value", "baseline", "status")

    def __init__(
        self,
        *,
        value: float | None,
        baseline: float | None,
        status: DataStatus,
    ) -> None:
        self.value = value
        self.baseline = baseline
        self.status = status


class DataProvider(Protocol):
    """Fetch one observation for a metric at an injected clock time `now`."""

    def observe(self, metric: str, *, now: float) -> Observation: ...


# ---------------------------------------------------------------------------
# Synthetic provider (deterministic; drives the whole loop + integration test)
# ---------------------------------------------------------------------------
class SimDataProvider:
    """Deterministic provider backed by `NeonatalSim`.

    It generates enough history to compute a healthy baseline and returns the
    sample whose intended tick time matches `now`. Scenarios (rso2_desat,
    source_degraded, unreachable, stale, short_history) let the integration
    test exercise the full lifecycle with an injected clock.
    """

    def __init__(
        self,
        *,
        scenario: Scenario = Scenario.healthy,
        start_ts: float = 0.0,
        step_s: float = 30.0,
        n_ticks: int = 60,
        patient_id: str = "neo-001",
        history_coverage_s: float = 24 * 3600.0,
    ) -> None:
        self.scenario = Scenario(scenario)
        self.start_ts = float(start_ts)
        self.step_s = float(step_s)
        self.n_ticks = int(n_ticks)
        self.patient_id = patient_id
        self.history_coverage_s = float(history_coverage_s)
        self._sim = NeonatalSim(
            patient_id=patient_id,
            scenario=self.scenario,
            start_ts=self.start_ts,
            step_s=self.step_s,
        )
        # Pre-generate the full trace so lookups are O(1) and byte-stable.
        self._series_cache: dict[str, list] = {}

    def _series(self, metric: str) -> list:
        if metric not in self._series_cache:
            self._series_cache[metric] = self._sim.series(metric, n=self.n_ticks)
        return self._series_cache[metric]

    def _index_for(self, now: float) -> int:
        if self.step_s <= 0:
            return 0
        idx = int(round((now - self.start_ts) / self.step_s))
        return max(0, min(idx, self.n_ticks - 1))

    def observe(self, metric: str, *, now: float) -> Observation:
        series = self._series(metric)
        idx = self._index_for(now)
        s = series[idx]
        # A live provider reports the freshest sample it actually has.
        status = DataStatus(
            reachable=s.reachable,
            returns_data=s.has_data,
            source=s.source,
            sample_ts=s.ts,
            history_coverage_s=self.history_coverage_s,
        )
        value = s.value if (s.reachable and s.has_data) else None
        baseline = self._sim.baseline(metric)
        return Observation(value=value, baseline=baseline, status=status)


# ---------------------------------------------------------------------------
# Prometheus provider (reads the live client AND its `source` flag)
# ---------------------------------------------------------------------------
class PrometheusDataProvider:
    """Provider backed by the real `PrometheusClient`.

    Reads BOTH the instant value and the 24h avg_over_time baseline, and — the
    load-bearing safety step — reads the `source` flag on each response so the
    core can fail closed on the silent mock fallback.
    """

    def __init__(
        self,
        client: PrometheusClient | None = None,
        *,
        staleness_budget_s: float = 60.0,
        history_coverage_s: float = 24 * 3600.0,
    ) -> None:
        self.client = client or PrometheusClient()
        self.staleness_budget_s = float(staleness_budget_s)
        self.history_coverage_s = float(history_coverage_s)

    @staticmethod
    def _label_selector(metric: str) -> str:
        # Metric names are already validated by AlertRuleSpec (PromQL token
        # denylist + identifier shape), so composing the query is safe.
        return metric

    def observe(self, metric: str, *, now: float) -> Observation:
        sel = self._label_selector(metric)
        instant = self.client.query(sel)
        baseline_resp = self.client.query(f"avg_over_time({sel}[24h])")

        # READ the source flag — this is the whole point. A mock fallback is
        # reported honestly as source=="mock"; the core refuses to alert on it.
        src = str(instant.get("source", "mock"))
        reachable = src == "prometheus"

        value, sample_ts = _first_scalar(instant)
        baseline, _ = _first_scalar(baseline_resp)
        base_src = str(baseline_resp.get("source", "mock"))
        # If either the value or the baseline came from mock, treat provenance
        # as mock — a half-mock observation is not trustworthy.
        if base_src != "prometheus":
            src = base_src if base_src != "prometheus" else src
            if base_src == "mock":
                src = "mock"

        returns_data = value is not None
        # If we never saw a sample ts, treat the sample as maximally stale so
        # the freshness gate trips rather than silently passing `now`.
        if sample_ts is None:
            sample_ts = now - (self.staleness_budget_s + 1.0)

        status = DataStatus(
            reachable=reachable,
            returns_data=returns_data,
            source=src,
            sample_ts=float(sample_ts),
            history_coverage_s=self.history_coverage_s,
        )
        return Observation(value=value, baseline=baseline, status=status)


def _first_scalar(resp: dict[str, Any]) -> tuple[float | None, float | None]:
    """Pull (value, sample_ts) from a Prometheus instant-vector response."""
    try:
        results = resp.get("data", {}).get("result", [])
        if not results:
            return None, None
        pair = results[0].get("value")  # [ts, "value"]
        if not pair or len(pair) < 2:
            return None, None
        return float(pair[1]), float(pair[0])
    except Exception:
        return None, None
