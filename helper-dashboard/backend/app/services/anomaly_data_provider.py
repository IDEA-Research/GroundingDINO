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

import math
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
    """Fetch one observation for a metric at an injected clock time `now`.

    `labels` is the rule's validated label set (e.g. {"patient": "neo-001"});
    a live provider MUST use it to select exactly one series — evaluating an
    arbitrary series of a multi-patient metric is a wrong-patient hazard.
    Single-series providers may ignore it.
    """

    def observe(
        self,
        metric: str,
        *,
        now: float,
        labels: dict[str, str] | None = None,
    ) -> Observation: ...


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

    def observe(
        self,
        metric: str,
        *,
        now: float,
        labels: dict[str, str] | None = None,
    ) -> Observation:
        # `labels` is accepted for protocol compatibility; the sim generates
        # exactly one series per metric (one synthetic patient), so there is
        # no ambiguity to resolve.
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

    Load-bearing safety steps (each fails CLOSED when unavailable):

    - Reads the `source` flag on every response so the core refuses the
      silent mock fallback.
    - Selects the rule's exact series via its validated `labels` and reports
      a multi-series answer as `ambiguous` (an arbitrary first-pick could
      evaluate the WRONG PATIENT) — the core gates it to SIGNAL_LOST.
    - Queries `timestamp(<sel>)` for the sample's OWN scrape time (the
      instant-vector pair carries the query-EVALUATION time, ~now, which
      would make the staleness gate structurally unable to trip). This
      detects a STOPPED scrape — it can NOT detect a frozen value behind a
      still-live /metrics endpoint, because Prometheus re-stamps the frozen
      value on every successful scrape.
    - For the frozen-publisher case, pass `freshness_metric`: a gauge whose
      VALUE is the measurement's own wall-clock time (the neonatal publisher
      exports `neonatal_sim_last_update_timestamp_seconds` exactly for
      this). The effective sample_ts is then the OLDER of scrape time and
      measurement time, so either failure mode trips the gate. Live wiring
      MUST set this; without it, staleness only covers scrape stoppage.
    - MEASURES baseline coverage (`count_over_time` x the scrape interval)
      instead of asserting a constant, so INSUFFICIENT_BASELINE stays
      load-bearing on live data; unmeasurable -> 0.0 coverage.

    Unconfirmable freshness fails to sample_ts = -inf — infinitely stale in
    ANY consumer's budget (a finite sentinel could pass a core configured
    with a larger staleness budget than the provider assumed).
    """

    # Rules lock the baseline window to 24h (BaselineSpec Literal["24h"]).
    _WINDOW = "24h"
    _WINDOW_S = 24 * 3600.0

    def __init__(
        self,
        client: PrometheusClient | None = None,
        *,
        staleness_budget_s: float = 60.0,
        scrape_interval_s: float = 15.0,
        freshness_metric: str | None = None,
    ) -> None:
        self.client = client or PrometheusClient()
        self.staleness_budget_s = float(staleness_budget_s)
        # Converts a sample COUNT into covered seconds. Pass the actual
        # scrape interval of the job feeding these metrics; an over-estimate
        # inflates coverage, an under-estimate fails closed.
        self.scrape_interval_s = float(scrape_interval_s)
        # Measurement-time heartbeat gauge (see class docstring). REQUIRED
        # for live wiring to catch frozen-publisher freshness failures.
        self.freshness_metric = freshness_metric

    @staticmethod
    def _label_selector(metric: str, labels: dict[str, str] | None = None) -> str:
        # Metric names and label keys/values are already validated by
        # AlertRuleSpec (identifier shape + PromQL token denylist); values are
        # still quote-escaped here as defence in depth.
        if not labels:
            return metric
        pairs = ",".join(
            f'{k}="{_escape_label_value(v)}"' for k, v in sorted(labels.items())
        )
        return f"{metric}{{{pairs}}}"

    def observe(
        self,
        metric: str,
        *,
        now: float,
        labels: dict[str, str] | None = None,
    ) -> Observation:
        sel = self._label_selector(metric, labels)
        instant = self.client.query(sel)
        baseline_resp = self.client.query(f"avg_over_time({sel}[{self._WINDOW}])")

        # READ the source flag — a mock fallback is reported honestly as
        # source=="mock"; the core refuses to alert on it. A half-mock
        # observation (real value, mock baseline) is not trustworthy either.
        src = str(instant.get("source", "mock"))
        base_src = str(baseline_resp.get("source", "mock"))
        if src == "prometheus" and base_src != "prometheus":
            src = base_src
        reachable = src == "prometheus"

        value, n_series = _instant_scalar(instant)
        baseline, n_base = _instant_scalar(baseline_resp)
        # More than one series for this selector means the rule's labels
        # under-select; report it so the gate fails closed (never first-pick).
        ambiguous = n_series > 1 or n_base > 1
        returns_data = value is not None

        # Sample freshness: default to -inf (infinitely stale in ANY
        # consumer's budget) unless positively confirmed. Effective
        # freshness is the OLDER of scrape time (catches stopped scrapes)
        # and, when configured, measurement time (catches a frozen publisher
        # behind a live /metrics endpoint, which Prometheus re-stamps fresh
        # on every scrape).
        sample_ts = float("-inf")
        if reachable and returns_data and not ambiguous:
            ts_resp = self.client.query(f"timestamp({sel})")
            if str(ts_resp.get("source", "mock")) == "prometheus":
                ts_val, ts_n = _instant_scalar(ts_resp)
                if ts_val is not None and ts_n == 1:
                    sample_ts = float(ts_val)
            if self.freshness_metric and sample_ts != float("-inf"):
                hb_sel = self._label_selector(self.freshness_metric, labels)
                hb_resp = self.client.query(hb_sel)
                hb_ts = float("-inf")
                if str(hb_resp.get("source", "mock")) == "prometheus":
                    hb_val, hb_n = _instant_scalar(hb_resp)
                    # isfinite: a NaN heartbeat would poison min() into
                    # silently degrading to scrape-only freshness.
                    if (
                        hb_val is not None
                        and hb_n == 1
                        and math.isfinite(hb_val)
                    ):
                        hb_ts = float(hb_val)
                # Configured but unconfirmable heartbeat fails CLOSED too.
                sample_ts = min(sample_ts, hb_ts)

        # Coverage is MEASURED, never asserted; unmeasurable -> 0.0 so the
        # core reports INSUFFICIENT_BASELINE rather than trusting a constant.
        coverage_s = 0.0
        if reachable and not ambiguous:
            cnt_resp = self.client.query(
                f"count_over_time({sel}[{self._WINDOW}])"
            )
            if str(cnt_resp.get("source", "mock")) == "prometheus":
                cnt, cnt_n = _instant_scalar(cnt_resp)
                if cnt is not None and cnt_n == 1:
                    coverage_s = min(
                        float(cnt) * self.scrape_interval_s, self._WINDOW_S
                    )

        status = DataStatus(
            reachable=reachable,
            returns_data=returns_data,
            source=src,
            sample_ts=float(sample_ts),
            history_coverage_s=coverage_s,
            ambiguous=ambiguous,
        )
        return Observation(value=value, baseline=baseline, status=status)


def _escape_label_value(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"')


def _instant_scalar(resp: dict[str, Any]) -> tuple[float | None, int]:
    """Pull (first value, series count) from an instant-vector response."""
    try:
        results = resp.get("data", {}).get("result", [])
        if not results:
            return None, 0
        pair = results[0].get("value")  # [eval_ts, "value"]
        if not pair or len(pair) < 2:
            return None, len(results)
        return float(pair[1]), len(results)
    except Exception:
        return None, 0
