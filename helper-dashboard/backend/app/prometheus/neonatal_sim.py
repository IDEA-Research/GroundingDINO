"""Synthetic neonatal rSO2 data source (generator).

Real medical data is NOT flowing yet, so we control a synthetic exporter /
generator that emits realistic neonatal physiology so the whole anomaly
loop can run end-to-end automatically and so the golden harness has
deterministic, injectable traces.

This module is intentionally a PURE, importable generator in Increment 1.
It does not run an HTTP server yet (that is a later increment); it produces
in-memory time-series `Sample`s that fixtures and the evaluator core can
consume directly.

What it emits (per patient, channel-agnostic `Sample`s):

    rso2_left, rso2_right   cerebral oximetry (%), neonatal range ~55-85
    spo2                    pulse oximetry (%), neonatal target ~88-95
    hr                      heart rate (bpm), neonatal ~120-160
    map                     mean arterial pressure (mmHg), neonatal ~30-50
    fio2                    fraction inspired O2 (0.21-1.0)

Plus integrity / quality channels the evaluator's data-integrity gate and
OCR pipeline depend on:

    heartbeat               freshness signal — the sample timestamp itself;
                            also exposed as `monitor_heartbeat_seconds`.
    has_glare, has_occlusion   OCR-quality label channels (0/1).

Anomaly injection: `scenario=` selects a deterministic generator so goldens
are reproducible. A scenario can drive a sustained rSO2 desaturation, a
data-source degradation (real -> mock / unreachable / stale), or a short
history (<24h) to exercise INSUFFICIENT_BASELINE.

NOTE: neonatal ranges only. Do NOT assume adult physiology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


# Neonatal physiological reference ranges (used for realistic baselines and
# to keep injected anomalies inside *plausible* territory, not impossible).
NEONATAL_RANGES: dict[str, tuple[float, float]] = {
    "rso2_left": (55.0, 85.0),
    "rso2_right": (55.0, 85.0),
    "spo2": (88.0, 95.0),
    "hr": (120.0, 160.0),
    "map": (30.0, 50.0),
    "fio2": (0.21, 1.0),
}

# Healthy steady-state centres for the synthetic baby.
_BASELINE_CENTER: dict[str, float] = {
    "rso2_left": 72.0,
    "rso2_right": 71.0,
    "spo2": 92.0,
    "hr": 140.0,
    "map": 40.0,
    "fio2": 0.30,
}


class Scenario(str, Enum):
    """Deterministic, injectable trace shapes for the golden harness."""

    healthy = "healthy"  # stable -> MUST NOT fire
    rso2_desat = "rso2_desat"  # sustained >20% drop -> MUST fire at 5m
    source_degraded = "source_degraded"  # real -> mock -> SIGNAL_LOST
    unreachable = "unreachable"  # prometheus unreachable -> SIGNAL_LOST
    stale = "stale"  # data stops updating -> SIGNAL_LOST
    short_history = "short_history"  # <24h history -> INSUFFICIENT_BASELINE


@dataclass
class Sample:
    """One scalar observation at one instant, with provenance + quality.

    `source` mirrors `prometheus/client.py`'s `source` field; the evaluator
    MUST read it and refuse `source == "mock"`.
    """

    ts: float  # epoch seconds of THIS sample (freshness clock)
    metric: str
    value: float
    # The wall-clock tick time this sample belongs to. Normally == ts, but
    # for the `stale` scenario `ts` freezes while `intended_ts` keeps
    # advancing, so the freshness gate sees the sample go stale.
    intended_ts: float | None = None
    source: str = "prometheus"  # "prometheus" | "mock"
    reachable: bool = True
    has_data: bool = True
    has_glare: bool = False
    has_occlusion: bool = False
    labels: dict[str, str] = field(default_factory=dict)


def _sine_wander(center: float, t: float, *, amp: float, period_s: float) -> float:
    """Gentle physiological wander around a center value."""
    return center + amp * math.sin(2 * math.pi * (t % period_s) / period_s)


class NeonatalSim:
    """Deterministic synthetic neonatal data generator.

    Parameters are fixed-seeded by construction so every call with the same
    arguments yields byte-identical traces — a hard requirement for the
    golden suite.
    """

    def __init__(
        self,
        *,
        patient_id: str = "neo-001",
        scenario: Scenario = Scenario.healthy,
        start_ts: float = 0.0,
        step_s: float = 30.0,
    ) -> None:
        self.patient_id = patient_id
        self.scenario = Scenario(scenario)
        self.start_ts = float(start_ts)
        self.step_s = float(step_s)

    # ------------------------------------------------------------------
    def baseline(self, metric: str) -> float:
        """The healthy steady-state value for a metric (its 24h average)."""
        if metric not in _BASELINE_CENTER:
            raise KeyError(f"unknown neonatal metric: {metric!r}")
        return _BASELINE_CENTER[metric]

    # ------------------------------------------------------------------
    def sample_at(self, metric: str, i: int, *, n: int) -> Sample:
        """Build sample index `i` of an `n`-length series.

        Identical to ``series(metric, n=n)[i]`` without materializing the
        other samples — the sim is a pure function of ``(metric, i, n)``. The
        live publisher calls this once per tick; regenerating the full series
        there grows O(uptime²) on the Jetson.
        """
        if metric not in _BASELINE_CENTER:
            raise KeyError(f"unknown neonatal metric: {metric!r}")
        center = _BASELINE_CENTER[metric]
        t = self.start_ts + i * self.step_s
        value = _sine_wander(center, t, amp=center * 0.015, period_s=600.0)
        sample = Sample(
            ts=t,
            metric=metric,
            value=value,
            intended_ts=t,
            source="prometheus",
            reachable=True,
            has_data=True,
            labels={"patient": self.patient_id},
        )
        self._apply_scenario(sample, metric, i, n, center)
        return sample

    def series(self, metric: str, *, n: int) -> list[Sample]:
        """Generate `n` samples of `metric` from `start_ts`, step `step_s`.

        The shape is driven by `self.scenario`. For desaturation scenarios
        only the targeted rSO2 channel desaturates; other channels stay
        physiologic so a real evaluator wouldn't be confused by co-movement.
        """
        if metric not in _BASELINE_CENTER:
            raise KeyError(f"unknown neonatal metric: {metric!r}")
        return [self.sample_at(metric, i, n=n) for i in range(n)]

    # ------------------------------------------------------------------
    def _apply_scenario(
        self, s: Sample, metric: str, i: int, n: int, center: float
    ) -> None:
        sc = self.scenario

        if sc is Scenario.healthy:
            return

        if sc is Scenario.rso2_desat:
            # Sustained >20% relative drop on the rSO2 channels, starting
            # from the very first sample so a `for: 5m` window completes.
            if metric in ("rso2_left", "rso2_right"):
                # Drop to 75% of baseline -> below the 0.80 threshold.
                s.value = center * 0.75
            return

        if sc is Scenario.source_degraded:
            # First third real, then the silent mock fallback kicks in. The
            # mock samples are deliberately driven into DESATURATION territory
            # (below the 0.80x threshold) so the golden proves the integrity
            # GATE — not the numbers — is what blocks a clinical fire on fake
            # data. If the evaluator ever read mock values it would fire here.
            if i >= n // 3:
                s.source = "mock"
                if metric in ("rso2_left", "rso2_right"):
                    s.value = center * 0.70
            return

        if sc is Scenario.unreachable:
            # Prometheus stops answering partway through. The underlying value
            # is in breach territory so the gate (not the value) is what
            # blocks the fire.
            if i >= n // 3:
                s.reachable = False
                s.has_data = False
                if metric in ("rso2_left", "rso2_right"):
                    s.value = center * 0.70
            return

        if sc is Scenario.stale:
            # Data stops advancing: the freshness timestamp freezes after the
            # first third while the intended tick clock keeps moving, so the
            # staleness budget trips. The frozen sample is also in breach
            # territory so the gate is provably load-bearing.
            if i >= n // 3:
                frozen = self.start_ts + (n // 3) * self.step_s
                s.ts = frozen
                if metric in ("rso2_left", "rso2_right"):
                    s.value = center * 0.70
            return

        if sc is Scenario.short_history:
            # Caller asks for a long window but the generator only has a
            # little history; the evaluator decides INSUFFICIENT_BASELINE
            # from the coverage it is given (see anomaly_core).
            return

    # ------------------------------------------------------------------
    def ocr_quality(self, i: int, n: int) -> dict[str, bool]:
        """OCR-quality labels for sample index `i` (glare / occlusion)."""
        # Deterministic, sparse degradation so quality-aware logic has
        # something to react to without dominating the trace.
        return {
            "has_glare": (i % 17 == 0),
            "has_occlusion": (i % 23 == 0),
        }
