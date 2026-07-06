"""Synthetic neonatal rSO2 metrics PUBLISHER for the app's :8000 /metrics.

Increment 6 — "point to 8000".

Real medical data is NOT flowing yet, so we publish the deterministic
synthetic neonatal physiology from `NeonatalSim` onto the SAME
`prometheus_client` DEFAULT registry that the FastAPI app already exposes at
`/metrics` (via `Instrumentator().instrument(app).expose(app)` in
`main.py`). Prometheus already scrapes `http://localhost:8000/metrics`, so
once the backend is (separately, by the supervisor) restarted with the
publish gate ON, the rule's series resolve:

    rso2_left{patient="neo-001"}
    avg_over_time(rso2_left{patient="neo-001"}[24h])

This module is CODE ONLY. Importing it never starts anything; nothing runs
until `start()` is called, and `start()` is gated at the call site in
`main.py` behind the `ANOMALY_NEONATAL_SIM_PUBLISH` env var. `start()` /
`stop()` are idempotent.

Safety posture (unchanged by this module):
  - The evaluator still refuses `source=="mock"`; this publisher only ever
    emits real (synthetic) `prometheus`-sourced series, never mock.
  - Rules stay SHADOW; publishing metrics does not page anyone.
  - No secrets are handled here, so nothing to mask.

Registry robustness: gauges are registered lazily and idempotently. If a
gauge name already exists on the target registry (e.g. a hot reload, or a
re-import against the DEFAULT registry), we reuse the existing collector
instead of raising `Duplicated timeseries in CollectorRegistry`.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from prometheus_client import REGISTRY, CollectorRegistry, Gauge

from .neonatal_sim import NeonatalSim, Scenario


# Metrics we publish per patient. These are the physiological channels the
# synthetic sim produces; `rso2_left` is the one the default rule watches.
_PHYS_METRICS: tuple[str, ...] = (
    "rso2_left",
    "rso2_right",
    "spo2",
    "fio2",
    "hr",
    "map",
)

# Exported gauge names. We keep the metric name verbatim for the physiology
# channels (so PromQL like `rso2_left{patient="neo-001"}` resolves exactly),
# and add explicit-suffix gauges for freshness + OCR quality.
_HEARTBEAT_METRIC = "neonatal_sim_last_update_timestamp_seconds"
_GLARE_METRIC = "neonatal_sim_has_glare"
_OCCLUSION_METRIC = "neonatal_sim_has_occlusion"

_PATIENT_ID = "neo-001"

# How often the real-wall-clock updater refreshes the gauges.
_DEFAULT_INTERVAL_S = 5.0


def _get_or_create_gauge(
    registry: CollectorRegistry,
    name: str,
    documentation: str,
    labelnames: tuple[str, ...],
) -> Gauge:
    """Register a Gauge idempotently on `registry`.

    prometheus_client raises `ValueError: Duplicated timeseries ...` if the
    same metric name is registered twice on a registry. Since this module may
    be imported/started more than once against the process-wide DEFAULT
    registry (hot reload, repeated test runs in one process), we look up an
    existing collector by name and reuse it instead of raising.
    """
    existing = getattr(registry, "_names_to_collectors", {}).get(name)
    if existing is not None:
        return existing  # type: ignore[return-value]
    try:
        return Gauge(name, documentation, labelnames, registry=registry)
    except ValueError:
        # Lost a race (or a non-standard registry without the private map):
        # fall back to whatever is now registered under this name.
        existing = getattr(registry, "_names_to_collectors", {}).get(name)
        if existing is not None:
            return existing  # type: ignore[return-value]
        raise


class NeonatalPublisher:
    """Publishes synthetic neonatal series onto a prometheus_client registry.

    Deliberately decoupled from any HTTP server: the app's existing
    Instrumentator `/metrics` endpoint on :8000 scrapes the DEFAULT registry,
    so publishing onto that registry is all we need. Passing an explicit
    `registry` keeps it unit-testable in isolation.
    """

    def __init__(
        self,
        *,
        registry: CollectorRegistry | None = None,
        scenario: Scenario = Scenario.healthy,
        patient_id: str = _PATIENT_ID,
        interval_s: float = _DEFAULT_INTERVAL_S,
        step_s: float | None = None,
    ) -> None:
        self.registry = registry if registry is not None else REGISTRY
        self.scenario = Scenario(scenario)
        self.patient_id = patient_id
        self.interval_s = float(interval_s)
        # The sim steps in its own synthetic clock; drive it at the publish
        # interval so successive ticks walk the physiological wander/scenario.
        self.step_s = float(step_s if step_s is not None else interval_s)

        self._sim = NeonatalSim(
            patient_id=patient_id,
            scenario=self.scenario,
            start_ts=0.0,
            step_s=self.step_s,
        )

        # Register gauges up front (idempotently) so /metrics exposes the
        # series even before the first tick has run.
        self._phys_gauges: dict[str, Gauge] = {
            metric: _get_or_create_gauge(
                self.registry,
                metric,
                f"Synthetic neonatal {metric} (decision-support, non-diagnostic)",
                ("patient",),
            )
            for metric in _PHYS_METRICS
        }
        self._heartbeat = _get_or_create_gauge(
            self.registry,
            _HEARTBEAT_METRIC,
            "Unix timestamp of the last synthetic neonatal sim update (freshness)",
            ("patient",),
        )
        self._glare = _get_or_create_gauge(
            self.registry,
            _GLARE_METRIC,
            "OCR quality: glare present on the synthetic monitor image (0/1)",
            ("patient",),
        )
        self._occlusion = _get_or_create_gauge(
            self.registry,
            _OCCLUSION_METRIC,
            "OCR quality: occlusion present on the synthetic monitor image (0/1)",
            ("patient",),
        )

        self._tick_index = 0
        self._n_hint = 2  # arbitrary window for ocr_quality's (i, n) signature
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def update_once(self) -> dict[str, float]:
        """Pull one sim sample per metric and set the gauges. Returns the
        physiological values written (for tests / audit)."""
        written: dict[str, float] = {}
        # Generate a fresh single-index series for the current tick so the
        # scenario shape (e.g. rso2_desat) applies deterministically.
        idx = self._tick_index
        n = max(idx + 1, self._n_hint)
        for metric, gauge in self._phys_gauges.items():
            # sample_at builds only this tick's sample — regenerating the
            # whole series each tick is O(uptime²) on a long-running daemon.
            sample = self._sim.sample_at(metric, idx, n=n)
            gauge.labels(patient=self.patient_id).set(sample.value)
            written[metric] = sample.value

        # Freshness / heartbeat = real wall-clock now, so the evaluator's
        # staleness gate sees genuinely fresh data on a live scrape.
        self._heartbeat.labels(patient=self.patient_id).set(time.time())

        quality = self._sim.ocr_quality(idx, n)
        self._glare.labels(patient=self.patient_id).set(
            1.0 if quality["has_glare"] else 0.0
        )
        self._occlusion.labels(patient=self.patient_id).set(
            1.0 if quality["has_occlusion"] else 0.0
        )

        self._tick_index += 1
        return written

    # ------------------------------------------------------------------
    def _run(self) -> None:  # pragma: no cover - timing thread
        # Prime immediately so /metrics has data on the first scrape.
        try:
            self.update_once()
        except Exception:
            pass
        while not self._stop_event.wait(self.interval_s):
            try:
                self.update_once()
            except Exception:
                # Never let a single bad tick kill the publisher thread; the
                # next tick retries. Publishing is non-load-bearing anyway.
                pass

    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Start the background updater. Idempotent: a second call is a no-op
        and returns False. Returns True if it started a thread this call."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="neonatal-sim-publisher",
                daemon=True,
            )
            self._thread.start()
            return True

    # ------------------------------------------------------------------
    def stop(self, *, timeout: float = 2.0) -> None:
        """Stop the background updater. Idempotent and safe if never started."""
        with self._lock:
            thread = self._thread
            self._stop_event.set()
        if thread is not None:
            thread.join(timeout=timeout)
        with self._lock:
            self._thread = None


# ---------------------------------------------------------------------------
# Process-wide singleton helpers used by main.py's lifespan.
# ---------------------------------------------------------------------------
_PUBLISHER: NeonatalPublisher | None = None
_PUBLISHER_LOCK = threading.Lock()


def _scenario_from_env() -> Scenario:
    raw = os.getenv("ANOMALY_NEONATAL_SIM_SCENARIO", "healthy").strip()
    try:
        return Scenario(raw)
    except ValueError:
        # Unknown scenario name: default to healthy rather than crash the app.
        return Scenario.healthy


def _interval_from_env() -> float:
    try:
        return max(1.0, float(os.getenv("ANOMALY_NEONATAL_SIM_INTERVAL_S", "5")))
    except ValueError:
        return _DEFAULT_INTERVAL_S


def start_publisher(
    *, registry: CollectorRegistry | None = None
) -> NeonatalPublisher:
    """Idempotently create + start the process-wide publisher singleton.

    Reads the scenario + interval from env at start time. Safe to call more
    than once: subsequent calls return the existing singleton without spawning
    a second thread.
    """
    global _PUBLISHER
    with _PUBLISHER_LOCK:
        if _PUBLISHER is None:
            _PUBLISHER = NeonatalPublisher(
                registry=registry,
                scenario=_scenario_from_env(),
                interval_s=_interval_from_env(),
            )
        _PUBLISHER.start()
        return _PUBLISHER


def stop_publisher() -> None:
    """Idempotently stop the process-wide publisher singleton if running."""
    global _PUBLISHER
    with _PUBLISHER_LOCK:
        pub = _PUBLISHER
    if pub is not None:
        pub.stop()


def get_publisher() -> NeonatalPublisher | None:
    """Return the current singleton (or None). For inspection/tests."""
    return _PUBLISHER


def publisher_summary() -> dict[str, Any]:
    """Non-secret summary of publisher config for audit/inspection."""
    return {
        "gated_env_var": "ANOMALY_NEONATAL_SIM_PUBLISH",
        "scenario_env_var": "ANOMALY_NEONATAL_SIM_SCENARIO",
        "scenario": _scenario_from_env().value,
        "interval_s": _interval_from_env(),
        "patient": _PATIENT_ID,
        "phys_metrics": list(_PHYS_METRICS),
        "heartbeat_metric": _HEARTBEAT_METRIC,
        "quality_metrics": [_GLARE_METRIC, _OCCLUSION_METRIC],
    }
