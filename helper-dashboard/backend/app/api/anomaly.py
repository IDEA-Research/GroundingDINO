"""HTTP surface for the anomaly subsystem: alert reads + a demo driver.

Two deliberately-thin concerns live here:

1. ``GET /api/anomaly/alerts`` — the READ endpoint the ``decision_flow`` widget
   (``frontend/widget-toolkit/DecisionFlowWidget.tsx``) fetches. It projects the
   durable :class:`AlertStateStore` append-only history into the ``AlertEventDTO``
   shape the frontend expects. It is fail-LOUD: if no store is wired, or the
   store failed to load, it returns ``store_unavailable=true`` so the widget
   parks on its loud "signal unknown / lost" state rather than silently
   implying "normal".

2. ``POST /api/anomaly/demo/*`` — a DEMO-ONLY driver, mounted only when
   ``ANOMALY_DEMO=1``. It drives the REAL evaluator / store / notifier pipeline
   with an ACCELERATED INJECTED CLOCK (exactly how the golden harness steps it),
   so the locked ``for: 5m`` sustain window completes in ~10 instant ticks
   instead of five wall-clock minutes. It invents NO clinical logic — it only
   selects a synthetic scenario and advances the clock. Every Discord delivery
   still carries the ``[TEST TUNNEL · NON-DIAGNOSTIC]`` banner the notifier
   enforces.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.alert_state_store import AlertStateStore

router = APIRouter()


# ---------------------------------------------------------------------------
# READ: the alert history the decision_flow widget consumes.
# ---------------------------------------------------------------------------
def _read_store(request: Request) -> AlertStateStore | None:
    # Prefer the demo driver's live store (it is rebuilt per scenario, so the
    # app.state ref can otherwise go stale), else the background loop's store.
    demo = getattr(request.app.state, "anomaly_demo", None)
    if demo is not None:
        return demo.store
    return getattr(request.app.state, "anomaly_store", None)


@router.get("/alerts")
@router.get("/alerts/")
def get_alerts(request: Request) -> dict[str, Any]:
    store = _read_store(request)
    if store is None or getattr(store, "load_error", None):
        # Fail loud: never a fake "no events => normal".
        return {"events": [], "store_unavailable": True}
    try:
        events = store.history_records()
    except Exception:  # noqa: BLE001 — degrade loudly, do not 500 the widget
        return {"events": [], "store_unavailable": True}
    return {"events": events, "store_unavailable": False}


@router.get("/status")
def get_status(request: Request) -> dict[str, Any]:
    svc = getattr(request.app.state, "anomaly_service", None)
    if svc is None:
        raise HTTPException(status_code=404, detail="no evaluator running")
    return svc.status()


# ---------------------------------------------------------------------------
# DEMO DRIVER (mounted only when ANOMALY_DEMO=1).
# ---------------------------------------------------------------------------
class _LoadBody(BaseModel):
    scenario: str = "healthy"
    ticks: int = Field(default=0, ge=0, le=400)


class _AdvanceBody(BaseModel):
    ticks: int = Field(default=1, ge=1, le=400)


# Scenario -> synthetic trace length. Degradation scenarios use a SHORT trace so
# the degraded regime (source=="mock" / unreachable) is reached on the first
# tick; healthy/desat use a long trace so the injected clock never runs off the
# end of the series (which would false-trip the staleness gate).
_SCENARIO_NTICKS: dict[str, int] = {
    "healthy": 240,
    "rso2_desat": 240,
    "source_degraded": 2,
    "unreachable": 2,
    "stale": 240,
    "short_history": 240,
}


class DemoDriver:
    """Drives the real pipeline with an accelerated injected clock.

    NOT production code — mounted only under ``ANOMALY_DEMO=1``. It rebuilds a
    fresh store + service per scenario (so each demo act starts clean) and steps
    ``service.tick(now)`` with a clock advancing ``STEP_S`` per call, so the 5m
    sustain window fires in ~10 instant ticks. All clinical logic stays in the
    evaluator core; this only picks a synthetic scenario and moves the clock.
    """

    STEP_S = 30.0

    def __init__(self, app: Any, base_dir: str | os.PathLike[str]) -> None:
        self._app = app
        self.base_dir = Path(base_dir)
        self._notifier: Any = None
        self.load_scenario("healthy")

    # ------------------------------------------------------------------
    def load_scenario(self, scenario: str) -> None:
        from ..prometheus.neonatal_sim import Scenario
        from ..services.anomaly_data_provider import SimDataProvider
        from ..services.anomaly_evaluator_service import AnomalyEvaluatorService
        from ..services.anomaly_notification_dispatch import NotificationDispatcher
        from ..services.anomaly_notifier import build_default_notifier
        from ..tests_support.default_rules import promoted_neonatal_rso2_rule

        try:
            sc = Scenario(scenario)  # validates; raises ValueError on unknown
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        n = _SCENARIO_NTICKS.get(sc.value, 240)

        # Fresh store per act: wipe the DEMO state dir so "latest event" is
        # unambiguous. This is the demo dir only — never a real clinical store:
        # refuse outright if our dir resolves to the durable AlertStateStore
        # location (deleting state.json + the append-only history audit is a
        # clinical-data loss, fail loud instead).
        from ..services.alert_state_store import _DEFAULT_DIR as _REAL_STORE_DEFAULT

        real_dir = Path(
            os.getenv("ANOMALY_ALERT_STATE_DIR", str(_REAL_STORE_DEFAULT))
        ).resolve()
        if self.base_dir.resolve() == real_dir:
            raise RuntimeError(
                "DemoDriver refuses to wipe the REAL alert-state dir "
                f"({real_dir}): set ANOMALY_DEMO_STATE_DIR to a dedicated "
                "demo directory"
            )
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir, ignore_errors=True)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        store = AlertStateStore(base_dir=self.base_dir)
        provider = SimDataProvider(
            scenario=sc, start_ts=0.0, step_s=self.STEP_S, n_ticks=n
        )
        rule = promoted_neonatal_rso2_rule(metric="rso2_left")
        # Reuse one notifier across acts (its Discord target is fixed); a fresh
        # dispatcher per act re-arms page-once so every act delivers.
        if self._notifier is None:
            self._notifier = build_default_notifier()
        dispatcher = NotificationDispatcher(self._notifier, increment="demo")
        self.service = AnomalyEvaluatorService(
            [rule],
            provider,
            store=store,
            promoted=True,
            dispatcher=dispatcher,
            staleness_budget_s=60.0,
            # Audit tag: demo-driver records must never masquerade as a build
            # increment or as the live runtime loop.
            increment="demo",
        )
        self.store = store
        self.provider = provider
        self.rule = rule
        self.scenario = sc.value
        self.clock = 0.0
        self._last_dispatch: dict[str, Any] = {}

        # Keep app.state pointing at the CURRENT store/service so the read
        # endpoints never serve a stale (pre-scenario) store.
        self._app.state.anomaly_store = store
        self._app.state.anomaly_service = self.service

    # ------------------------------------------------------------------
    def advance(self, ticks: int) -> None:
        for _ in range(int(ticks)):
            self.clock += self.STEP_S
            res = self.service.tick(self.clock)
            self._last_dispatch = res.dispatch

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        rid = self.rule.id
        st = self.store.get(rid)
        delivered = 0
        would_fire = 0
        for outcome in self._last_dispatch.values():
            delivered += len(outcome.delivered)
            would_fire += len(outcome.would_fire_only)
        return {
            "scenario": self.scenario,
            "clock": self.clock,
            "promoted": True,
            "channel": self._notifier.masked_target() if self._notifier else None,
            "rule_id": rid,
            "state": st.state,
            "last_value": st.last_value,
            "last_baseline": st.last_baseline,
            "last_threshold": st.last_threshold,
            "paged_count": st.paged_count,
            "would_page_count": st.would_page_count,
            "last_tick_dispatch": {
                "delivered": delivered,
                "would_fire_only": would_fire,
            },
            "event_count": len(self.store.history_records()),
            "non_diagnostic": True,
        }


def _require_demo(request: Request) -> DemoDriver:
    demo = getattr(request.app.state, "anomaly_demo", None)
    if demo is None:
        raise HTTPException(
            status_code=409, detail="demo driver not enabled (set ANOMALY_DEMO=1)"
        )
    return demo


@router.post("/demo/load")
def demo_load(body: _LoadBody, request: Request) -> dict[str, Any]:
    demo = _require_demo(request)
    demo.load_scenario(body.scenario)
    if body.ticks:
        demo.advance(body.ticks)
    return demo.snapshot()


@router.post("/demo/advance")
def demo_advance(body: _AdvanceBody, request: Request) -> dict[str, Any]:
    demo = _require_demo(request)
    demo.advance(body.ticks)
    return demo.snapshot()


@router.get("/demo/state")
def demo_state(request: Request) -> dict[str, Any]:
    demo = _require_demo(request)
    return demo.snapshot()
