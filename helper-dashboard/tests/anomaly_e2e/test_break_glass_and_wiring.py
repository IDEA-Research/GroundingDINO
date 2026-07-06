"""INC5 end-to-end suite: break-glass + full wiring + audit completeness.

This is the final-wiring proof. It exercises, with an INJECTED CLOCK (no
wall-clock sleeps — pytest_asyncio is absent):

  A. FULL WIRING: the neonatal rSO2 rule (value < 0.80*avg_over_time[24h], for
     5m, neonatal) flowing evaluator -> alert-state store -> notifier (TEST
     TUNNEL local file) -> decision_flow widget projection, with the
     AnomalyEvaluationReport feeding the decide evidence.

  B. BREAK-GLASS: stop -> (tick refused, never a fake no-anomaly) ->
     force-signal-lost -> recover (restore last-known-good) + force-shadow.

  C. AUDIT COMPLETENESS: rule activation + every state transition
     (pending/firing/resolved/signal_lost) + delivery + ack + suppressed are
     all captured; a failed/never-acked delivery attempt is recorded as
     DELIVERY_FAILED (a channel failure, loud but attributable); a MISSED
     must-fire (a promoted page with NO delivery accounting at all) is
     recorded as the top-severity clinical event and fails closed.

A missed must-fire is the top-severity failure and blocks the increment.
"""

from __future__ import annotations

import pytest

from app.prometheus.neonatal_sim import Scenario
from app.services.alert_state_store import AlertStateStore
from app.services.anomaly_break_glass import BreakGlassController, BreakGlassError
from app.services.anomaly_decision_flow import (
    build_rso2_decision_flow_widget,
    project_report,
)
from app.services.anomaly_data_provider import SimDataProvider
from app.services.anomaly_evaluator_service import AnomalyEvaluatorService
from app.services.anomaly_notification_dispatch import NotificationDispatcher
from app.services.anomaly_notifier import (
    DeliveryReceipt,
    LocalFileReceiver,
    Notifier,
    build_default_notifier,
)
from app.services import anomaly_lifecycle_audit as lifecycle
from app.specs.anomaly_evaluation_report import AlertState, SignalLostReason
from app.tests_support.default_rules import (
    neonatal_rso2_rule,
    promoted_neonatal_rso2_rule,
)

STEP_S = 30.0
FOR_S = 300.0
N_TICKS = 40


def _sim(scenario, **kw):
    return SimDataProvider(
        scenario=scenario, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS, **kw
    )


def _run(ticker, n, *, start=0.0, step=STEP_S):
    out = []
    for i in range(n):
        now = start + i * step
        out.append(ticker.tick(now, mono=now))
    return out


def _kinds(records):
    return [r["kind"] for r in records]


# ---------------------------------------------------------------------------
# A. FULL WIRING: evaluator -> store -> notifier -> decision_flow widget
# ---------------------------------------------------------------------------
def test_full_wiring_shadow_desat_to_decision_flow(tmp_path):
    """A sustained desaturation flows all the way to the widget projection.

    In SHADOW the firing would-page but never pages; the decision_flow widget
    parks on the escalation node and carries the non-diagnostic banner.
    """
    rule = neonatal_rso2_rule(metric="rso2_left")
    store = AlertStateStore(base_dir=tmp_path / "state")
    receiver = LocalFileReceiver(path=str(tmp_path / "notif" / "out.jsonl"))
    dispatcher = NotificationDispatcher(receiver, increment="INC5")
    service = AnomalyEvaluatorService(
        [rule],
        _sim(Scenario.rso2_desat),
        store=store,
        dispatcher=dispatcher,
        monotonic=lambda: 0.0,
        promoted=False,
        increment="INC5",
    )

    results = _run(service, 15)  # long enough to cross the 5m for-window

    # --- clinical verdict: it MUST fire (missed must-fire is top severity) ---
    firing_ticks = [
        r.now for r in results if r.reports[rule.id].state == AlertState.firing
    ]
    assert firing_ticks, "MUST-FIRE MISSED: sustained desaturation never fired"
    # First fire is exactly 5m into a breach that started at t=0.
    assert firing_ticks[0] == pytest.approx(FOR_S)

    # --- notifier: shadow never pages the TEST TUNNEL ---
    assert receiver.messages == [], "SHADOW rule delivered a page — invariant violated"

    # --- decision_flow widget: valid, and the projection reflects the fire ---
    widget = build_rso2_decision_flow_widget(rso2_metric="rso2_left")
    assert widget.type.value == "decision_flow"
    # first node is the data-integrity gate (structural safety invariant)
    assert widget.decision_flow.nodes[0].kind.value == "data_integrity_gate"

    firing_report = next(
        r.reports[rule.id]
        for r in results
        if r.reports[rule.id].state == AlertState.firing
    )
    runtime = project_report(firing_report, widget_id=widget.id)
    assert runtime.active_node == "act"  # escalation branch
    assert runtime.would_page is True  # firing edge would page
    assert runtime.paged is False  # but shadow does not
    assert runtime.non_diagnostic is True
    assert "not a diagnosis" in runtime.banner.lower()

    # --- store reflects the durable firing lifecycle ---
    st = store.get(rule.id)
    assert st.would_page_count >= 1
    assert st.paged_count == 0


def test_signal_lost_projects_loud_degraded_flow(tmp_path):
    """A source degradation projects a LOUD signal-lost flow, never a verdict."""
    rule = neonatal_rso2_rule(metric="rso2_left")
    store = AlertStateStore(base_dir=tmp_path / "state")
    service = AnomalyEvaluatorService(
        [rule], _sim(Scenario.source_degraded), store=store, monotonic=lambda: 0.0
    )
    results = _run(service, N_TICKS)

    lost_report = next(
        (
            r.reports[rule.id]
            for r in results
            if r.reports[rule.id].state == AlertState.signal_lost
        ),
        None,
    )
    assert lost_report is not None, "source degradation did not SIGNAL_LOST"
    assert not any(
        r.reports[rule.id].state == AlertState.firing for r in results
    ), "FAIL-OPEN: fired on mock data — gate did not run first (top-severity)"

    runtime = project_report(lost_report)
    assert runtime.signal_lost is True
    assert runtime.active_node == "lost"  # loud terminal, never 'ok'
    assert runtime.signal_lost_reason == SignalLostReason.mock_source.value
    assert "signal lost" in runtime.banner.lower()


# ---------------------------------------------------------------------------
# B. BREAK-GLASS: stop -> force-signal-lost -> recover
# ---------------------------------------------------------------------------
def test_break_glass_stop_force_signal_lost_recover(tmp_path):
    rule = neonatal_rso2_rule(metric="rso2_left")
    state_dir = tmp_path / "shared"
    provider = _sim(Scenario.healthy)
    service = AnomalyEvaluatorService(
        [rule],
        provider,
        store=AlertStateStore(base_dir=state_dir),
        monotonic=lambda: 0.0,
        increment="INC5",
    )
    bg = BreakGlassController(service, provider, increment="INC5")

    # Run a few healthy ticks, then checkpoint a known-good state.
    _run(bg, 4)
    healthy_state = service.store.get(rule.id).state
    bg.backup(by="charge_nurse")

    # --- STOP: a stopped loop must be LOUD, never a silent 'no anomaly'. ---
    bg.stop(by="charge_nurse")
    assert bg.status()["enabled"] is False
    with pytest.raises(BreakGlassError):
        bg.tick(999.0, mono=999.0)

    # --- FORCE SIGNAL_LOST: conspicuous manual 'data untrustworthy'. ---
    bg.start(by="charge_nurse")
    bg.force_signal_lost(by="charge_nurse")
    forced = bg.tick(200.0, mono=200.0)
    assert forced.reports[rule.id].state == AlertState.signal_lost, (
        "force_signal_lost did not produce a conspicuous SIGNAL_LOST"
    )
    runtime = project_report(forced.reports[rule.id])
    assert runtime.signal_lost is True

    # --- RECOVER: restore last-known-good and return to automatic. ---
    restored = bg.recover(by="charge_nurse")
    assert rule.id in (restored.get("rules") or {}), "LKG did not include the rule"
    assert bg.status()["enabled"] is True
    assert bg.status()["force_signal_lost"] is False
    # The restored durable state matches the checkpointed healthy state.
    assert service.store.get(rule.id).state == healthy_state

    # After recovery the loop ticks normally again (healthy scenario).
    post = bg.tick(300.0, mono=300.0)
    assert post.reports[rule.id].state in (
        AlertState.resolved,
        AlertState.pending,
    )

    # --- COMPLETE AUDIT TRAIL for the break-glass sequence. ---
    from app.services import anomaly_build_audit as ba
    import json
    from datetime import datetime, timezone

    day = f"{datetime.now(tz=timezone.utc):%Y-%m-%d}"
    audit_path = ba._audit_dir() / f"{day}.jsonl"
    lines = [json.loads(x) for x in audit_path.read_text().splitlines() if x.strip()]
    outcomes = [x["outcome"] for x in lines]
    for expected in (
        "backed_up",
        "stopped",
        "started",
        "forced_signal_lost",
        "recovered",
    ):
        assert expected in outcomes, f"break-glass action {expected!r} not audited"
    # Every break-glass line records the operator identity.
    bg_lines = [x for x in lines if x["target"].startswith("break_glass/")]
    assert bg_lines, "no break-glass audit lines"
    assert all("operator=charge_nurse" in x["summary"] for x in bg_lines)


def test_break_glass_recover_without_lkg_fails_loud(tmp_path):
    """Recover with no last-known-good must FAIL LOUD, never blank the state."""
    rule = neonatal_rso2_rule(metric="rso2_left")
    provider = _sim(Scenario.healthy)
    service = AnomalyEvaluatorService(
        [rule],
        provider,
        store=AlertStateStore(base_dir=tmp_path / "state"),
        monotonic=lambda: 0.0,
    )
    bg = BreakGlassController(service, provider, increment="INC5")
    # Delete the LKG the controller captured at init to simulate 'no baseline'.
    lkg = service.store._lkg_path  # noqa: SLF001
    if lkg.exists():
        lkg.unlink()
    with pytest.raises(Exception):
        bg.recover(by="operator")


def test_break_glass_force_shadow_kills_paging(tmp_path):
    """force_shadow drops a PROMOTED rule to non-paging — loudly, recorded."""
    rule = promoted_neonatal_rso2_rule(metric="rso2_left")
    provider = _sim(Scenario.rso2_desat)
    receiver = LocalFileReceiver(path=str(tmp_path / "notif" / "out.jsonl"))
    dispatcher = NotificationDispatcher(receiver, increment="INC5")
    service = AnomalyEvaluatorService(
        [rule],
        provider,
        store=AlertStateStore(base_dir=tmp_path / "state"),
        dispatcher=dispatcher,
        monotonic=lambda: 0.0,
        promoted=True,  # this rule is promoted -> would page
        increment="INC5",
    )
    bg = BreakGlassController(service, provider, increment="INC5")

    # Kill paging BEFORE the firing edge.
    bg.force_shadow(rule.id, by="safety_officer")
    _run(bg, 15)  # crosses 5m

    # Even though the desaturation fires, paging was killed: nothing delivered.
    assert receiver.messages == [], (
        "force_shadow failed to kill paging — a promoted rule still paged"
    )
    st = service.store.get(rule.id)
    assert st.would_page_count >= 1  # still recorded (visible), never silent
    assert st.paged_count == 0

    recs = lifecycle.read_records()
    assert any(r["kind"] == "rule_shadowed" for r in recs), (
        "force_shadow was not recorded in the lifecycle audit"
    )


# ---------------------------------------------------------------------------
# C. AUDIT COMPLETENESS
# ---------------------------------------------------------------------------
def test_audit_captures_activation_transitions_delivery_ack(tmp_path):
    """A PROMOTED rule's full lifecycle is captured: activation, transitions,
    delivery to the test tunnel, and a human ack."""
    rule = promoted_neonatal_rso2_rule(metric="rso2_left")
    receiver = LocalFileReceiver(path=str(tmp_path / "notif" / "out.jsonl"))
    dispatcher = NotificationDispatcher(receiver, increment="INC5")
    service = AnomalyEvaluatorService(
        [rule],
        _sim(Scenario.rso2_desat),
        store=AlertStateStore(base_dir=tmp_path / "state"),
        dispatcher=dispatcher,
        monotonic=lambda: 0.0,
        promoted=True,
        increment="INC5",
    )
    _run(service, 15)

    # A promoted rule that fired MUST have paged the TEST TUNNEL exactly once.
    assert len(receiver.messages) == 1, "promoted firing did not page once"
    assert "[TEST TUNNEL · NON-DIAGNOSTIC]" in receiver.messages[0]["text"]

    # Human ack via the service wrapper (durable + lifecycle audit).
    service.acknowledge(rule.id, by="dr_smith")
    assert service.store.get(rule.id).acked is True

    recs = lifecycle.read_records()
    kinds = set(_kinds(recs))
    for k in ("rule_activated", "transition", "delivered", "acked"):
        assert k in kinds, f"lifecycle audit missing {k!r}"

    # The delivery record is masked (never the raw token/URL) and non-diagnostic.
    deliv = [r for r in recs if r["kind"] == "delivered"]
    assert deliv, "no delivery lifecycle record"
    assert deliv[0]["payload"]["acked"] is True
    assert deliv[0]["non_diagnostic"] is True
    # Every transition state observed appears in the trail.
    trans_states = {r["payload"]["state"] for r in recs if r["kind"] == "transition"}
    assert "firing" in trans_states
    assert "pending" in trans_states


def test_channel_failure_is_delivery_failed_not_missed(tmp_path):
    """A delivery ATTEMPT that fails/never acks is DELIVERY_FAILED, not MISSED.

    We wire a notifier that reports every delivery as un-acked (a channel that
    silently drops). The dispatcher records the attempt; the service must
    surface the channel failure loudly — but as a delivery_failed record with
    the error attached, NEVER as the worst-case clinical MISSED signal (a
    webhook outage must not masquerade as the pipeline losing a page).
    """

    class DroppingNotifier(Notifier):
        channel_name = "dropping_test"

        def deliver(self, message):
            # Delivery 'attempted' but never acks — surfaced, not swallowed.
            return DeliveryReceipt(
                dedup_key=message.dedup_key,
                channel=self.channel_name,
                delivered=False,
                acked=False,
                attempts=3,
                error="channel never acked",
            )

    rule = promoted_neonatal_rso2_rule(metric="rso2_left")
    dispatcher = NotificationDispatcher(DroppingNotifier(), increment="INC5")
    service = AnomalyEvaluatorService(
        [rule],
        _sim(Scenario.rso2_desat),
        store=AlertStateStore(base_dir=tmp_path / "state"),
        dispatcher=dispatcher,
        monotonic=lambda: 0.0,
        promoted=True,
        increment="INC5",
    )
    _run(service, 15)

    recs = lifecycle.read_records()
    failed = [r for r in recs if r["kind"] == "delivery_failed"]
    assert failed, (
        "a promoted firing whose page never acked was NOT recorded as a "
        "delivery_failed channel failure"
    )
    assert failed[0]["payload"]["state"] == "firing"
    assert failed[0]["payload"]["severity"] == "critical"
    assert failed[0]["payload"]["acked"] is False
    assert failed[0]["payload"]["error"] == "channel never acked"
    # The failure is attributable to the channel — it must NOT be conflated
    # with the top-severity "pipeline lost a page" signal.
    assert not [r for r in recs if r["kind"] == "missed"], (
        "a channel delivery failure was recorded as a clinical MISSED "
        "must-fire — audit-honesty regression (A_DIAGNOSIS E6b)"
    )


def test_activation_reasoning_reflects_actual_mode(tmp_path):
    """E6(a): an ACTIVE activation must never carry canned shadow reasoning.

    An audit reader who trusts the record verbatim must not conclude a paging
    rule is in shadow — the reasoning is derived from the actual mode.
    """
    AnomalyEvaluatorService(
        [neonatal_rso2_rule(metric="rso2_left")],
        _sim(Scenario.healthy),
        store=AlertStateStore(base_dir=tmp_path / "s1"),
        monotonic=lambda: 0.0,
        promoted=False,
        increment="INC5",
    )
    AnomalyEvaluatorService(
        [promoted_neonatal_rso2_rule(metric="rso2_left")],
        _sim(Scenario.healthy),
        store=AlertStateStore(base_dir=tmp_path / "s2"),
        monotonic=lambda: 0.0,
        promoted=True,
        increment="INC5",
    )

    activations = [
        r for r in lifecycle.read_records() if r["kind"] == "rule_activated"
    ]
    assert len(activations) == 2
    shadow_rec = next(r for r in activations if "SHADOW" in r["summary"])
    active_rec = next(r for r in activations if "ACTIVE" in r["summary"])
    assert "shadow by default" in shadow_rec["reasoning"]
    # The paging activation must say what actually happened, not the shadow
    # boilerplate (A_DIAGNOSIS E6a).
    assert "shadow by default" not in active_rec["reasoning"]
    assert "PROMOTED" in active_rec["reasoning"]
    assert active_rec["payload"]["promoted"] is True


def test_unaccounted_page_is_recorded_as_missed_top_severity(tmp_path):
    """A paged event with NO delivery accounting at all is a MISSED must-fire.

    We wire a notifier whose receipt comes back keyed to the WRONG dedup key,
    so the reconciliation finds no receipt for the paged firing — the pipeline
    lost track of a page. That is the top-severity clinical event.
    """

    class MisaccountingNotifier(Notifier):
        channel_name = "misaccounting_test"

        def deliver(self, message):
            return DeliveryReceipt(
                dedup_key="wrong-key|not-the-event",
                channel=self.channel_name,
                delivered=True,
                acked=True,
                attempts=1,
                error=None,
            )

    rule = promoted_neonatal_rso2_rule(metric="rso2_left")
    dispatcher = NotificationDispatcher(MisaccountingNotifier(), increment="INC5")
    service = AnomalyEvaluatorService(
        [rule],
        _sim(Scenario.rso2_desat),
        store=AlertStateStore(base_dir=tmp_path / "state"),
        dispatcher=dispatcher,
        monotonic=lambda: 0.0,
        promoted=True,
        increment="INC5",
    )
    _run(service, 15)

    recs = lifecycle.read_records()
    missed = [r for r in recs if r["kind"] == "missed"]
    assert missed, (
        "TOP-SEVERITY: a promoted firing with no delivery accounting was NOT "
        "recorded as a MISSED must-fire"
    )
    assert missed[0]["payload"]["state"] == "firing"
    assert missed[0]["payload"]["severity"] == "critical"


def test_default_notifier_falls_back_to_local_file_when_no_webhook(tmp_path):
    """No ANOMALY_TEST_WEBHOOK_URL -> local file receiver, loop still runs."""
    notifier = build_default_notifier(local_path=str(tmp_path / "n" / "out.jsonl"))
    assert notifier.channel_name == "local_file"
    # masked_target never leaks a token/url.
    assert "http" not in notifier.masked_target().lower()


def test_demo_driver_refuses_to_wipe_real_alert_state_dir(tmp_path, monkeypatch):
    """The demo's per-scenario wipe must never touch the REAL clinical store.

    An env-var collision (demo dir resolving to ANOMALY_ALERT_STATE_DIR)
    previously rmtree'd state.json + the append-only history audit on app
    startup. The driver now refuses outright.
    """
    from types import SimpleNamespace

    from app.api.anomaly import DemoDriver

    real_dir = tmp_path / "real_state"
    real_dir.mkdir(parents=True)
    (real_dir / "state.json").write_text("{}")
    (real_dir / "history.jsonl").write_text("")
    monkeypatch.setenv("ANOMALY_ALERT_STATE_DIR", str(real_dir))

    app = SimpleNamespace(state=SimpleNamespace())
    with pytest.raises(RuntimeError, match="refuses to wipe"):
        DemoDriver(app, base_dir=str(real_dir))

    # The real store survived untouched.
    assert (real_dir / "state.json").exists()
    assert (real_dir / "history.jsonl").exists()


def test_persist_failure_at_firing_edge_does_not_swallow_the_page(tmp_path):
    """E11: a one-tick store failure at the firing edge must not lose the page.

    evaluate() consumes the firing edge before persist/dispatch run; on a
    persist failure the service rolls the edge back so the next tick
    re-emits the firing event and the page is re-attempted (at-least-once:
    a duplicate page is acceptable, a lost page is not).
    """

    class FailingOnceStore(AlertStateStore):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.failed_once = False

        def record_report(self, report, **kw):
            if not self.failed_once and report.state == AlertState.firing:
                self.failed_once = True
                raise RuntimeError("simulated disk failure at the firing edge")
            return super().record_report(report, **kw)

    rule = promoted_neonatal_rso2_rule(metric="rso2_left")
    receiver = LocalFileReceiver(path=str(tmp_path / "notif" / "out.jsonl"))
    dispatcher = NotificationDispatcher(receiver, increment="INC5")
    store = FailingOnceStore(base_dir=tmp_path / "state")
    service = AnomalyEvaluatorService(
        [rule],
        _sim(Scenario.rso2_desat),
        store=store,
        dispatcher=dispatcher,
        monotonic=lambda: 0.0,
        promoted=True,
        increment="INC5",
    )
    _run(service, 15)

    assert store.failed_once, "the simulated failure never triggered"
    assert service.heartbeat.error_count == 1, "the failure was not loud"
    assert len(receiver.messages) == 1, (
        "SWALLOWED PAGE: the firing edge consumed by the failed tick was "
        "never re-emitted — the page for this episode is lost (E11)"
    )


def test_break_glass_restart_preserves_signal_lost_hold(tmp_path):
    """E13(i): cycling the loop must NOT lift a force_signal_lost hold."""
    rule = neonatal_rso2_rule(metric="rso2_left")
    provider = _sim(Scenario.healthy)
    service = AnomalyEvaluatorService(
        [rule],
        provider,
        store=AlertStateStore(base_dir=tmp_path / "state"),
        monotonic=lambda: 0.0,
        increment="INC5",
    )
    bg = BreakGlassController(service, provider, increment="INC5")

    bg.force_signal_lost(by="charge_nurse")
    bg.restart(by="charge_nurse")
    assert bg.status()["force_signal_lost"] is True, (
        "restart() silently lifted the operator's data-untrustworthy hold"
    )
    held = bg.tick(30.0, mono=30.0)
    assert held.reports[rule.id].state == AlertState.signal_lost

    # Only the EXPLICIT clear lifts the hold.
    bg.clear_force_signal_lost(by="charge_nurse")
    freed = bg.tick(60.0, mono=60.0)
    assert freed.reports[rule.id].state != AlertState.signal_lost


def test_clear_force_shadow_restores_pre_hold_posture(tmp_path):
    """E13(ii): lifting a shadow-hold restores the pre-hold paging posture
    (an undo, not a promotion), and status reports the EFFECTIVE posture."""
    rule = promoted_neonatal_rso2_rule(metric="rso2_left")
    provider = _sim(Scenario.rso2_desat)
    receiver = LocalFileReceiver(path=str(tmp_path / "notif" / "out.jsonl"))
    dispatcher = NotificationDispatcher(receiver, increment="INC5")
    service = AnomalyEvaluatorService(
        [rule],
        provider,
        store=AlertStateStore(base_dir=tmp_path / "state"),
        dispatcher=dispatcher,
        monotonic=lambda: 0.0,
        promoted=True,
        increment="INC5",
    )
    bg = BreakGlassController(service, provider, increment="INC5")

    bg.force_shadow(rule.id, by="safety_officer")
    assert bg.status()["effective_promoted"][rule.id] is False
    _run(bg, 15)  # fires in shadow — nothing delivered
    assert receiver.messages == []

    bg.clear_force_shadow(rule.id, by="safety_officer")
    assert bg.status()["effective_promoted"][rule.id] is True, (
        "pre-hold paging posture was not restored — the rule would stay "
        "silently demoted until process restart (E13)"
    )
    assert rule.id not in bg.status()["forced_shadow_rules"]


def test_persist_failure_at_resolved_edge_does_not_mute_next_episode(tmp_path):
    """Verifier-found defect in the first E11 rollback: a blanket rollback at
    a failed RESOLVED edge restored _fired=True, so a re-breach entered
    firing with no event — an entire new episode paged nothing. The rollback
    must apply ONLY to a consumed firing edge."""
    from app.services.anomaly_core import DataStatus
    from app.services.anomaly_data_provider import Observation

    class BreachRecoverBreachProvider:
        """Breach ticks 0-11 (fires ~t=300s), healthy 12-13 (resolved edge),
        breach again 14+ (second episode fires ~tick 24)."""

        def observe(self, metric, *, now, labels=None):
            i = int(now // STEP_S)
            healthy = 12 <= i < 14
            status = DataStatus(
                reachable=True, returns_data=True,
                source="prometheus", sample_ts=now,
                history_coverage_s=24 * 3600.0,
            )
            return Observation(
                value=80.0 if healthy else 40.0,  # 40 < 0.8*80: breaching
                baseline=80.0,
                status=status,
            )

    class FailingAtResolvedStore(AlertStateStore):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.failed_once = False

        def record_report(self, report, **kw):
            if not self.failed_once and report.state == AlertState.resolved:
                self.failed_once = True
                raise RuntimeError("simulated disk failure at the resolved edge")
            return super().record_report(report, **kw)

    rule = promoted_neonatal_rso2_rule(metric="rso2_left")
    receiver = LocalFileReceiver(path=str(tmp_path / "notif" / "out.jsonl"))
    dispatcher = NotificationDispatcher(receiver, increment="INC5")
    store = FailingAtResolvedStore(base_dir=tmp_path / "state")
    service = AnomalyEvaluatorService(
        [rule],
        BreachRecoverBreachProvider(),
        store=store,
        dispatcher=dispatcher,
        monotonic=lambda: 0.0,
        promoted=True,
        increment="INC5",
    )
    _run(service, 30)

    assert store.failed_once, "the simulated resolved-edge failure never triggered"
    # Both distinct firing episodes must page (at-least-once), and the
    # resolved-edge failure must be loud.
    assert len(receiver.messages) == 2, (
        f"expected 2 pages (one per firing episode); got "
        f"{len(receiver.messages)} — a failed resolved edge muted an episode"
    )
    assert service.heartbeat.error_count == 1
