"""INC3 suite — channel-agnostic notifier + Discord TEST-TUNNEL adapter.

Behavioural gates for the out-of-band notification path:

  (1) a REAL firing on a PROMOTED rule delivers EXACTLY ONE notification,
  (2) a SHADOW firing delivers ZERO (would-fire recorded only),
  (3) a SIGNAL_LOST-only trace delivers ZERO clinical firings and raises its
      own distinct signal-lost page (promoted) / records would-fire (shadow),
  (4) a flapping breach pages ONCE per firing edge (dedup + hysteresis),
  (5) suppression of a critical/firing alert is REFUSED (never silent),
  (6) every message carries the [TEST TUNNEL · NON-DIAGNOSTIC] banner,
  (7) the Discord adapter never hits the network in tests (injected opener),
      masks the token, POSTs via urllib (shell=False, no argv), retries+acks,
  (8) the env-driven factory falls back to LocalFileReceiver with no webhook.

A MISSED must-fire delivery on a PROMOTED rule is the top-severity failure.
Tests NEVER touch a real Discord endpoint (conftest clears the env var; the
Discord adapter is only ever exercised with an injected fake opener).
"""

from __future__ import annotations

import json

import pytest

from app.prometheus.neonatal_sim import Scenario
from app.services.alert_state_store import AlertStateStore
from app.services.anomaly_data_provider import SimDataProvider
from app.services.anomaly_evaluator_service import AnomalyEvaluatorService
from app.services.anomaly_notification_dispatch import (
    NotificationDispatcher,
    SuppressionRefused,
)
from app.services.anomaly_notifier import (
    TEST_TUNNEL_BANNER,
    DeliveryReceipt,
    DiscordWebhookNotifier,
    LocalFileReceiver,
    NotificationMessage,
    Notifier,
    build_default_notifier,
)
from app.specs.anomaly_evaluation_report import (
    AlertEvent,
    AlertState,
    SignalLostReason,
)
from app.tests_support.default_rules import (
    neonatal_rso2_rule,
    promoted_neonatal_rso2_rule,
)

STEP_S = 30.0
FOR_S = 300.0
N_TICKS = 40


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeReceiver(Notifier):
    """In-memory fake channel — records deliveries, never touches network."""

    channel_name = "fake"

    def __init__(self, *, ack: bool = True) -> None:
        self.deliveries: list[NotificationMessage] = []
        self._ack = ack

    def deliver(self, message: NotificationMessage) -> DeliveryReceipt:
        self.deliveries.append(message)
        return DeliveryReceipt(
            dedup_key=message.dedup_key,
            channel=self.channel_name,
            delivered=self._ack,
            acked=self._ack,
            attempts=1,
        )


class _FakeResp:
    def __init__(self, status: int) -> None:
        self.status = status

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def _service(rule, provider, tmp_path, dispatcher, *, promoted, name="state"):
    return AnomalyEvaluatorService(
        [rule],
        provider,
        store=AlertStateStore(base_dir=tmp_path / name),
        staleness_budget_s=60.0,
        monotonic=lambda: 0.0,
        promoted=promoted,
        dispatcher=dispatcher,
    )


def _run(service, n=N_TICKS, step=STEP_S, start=0.0):
    for i in range(n):
        now = start + i * step
        service.tick(now, mono=now)


# ---------------------------------------------------------------------------
# (1) PROMOTED firing => EXACTLY ONE delivery
# ---------------------------------------------------------------------------
def test_promoted_firing_delivers_exactly_once(tmp_path):
    rule = promoted_neonatal_rso2_rule(metric="rso2_left")
    receiver = FakeReceiver()
    disp = NotificationDispatcher(receiver)
    provider = SimDataProvider(
        scenario=Scenario.rso2_desat, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS
    )
    service = _service(rule, provider, tmp_path, disp, promoted=True)

    _run(service, n=20)  # well past the 5m firing point

    assert len(receiver.deliveries) == 1, (
        "MUST-FIRE DELIVERY: a promoted, sustained desaturation must page "
        f"EXACTLY once (got {len(receiver.deliveries)})"
    )
    msg = receiver.deliveries[0]
    assert msg.state == AlertState.firing.value
    assert msg.severity == "critical"
    # Durable page accounting agrees.
    assert service.store.get(rule.id).paged_count == 1


# ---------------------------------------------------------------------------
# (2) SHADOW firing => ZERO deliveries (would-fire recorded only)
# ---------------------------------------------------------------------------
def test_shadow_firing_delivers_zero(tmp_path):
    rule = neonatal_rso2_rule(metric="rso2_left")  # mode=shadow
    receiver = FakeReceiver()
    disp = NotificationDispatcher(receiver)
    provider = SimDataProvider(
        scenario=Scenario.rso2_desat, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS
    )
    service = _service(rule, provider, tmp_path, disp, promoted=False)

    _run(service, n=20)

    assert receiver.deliveries == [], "SHADOW rule PAGED — invariant violated"
    st = service.store.get(rule.id)
    # would-fire is recorded, never delivered — visible, not silent.
    assert st.would_page_count >= 1
    assert st.paged_count == 0


# ---------------------------------------------------------------------------
# (3) SIGNAL_LOST-only => zero clinical firing; its own distinct page
# ---------------------------------------------------------------------------
def test_signal_lost_only_no_clinical_fire_promoted_pages_signal_lost(tmp_path):
    rule = promoted_neonatal_rso2_rule(metric="rso2_left")
    receiver = FakeReceiver()
    disp = NotificationDispatcher(receiver)
    # source_degrades to mock; the numeric values breach but the gate blocks.
    provider = SimDataProvider(
        scenario=Scenario.source_degraded, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS
    )
    service = _service(rule, provider, tmp_path, disp, promoted=True)

    _run(service, n=N_TICKS)

    # No clinical firing was ever delivered.
    assert all(m.state != AlertState.firing.value for m in receiver.deliveries), (
        "FAIL-OPEN: a clinical firing was delivered on mock/fake data"
    )
    # A distinct signal-lost page WAS delivered (promoted, non-suppressible).
    sl = [m for m in receiver.deliveries if m.state == AlertState.signal_lost.value]
    assert sl, "SIGNAL_LOST did not raise its own distinct page"
    assert sl[0].signal_lost_reason == SignalLostReason.mock_source.value
    assert sl[0].banner == TEST_TUNNEL_BANNER


def test_signal_lost_only_shadow_delivers_zero(tmp_path):
    rule = neonatal_rso2_rule(metric="rso2_left")  # shadow
    receiver = FakeReceiver()
    disp = NotificationDispatcher(receiver)
    provider = SimDataProvider(
        scenario=Scenario.unreachable, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS
    )
    service = _service(rule, provider, tmp_path, disp, promoted=False)

    _run(service, n=N_TICKS)

    assert receiver.deliveries == [], "shadow signal_lost must not page"
    # but the would-fire (signal_lost is conspicuous) is recorded.
    assert service.store.get(rule.id).would_page_count >= 1


# ---------------------------------------------------------------------------
# (4) flap => page ONCE per firing edge (dedup + hysteresis)
# ---------------------------------------------------------------------------
def test_flapping_breach_pages_once_per_edge(tmp_path):
    rule = promoted_neonatal_rso2_rule(metric="rso2_left")
    receiver = FakeReceiver()
    disp = NotificationDispatcher(receiver)

    healthy = SimDataProvider(
        scenario=Scenario.healthy, start_ts=0.0, step_s=STEP_S, n_ticks=200
    )
    desat = SimDataProvider(
        scenario=Scenario.rso2_desat, start_ts=0.0, step_s=STEP_S, n_ticks=200
    )

    class Phased:
        """Two separate sustained breaches with a full recovery between."""

        def observe(self, metric, *, now):
            # Breach window A: [60, 60+FOR+90); recover; breach B: [700, 700+FOR+90)
            if 60.0 <= now < 60.0 + FOR_S + 90.0:
                return desat.observe(metric, now=now)
            if 700.0 <= now < 700.0 + FOR_S + 90.0:
                return desat.observe(metric, now=now)
            return healthy.observe(metric, now=now)

    service = _service(rule, Phased(), tmp_path, disp, promoted=True)
    _run(service, n=45)  # covers both breach windows and the recovery between

    fires = [m for m in receiver.deliveries if m.state == AlertState.firing.value]
    # Two genuine firing EDGES => exactly two pages, not one per tick.
    assert len(fires) == 2, (
        f"expected 2 pages for 2 firing edges (dedup+hysteresis), got {len(fires)}"
    )
    # Distinct dedup keys (different transition ts) prove hysteresis re-armed.
    assert fires[0].dedup_key != fires[1].dedup_key


def test_same_firing_edge_never_double_pages(tmp_path):
    """Re-dispatching the SAME firing event twice delivers only once."""
    receiver = FakeReceiver()
    disp = NotificationDispatcher(receiver)
    ev = AlertEvent(
        ts="2026-07-01T00:05:00+00:00",
        rule_id="neo-rso2-left-desat",
        state=AlertState.firing,
        value=40.0,
        threshold=56.0,
        severity="critical",
        would_page=True,
        paged=True,  # promoted edge
    )
    from app.specs.anomaly_evaluation_report import AnomalyEvaluationReport

    rpt = AnomalyEvaluationReport(
        rule_id=ev.rule_id, ts=ev.ts, state=AlertState.firing, events=[ev]
    )
    out1 = disp.dispatch_report(rpt)
    out2 = disp.dispatch_report(rpt)  # same edge again
    assert out1.delivery_count == 1
    assert out2.delivery_count == 0
    assert out2.deduped, "second dispatch of the same edge must hit the dedup path"
    assert len(receiver.deliveries) == 1


# ---------------------------------------------------------------------------
# (5) suppression of a critical / firing alert is REFUSED (never silent)
# ---------------------------------------------------------------------------
def test_suppression_of_critical_refused_via_api(tmp_path):
    receiver = FakeReceiver()
    disp = NotificationDispatcher(receiver)
    ev = AlertEvent(
        ts="2026-07-01T00:05:00+00:00",
        rule_id="neo-rso2-left-desat",
        state=AlertState.firing,
        severity="critical",
        would_page=True,
        paged=True,
    )
    with pytest.raises(SuppressionRefused):
        disp.refuse_suppression(ev)
    assert receiver.deliveries == [], "nothing should have been delivered on refusal"


def test_paging_event_marked_suppressed_is_refused(tmp_path):
    """An inconsistent 'paged AND suppressed' firing must fail loud, not silence."""
    receiver = FakeReceiver()
    disp = NotificationDispatcher(receiver)
    ev = AlertEvent(
        ts="2026-07-01T00:05:00+00:00",
        rule_id="neo-rso2-left-desat",
        state=AlertState.firing,
        severity="critical",
        would_page=True,
        paged=True,
        suppressed_reason="display_filter",  # someone tried to silence a page
    )
    from app.specs.anomaly_evaluation_report import AnomalyEvaluationReport

    rpt = AnomalyEvaluationReport(
        rule_id=ev.rule_id, ts=ev.ts, state=AlertState.firing, events=[ev]
    )
    with pytest.raises(SuppressionRefused):
        disp.dispatch_report(rpt)


# ---------------------------------------------------------------------------
# (6) banner + non-diagnostic disclaimer on every message
# ---------------------------------------------------------------------------
def test_every_message_carries_banner_and_disclaimer():
    ev = AlertEvent(
        ts="2026-07-01T00:05:00+00:00",
        rule_id="neo-rso2-left-desat",
        state=AlertState.firing,
        severity="critical",
        message="FIRING — value 40 < threshold 56",
        would_page=True,
        paged=True,
    )
    msg = NotificationMessage.from_event(ev)
    text = msg.rendered()
    assert msg.banner == TEST_TUNNEL_BANNER
    assert text.startswith(TEST_TUNNEL_BANNER)
    assert "NOT a diagnosis" in text
    assert msg.non_diagnostic is True


# ---------------------------------------------------------------------------
# (7) Discord adapter — injected opener, masking, retries, ack; NO network
# ---------------------------------------------------------------------------
def test_discord_adapter_uses_injected_opener_and_masks_token():
    calls = []

    def fake_opener(req, timeout=None):
        # Assert we POST JSON, not shell out; capture the payload.
        assert req.method == "POST"
        calls.append(json.loads(req.data.decode("utf-8")))
        return _FakeResp(204)  # Discord webhook success

    secret = "https://discord.com/api/webhooks/123456789/SUPER_SECRET_TOKEN_abcd"
    notifier = DiscordWebhookNotifier(secret, opener=fake_opener)

    ev = AlertEvent(
        ts="2026-07-01T00:05:00+00:00",
        rule_id="neo-rso2-left-desat",
        state=AlertState.firing,
        severity="critical",
        message="FIRING",
        would_page=True,
        paged=True,
    )
    receipt = notifier.deliver(NotificationMessage.from_event(ev))

    assert receipt.delivered and receipt.acked
    assert receipt.attempts == 1
    # Exactly one POST; content carries the banner.
    assert len(calls) == 1
    assert calls[0]["content"].startswith(TEST_TUNNEL_BANNER)
    # The masked target NEVER exposes the token.
    masked = notifier.masked_target()
    assert "SUPER_SECRET_TOKEN" not in masked
    assert secret not in masked


def test_discord_adapter_retries_then_reports_unacked():
    attempts = {"n": 0}

    def flaky_opener(req, timeout=None):
        attempts["n"] += 1
        raise OSError("connection reset")  # never acks

    notifier = DiscordWebhookNotifier(
        "https://discord.com/api/webhooks/1/x", opener=flaky_opener, max_attempts=3
    )
    ev = AlertEvent(
        ts="2026-07-01T00:05:00+00:00",
        rule_id="r",
        state=AlertState.firing,
        severity="critical",
        would_page=True,
        paged=True,
    )
    receipt = notifier.deliver(NotificationMessage.from_event(ev))
    # At-least-once: it TRIED max_attempts times; ack failed => surfaced.
    assert attempts["n"] == 3
    assert receipt.delivered is False
    assert receipt.acked is False
    assert receipt.error is not None


# ---------------------------------------------------------------------------
# (8) env-driven factory falls back to LocalFileReceiver when no webhook
# ---------------------------------------------------------------------------
def test_factory_falls_back_to_local_file_when_no_webhook(tmp_path):
    # conftest already cleared ANOMALY_TEST_WEBHOOK_URL.
    notifier = build_default_notifier(local_path=tmp_path / "notif" / "out.jsonl")
    assert isinstance(notifier, LocalFileReceiver)


def test_factory_uses_discord_when_webhook_present(monkeypatch):
    monkeypatch.setenv(
        "ANOMALY_TEST_WEBHOOK_URL", "https://discord.com/api/webhooks/9/TOKENZ"
    )
    notifier = build_default_notifier()
    assert isinstance(notifier, DiscordWebhookNotifier)
    # Never expose the token in the masked target.
    assert "TOKENZ" not in notifier.masked_target()


def test_local_file_receiver_persists_message(tmp_path):
    path = tmp_path / "notif" / "out.jsonl"
    receiver = LocalFileReceiver(path=path)
    ev = AlertEvent(
        ts="2026-07-01T00:05:00+00:00",
        rule_id="r",
        state=AlertState.firing,
        severity="critical",
        message="FIRING",
        would_page=True,
        paged=True,
    )
    receipt = receiver.deliver(NotificationMessage.from_event(ev))
    assert receipt.ok
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["banner"] == TEST_TUNNEL_BANNER
    assert rec["text"].startswith(TEST_TUNNEL_BANNER)


# ---------------------------------------------------------------------------
# integration: promoted service end-to-end with the LOCAL receiver (no net)
# ---------------------------------------------------------------------------
def test_end_to_end_promoted_local_receiver_one_delivery(tmp_path):
    rule = promoted_neonatal_rso2_rule(metric="rso2_left")
    receiver = LocalFileReceiver(path=tmp_path / "notif" / "out.jsonl")
    disp = NotificationDispatcher(receiver)
    provider = SimDataProvider(
        scenario=Scenario.rso2_desat, start_ts=0.0, step_s=STEP_S, n_ticks=N_TICKS
    )
    service = _service(rule, provider, tmp_path, disp, promoted=True)
    _run(service, n=20)
    assert len(receiver.messages) == 1
    assert receiver.messages[0]["text"].startswith(TEST_TUNNEL_BANNER)
