"""Channel-agnostic alert Notifier + Discord TEST-TUNNEL adapter (INC3).

The evaluator core decides *what* happened (an `AlertEvent`); this module is
the ONLY place that turns a firing/signal-lost event into an out-of-band
delivery. It is deliberately separated behind an interface so the real
clinical alarm channel can later be swapped in without touching the
evaluator.

Safety posture (inviolable — see the anomaly-builder charter):

- **TEST TUNNEL ONLY.** The only shipped remote adapter is a Discord *incoming
  webhook*. It is a test tunnel, never a clinical alarm channel and never
  load-bearing for patient safety. EVERY message carries a visible
  `[TEST TUNNEL · NON-DIAGNOSTIC]` banner (:data:`TEST_TUNNEL_BANNER`) and a
  persistent non-diagnostic disclaimer. Delivery here must NEVER be recorded
  or implied as "a clinician was notified".
- **Never hardcode / commit / log the token.** The webhook URL comes from the
  env var ``ANOMALY_TEST_WEBHOOK_URL``. It is never written to source, and
  every audit/log line masks it via the INC1 ``mask_url`` helper. If the env
  var is absent we DO NOT block: we fall back to a local `LocalFileReceiver`
  so the whole loop still runs end to end.
- **shell=False, no user-controlled argv, no data-exfil.** The Discord adapter
  POSTs JSON with the stdlib ``urllib.request`` — there is no subprocess, no
  shell, and no user-controlled command line anywhere in the delivery path.
- **At-least-once + ack.** Every delivery attempt is logged (masked). A
  transient failure is retried up to a bounded count; the receipt records
  whether the channel acked. A delivery that never acks is surfaced, not
  swallowed into a false "sent".

This module does NOT decide shadow-vs-active or dedup policy — that is the
`NotificationDispatcher`'s job (see :mod:`anomaly_notification_dispatch`). A
Notifier just delivers a message it was handed and reports honestly.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..specs.anomaly_evaluation_report import AlertEvent
from . import anomaly_build_audit as audit

# The banner that MUST prefix every message this module emits. It makes the
# test-tunnel + non-diagnostic status impossible to miss on any surface.
TEST_TUNNEL_BANNER = "[TEST TUNNEL · NON-DIAGNOSTIC]"
_NON_DIAGNOSTIC_DISCLAIMER = (
    "Decision-support signal for a neonatal rSO2 monitor — NOT a diagnosis and "
    "NOT a clinical alarm. This channel is a test tunnel only; do not treat a "
    "message here as a clinician having been notified."
)

_DEFAULT_RECEIVER_DIR = (
    Path(__file__).resolve().parent.parent / "storage" / "anomaly_notifications"
)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Message + receipt value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NotificationMessage:
    """A single out-of-band notification derived from an `AlertEvent`.

    `body` is the human-facing text; `banner` is ALWAYS present and prepended
    by :meth:`rendered`. `dedup_key` lets the channel/adapter recognise a
    retry of the same logical alert (idempotency hint).
    """

    dedup_key: str
    rule_id: str
    state: str
    severity: str
    title: str
    body: str
    ts: str = field(default_factory=_now_iso)
    banner: str = TEST_TUNNEL_BANNER
    non_diagnostic: bool = True
    signal_lost_reason: str | None = None

    def rendered(self) -> str:
        """The full text delivered to a channel — banner is NON-optional."""
        return (
            f"{self.banner} {self.title}\n"
            f"{self.body}\n"
            f"— {_NON_DIAGNOSTIC_DISCLAIMER}"
        )

    @classmethod
    def from_event(cls, event: AlertEvent) -> "NotificationMessage":
        """Build a banner-carrying message from an evaluator `AlertEvent`."""
        slr = (
            event.signal_lost_reason.value
            if event.signal_lost_reason is not None
            else None
        )
        # dedup_key keys on rule + state + the transition ts so a re-render of
        # the SAME firing edge is idempotent, but a genuinely new edge (new ts)
        # is a new notification. SIGNAL_LOST keys additionally on the reason.
        parts = [event.rule_id, event.state.value, event.ts]
        if slr:
            parts.append(slr)
        dedup_key = "|".join(parts)
        title = f"{event.severity.upper()} · {event.rule_id} · {event.state.value}"
        return cls(
            dedup_key=dedup_key,
            rule_id=event.rule_id,
            state=event.state.value,
            severity=event.severity,
            title=title,
            body=event.message or f"{event.state.value} on {event.rule_id}",
            ts=event.ts,
            signal_lost_reason=slr,
        )


@dataclass
class DeliveryReceipt:
    """Outcome of ONE delivery attempt sequence (with retries)."""

    dedup_key: str
    channel: str
    delivered: bool
    acked: bool
    attempts: int
    error: str | None = None
    detail: str | None = None  # channel-specific, MUST be secret-free

    @property
    def ok(self) -> bool:
        return self.delivered and self.acked


# ---------------------------------------------------------------------------
# Notifier interface + adapters
# ---------------------------------------------------------------------------
class Notifier(ABC):
    """Channel-agnostic delivery interface.

    An adapter delivers a `NotificationMessage` and returns a
    `DeliveryReceipt`. It must NEVER raise for an ordinary delivery failure —
    it reports the failure in the receipt so the dispatcher can decide.
    """

    #: Human name of the channel, safe to log (no secrets).
    channel_name: str = "notifier"

    @abstractmethod
    def deliver(self, message: NotificationMessage) -> DeliveryReceipt:  # pragma: no cover - interface
        raise NotImplementedError

    # A masked identity for logs/audit (never the raw token/URL).
    def masked_target(self) -> str:
        return self.channel_name


class LocalFileReceiver(Notifier):
    """Default fallback channel — appends each message to a JSONL file.

    Used automatically when ``ANOMALY_TEST_WEBHOOK_URL`` is absent so the loop
    still runs end to end without a network or a real webhook. It is also the
    fake receiver the tests assert against. Delivery is ack'd iff the append
    (fsync'd) succeeded. It NEVER hits the network.
    """

    channel_name = "local_file"

    def __init__(self, path: str | Path | None = None, *, also_print: bool = False) -> None:
        self._dir = Path(path).parent if path else _DEFAULT_RECEIVER_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = (
            Path(path)
            if path
            else self._dir / f"{datetime.now(tz=timezone.utc):%Y-%m-%d}.jsonl"
        )
        self.also_print = bool(also_print)
        self.messages: list[dict[str, Any]] = []

    def deliver(self, message: NotificationMessage) -> DeliveryReceipt:
        record = {
            "logged_at": _now_iso(),
            "dedup_key": message.dedup_key,
            "rule_id": message.rule_id,
            "state": message.state,
            "severity": message.severity,
            "banner": message.banner,
            "non_diagnostic": message.non_diagnostic,
            "signal_lost_reason": message.signal_lost_reason,
            "text": message.rendered(),
        }
        try:
            import os

            line = json.dumps(record, ensure_ascii=False, default=str)
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.messages.append(record)
            if self.also_print:
                print(f"[anomaly-notifier:{self.channel_name}] {message.rendered()}")
            return DeliveryReceipt(
                dedup_key=message.dedup_key,
                channel=self.channel_name,
                delivered=True,
                acked=True,
                attempts=1,
                detail=str(self._path),
            )
        except Exception as exc:  # noqa: BLE001 — report, never raise
            return DeliveryReceipt(
                dedup_key=message.dedup_key,
                channel=self.channel_name,
                delivered=False,
                acked=False,
                attempts=1,
                error=f"{type(exc).__name__}: {exc}",
            )

    def masked_target(self) -> str:
        return f"local_file://{self._path.name}"


class DiscordWebhookNotifier(Notifier):
    """Discord incoming-webhook adapter — TEST TUNNEL ONLY.

    POSTs a JSON body via ``urllib.request`` (no subprocess, no shell, no
    user-controlled argv). The URL is read from ``ANOMALY_TEST_WEBHOOK_URL``
    and NEVER logged in full — every audit/log reference is masked. Every
    message carries the test-tunnel banner. A 2xx (Discord returns 204 on a
    webhook) counts as an ack; anything else is retried up to `max_attempts`
    and, if it never acks, reported as un-acked (surfaced, not swallowed).
    """

    channel_name = "discord_test_tunnel"

    def __init__(
        self,
        webhook_url: str,
        *,
        max_attempts: int = 3,
        timeout_s: float = 5.0,
        opener: Any = None,
    ) -> None:
        if not webhook_url:
            raise ValueError("webhook_url must be a non-empty string")
        self._url = webhook_url
        self.max_attempts = max(1, int(max_attempts))
        self.timeout_s = float(timeout_s)
        # `opener` is injectable so tests NEVER hit the real Discord endpoint;
        # default uses urllib.request.urlopen.
        self._opener = opener or urllib.request.urlopen

    def masked_target(self) -> str:
        return audit.mask_url(self._url) or "****"

    def _payload(self, message: NotificationMessage) -> bytes:
        # `content` is what Discord renders; banner + disclaimer are inside it.
        body = {
            "content": message.rendered()[:1900],  # Discord content limit guard
            "username": "anomaly-test-tunnel (NON-DIAGNOSTIC)",
        }
        return json.dumps(body).encode("utf-8")

    def deliver(self, message: NotificationMessage) -> DeliveryReceipt:
        data = self._payload(message)
        last_err: str | None = None
        attempts = 0
        for attempt in range(1, self.max_attempts + 1):
            attempts = attempt
            req = urllib.request.Request(
                self._url,
                data=data,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                resp = self._opener(req, timeout=self.timeout_s)
                status = getattr(resp, "status", None)
                if status is None:
                    # http.client.HTTPResponse exposes .getcode() on older pys
                    status = resp.getcode() if hasattr(resp, "getcode") else 0
                try:
                    resp.close()
                except Exception:  # noqa: BLE001
                    pass
                if 200 <= int(status) < 300:
                    return DeliveryReceipt(
                        dedup_key=message.dedup_key,
                        channel=self.channel_name,
                        delivered=True,
                        acked=True,
                        attempts=attempts,
                        detail=f"http {status}",
                    )
                last_err = f"non-2xx status {status}"
            except urllib.error.HTTPError as exc:  # 4xx/5xx
                last_err = f"HTTPError {exc.code}"
            except Exception as exc:  # noqa: BLE001 — transient/network
                last_err = f"{type(exc).__name__}: {exc}"
        # Exhausted attempts without an ack: delivered=False (surface it).
        return DeliveryReceipt(
            dedup_key=message.dedup_key,
            channel=self.channel_name,
            delivered=False,
            acked=False,
            attempts=attempts,
            error=last_err,
        )


# ---------------------------------------------------------------------------
# Factory — env-driven channel selection (webhook if present, else local file)
# ---------------------------------------------------------------------------
def build_default_notifier(
    *,
    webhook_url: str | None = None,
    local_path: str | Path | None = None,
) -> Notifier:
    """Return the shipping default channel.

    If a webhook URL is provided (or ``ANOMALY_TEST_WEBHOOK_URL`` is set), use
    the Discord TEST-TUNNEL adapter; otherwise fall back to a local file
    receiver so the loop still runs end to end. The URL is NEVER logged in
    full — the audit records only the masked target.
    """
    import os

    url = webhook_url if webhook_url is not None else os.getenv("ANOMALY_TEST_WEBHOOK_URL")
    if url:
        notifier: Notifier = DiscordWebhookNotifier(url)
        chosen = "discord_test_tunnel"
        masked = notifier.masked_target()
    else:
        notifier = LocalFileReceiver(path=local_path)
        chosen = "local_file"
        masked = notifier.masked_target()

    audit.append(
        increment="INC3",
        stage="code",
        action="run",
        target="build_default_notifier",
        summary=f"Selected notification channel: {chosen}",
        reasoning=(
            "TEST TUNNEL only; webhook used when env var present else local file "
            "fallback so the loop always runs end to end. URL is masked."
        ),
        evidence={"channel": chosen, "target": masked},
        outcome="notifier_built",
    )
    return notifier
