# LOCKED_DECISIONS.md — settled user decisions (single in-repo source)

> These are decisions the user has made explicitly. **Do not re-litigate, water down,
> or "improve" them.** If new work seems to conflict with one, stop and ask the user,
> quoting the entry ID. Only the user adds/changes entries here (an agent may draft the
> wording, the user approves). See `MAINTENANCE_PROTOCOL.md` §2.
> If docs elsewhere (README/AGENTS/DESIGN/ARCHITECTURE/SECURITY_BOUNDARIES) contradict
> this file, THIS FILE WINS. Precedence: LOCKED_DECISIONS > CLAUDE.md > other docs.

## LD-1. Widget toolkit is a cache, not a cap (2026-05-29)
The `WidgetType` enum / widget toolkit is a set of **pre-built widgets so agents code
less** — not a fence limiting what may be built. When a user asks for a widget type
that doesn't exist (e.g. sankey), the correct behavior is: **an agent codes it**
(in-app: the `rescue_extend` path; dev-time: a Claude session/subagent edits the six
widget files directly). Refusing with "flagged for the team" / DeveloperTicket-and-wait
is explicitly wrong. The six files (authoritative list: `bin/opencode` ~line 1362):
`backend/app/specs/widget_spec.py`, `backend/app/specs/widget_schema_doc.py`,
`frontend/lib/spec-schema.ts`, `frontend/lib/renderer.tsx`,
`frontend/widget-toolkit/<Type>Widget.tsx`, `tests/spec_validation/test_extend_<type>.py`.
- The 5 safety gates in the extend path (denylist, regex, prompt-injection check,
  quota, audit) stay — they are safety, not policy.
- Do not propose narrowing the toolkit or adding refusal paths; extend-by-default.
- Docs still describing ticket-first as the only path are stale (see
  `A_DIAGNOSIS.md` F1); trust this entry.

## LD-2. Anomaly system is fully automatic — no human in the build path (2026-07-01)
The neonatal anomaly-detection system is built and maintained by **Claude sessions and
the `anomaly-builder` subagent writing code directly** (`.claude/agents/anomaly-builder.md`),
NOT via the in-app `developer_fix` / human-developer route. "We are never going the
human-only path."
- Required in exchange: **full durable audit** of agent reads/edits/decisions
  (`backend/app/storage/anomaly_build_audit/`) and of Helper↔Big-guy interactions.
- Required: **break-glass manual control surface** (start/stop/recover, force-shadow,
  force-SIGNAL_LOST) — a kill switch, never part of normal operation.

## LD-3. Anomaly rule semantics (2026-07-01)
- Breach = `value < 0.80 × avg_over_time(metric[24h])` — a 20% **relative** drop below
  the 24-hour baseline, sustained `for: 5m`. (NOT `< 0.20×` — a past misreading;
  0.80× is correct.)
- Population: **neonatal only**. Use neonatal physiological ranges, never adult.
- `<24h` of history ⇒ `INSUFFICIENT_BASELINE`, not a fire and not a pass.

## LD-4. Notification channel is a TEST TUNNEL, not a clinical channel (2026-07-01)
- Discord incoming webhook; URL only from env var `ANOMALY_TEST_WEBHOOK_URL`
  (git-ignored; never hardcode, commit, paste, or log unmasked).
- Every message carries `[TEST TUNNEL · NON-DIAGNOSTIC]`. Delivery must never be
  recorded or implied as "a clinician was notified."
- Notifier stays channel-agnostic (interface + adapters) so a real clinical channel
  can swap in later without touching the evaluator.
- Missing URL ⇒ fall back to the local fake receiver; never block on it.

## LD-5. Synthetic data is authorized while real data is dead (2026-07-01)
Real medical telemetry is not flowing (exporter dead). Building/running a synthetic
neonatal rSO₂ data source (plus SpO₂/HR/MAP/FiO₂) to drive the whole loop is
explicitly approved. Constraints:
- Synthetic series must remain distinguishable from real ones at the metric level
  (label or prefix). Before real data is enabled, a cutover plan must be agreed with
  the user (`A_DIAGNOSIS.md` E5).
- Evaluator never treats mock/stale/unreachable data as clinical truth: those become
  `SIGNAL_LOST`, and `source=="mock"` never alerts.

## LD-6. Shadow-first alerting (2026-07-01)
Every new or edited clinical rule starts in SHADOW mode (records would-fire events,
never pages). Promotion out of shadow requires the **user's** explicit instruction,
gated on the behavioural golden suite being green (must-fire traces fire;
must-not-fire traces don't; mock/stale ⇒ SIGNAL_LOST; short history ⇒
INSUFFICIENT_BASELINE). ("Supervisor" in the anomaly-builder charter means the main
Claude session, which may RELAY the user's approval to the sub-agent but can never
originate a promotion on its own authority. No Claude session, of any size, promotes
a rule to paging by itself.)

## LD-7. Certification posture (2026-07-01)
The hospital handles regulatory clearance. The software still ships the persistent
non-suppressible "decision-support, not a diagnosis" disclaimer on every alerting
surface, and the audit trail — so clearance stays possible.

---

### Change log
- 2026-07-03: file created by institutional-review session; content promoted verbatim
  from the user's auto-memory entries (`project_anomaly_detection_decisions`,
  `feedback_toolkit_philosophy`, `feedback_fully_automatic_no_human_path`).
- 2026-07-04: adversarial-review clarity fixes, NO policy change: LD-6 now states
  explicitly that only the USER authorizes shadow→paging promotion (a Claude
  "supervisor" only relays); LD-1 now enumerates the six extend-path files.
