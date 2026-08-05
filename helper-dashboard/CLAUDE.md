# CLAUDE.md — helper-dashboard

Chat-to-dashboard system for Prometheus on a hospital Jetson AGX Orin, plus a
**neonatal cerebral-oximetry (rSO₂) anomaly-detection subsystem** — patient-safety-
critical: a missed real desaturation is worse than a false alarm. FastAPI backend
(`backend/app/`), Next.js frontend (`frontend/`), in-app OpenCode agents (`.opencode/`).

## Rule precedence (read this first)

`docs/agent-ops/LOCKED_DECISIONS.md` > this file > everything else in `docs/`,
`README.md`, `AGENTS.md`.

Known trap: README/AGENTS/DESIGN/ARCHITECTURE/SECURITY_BOUNDARIES still say
"unsupported widget type ⇒ DeveloperTicket ⇒ wait for Big guy". That policy is
**overturned** (LD-1): the widget toolkit is a cache of pre-built widgets, not a cap —
out-of-toolkit requests default to code-gen (in-app `rescue_extend`, or a Claude
session coding it directly). Never refuse widget work with "flagged for the team".
The `WidgetType` enum has **9** members (`backend/app/specs/widget_spec.py:33-42`,
incl. pie_chart, bar_chart, heatmap, decision_flow); docs and the in-app prompts
(`AGENTS.md:44`, `dashboard-spec-agent.md:49`) still hardcode the original 5 — trust
the enum. `alert_list` renders hardcoded MOCK data (`AlertListWidget.tsx:10-25`), it
is not a real alerting surface.

## Two agent layers — don't conflate them

| | In-app product agents | Dev-time Claude agents |
|---|---|---|
| Who | Helper (user-facing) + Big guy (internal), small OpenCode models | You (Claude Code session) + subagents like `anomaly-builder` |
| Config | `.opencode/agent/*.md`, `opencode.json` | `.claude/agents/*.md` |
| Invoked by | `backend/app/helper/runtime.py` operation allow-list | Claude Code Agent tool |
| May edit repo code | Helper: never. Big guy: via dev-token `developer_fix`, or the gated background auto-fix (`helper/auto_fix.py`, SECURITY_BOUNDARIES §5b; user-approved 2026-07-06) | **Yes** — that's your job (LD-2) |
| Safety rules | `docs/SECURITY_BOUNDARIES.md` | This file + `.claude/agents/anomaly-builder.md` |

"Helper can't edit code" is a rule for the product layer only. Do not apply it to
yourself, and do not grant Helper agents code-edit paths.

## Clinical-safety code — protected paths

`backend/app/services/anomaly_*.py`, `services/alert_state_store.py`,
`backend/app/prometheus/neonatal_*.py`, `prometheus/client.py`,
`backend/app/specs/alert_rule_spec.py`, `backend/app/api/anomaly.py`.

Inviolable invariants (full text: `.claude/agents/anomaly-builder.md`):
1. Data-integrity gate before any threshold check: reachable ∧ returns-data ∧
   `source=="prometheus"` ∧ fresh. Any failure ⇒ `SIGNAL_LOST`, never a clinical verdict.
2. `source=="mock"` never alerts; absence of data is alarmable, not "no anomaly".
3. No silent suppression — every withheld alert is logged; critical alerts can't be filtered away.
4. New/edited rules start in SHADOW (LD-6); promotion needs the user + green goldens.
5. Every alerting surface keeps the non-suppressible "decision-support, not a diagnosis" label.

Rules for these paths: never weaken an invariant to make something pass; run
`python3 -m pytest tests/ -q` before claiming done; if a requested change conflicts
with an invariant, stop and ask the user instead of working around it.

**Any new consumer of `prometheus/client.py` must branch on the `source` field**
(non-`"prometheus"` ⇒ mock badge / SIGNAL_LOST, never real data). Copy the check from
`services/anomaly_data_provider.py`, not from dashboard widget code.

## Verification contract

| Change touches | Mandatory before "done" |
|---|---|
| Anything in `backend/` or `tests/` | `python3 -m pytest tests/ -q` (Tier 0, no network) |
| Orchestrator / API routes | Tier 0 + `python3 scripts/tier2_backend_e2e.py` (mock mode) |
| Frontend build files | `cd frontend && npx tsc --noEmit` (fast) — full `npm run build` only when shipping |
| Anomaly evaluator / rules | Tier 0; goldens = `tests/anomaly_golden/` (targeted: `python3 -m pytest tests/anomaly_golden -q`) must stay green |

**Tier 0 baseline (measured 2026-07-03): 407 passed, 1 failed, ~100s.** The 1 failure
is `tests/browser_evaluation/test_backend_down.py` — known-environmental (needs
next+Playwright), NOT a regression; run with
`--ignore=tests/browser_evaluation/test_backend_down.py` or expect exactly it.
Never use `-x` on the full suite (it dies at that test with 350+ tests unrun).
Any OTHER failure, or a second failure, is a real regression — report it.

Tier 3 (`scripts/live_full_demo.py`) is a stress demo that legitimately fails on this
Jetson — **never** use it as a correctness signal. README:71 saying "Tiers 1 and 2 are
the authoritative proof" is wrong (A_DIAGNOSIS F6); Tier 0 is mandatory. If a test
fails, report the real output; never claim green without running the command.

## Git rules

- Work happens on branch `uiagent_with_newwidget`; `main` is far behind and lacks the
  anomaly subsystem. Don't "fix" that by merging without the user.
- **Trap:** parent-repo `.gitignore:21` (`lib/`) silently ignores `frontend/lib/`
  (renderer.tsx, spec-schema.ts). Disk, not git, is ground truth there. Don't
  re-implement "missing" renderer code — check the filesystem first.
- Never `git add`: `*.log`, `*.pid`, `*.bak`, `.env*` (except `.env.example`),
  `backend/app/storage/**` runtime artifacts, audit JSONL
  (`anomaly_build_audit/`, `anomaly_lifecycle_audit/`, `extend_audit/`).
- Scope git commands to your subtree (`git status -- .` from helper-dashboard) — the
  repo root is noisy (see `docs/agent-ops/A_DIAGNOSIS.md` T5).
- Commit policy (canonical — other files defer here): commit ONLY files you yourself
  changed this session, staged by explicit path — never `-a`/`-A`/`git add .`.
  Pre-existing dirty files and tracked audit JSONL stay uncommitted (T3, ASK-USER).
  If the user hasn't authorized commits this session, hand them the exact
  `git add <paths> && git commit` command instead. Subagents never commit.
  Nothing here is pushed to a remote (no gh auth) — warn the user when a session ends
  with valuable unpushed work.
- Secrets: `OPENROUTER_API_KEY`, `ANOMALY_TEST_WEBHOOK_URL`, `HELPER_DASHBOARD_DEV_TOKEN`
  live in `.env` (git-ignored). Never print, commit, or log their values; mask them in
  any audit or report output.

## Orientation map

- `backend/app/helper/` — orchestrator (1.1k lines), runtime allow-list (1.3k lines),
  retry loop. Read the target function ± callers before editing; never bulk-reformat.
- `backend/app/specs/` — Pydantic schemas (strict, `extra='forbid'`). `widget_spec.py`
  holds the WidgetType enum (9 types).
- `backend/app/services/` — validators, stores, the anomaly subsystem.
- `frontend/widget-toolkit/` + `frontend/lib/renderer.tsx` (NOT `renderer.ts` — docs
  citing that name are wrong) — fixed widget renderer; spec fields never executed as code.
- `docs/USER_SESSION_LOG.md` — machine-appended by `helper/memory.py` on every
  create/patch INCLUDING pytest runs; ~90% test-fixture noise; nothing reads it back.
  **Never read it as documentation.** Same caution for `docs/SESSION_CHAT_HISTORY_*.md`
  (superseded snapshot, contains PII, never commit).
- Doc ownership: invariants → `docs/SECURITY_BOUNDARIES.md`; how-to-run → `README.md`;
  agent roles → `AGENTS.md`. If two docs disagree, follow precedence above and note the
  contradiction in `docs/agent-ops/A_DIAGNOSIS.md`.
- Repo-wide searches: always add `--exclude-dir=worktrees --exclude-dir=snapshots
  --exclude-dir=node_modules` — a 4.8GB stale worktree (`.claude/worktrees/`) and audit
  snapshots (`backend/app/storage/extend_audit/snapshots/`) return duplicate/stale copies.

## Known runtime landmines (details: A_DIAGNOSIS S1–S3, E6–E10)

- Port 8000 is held by an **orphaned demo backend** from a past session; the real
  medical exporter is dead; system Prometheus scrapes synthetic data as `nicu_jetson`
  (S1). Do NOT kill/restart either side without the user.
- Background evaluator (`ANOMALY_EVALUATOR_ENABLED=1`) would be permanently SIGNAL_LOST
  due to a sim-clock/wall-clock mismatch (E7); not currently enabled anywhere (the S1
  orphan runs demo-only). Do not "fix" it by weakening the staleness gate.
- `/api/anomaly/alerts` returns data only in demo mode (E10). Fails safe — don't
  default-construct a store to make it "work".
- The audit JSONL has known lies: canned shadow reasoning on ACTIVE activations, fake
  MISSED records from webhook 403s (E6). Don't take it at face value in incident review.
- Break-glass (`services/anomaly_break_glass.py`) is not mounted anywhere — REPL-only (E8).

## Working agreements (any session, any model)

1. Before non-trivial work: read `docs/agent-ops/LOCKED_DECISIONS.md` (short) and skim
   `docs/agent-ops/A_DIAGNOSIS.md` findings list.
2. Delegating to subagents? Use the fill-in templates in
   `docs/agent-ops/DELEGATION_TEMPLATES.md` (they encode acceptance criteria + report format).
3. Updating rules/docs/memory: follow `docs/agent-ops/MAINTENANCE_PROTOCOL.md` —
   it says what you may change alone vs. what needs the user, and where lessons go.
4. Judgment calls a Sonnet-level model should NOT make alone: relaxing any safety
   invariant, changing alert thresholds, promoting rules out of shadow, deleting
   audit data, clinical-plausibility calls. Present evidence + options to the user;
   a 3-subagent judge panel may be convened as ADVISORY input, but user sign-off is
   required either way — the decision itself is never delegated.
5. New session with no context? Start at `docs/agent-ops/LETTER_TO_FUTURE_SESSIONS.md`.
