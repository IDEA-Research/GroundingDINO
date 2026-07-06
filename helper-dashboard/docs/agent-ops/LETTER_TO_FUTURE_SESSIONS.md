# Letter to future sessions

> Written 2026-07-03 by the one-time Claude Fable 5 institutional-review session.
> You are probably a smaller model (Sonnet/Opus/Haiku). That is fine — everything in
> `docs/agent-ops/` was written so that you don't need to be smarter, just more
> disciplined. Read `LOCKED_DECISIONS.md` first, then this letter's "Current state".

## Three things the user didn't ask about, but matter most here

### 1. This is a hospital system whose data path is currently self-poisoned — treat "what is actually running" as unknown until you check
The single most dangerous assumption in this repo is that docs describe reality.
As of 2026-07-03: the real vitals exporter is dead, an orphaned demo backend owns its
port, the working Prometheus ingests synthetic neonatal data under the production job
name, and the AI-analysis service is a zombie (A_DIAGNOSIS S1–S3). Every one of those
contradicts the runbooks. **Habit to keep:** before touching anything runtime-related,
run `systemctl status <unit>`, `docker ps`, `ss -tlnp`, and compare against what the
docs claim; report divergence to the user instead of silently working around it. The
cost of a wrong assumption here is not a broken build — it is fake patient data
looking real.

### 2. The audit trail is the product — guard its honesty above its volume
The user's core requirement for the anomaly system is that a human can later
reconstruct what the machine did and why (LD-2). Right now the trail exists but lies
in places (canned "shadow" reasoning on paging activations, webhook failures recorded
as clinical MISSED events, wrong increment tags — E6). When you touch anything that
writes audit records, the bar is: **a reader who trusts the record verbatim must not
be misled.** That is a stricter bar than "the data is technically present". Never
trade it for convenience, and never delete audit data to "clean up" (ASK-USER always).

### 3. Files are the only memory that survives — write like you'll be replaced tomorrow
This environment loses everything between sessions except: the repo, the auto-memory
dir, and these agent-ops files. The Fable session you're reading was possible because
three memory entries survived from May–July. Whatever you learn that took >10 minutes
to figure out: write it to `LESSONS.md` (verified mistakes) or the letter's "Current
state" (facts), per `MAINTENANCE_PROTOCOL.md` §3, BEFORE you end the session. An
unrecorded discovery is a discovery the project never made. Also: nothing is pushed to
any remote — until the user sets up push access, this Jetson's disk is the only copy
of patient-safety-relevant code. Remind them occasionally.

## How this institution will most likely decay, and the countermeasures

1. **Drift between rules and reality** (the way the old docs decayed): code moves,
   CLAUDE.md doesn't, and in ~3 months CLAUDE.md is the new trap. Countermeasure:
   MAINTENANCE_PROTOCOL §1 lets any session fix verified-wrong facts immediately, and
   §5 forces a weak-reader test after edits. When you catch CLAUDE.md being wrong,
   fixing it IS the task — don't route around it.
2. **Append-only bloat** (the way USER_SESSION_LOG decayed): LESSONS.md grows until
   nobody reads it. Countermeasure: §4 compaction thresholds. If you notice a file
   over its budget, compact it (that's self-serve work).
3. **Authority erosion**: a future session, unaware of precedence, "fixes" LOCKED
   decisions or re-adds refusal paths because stale docs sound authoritative.
   Countermeasure: the precedence line at the top of CLAUDE.md and LOCKED_DECISIONS.
   If you ever find yourself about to enforce DeveloperTicket-first widget policy,
   stop — that's the known failure mode (LESSONS 2026-05-29).
4. **Silent rule violations under pressure**: mid-incident, a session bypasses
   ASK-USER gates "just this once" (that is exactly how the orphan process in S1 was
   born — a demo left running). Countermeasure: the gates in MAINTENANCE_PROTOCOL §2
   are cheap to honor (one question to the user); when you feel the urge to bypass,
   write the question down instead.
5. **Institution ignored entirely**: a session starts in a directory where no
   CLAUDE.md loads, or an assistant that doesn't read memory. Countermeasure: both
   CLAUDE.md levels exist (root + dashboard) and auto-memory points here; if you're
   reading this from an entry point that got NO other context, the map is:
   `helper-dashboard/docs/agent-ops/` has everything; start with LOCKED_DECISIONS.md.

## What quality this setup can and cannot reach (honesty clause)

With templates + verification gates + adversarial review, a Sonnet-level session can
execute scoped code changes, searches, and audits at high reliability. It cannot
reliably make: clinical-plausibility calls, novel security judgments, taste decisions
on ambiguous specs, or "is this safe to simplify" calls in the anomaly core. For
those: (a) ask the user with options+evidence, (b) if the user insists on delegation,
run 3 independent subagent judges and require unanimity, and (c) if still ambiguous,
say plainly "this needs a stronger model or a human expert" — that sentence is allowed
and expected here. Do not bluff confidence.

## Current state (update me when it changes — self-serve per MAINTENANCE_PROTOCOL §1)

As of 2026-07-05 (continuation of the review-and-fix session; user directed
"try to fix it" on the E11–E14 queue):
- E11–E14 FIXED in code (details + residuals inline in A_DIAGNOSIS at each
  finding). Method: implement → 8 adversarial verifier agents (2 per fix) →
  3 confirmed defects found IN MY OWN FIXES (blanket-rollback hazard;
  single-slot signal_lost key leak; timestamp()-can't-see-frozen-publisher)
  → revised per verifier prescriptions → second 3-skeptic round → 1 residual
  (exception-path re-arm) fixed → all suites green. Lesson: adversarial
  verification of protected-path changes is NOT optional here — it caught
  three real patient-safety bugs tests missed.
- Tier 0: 434 pass + 1 known-env failure (was 407 at the 2026-07-03 review;
  +27 tests over the two fix sessions). Tier 2: 10/10. Goldens green.
- E12 leftovers for INC6 wiring: MUST pass
  freshness_metric="neonatal_sim_last_update_timestamp_seconds" + the real
  scrape_interval_s to PrometheusDataProvider. E11 residual: crash-window
  needs a durable undelivered-page ledger (ASK-USER). S1–S4 still untouched
  and still need user decisions.

As of 2026-07-04 (review-and-fix session; supersedes the 2026-07-03 block below):
- E6 audit-honesty fixes (a,b,c,e) LANDED; new findings E11–E15 added to
  A_DIAGNOSIS (E15 = fixed batch: demo rmtree guard + ANOMALY_DEMO_STATE_DIR,
  decision_flow by_alias completion, for>0 validator, honest core messages,
  publisher O(n²) fix, tick off the event loop). E11–E14 are new ASK-USER items.
- Tier 0 tests: 413 pass + 1 known-env failure (test_backend_down.py) after this
  session's +6 tests (was 407). Tier 2 (`scripts/tier2_backend_e2e.py`) was
  silently broken — it predated the /api/chat/message enqueue+poll conversion and
  failed 4/10 on shape mismatch; fixed (polls the job endpoint), now 10/10.
- Doc fixes landed: README tier authority (F6), 9-type widget lists + LD-1
  stale-markers in README/AGENTS/DESIGN/SECURITY_BOUNDARIES (F1 docs half);
  `.opencode/agent/*` prompts still stale (ASK-USER).
- Everything below from 2026-07-03 otherwise still holds (S1–S4 live incidents
  still unresolved and still un-asked; nothing pushed).

As of 2026-07-03:
- Branch `uiagent_with_newwidget`, 6 commits unpushed, no remote auth. Local `main`
  diverged from origin/main (2 ahead / 1 behind). Working tree has uncommitted
  changes predating this review (api/dashboard.py, main.py, anomaly_notifier.py,
  widget_spec.py + untracked anomaly files) — from the INC-era sessions; do not
  discard.
- Anomaly subsystem INC1–INC5 complete in code; runtime wiring incomplete: the
  "INC6" that would make it real (neonatal_publisher + PrometheusDataProvider in the
  live loop, synthetic-data labeling per LD-5) is NOT built. Background loop currently
  useless (E7). Break-glass unmounted (E8). INC6 is now additionally blocked on E12
  (provider staleness/labels/coverage defects) — do not wire it as-is.
- Tier 0 tests: 407 pass + 1 known-env failure (test_backend_down.py).
- Live incidents S1–S3 unresolved; user has not yet been asked about them (first
  session after 2026-07-03: surface them).
- The Discord webhook token should be treated as leaked (S4 + E6e) — recommend
  rotation when the user next appears.

## Unfinished items handed off by the review session

1. `[ASK-USER]` decisions queue — present these with A_DIAGNOSIS evidence: S1 port-8000
   ownership + cleanup; S2 which Prometheus is canonical; S3 key provisioning +
   mediamtx; S4 webhook rotation + chat-history scrubbing; F4 one-line .gitignore fix
   (`!helper-dashboard/frontend/lib/` — then commit those 6 files); T3 audit
   git-tracking policy; T6 pruning the 4.8GB worktree; E9 alert_list realness; gh auth
   + push.
2. `[MECHANICAL, unclaimed]` — safe for any session: E6 audit-honesty fixes (derive
   reasoning from actual mode; separate delivery_failed from missed; real increment
   tags; full-mask URLs) under full Tier 0 + goldens; opportunistic doc de-duplication
   (T2); stale 5-type list fixes in docs (F1) — but `.opencode/agent/*` prompt files
   are product behavior ⇒ ASK-USER first.
3. Not done at all: measuring in-app agent prompt quality (the 40KB `.opencode/agent/`
   prompts were only inventoried, not rewritten — they duplicate rules and carry the
   stale widget list; rewriting them changes product behavior ⇒ needs user + Tier 1/2
   verification runs).
4. **Fresh-context adversarial review status: `[REVIEWED 2026-07-04]`.** Three
   independent reviewers (rule-conflicts / mechanical fact-check / weak-reader
   simulation) audited every institution file. Of ~80 sampled factual claims, 3 were
   defective (all fixed); 2 blocker-level rule ambiguities (shadow-promotion
   authority; commit policy) and ~12 smaller conflicts/gaps were found and fixed the
   same night. Re-run this review after any LARGE institution change
   (DELEGATION_TEMPLATES template 5, 3 reviewers), and update this marker.
5. The institution files are **not yet committed to git** (this review session chose
   to leave the commit for user approval; the standing rule is CLAUDE.md §Git rules).
   They are untracked on disk — protect them from `git clean`. Ask the user, then from
   `helper-dashboard/`: `git add CLAUDE.md ../CLAUDE.md docs/agent-ops/*.md
   .claude/agents/anomaly-builder.md && git commit` (never stage
   `docs/agent-ops/backups/` — one backup contains PII).

Good luck. Check twice, change once, write it down.
