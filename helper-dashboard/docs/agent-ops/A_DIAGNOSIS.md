# A. Project Diagnosis — token leaks, focus drift, error-prone spots

> Written 2026-07-03 by a Claude Fable 5 institutional-review session (v2 — merges a
> 6-agent parallel deep-read: session logs, audit trails, docs cross-check, backend
> code, parent-repo runtime, config/tests; ~530k subagent tokens of evidence).
> Audience: future agent sessions (Sonnet-level and up) and the user.
> Other files in `docs/agent-ops/` cite findings here by ID (S1, T1, F1, E1…).
> When a finding is fixed, mark it `[FIXED <date> — how]` in place; don't delete it.

Categories: **S** live incidents (user decision needed now) · **T** token leaks ·
**F** focus drift · **E** error-prone spots. Each finding: evidence → cost → fix.
`[MECHANICAL]` = safe for any session to apply. `[ASK-USER]` = policy/destructive;
get explicit approval (see `MAINTENANCE_PROTOCOL.md` §2).

---

## S. Live incidents found during this review (do not "fix" without the user)

### S1. An orphaned demo backend replaced the production metrics endpoint; Prometheus is ingesting synthetic neonatal data as the NICU feed
- **Evidence (verified live 2026-07-03):** pid 4107710 `uvicorn app.main:app --port
  8000`, running 2d5h, started by a previous Claude session's shell with
  `ANOMALY_DEMO=1` and alert state under a `/tmp` scratchpad; it holds 127.0.0.1:8000.
  The real medical exporter is dead — `../medical_exporter.log` ends with
  `OSError: [Errno 98] Address already in use`; its tracked pidfile points to a
  nonexistent pid. Host `/etc/prometheus/prometheus.yml` job `nicu_jetson` scrapes
  `localhost:8000` every 5s ⇒ the working Prometheus has been storing the dashboard's
  **synthetic** rSO₂/SpO₂/HR series in place of the real patient-vitals series since
  ~2026-07-01 17:47.
- **Cost:** Real-vitals export is down; the TSDB now contains synthetic data under
  production job labels (poisons any 24h-baseline math and any later retrospective
  analysis); demo alert state dies whenever /tmp is cleaned.
- **Fix `[ASK-USER]`:** Decide the intended owner of port 8000 (the
  `neonatal_publisher` "point to 8000" design suggests the dashboard backend may be
  intended — but then the real exporter needs a new port and the Prometheus job needs
  relabeling, and synthetic series need a distinguishing label per LD-5). Then: kill or
  service-ify the orphan, restart `medical-monitoring-base`, and record the chosen
  topology in the repo-root CLAUDE.md. Until decided, agents must not restart either
  side "to fix it".

### S2. Two Prometheus instances; the documented one crash-loops, an undocumented one serves
- **Evidence (agent-verified, rerunnable):** docker `medical-prometheus`
  (v2.40.0) restart-loops on the committed `../prometheus.yml` (invalid `storage:` /
  `web:` sections → `field ... not found in type config.plain` every ~62s). The
  Prometheus actually answering :9090 is the Ubuntu **apt** `prometheus.service`
  v2.31.2 (up since 2026-03-09) with its own config at `/etc/prometheus/prometheus.yml`.
- **Cost:** An agent "fixing Prometheus" edits the repo yaml and restarts docker with
  zero effect on the live :9090 — or worse, removes the apt service as "undocumented"
  and takes down the only working TSDB.
- **Fix `[ASK-USER]`:** pick one instance as canonical. `[MECHANICAL]` meanwhile:
  repo-root CLAUDE.md documents the split; any Prometheus work must first check
  `systemctl status prometheus` AND `docker ps | grep prometheus`.

### S3. `medical-ai-analysis` is a zombie: placeholder API key on world-readable argv, RTSP target that nothing serves
- **Evidence (agent-verified):** `ps` shows `--api_key YOUR_OPENROUTER_API_KEY_HERE`
  (interpolated from untracked `../.env.systemd`, mode 664 not 600); nothing listens
  on 8554 and no mediamtx unit exists, so the service retry-logs several times per
  second into journald (`journalctl -u medical-ai-analysis`).
- **Cost:** The flagship OCR-analysis service does no work while `Restart=always`
  keeps it alive; journald churn; if a real key lands in `.env.systemd` it becomes
  ps-visible to every local process.
- **Fix `[ASK-USER]`:** provision the key properly (mode 600, or pass via stdin/config
  not argv — requires editing `../systemd/medical-ai-analysis.service`), and decide
  who starts mediamtx (binary + `mediamtx.yml` exist at repo root; no unit references
  them). Until then: don't "helpfully" restart it.

### S4. PII + live webhook channel ID sitting in an uncommitted docs file
- **Evidence:** `docs/SESSION_CHAT_HISTORY_2026-07-01.md` contains the user's email,
  a Discord webhook URL with real channel ID (token masked), and the transcript's own
  note that the raw URL leaked into that conversation. Locate with greppable anchors
  (line numbers shift as the file is edited): `grep -n '@gmail\|discord.com/api/webhooks'
  docs/SESSION_CHAT_HISTORY_2026-07-01.md`. File is untracked but sits in `docs/` on a
  branch that will eventually be pushed.
- **Cost:** One `git add docs/` away from permanent history exposure; webhook token
  should be assumed compromised already (rotate).
- **Fix `[ASK-USER]`:** rotate the Discord webhook; move the transcript out of `docs/`
  (e.g. `docs/history/` + gitignore) or scrub PII. `[MECHANICAL]`: never `git add`
  this file; also note the committed audit JSONL leaks the webhook token's last 4
  chars (E6, item e).

---

## T. Token leaks (context wasted every session)

### T1. No CLAUDE.md existed — every session rediscovered the project from scratch
- **Evidence:** Before 2026-07-03 there was no CLAUDE.md anywhere in the repo. Rules
  lived scattered: `AGENTS.md` (192 ln), `README.md` (430 ln), `docs/ARCHITECTURE.md`
  (196), `docs/SECURITY_BOUNDARIES.md` (247), `docs/DESIGN.md` (118),
  `.claude/agents/anomaly-builder.md` (153), plus auto-memory.
- **Cost:** Thousands of tokens of re-reading per session; wrong-policy onboarding
  (F1) because the freshest rules weren't in any file a session auto-loads.
- **Fix `[FIXED 2026-07-03]`:** `helper-dashboard/CLAUDE.md` + thin repo-root
  `CLAUDE.md` created; `docs/agent-ops/` institution added. Keep dashboard CLAUDE.md
  ≤150 lines (MAINTENANCE_PROTOCOL §4).

### T2. The same rules are written 3–5× across docs — ~⅓ of the doc set is duplicated boilerplate
- **Evidence:** "Two kinds of widget operations" appears near-verbatim in `DESIGN.md:33-49`,
  `AGENTS.md:28-72`, `SECURITY_BOUNDARIES.md:53-91`, `README.md:15-24`. Runtime modes +
  "no silent fallback" in `SECURITY_BOUNDARIES.md:170-190`, `AGENTS.md:121-154`,
  `README.md:313-327`. Subprocess invariants ×3. Permission layers ×3 in docs and ×3
  again in config (`opencode.json` + agent frontmatter + AGENTS.md prose). The in-app
  prompts duplicate too: the ~35-line DeveloperTicket extend-format block is verbatim
  in `dashboard-spec-agent.md:223-259` AND `patch-agent.md:180-215`.
- **Cost:** ~15KB redundant reading per doc tour; copies have already diverged (F1,
  F6) — every future edit must hit 3-5 places or create a new contradiction.
- **Fix `[MECHANICAL, opportunistic]`:** one owner per topic (invariants →
  SECURITY_BOUNDARIES; how-to-run → README; roles → AGENTS.md); when touching a
  duplicated section, replace the copy with a pointer to the owner. No big-bang rewrite.

### T3. Generated audit JSONL is committed to git and mutates at runtime
- **Evidence:** `anomaly_build_audit/2026-06-30.jsonl` + `2026-07-01.jsonl` are
  tracked and currently modified (+57 lines of runtime lifecycle mirrors). The other
  audit dirs (`anomaly_lifecycle_audit/`, `extend_audit/` = 6.2MB incl. 70 snapshot
  dirs) are untracked but NOT gitignored — one `git add -A` commits 6MB of generated
  snapshots. `frontend/tsconfig.tsbuildinfo` likewise unignored.
- **Cost:** Permanently dirty tree; every status/diff drags generated JSONL into
  context; snapshots pollute grep (T6).
- **Fix `[ASK-USER]`** (user requires durable audit — never delete without approval):
  gitignore the three audit dirs + `git rm --cached` the two tracked JSONLs (keep on
  disk), or declare "audit is part of the deliverable" and commit them at session end
  only. `[MECHANICAL]` until decided: never `git add` audit paths; diff with
  `git diff -- ':!*.jsonl'`.

### T4. `docs/USER_SESSION_LOG.md` is a machine-written log that is 90% test noise — and its own header lies about it
- **Evidence (deep-read, full file):** 3,875 lines / 643 entries, appended by
  `backend/app/helper/memory.py:53-59` (call site `orchestrator.py:812`) on every
  dashboard create/patch — including from pytest: 577/643 entries are test-fixture
  session IDs (`s-widget-test` ×154, `sp-3` ×90 …); "Show me a CPU dashboard" appears
  359 times. Header claims it's a "rolling log … used by Helper agents to recall
  context" — **no code reads it back and nothing rolls it**. Currently +1,104
  uncommitted test-noise lines. It contains zero decisions (nothing about anomaly,
  OCR, or device templates).
- **Cost:** ~20k tokens for near-zero information for any agent that trusts the
  header; unbounded growth; every pytest run dirties a committed doc.
- **Fix `[MECHANICAL]`:** never read it (if you must: last ~200 lines). Real history
  lives in git log, audit JSONL, and `docs/agent-ops/`. `[ASK-USER]` (code change):
  gate `memory.record()` off under pytest, move the file to
  `backend/app/storage/` (gitignored) with rotation, and fix its lying header.
  Same treatment for `docs/SESSION_CHAT_HISTORY_2026-07-01.md` (42KB, see S4/F5).

### T5. Repo-root git noise
- **Evidence:** Tracked: `medical_exporter.pid` + `medical_exporter_simple.pid`
  (rewritten every service start ⇒ always dirty), `performance_log.txt` (1.7MB), ~600KB
  of JPEG frames under `test_monitoring_verification/`, 2 stray files inside
  `node_modules/`, junk file `device_templates/Untitled`. Half-staged
  device_templates rename (5 unstaged deletes + 6 untracked human-named files that are
  the ONLY copies of live templates). ~18 overlapping root-level .md guides, two of
  them contradicting each other on Grafana's port.
- **Cost:** Root git status is unreadable; `git clean -fd` would DELETE the only
  copies of live device templates and `.env.systemd`; doc sprawl sends agents to wrong
  ports/paths.
- **Fix `[ASK-USER]`:** stage the rename; untrack pid/log/jpeg noise; extend root
  .gitignore (`*.log`, `*.pid`, `*.bak`, `mongodb_data*/`). `[MECHANICAL]`: never run
  `git clean` at repo root; never commit files matching those patterns; scope git
  commands to your subtree.

### T6. 4.8GB stale worktree + 6.2MB audit snapshots poison repo-wide search
- **Evidence:** `helper-dashboard/.claude/worktrees/goofy-yalow-f9d4d7/` is a full
  duplicate of the repo (incl. 3.5GB Test_video, 662MB model checkpoint). Repo-wide
  grep returns doubled hits; `grep -rn 'class WidgetType' backend/app` returns stale
  snapshot copies under `extend_audit/snapshots/` BEFORE the live file.
- **Cost:** Agents read/edit the wrong copy; duplicated grep output wastes context;
  5GB gone on an embedded Jetson.
- **Fix `[ASK-USER]`:** `git worktree list` + prune the stale worktree (verify no
  unmerged work first — it may hold uncommitted changes). `[MECHANICAL]`: always
  exclude `worktrees` and `snapshots` from sweeps:
  `grep -rn --exclude-dir=worktrees --exclude-dir=snapshots …`.

---

## F. Focus drift (things that send an agent down the wrong path)

### F1. Docs teach a widget policy the user explicitly overturned — biggest documentation trap
- **Evidence:** All five design docs say: unsupported widget type ⇒ `DeveloperTicket`
  ⇒ wait for Big guy (`DESIGN.md:30-31,60-65`, `AGENTS.md:55-68`,
  `SECURITY_BOUNDARIES.md:74-91`, `README.md:20-24`). Reality:
  `backend/app/helper/extend_gate.py:1-23` — "extend runs by default, no opt-in
  required"; `rescue_extend` is a user-facing operation (`runtime.py:83,100`) that
  writes 6 widget files with no ticket; the user's locked decision LD-1 says the
  toolkit is a cache, not a cap. Even the docs' example is stale:
  `SECURITY_BOUNDARIES.md:77` names `heatmap` as un-representable — heatmap shipped
  (`widget_spec.py` enum, `HeatmapWidget.tsx`). Widget enum has **9** types;
  `AGENTS.md:44,84-85`, `dashboard-spec-agent.md:49,249`, `patch-agent.md:206` still
  hardcode the original 5.
- **Cost:** A fresh session enforcing docs will refuse supported widgets, ticket
  instead of extending, or "revert" the anomaly build as a boundary violation. The
  in-app agents' own prompts push the product's LLMs into the same refusals.
- **Fix `[FIXED 2026-07-03 for Claude sessions]`:** precedence rule + this exact trap
  named in `CLAUDE.md`; LD-1 in `LOCKED_DECISIONS.md`. `[ASK-USER, high value]`: update
  the 5-type lists inside `.opencode/agent/dashboard-spec-agent.md` and
  `patch-agent.md` (these are product prompt files — product behavior change).
  `[FIXED 2026-07-04 — docs only]`: README/AGENTS/DESIGN/SECURITY_BOUNDARIES now list
  all 9 types, carry dated LD-1 stale-markers, and SECURITY_BOUNDARIES no longer cites
  heatmap as un-representable. The `.opencode/agent/*` prompt files remain stale
  (still ASK-USER).

### F2. Two agent systems with confusingly similar vocabulary
- **Evidence:** In-app OpenCode product agents (`.opencode/agent/`, invoked through
  `runtime.py`) vs dev-time Claude agents (`.claude/agents/anomaly-builder.md`,
  supervisor = main session). Memory records the user explicitly correcting the
  conflation once already.
- **Cost:** Sessions apply product-layer rules ("Helper can't edit code") to
  themselves, or grant product agents dev-layer powers; whole budgets burned in the
  wrong layer.
- **Fix `[FIXED 2026-07-03]`:** disambiguation table in `CLAUDE.md`.

### F3. Locked decisions lived only in one user's auto-memory
- **Evidence:** Rule semantics (0.80×, 24h, 5m), neonatal-only scope, TEST-TUNNEL-only
  webhook, synthetic-data authorization existed only in
  `/home/cluster/.claude/projects/-etc-GroundDino-GroundingDINO-GroundingDINO/memory/`
  — invisible to any other entry point (OpenCode agents, other machines, fresh clones).
- **Cost:** Wrong thresholds re-derived (the `<0.20×` misreading), settled decisions
  re-litigated.
- **Fix `[FIXED 2026-07-03]`:** promoted to `docs/agent-ops/LOCKED_DECISIONS.md`
  (LD-1…LD-7), cited from CLAUDE.md; memory now points at the repo file.

### F4. "It's committed" ≠ "it's in the repo": the gitignored-frontend trap
- **Evidence (verified):** parent `.gitignore:21` = `lib/` (Python packaging pattern)
  silently ignores `helper-dashboard/frontend/lib/` — `git check-ignore -v
  frontend/lib/renderer.tsx` → `.gitignore:21:lib/`; `git ls-files frontend/lib/` is
  empty while 6 files exist on disk (renderer.tsx, spec-schema.ts, api.ts, patch.ts,
  useBackendStatus.ts, useQueryData.ts). A fresh clone cannot build the frontend.
  Compounding it, docs cite the file under two names (`renderer.ts` in
  `DESIGN.md:80`/`AGENTS.md:161`/big-guy prompt `:250`; `renderer.tsx` elsewhere) — a
  session following the wrong name will create a duplicate file.
- **Cost:** Re-implementation of existing code; `git stash/clean/checkout` can
  irrecoverably destroy the renderer; broken fresh clones.
- **Fix `[ASK-USER, one line]`:** add `!helper-dashboard/frontend/lib/` to root
  .gitignore, then commit the 6 files. `[MECHANICAL]` until then: filesystem, not git,
  is ground truth for `frontend/lib/`; the correct filename is **renderer.tsx**.

### F5. Stale "verified facts" documents with no supersession banner
- **Evidence:** `docs/SESSION_CHAT_HISTORY_2026-07-01.md` presents P1–P24 as verified
  problems, several now falsified by INC1–INC5 (INC1–INC5 = the anomaly subsystem's
  numbered build increments, see `.claude/agents/anomaly-builder.md`; "INC6" = the
  not-yet-built live-wiring increment): "source flag written but never read"
  (now read — widget mock badges, `StatCardWidget.tsx:17`, and the anomaly gate), "no
  background scheduler" (exists, `main.py:127-137`), "no notification channel" (exists,
  `anomaly_notifier.py`). Similarly `neonatal_publisher.py:17-19` documents main.py
  wiring behind `ANOMALY_NEONATAL_SIM_PUBLISH` that was never landed.
- **Cost:** Fresh agents re-litigate solved problems or hunt for wiring that doesn't
  exist; the publisher docstring makes "enable the publisher" a dead-end task.
- **Fix `[MECHANICAL]`:** supersession banner added at top of the chat-history file
  (done 2026-07-03); rule: any point-in-time snapshot doc gets a dated banner. The
  publisher docstring fix is a code comment change — safe, but do it together with the
  real INC6 wiring decision `[ASK-USER]`.

### F6. Docs contradict each other on which test tier is authoritative
- **Evidence:** `README.md:71-73` "Tiers 1 and 2 are the authoritative proof" vs
  `README.md:85` Tier 0 "canonical correctness proof" vs `ARCHITECTURE.md:159-160`
  "Only Tiers 0–2 are authoritative" vs `SECURITY_BOUNDARIES.md:228-231` Tier 0 =
  authoritative safety proof.
- **Cost:** A weak model can legitimately cite README:71 to skip pytest and run only
  LLM-dependent tiers — the opposite of intent.
- **Fix `[FIXED 2026-07-03 for Claude sessions]`:** CLAUDE.md owns the verification
  contract (Tier 0 mandatory). `[MECHANICAL, opportunistic]`: fix README:71 wording
  when next editing README. `[FIXED 2026-07-04 — README §Verification now names
  Tier 0 as the mandatory, authoritative proof; header says four tiers]`.

### F7. Graveyard runtime scripts sit beside live ones with no markers
- **Evidence:** Live path: systemd units → `start/stop_medical_monitoring.sh`,
  `run.sh`. Graveyard at the same level: `start_all_test.sh`,
  `start/stop_medical_monitoring_nodocr.sh`, `run.sh.bak`, `stop_all.sh` (which pkills
  a process systemd immediately respawns per `medical-ai-analysis.service`
  Restart=always), 5 dead `medical_exporter_*.log/pid` variants.
- **Cost:** "Stop the extractor" via `stop_all.sh` → systemd respawns it → agent
  concludes the process is haunted; `start_all_test.sh` double-launches services.
- **Fix `[FIXED 2026-07-03]`:** repo-root CLAUDE.md documents the live path and names
  the graveyard. `[ASK-USER]`: delete/move the graveyard scripts.

---

## E. Error-prone spots (where a weak model will predictably break things)

### E1. Clinical-safety code with unmarked blast radius
- **Evidence:** `backend/app/services/anomaly_*.py`, `alert_state_store.py`,
  `prometheus/neonatal_*.py`, `specs/alert_rule_spec.py` encode the inviolable
  invariants; good news from deep-read: the invariants ARE implemented (mock rejected
  in 3 places incl. `anomaly_core.py:148-160`; SIGNAL_LOST fail-closed
  `anomaly_core.py:162-183`; shadow gating `anomaly_core.py:98`; banner non-optional
  `anomaly_notifier.py:51,90-96`; breach-timer rehydration
  `anomaly_evaluator_service.py:159-171`). But nothing in the code tree marks these
  files as special for a generic session.
- **Cost:** A "simplify this evaluator" refactor can delete a patient-safety gate.
- **Fix `[FIXED 2026-07-03]`:** protected-paths list + invariants in CLAUDE.md.

### E2. The mock-fallback data source is a standing foot-gun — and its safe consumer is dead code
- **Evidence:** `prometheus/client.py:48-69` silently returns mock data with
  `source='mock'` on ANY exception (by design for dashboard UX).
  `PrometheusDataProvider` (`anomaly_data_provider.py:128-186`) reads the flag
  correctly — but grep shows **nothing instantiates it**; both wired providers are
  `SimDataProvider` (`main.py:43`, `api/anomaly.py:138`). Also
  `anomaly_data_provider.py:166-171` conflates mock with unreachable and contains a
  dead self-assignment branch.
- **Cost:** New consumers copy dashboard call sites (no source check) instead of the
  anomaly provider; the "central fix" for the charter hazard is unexercised by any
  running path.
- **Fix `[MECHANICAL rule]`:** every new consumer of `prometheus/client.py` must
  branch on `source` (copy `anomaly_data_provider.py`'s check). `[ASK-USER]`: wiring
  `PrometheusDataProvider` into the background loop = the real INC6 (see letter).

### E3. Test ground truth vs what a naive session runs
- **Evidence (measured):** `python3 -m pytest tests/ -q` → **407 passed, 1 failed in
  ~102s**. The 1 failure is `tests/browser_evaluation/test_backend_down.py`
  ("connection dot did not turn red") — known-environmental (needs next+Playwright;
  `.claude/settings.local.json:33,49` shows the team habitually `--ignore`s it). With
  `-x` the run dies at 51 tests and a fresh agent concludes the suite is broken.
- **Cost:** False "suite broken" alarms; debugging a known-environmental Playwright
  test on a Jetson; or skipping tests entirely because README:71 blessed Tiers 1–2 (F6).
- **Fix `[FIXED 2026-07-03]`:** CLAUDE.md states the canonical command, the expected
  baseline (407/1 known), and bans `-x` for full runs.

### E4. Three files >1,000 lines hold the security allow-lists
- **Evidence:** `helper/runtime.py` (1,350), `helper/orchestrator.py` (1,151);
  `specs/widget_spec.py` (434, grows per widget type).
- **Cost:** Mis-anchored edits by partial-context editing in exactly the files where a
  bad edit is a security event.
- **Fix `[MECHANICAL rule]`:** Grep the symbol, read target function ± callers only,
  edit surgically, run Tier 0; never bulk-reformat these files.

### E5. Synthetic vs real data ambiguity — no longer hypothetical (see S1)
- **Evidence:** Synthetic generator publishes realistic vitals; S1 shows Prometheus
  already ingesting them under the production scrape job. No metric-level label
  distinguishes synthetic from real.
- **Cost:** Baseline math and any retrospective analysis silently poisoned; a future
  clinician-facing surface could show synthetic numbers as real.
- **Fix `[ASK-USER, before real data ever flows]`:** enforce a `synthetic="true"`
  label (or dedicated prefix) on every synthetic series + a cutover plan (LD-5).

### E6. The audit trail currently misleads its readers (for a system whose point is auditability)
- **Evidence (verified):** (a) `anomaly_lifecycle_audit.py:189-192` hardcodes
  reasoning "shadow by default; promotion … supervisor-gated" onto every
  rule-activation record — including 13 records that say `activated in ACTIVE (paging)
  mode, promoted=true` (from the demo driver, `api/anomaly.py:141-151`). (b) 6
  top-severity "MISSED must-fire" records were generated purely by a misconfigured
  Discord webhook (HTTP 403 ×12) during a 10-minute demo churn — webhook delivery
  failure is being recorded as the *clinical* worst-case signal. (c) Runtime events
  carry build-increment tags (`increment="INC2"/"INC3"/"INC5"` hardcoded defaults,
  e.g. `anomaly_evaluator_service.py:111`). (d) One state transition is logged up to 3×
  (evaluator report + lifecycle stream + build-audit mirror,
  `anomaly_lifecycle_audit.py:154-167`), contradicting that module's own docstring.
  (e) `anomaly_build_audit.py:44` `mask_url` keeps the webhook token's last 4 chars in
  git-committed files while claiming to drop the secret. (f) Records mix wall-clock and
  1970 sim-clock timestamps (`2026-07-01.jsonl:24`).
- **Cost:** A reviewer reading the audit concludes a paging rule is in shadow (the
  exact property the audit exists to prove); real missed pages are indistinguishable
  from webhook 403 noise; incident timelines misordered.
- **Fix:** `[MECHANICAL]` (behavior-preserving, well-tested): derive the reasoning
  string from the actual mode; separate `delivery_failed` from `missed`; pass real
  increment/phase tags; drop the 4-char tail from mask_url. Each is small but touches
  protected paths ⇒ run full Tier 0 + golden suite; `[ASK-USER]` for the
  triple-logging design (it's arguably intentional mirroring).
  `[FIXED 2026-07-04 — items a,b,c,e]`: (a) `record_rule_activated` reasoning now
  derived from the actual promoted flag; (b) new `LifecycleKind.delivery_failed`
  (clinical, fail-closed) for attempted-but-failed deliveries — `missed` reserved for
  paged events with NO receipt (evaluator `_audit_delivery_lifecycle` split; e2e tests
  updated + new); (c) runtime/demo constructions now tag `increment="runtime"`/"demo",
  library defaults changed INC2/INC3/INC5→"runtime" (explicit INCn from build loops
  still honored); (e) `mask_url` keeps scheme+host only (regression test added).
  Items (d) triple-logging and (f) sim-clock timestamps remain open (d=ASK-USER,
  f=E7 territory).

### E7. Background evaluator loop is permanently SIGNAL_LOST while spamming disk
- **Evidence:** `main.py:43` builds `SimDataProvider(scenario=healthy)` with default
  `start_ts=0.0` (1970 sim clock) but ticks with `now=time.time()` (`main.py:53-56`)
  ⇒ staleness gate (`anomaly_core.py:158`) trips every tick, forever. Each tick emits
  SIGNAL_LOST events with no edge-detection into 3 fsync'd JSONL sinks — ~11.5k
  lines/day/file when `ANOMALY_EVALUATOR_ENABLED=1`. (Corrected 2026-07-04: NOT
  currently running anywhere — the S1 orphan was started with `ANOMALY_DEMO=1` only,
  without `ANOMALY_EVALUATOR_ENABLED`; verify via `tr '\0' '\n'
  </proc/<pid>/environ | grep ANOMALY`.)
- **Cost:** The "running" evaluator never produces a clinical verdict; disk churn on
  the Jetson; a session reading the audit sees a wall of stale SIGNAL_LOST and may
  "fix" it by weakening the staleness gate — the wrong fix.
- **Fix `[ASK-USER]`** (= the real INC6 decision): wire a real-time provider
  (PrometheusDataProvider + neonatal publisher, per the unlanded design) or align the
  sim clock; add edge-detection so state *changes* log, not every tick. Never weaken
  the staleness gate itself.

### E8. Break-glass exists only as an unmounted class
- **Evidence:** `BreakGlassController` (`services/anomaly_break_glass.py:104`) has
  start/stop/force-shadow/force-signal-lost/backup/recover — referenced only by its
  own tests; no router, no CLI, no main.py import. The charter
  (`anomaly-builder.md:123-133`) requires an operator-usable surface.
- **Cost:** In an incident, an operator (or agent) hunting for break-glass endpoints
  finds none and improvises riskier interventions (cf. S1's orphan-process situation).
- **Fix `[ASK-USER]`:** decide the surface (authenticated HTTP routes vs CLI script)
  and mount it; until then CLAUDE.md notes it's REPL-only.

### E9. `alert_list` widget renders hardcoded fake alerts
- **Evidence:** `frontend/widget-toolkit/AlertListWidget.tsx:10-25` renders a
  hardcoded `MOCK_ALERTS` array (HighCPU/LowDiskSpace) with a permanent mock badge;
  docs list `alert_list` as a supported type with no disclaimer.
- **Cost:** In a neonatal product, a plausible-looking fake alert surface; a session
  might wire anomaly output "into the existing alert widget" that displays nothing real.
- **Fix `[ASK-USER]`:** implement real alert sourcing (natural companion to the
  decision_flow/anomaly API) or visibly mark the type as demo-only in the widget
  gallery. `[MECHANICAL]`: CLAUDE.md warns it's mock.

### E10. Misc traps confirmed by deep-read
- `/api/anomaly/alerts` serves data ONLY in demo mode (store set solely by DemoDriver,
  `api/anomaly.py:47,164`; background loop sets `anomaly_service` not `anomaly_store`,
  `main.py:50`) — fails safe but looks broken; don't "fix" by defaulting a store in
  (that converts fail-safe into fake-empty-OK).
- `anomaly_core.py:225` `resolved if events else resolved` — both branches identical;
  a "fix" without reading the golden suite could alter gate order. Leave unless doing
  E6/E7 work with full tests.
- Extend-gate daily quota is per-process in-memory (`extend_gate.py:129-131`) —
  restarts reset it; weaker than the audit implies.
- `.opencode/command/` (singular; holds unregistered `rescue-extend.md`) vs
  `.opencode/commands/` (the 5 registered) — check `opencode.json` before editing either.
- `.claude/settings.local.json` allowlists `Bash(pip install *)` on a clinical device
  and carries dead PID-specific kill entries `[ASK-USER to tighten]`.
- Local `main` is 2 ahead / 1 behind `origin/main`; 6 commits unpushed on
  `uiagent_with_newwidget`. The entire anomaly subsystem + GPU fix exists only on this
  Jetson `[ASK-USER: auth gh + push]`.

### E11–E15. Findings added 2026-07-04 (multi-agent review session; sweep evidence re-verifiable at cited lines)

- **E11 `[ASK-USER — anomaly-core design]` Persist/audit failure at the firing edge
  permanently swallows the page.** `tick()` order: `core.evaluate()` consumes the
  firing edge (`_fired=True`, `anomaly_core.py:286`) BEFORE persist/audit/dispatch;
  if `store.record_report` or `lifecycle.record_transition` raises (disk full), the
  per-rule except records `tick_error` and dispatch never runs — and no later tick
  re-emits the firing event for that episode. For a promoted rule that is a lost page
  with only a tick_error trace. Fix options (rollback the edge on persist failure vs
  durable undelivered-page ledger) change core state-machine semantics ⇒ user + goldens.
  `[FIXED 2026-07-05 — user-directed, 2 adversarial-verify rounds]`: edge rollback in
  the except handler, applied ONLY when the failed tick consumed a NEW firing edge
  (a blanket rollback was itself a verifier-confirmed hazard: restoring `_fired=True`
  after a failed RESOLVED edge muted the next episode), plus a symmetric
  `rearm_signal_lost` on the exception path. Regression tests:
  `test_persist_failure_at_firing_edge_*`, `test_persist_failure_at_resolved_edge_*`,
  `test_failed_recovery_tick_still_rearms_signal_lost`. RESIDUAL (documented, not a
  regression — pre-existing, window now much narrower): store-committed-then-
  dispatch-failed followed by a process crash BEFORE the retry tick rehydrates
  `fired=True` and loses the page; closing it fully needs a durable
  undelivered-page ledger ⇒ still ASK-USER.
- **E12 `[ASK-USER — blocks INC6]` `PrometheusDataProvider` is unsafe to wire as-is
  (three related defects).** (i) staleness gate reads the instant-vector eval
  timestamp (~now), so frozen/stale series pass as fresh (`anomaly_data_provider.py:
  189-200` vs gate `anomaly_core.py:158`); (ii) rule `labels` are validated but never
  sent — unlabeled query + `results[0]` picks an arbitrary series (wrong-patient
  hazard with >1 series per metric, e.g. S1's synthetic job coexisting with a real
  exporter); (iii) `history_coverage_s` is hardcoded to the constructor's 24h constant,
  structurally defeating INSUFFICIENT_BASELINE on live data. All three must be fixed
  in the INC6 design (E2/E7's `[ASK-USER]` wiring decision) before this provider ever
  feeds the evaluator.
  `[FIXED 2026-07-05 — user-directed, 2 adversarial-verify rounds]`: (i) labels now
  compose an exact selector; >1 series ⇒ new `SignalLostReason.ambiguous_series`
  (gate fails closed, never first-pick); (ii) freshness = the OLDER of `timestamp()`
  scrape time AND (when `freshness_metric` is set) the measurement-time heartbeat
  gauge value — a verifier proved `timestamp()` alone cannot catch a frozen publisher
  behind a live /metrics endpoint (Prometheus re-stamps on every scrape);
  unconfirmable ⇒ `sample_ts=-inf` (stale in ANY budget); (iii) coverage measured via
  `count_over_time × scrape_interval_s`, unmeasurable ⇒ 0.0. Tests:
  `tests/anomaly_service/test_prometheus_provider.py` (13 cases). **INC6 wiring MUST
  pass `freshness_metric="neonatal_sim_last_update_timestamp_seconds"` and the real
  job's `scrape_interval_s`** — both default to safe-but-weaker behavior. The
  provider remains unwired (INC6 itself still ASK-USER).
- **E13 `[ASK-USER — break-glass semantics]`** (i) `start()`/`restart()`
  unconditionally clear a `force_signal_lost` hold (`anomaly_break_glass.py:140-142`)
  — an unrelated loop-cycle silently lifts an operator's "data untrustworthy" safety
  override; (ii) `force_shadow` sets `core.promoted=False` and nothing (incl.
  `recover()`) can restore it until process restart, while `service.status()` still
  reports promoted=True — status misrepresents paging posture. Break-glass is
  REPL-only today (E8), so exposure is low until mounted.
  `[FIXED 2026-07-05 — user-directed, verified clean by 2 independent skeptics]`:
  `start()`/`restart()` now PRESERVE a force_signal_lost hold (only explicit
  `clear_force_signal_lost`, or an explicit `recover()`, lifts it — loudly, with
  every lifted hold named in the audit summary); `force_shadow` records the
  pre-hold `core.promoted` and `clear_force_shadow`/`recover()` restore it (an
  undo, not a promotion — verifiers confirmed a never-promoted rule can never
  become paging via this path, LD-6 intact); `status()` gained
  `effective_promoted` so posture is never misreported. Tests:
  `test_break_glass_restart_preserves_signal_lost_hold`,
  `test_clear_force_shadow_restores_pre_hold_posture`.
- **E14 `[ASK-USER — coordinate with E7 edge-detection]` SIGNAL_LOST notification
  storm by design.** Dedup keys on `rule|state|ts` and the core emits a fresh-ts
  signal_lost event EVERY degraded tick ⇒ a promoted rule pages every tick
  (~5760/day at 15s) — alarm fatigue. Fix (key on rule|state|reason + page on edge)
  changes paging semantics ⇒ user sign-off.
  `[FIXED 2026-07-05 — user-directed, 2 adversarial-verify rounds]`: one shared
  `event_dedup_key()` (`anomaly_notifier.py`) used by BOTH dispatcher and the
  evaluator's delivery reconciliation (divergence would fabricate MISSED records);
  signal_lost keys drop the ts (rule|state|reason ⇒ one page per loss EPISODE per
  reason); re-arm is state-driven (`rearm_signal_lost` when a tick's verdict is not
  signal_lost — a quiet recovery emits no events) including on the exception path,
  and tracks a SET of keys per rule (round-1 verifiers proved a single-slot tracker
  leaked earlier-reason keys, permanently muting a future episode). Every deduped
  repeat is still logged (suppressed/deduped_page_once — never silent). Tests: the
  three `test_signal_lost_*` cases + `test_failed_recovery_tick_still_rearms_*` in
  `tests/anomaly_notification/`. Known cosmetic residual: ts-based FIRING keys of
  episodes that end via signal_lost (never resolved) accumulate in the in-memory
  dedup set — bounded growth only, can never mute (new firing edges get new ts keys).
- **E15 `[FIXED 2026-07-04]`** (i) demo driver could rmtree the REAL alert-state dir
  via the shared `ANOMALY_ALERT_STATE_DIR` env var — now a dedicated
  `ANOMALY_DEMO_STATE_DIR` + a hard refusal guard in `DemoDriver.load_scenario`
  (+ regression test); (ii) decision_flow `from_`/`from` aliasing was incomplete —
  chat path (`orchestrator.py` `_finalize`/patch), `/api/dashboard/validate`, store
  saves and LLM context now all dump `by_alias=True` (edges no longer silently vanish
  from chat-created dashboards); (iii) `for: "0s"` passed spec validation, nullifying
  the LD-3 sustain window — validator now requires a positive duration (+ golden
  test); (iv) firing / insufficient-baseline messages hardcoded "0.80x baseline" /
  "<24h" — now interpolate the rule's actual ratio/window; (v) neonatal publisher
  regenerated the full sim series every tick (O(uptime²)) — `NeonatalSim.sample_at`
  + equivalence test; (vi) background loop ran the synchronous tick (fsync + up to
  15s webhook retries) on the asyncio event loop — now `asyncio.to_thread`, ticks
  still strictly sequential.

---

## Harness limits (honesty clause)

What this institution CAN give a Sonnet-level session: reliable recall of decisions,
correct file targeting, mandatory verification gates, protected-path discipline, and
delegation patterns with built-in acceptance checks. What it CANNOT give:

1. **Taste/ambiguity judgments** ("is this threshold clinically sane?", "does this
   rewrite change the safety posture?"). Protocol: don't decide solo — lay out
   options + evidence and ask the user; or run 3 independent subagent judges and
   still get user sign-off for anything safety-adjacent. Say "low confidence" out loud.
2. **Novel security reasoning.** Verifying listed invariants is in scope; inventing or
   relaxing invariants is not — user only.
3. **Clinical correctness.** No model here is a clinician; every alerting surface
   stays decision-support-only (LD-7).
4. **Cross-session state.** Files are the only memory that survives. If it's not
   written to a file the next session loads or is told to read, it doesn't exist.

## Evidence sources
- First-hand verification by the review session (S1 process/ports, F4 check-ignore,
  E6(a) audit record + demo promoted flag, doc duplication, file sizes).
- 6 parallel deep-read agents (2026-07-03): session-log, docs-crosscheck, audit-trails,
  backend-code, parent-runtime, config/tests. Claims sourced only from an agent are
  phrased with their evidence (file:line / command) so any session can re-verify.
