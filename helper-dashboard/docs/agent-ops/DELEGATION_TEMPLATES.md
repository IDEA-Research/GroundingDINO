# DELEGATION_TEMPLATES.md — fill-in prompts for delegating to subagents

> For the main session (any model) when spawning subagents (Agent tool / Task tool).
> Copy the template, replace every `⟪…⟫`, delete inapplicable optional blocks.
> Rules that make delegation work at Sonnet level:
> 1. A subagent knows NOTHING about your conversation. Every template is
>    self-contained: absolute paths, exact commands, exact deliverable shape.
> 2. Always state acceptance criteria the subagent can CHECK ITSELF (a command whose
>    output proves done-ness), and a report format you can consume without re-reading
>    the files it touched.
> 3. Never delegate THE DECISION on LOCKED_DECISIONS topics, safety-invariant changes,
>    or anything requiring user judgment — collect evidence, then ask the user yourself.
>    (Advisory judge panels per CLAUDE.md working agreement 4 are fine; the decision
>    still needs user sign-off.)
> 4. Read-only tasks → subagent type `Explore` (or `general-purpose`). Code-writing
>    tasks in the anomaly subsystem → `anomaly-builder`. Everything else → `general-purpose`.
> 5. If the result matters, verify one spot-check claim yourself (open one cited file
>    at the cited line) before acting on the report.

Base paths for all templates:
`REPO=/etc/GroundDino/GroundingDINO/GroundingDINO`, `DASH=$REPO/helper-dashboard`.

---

## 1. SEARCH / LOCATE (read-only)

```
You are a read-only search agent in ⟪DASH or REPO⟫. Do not modify files.

GOAL: Find ⟪what: e.g. "every code path that reads prometheus/client.py query output"⟫.

CONTEXT YOU NEED: ⟪1-3 lines: why this is needed; known aliases/synonyms; where it is
NOT (already checked)⟫.

SEARCH PLAN (do all, not just the first hit):
- Grep for ⟪primary symbols/strings⟫ and ⟪alternate spellings⟫.
- Check these likely dirs first: ⟪dirs⟫; then sweep the rest of ⟪scope⟫.
- For each hit, read enough surrounding lines to classify it (definition / call site /
  test / doc mention / dead code).

ACCEPTANCE: You are done only when a second, different search strategy (different
keyword or structural pattern) finds no NEW hits.

REPORT FORMAT (your final message, nothing else):
- `path:line` — one-line classification, for every hit
- 2-4 sentence answer to the GOAL question
- "Not found in: ⟪scopes swept⟫" for negative space
- Confidence: high/medium/low + what would raise it
```

## 2. IMPLEMENT (write code)

```
You are a coding agent in DASH (=/etc/GroundDino/GroundingDINO/GroundingDINO/helper-dashboard).
Branch: uiagent_with_newwidget. Read DASH/CLAUDE.md first and obey it, especially the
protected clinical paths and the rule precedence.

TASK: ⟪one paragraph: exact behavior to add/change, user-visible outcome⟫.

SCOPE: You may edit: ⟪explicit file list or dirs⟫. Do NOT touch: ⟪files/dirs⟫ —
if the task seems to require it, STOP and report back instead of proceeding.
⟪If task touches backend/app/services/anomaly_* or prometheus/*: "This is
patient-safety code: the 5 invariants in CLAUDE.md are inviolable; if the task
conflicts with one, stop and report."⟫

DESIGN CONSTRAINTS: ⟪existing patterns to follow, e.g. "copy the store pattern from
services/alert_state_store.py"; schema strictness (extra='forbid'); no new deps
without approval⟫.

ACCEPTANCE (all must pass; run them, paste real output):
1. `cd $DASH && python3 -m pytest tests/ -q` — no new failures (record before/after counts).
2. ⟪task-specific check: new test you must write, or exact command + expected output⟫.
3. `git diff --stat` matches SCOPE (no stray files, no *.log/*.pid/audit JSONL staged).

REPORT FORMAT:
- What changed: file-by-file, one line each
- Test evidence: exact commands + pass/fail counts (before → after)
- Anything you noticed but did NOT fix (leave it; just list it)
- Blocked/partial? Say so explicitly — never report done with failing checks.
Do NOT commit; leave the working tree for the supervisor to review and commit.
```

## 3. REFACTOR (behavior-preserving)

```
You are a refactoring agent in DASH. Read DASH/CLAUDE.md first.
Branch: uiagent_with_newwidget.

TASK: Refactor ⟪target⟫ to ⟪goal: e.g. "extract X into its own module"⟫ with ZERO
behavior change.

HARD RULES:
- Behavior-preserving means: same public function signatures unless listed below, same
  test results, same API responses. Allowed interface changes: ⟪list or "none"⟫.
- Forbidden: touching ⟪protected paths⟫; reordering/reformatting code you didn't
  otherwise change; "improving" validation, error messages, or security checks in
  passing; editing >⟪N⟫ files (if needed, stop and report).
- In files >1000 lines (helper/runtime.py, helper/orchestrator.py): Grep the symbol,
  read only the needed ranges ± callers, edit surgically.

ACCEPTANCE:
1. `python3 -m pytest tests/ -q` — identical pass count before vs after (run BOTH,
   record both numbers).
2. `git diff --stat` — only the agreed files.
3. Grep proves no caller still references the old ⟪symbol/path⟫.

REPORT: files changed + why; before/after test counts; every call site you updated
(path:line); anything that smelled wrong but was out of scope.
```

## 4. RESEARCH / INVESTIGATE (read-only, answer a question)

```
You are a research agent. Read-only — do not modify files.

QUESTION: ⟪the precise question, phrased so a yes/no/value answer is possible⟫.
WHY IT MATTERS: ⟪1-2 lines so you can judge relevance of what you find⟫.

EVIDENCE SOURCES (check all that apply):
- Code: ⟪dirs/files⟫
- Runtime state: ⟪commands, e.g. `systemctl status medical-webapp`, `curl -s
  localhost:9090/api/v1/label/__name__/values`⟫ — read-only commands only.
- History: `git log --oneline -20 -- ⟪path⟫`. USER_SESSION_LOG.md is ~90% test-fixture
  noise (A_DIAGNOSIS T4) — usually skip it; last ~200 lines only if truly needed.
  Audit JSONL (storage/anomaly_*audit/) has known honesty defects (A_DIAGNOSIS E6:
  fake MISSED records, canned reasoning) — corroborate every audit-derived claim
  against code/git before reporting it.
- Docs: ⟪specific docs⟫ — but docs may be stale; code and runtime state outrank them
  (see DASH/docs/agent-ops/LOCKED_DECISIONS.md precedence).

ACCEPTANCE: Every claim in your answer carries evidence (path:line, command output,
or commit hash). Unverifiable ⇒ mark "UNVERIFIED". Never fill gaps with plausible
guesses — an explicit "could not determine" is a valid finding.

REPORT: Answer first (≤5 sentences). Then evidence list. Then "what I could not
verify and how the user could". Confidence: high/medium/low.
```

## 5. REVIEW (adversarial check of work)

```
You are a fresh-context adversarial reviewer. You did NOT write this work; your job is
to find real problems, not to approve it. Read-only.

WORK UNDER REVIEW: ⟪diff range `git diff ⟪base⟫..HEAD`, or file list, or doc paths⟫.
INTENT: ⟪what the work claims to do, 2-3 lines⟫.

CHECK, in order:
1. Correctness: does the code/doc actually do what INTENT says? Trace the main path
   by reading it, don't trust names.
2. Safety: any weakening of the invariants in DASH/CLAUDE.md §Clinical-safety? Any
   secret printed/committed? Any new consumer of prometheus/client.py that ignores
   `source`?
3. Contradictions: does it conflict with DASH/docs/agent-ops/LOCKED_DECISIONS.md or
   with other rules in the same document set? Quote both sides.
4. Weak-reader test: any instruction/naming a Sonnet-level model could misread? Any
   wrong path, wrong command, wrong tool name? VERIFY paths/commands actually exist
   (ls / --help), don't eyeball.
5. Tests: are the claimed test results reproducible? Re-run the cheapest one.

For each finding: severity (blocker/major/minor), evidence (path:line + quote),
concrete fix suggestion. Zero findings is a valid outcome — do not invent nitpicks to
seem thorough. End with: verdict `approve` / `fix-then-approve` / `reject`.
```

---

## Choosing scale (how many agents, when to fan out)

- Single question, known location → no subagent; just look.
- Sweep-shaped work (find all X, audit all Y) → 1 Explore agent per independent slice;
  slices = directories or modality (by-name / by-content / by-history).
- Anything you'll act on irreversibly, or any "all clear" claim → add one independent
  REVIEW agent (template 5) before acting; for safety-relevant changes use 3 reviewers
  and require unanimity for `approve`.
- If two agents disagree, don't average: read the disputed evidence yourself and decide,
  or escalate to the user with both sides quoted.
