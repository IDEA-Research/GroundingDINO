# Helper Dashboard

A web system where users chat with small **Helper** agents to create, modify,
evaluate, and debug Prometheus-based dashboards.

Helper agents never edit code. They produce safe structured JSON
(`DashboardSpec`, `PatchSpec`, `BugReport`, `DeveloperTicket`) which the
backend validates, the frontend renders with a fixed widget toolkit, and a
Playwright evaluator checks in a real browser. If a problem needs real code
changes, a `DeveloperTicket` is routed to **Big guy**, the internal developer
agent.

## Roles

| Role | Who | Can talk to | Can author widget instances | Can edit source code |
| --- | --- | --- | --- | --- |
| User | End user | Helper agents only | no (asks Helper) | no |
| Helper | Small user-facing OpenCode agents | User, orchestrator | **yes** (via `DashboardSpec` / `PatchSpec`) | no |
| Big guy | Internal developer agent | Developer, DeveloperTickets | yes | yes |

Users never talk to Big guy directly. Big guy never becomes the chatbot.
Helper is the widget-instance author; Big guy is only pulled in when
the **widget toolkit itself** needs a new kind of visualization.
See `docs/SECURITY_BOUNDARIES.md` for the rationale and enforcement.

> **Stale (LD-1, 2026-07-04):** out-of-toolkit widget requests now default to agent code-gen (in-app `rescue_extend`, or a dev-time Claude session) — not DeveloperTicket-and-wait. See `docs/agent-ops/LOCKED_DECISIONS.md` LD-1.

## Data flow

```
User
  -> Web Chat UI (frontend/components/chat)
  -> helper-chat-agent
  -> dashboard-spec-agent / patch-agent / prometheus-agent
  -> DashboardSpec / PatchSpec JSON
  -> Backend validation (services/spec_validator.py)
  -> Frontend renderer (widget-toolkit/*)
  -> Dashboard preview
  -> browser-eval-agent (Playwright)
  -> BugReport / DeveloperTicket
  -> big-guy-developer-agent (code fix, internal only)
```

## Tech stack

- Backend: FastAPI + Pydantic
- Frontend: Next.js + React + Tailwind
- Charts: Recharts
- Browser evaluation: Playwright
- Data source: Prometheus HTTP API (with mock fallback)
- Agent runtime: OpenCode
- Storage: local JSON files (`backend/app/storage/`)

## Layout

```
helper-dashboard/
  README.md              # this file
  AGENTS.md              # role description of all agents
  opencode.json          # command -> agent routing
  docs/                  # design, architecture, memory, session log, boundaries
  .opencode/
    agent/               # agent prompt/config files
    commands/            # slash commands that route to agents
  backend/app/           # FastAPI backend, schemas, services, storage
  frontend/              # Next.js frontend, widget toolkit
  tests/                 # validation, patching, prometheus, browser, security
```

## Verification strategy (four tiers, 0-3)

Correctness is verified in four tiers, from deterministic-fast to
optional-stress. **Tier 0 (pytest) is the mandatory, authoritative
proof; Tiers 1 and 2 are supplementary live checks. Tier 3 is an
acceptance/stress test and is expected to fail in under-resourced or
disconnected environments.**

### Tier 0 — unit + integration tests (always run these)

```bash
cd helper-dashboard
python3 -m pytest tests/ -q
```

Runs 100+ tests across schema validation, patching, security
boundaries, runtime modes, review loop, saved dashboards, draft
shadow, browser-evaluator fallback, and the backend-down UX. No
network, no browser required. This is the canonical correctness
proof.

### Tier 1 — CLI-only real-model verification (~1–2 min)

```bash
cd helper-dashboard
set -a; source .env; set +a            # load OPENROUTER_API_KEY
python3 scripts/tier1_cli_check.py
```

Exercises `bin/opencode` directly against the configured LLM
(Kimi K2.6 / GLM 5.1 / whatever is in `.env`) for:

- `user_message` → DashboardIntent / UserResponse / …
- `generate_dashboard` → DashboardSpec (schema-revalidated locally)
- `patch_dashboard` → PatchSpec
- `review_rendered` → ReviewDecision

No backend, no frontend, no Playwright. Confirms the provider is
reachable, the subprocess protocol is intact, and each agent is
producing well-typed JSON.

To run without an LLM key:
```bash
OPENCODE_LLM_PROVIDER=heuristic python3 scripts/tier1_cli_check.py
```

### Tier 2 — backend-only HTTP E2E (~5 s in mock, ~1–3 min with real LLM)

```bash
cd helper-dashboard
python3 scripts/tier2_backend_e2e.py                   # mock, review off
TIER2_LLM=opencode python3 scripts/tier2_backend_e2e.py        # real LLM
TIER2_REVIEW=1 python3 scripts/tier2_backend_e2e.py            # review ON
```

Runs the FastAPI app in-process (TestClient) and drives the full
orchestrator pipeline through real HTTP calls:

- Create a dashboard via `/api/chat/message`.
- Receive the save-prompt and answer it.
- Confirm the library got the entry.
- Apply a patch via chat.
- Check that the developer endpoint is gated (403 without token,
  200 with correct token, 403 with wrong token).
- In `opencode` mode: verify unsupported-widget requests produce a
  DeveloperTicket or ClarificationRequest.
  (Historical behavior — superseded by the LD-1 code-gen default;
  `rescue_extend` now handles out-of-toolkit requests.)

**This is the primary reliable live verification.** No browser, no
Playwright, no resource pressure.

### Tier 3 — optional full UI stress demo (5–15 min, resource-heavy)

```bash
cd helper-dashboard
set -a; source .env; set +a
python3 scripts/live_full_demo.py
```

Starts uvicorn + `next start` + Playwright and drives the full UI
flow through Chromium against the real LLM with the review loop on.
Expected to be slow on a reasoning model, and can fail cleanly for
reasons that are **not** system bugs:

- OpenRouter credits exhausted / rate-limited / slow.
- Jetson / sandboxed CPU can't sustain Chromium + LLM + backend.
- Review loop legitimately iterating when Helper's first attempt
  doesn't satisfy the user intent.

The script detects these conditions and prints guidance. If this
tier fails, **Tiers 0, 1, and 2 remain the authoritative answer**
to "does the system work?".

## Demo scenario — API observability

The canonical showcase is an **API observability dashboard** (HTTP
request rate, 5xx error rate, p95/p99 latency, availability, CPU,
memory, firing alerts). A golden reference lives at
`backend/app/samples/api_observability.json`.

You can exercise the full scenario without a browser. Both tiers
below are deterministic and fast.

### Scenario Tier 1 — CLI-only

```bash
cd helper-dashboard
set -a; source .env; set +a                         # real LLM
python3 scripts/tier1_api_observability.py
# or, no LLM key needed:
OPENCODE_LLM_PROVIDER=heuristic \
  python3 scripts/tier1_api_observability.py
```

Verifies that the configured LLM can:
- generate an API-observability `DashboardSpec` (≥ 6 widgets,
  allowed types, `histogram_quantile(...) by (le)` for percentiles,
  `rate(http_requests_total[5m])` for request rate, at least one
  threshold);
- produce a `reorder_widgets` patch for "move critical widgets to
  the top";
- produce an `update_widget` patch on a latency widget for "make
  latency more prominent";
- produce an `update_widget` patch on the error-rate widget with a
  `0.02` threshold for "add error-rate threshold at 2%";
- produce an `update_widget` patch on the CPU widget with memory
  PromQL for "change CPU chart to memory chart".

### Scenario Tier 2 — backend HTTP E2E

```bash
cd helper-dashboard
python3 scripts/tier2_api_observability.py           # mock (default)
TIER2_LLM=opencode python3 scripts/tier2_api_observability.py   # real LLM
```

Drives the FastAPI app via TestClient through the full user journey:
1. `POST /api/chat/message` to build the API observability dashboard.
2. Assert widget quality (count, types, promql patterns, thresholds).
3. Answer the save-prompt with `"api obs"`.
4. Confirm `GET /api/saved_dashboards/` lists the entry.
5. Run three sequential patches via chat (`move critical …`,
   `add error-rate threshold at 2%`, `change CPU chart to memory chart`).
6. `GET /api/dashboard/<id>` and verify every patch landed in the
   stored spec.

### Golden sample

`backend/app/samples/api_observability.json` is a hand-written
reference matching the 8-widget composition the heuristic and the
tuned agent prompts converge on. Tests in
`tests/demo_scenarios/test_api_observability.py` assert this sample
validates, uses correct PromQL patterns, and carries thresholds on
all critical widgets.

## Running the app

### Agent validation retry

User-originated `generate_dashboard` and `patch_dashboard` outputs go
through a bounded validation-retry loop in the orchestrator. If an LLM
reply fails schema or semantic-contract checks for the user's request,
the orchestrator retries the same agent with a structured feedback
message up to `HELPER_AGENT_MAX_ATTEMPTS` times (default `3`, clamped
to `[1, 5]`).

- **Scope**: only user-originated generation and patch. The review
  loop calls the runtime directly and does not stack additional
  retries (prevents combinatorial explosion).
- **Short-circuit**: deterministic providers (`mock`,
  `mock_fallback`, `heuristic`) do not retry — if the first attempt
  fails, the loop stops and the failure is surfaced.
- **Feedback is data, not a prompt tweak**: errors are injected via
  `args["_prior_errors"]` and `args["_prior_feedback_message"]`, which
  the CLI forwards into the user message. No `.opencode/agent/*.md`
  files are modified at runtime.
- **Warnings on retry-accepted responses**: if any check accepts only
  after >1 attempt, the response includes a warning and the Tier 1
  script reports `PASS (WARN) attempts_used=N/M` so quality drift is
  visible.
- **Artifacts**: when `HELPER_DASHBOARD_PERSIST_RETRY_LOGS=1`, each
  loop writes its attempt log to
  `backend/app/storage/retry_logs/<ts>_<op>.json` (git-ignored).
  Secrets are masked before serialization.

### Running

Fastest path — one command, both processes:

```bash
./scripts/dev.sh            # backend :8000 + frontend :3000
./scripts/dev.sh --mode opencode
./scripts/dev.sh --mode mock
```

Manual two-terminal path:

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Browser evaluator (optional):

```bash
cd backend
python -m playwright install chromium
```

## Secrets

Copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY` (or
the key env for whichever LLM you wire into `bin/opencode`). `.env`
is git-ignored. Rotate the key if it's ever pasted into chat logs
or shared over insecure channels.

## OpenCode runtime modes

Helper and Big guy agents are invoked through
`backend/app/helper/runtime.py` via a fixed **operation allow-list**:

| Operation | Agent | Reachable from |
| --- | --- | --- |
| `user_message` | `helper-chat-agent` | `/api/chat` |
| `generate_dashboard` | `dashboard-spec-agent` | `/api/chat` |
| `patch_dashboard` | `patch-agent` | `/api/chat` |
| `prometheus_query` | `prometheus-agent` | `/api/chat` |
| `evaluate_dashboard` | `browser-eval-agent` | `/api/chat` and `/api/evaluate` |
| `review_rendered` | `helper-review-agent` | internal (review loop) |
| `rescue_review` | `big-guy-developer-agent` (JSON-only, no edits) | internal (review loop) |
| `developer_fix` | `big-guy-developer-agent` (full perms) | **dev-only** `/api/developer/*` |

Callers never pass free-form agent names. `developer_fix` requires
`invoke_operation(..., developer=True)` and is not reachable from
the user chat flow.

Three runtime modes, selected by environment variables:

| Env var | Default | Meaning |
| --- | --- | --- |
| `HELPER_DASHBOARD_OPENCODE` | `mock` | `mock` — always use the in-process `MockHelperRuntime`. `opencode` — must use the real subprocess adapter; any failure raises. `auto` — optional dev convenience; may fall back to mock, but the response carries `runtime_used="mock_fallback"` and `fallback_reason`. |
| `HELPER_DASHBOARD_OPENCODE_BIN` | `opencode` | Path or name of the CLI binary. |
| `HELPER_DASHBOARD_OPENCODE_CMD` | `["{bin}","run","--agent","{agent}","--json"]` | JSON array for the subprocess argv. Only `{bin}` and `{agent}` are substituted. `{agent}` always comes from the allow-list, never from user input. |
| `HELPER_DASHBOARD_OPENCODE_TIMEOUT_SECONDS` | `30` | Seconds per invocation. |

**Why `opencode` mode does not silently fall back.** If you've asked
for the real runtime, silent mock fallback would mean the user sees
plausible output from the wrong runtime. That makes debugging
impossible and quietly hides production problems. Use `auto` if you
want graceful degradation during development; use `opencode` in CI
and production.

**Subprocess safety:**

- `shell=False`, always.
- cwd is pinned to the project root.
- Command tokens come from the operator-set template; `{agent}` is
  substituted from the operation allow-list; user input never
  reaches argv.
- User args go on **stdin** as a JSON object `{operation, agent, args}`.
- stdout must be a single JSON object with a `type` field whose value
  is in the operation's expected set (see
  `EXPECTED_OUTPUT_TYPES` in `runtime.py`).
- Timeout and non-zero exit are structured errors with `kind` tags.

**Permissions are not enforced by CLI flags.** We do not assume
OpenCode supports permission flags. The enforcement layers are:

1. `.opencode/agent/*.md` frontmatter (`edit: false` etc. on Helpers).
2. `opencode.json` `permissions` block (defense in depth).
3. Backend operation allow-list (this file).
4. Backend schema validation (`services/spec_validator.py`).
5. Developer-only endpoint protection (`HELPER_DASHBOARD_DEV_TOKEN`).
6. No direct user path to `developer_fix`.

Big guy is **never** callable through `invoke_operation` from the
chat flow. He only runs via developer tooling.

## Pre-output review loop

Before a `DashboardSpec` or `PatchSpec` reaches the user, the
orchestrator runs a synchronous review:

```
draft -> render (headless browser) -> Helper review (3 attempts max)
       -> if still bad, Big guy rescue (one-shot, JSON-only)
       -> approved | clarify | ticket
```

- `review_rendered` (Helper small model): approves, emits a minimal
  `PatchSpec`, or escalates.
- `rescue_review` (Big guy, bigger model, edit=false, shell=false):
  emits a final `PatchSpec`, asks the user clarifying questions, or
  files a `DeveloperTicket`.

Gated by `HELPER_DASHBOARD_PRE_OUTPUT_REVIEW`:

- Unset + mode `opencode`/`auto` → review is **on**.
- Unset + mode `mock` → review is **off** (keeps CI fast).
- `=1` / `=0` → force override.

The `ChatResponse` includes a `review_trail` array the UI renders in
a dedicated "Review" tab, so you can see Helper↔Big guy↔renderer
interactions in real time.

## Saved dashboards

After a dashboard passes review, Helper asks in chat:
*"Would you like to save this dashboard to your library? Reply with
a name, or 'no' to skip."*

Saving writes to `backend/app/storage/saved_dashboards/*.json` and
is exposed at:

- `POST /api/saved_dashboards/save` — `{dashboard_id, name, tags?}`
- `GET /api/saved_dashboards/` — list summaries
- `GET /api/saved_dashboards/{saved_id}` — full entry
- `DELETE /api/saved_dashboards/{saved_id}`

At session start, Helper gets the library summaries in its prompt so
it can semantically suggest reuse.

## The opencode CLI

`bin/opencode` is the real executable. Given `{operation, agent, args}`
on stdin it returns a single JSON object on stdout.

Providers (selected by `OPENCODE_LLM_PROVIDER`):

- `openrouter` (default when `OPENROUTER_API_KEY` is set) — calls
  `https://openrouter.ai/api/v1/chat/completions`. Models picked from
  `OPENCODE_HELPER_MODEL` and `OPENCODE_BIG_GUY_MODEL` (default
  `moonshotai/kimi-k2-thinking`).
- `heuristic` — deterministic rules identical to the in-process mock.
  Useful in CI and when no key is configured.

This CLI never gets tool access from within a single invocation — it
reads stdin, talks to the LLM, writes stdout, exits. The "Helper
can't edit code" invariant is preserved by the protocol itself, not
by CLI flags.

## MVP success criteria

1. User asks Helper to create a dashboard.
2. Helper returns a `DashboardSpec` JSON.
3. Backend validates the spec.
4. Frontend renders the dashboard safely.
5. User asks Helper to modify a widget.
6. Helper returns a `PatchSpec` JSON.
7. Backend applies the patch and re-validates.
8. Browser evaluator inspects the rendered page.
9. A rendering bug can create a `DeveloperTicket`.
10. Big guy fixes the bug internally without exposing anything to the user.
