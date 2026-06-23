# ARCHITECTURE.md

## Components

```
              +--------------------------+
              |  User (browser)          |
              +------------+-------------+
                           |
                           v
              +--------------------------+
              |  Next.js frontend        |
              |   - chat panel           |
              |   - dashboard preview    |
              |   - JSON inspector       |
              +------------+-------------+
                           | HTTP (JSON)
                           v
+--------------------------+-----------------------------+
|  FastAPI backend (backend/app)                         |
|                                                        |
|  api/chat.py        api/dashboard.py                   |
|  api/evaluate.py    api/developer.py                   |
|       \               |           |                    |
|        v              v           v                    |
|  helper/orchestrator.py                                |
|    - routing.py  (user msg -> Helper agent)            |
|    - runtime.py  (OpenCode runtime wrapper)            |
|    - memory.py   (session + helper memory)             |
|    - debug_loop.py (validate -> render -> eval -> fix) |
|                                                        |
|  services/spec_validator.py                            |
|  services/patch_service.py                             |
|  services/browser_evaluator.py (Playwright)            |
|  services/dashboard_store.py                           |
|  services/prometheus_validator.py                      |
|                                                        |
|  prometheus/client.py   (HTTP + mock fallback)         |
|  prometheus/catalog.py  (metric name catalog)          |
|                                                        |
|  specs/*.py             (Pydantic models)              |
|                                                        |
|  storage/                                              |
|    dashboards/   *.json                                |
|    tickets/      *.json                                |
|    evaluation_reports/ *.json                          |
+--------------------------+-----------------------------+
                           |
                           v
              +--------------------------+
              |  OpenCode runtime        |
              |   - helper-chat-agent    |
              |   - dashboard-spec-agent |
              |   - patch-agent          |
              |   - prometheus-agent     |
              |   - browser-eval-agent   |
              |   - big-guy (internal)   |
              +--------------------------+
```

## Request lifecycles

### Chat message (user creates a dashboard)

```
POST /api/chat { message, session_id }
 -> orchestrator.handle_user_message
 -> runtime.invoke(helper-chat-agent)        # classify intent
 -> routing.dispatch(intent)
      if DashboardIntent:
        runtime.invoke(dashboard-spec-agent) -> DashboardSpec
 -> spec_validator.validate(DashboardSpec)
 -> dashboard_store.save(DashboardSpec)
 -> return { user_reply, dashboard_id, spec }
```

### Patch message (user edits a widget)

```
POST /api/chat { message, session_id, dashboard_id }
 -> helper-chat-agent -> PatchIntent
 -> patch-agent + existing spec -> PatchSpec
 -> spec_validator.validate(PatchSpec)
 -> patch_service.apply(PatchSpec, existing_spec) -> new spec
 -> spec_validator.validate(new_spec)
 -> dashboard_store.save(new_spec)
 -> return { user_reply, patch, spec }
```

### Evaluation

```
POST /api/evaluate { dashboard_id }
 -> browser_evaluator.run(dashboard_id) -> BrowserEvaluationReport
 -> storage/evaluation_reports/<id>.json
 -> browser-eval-agent -> BugReport | PatchSpec | DeveloperTicket
 -> if DeveloperTicket: storage/tickets/<id>.json
 -> return { report, bug_report?, patch?, ticket? }
```

### Developer fix (Big guy only)

```
GET /api/developer/tickets              # internal endpoint
POST /api/developer/tickets/{id}/ack    # internal endpoint
 -> big-guy-developer-agent takes the ticket
 -> edits code (widget toolkit, renderer, schema, service)
 -> runs focused tests
 -> marks ticket resolved
```

The `/api/developer/*` endpoints are mounted behind a developer-only
guard. They are **not** reachable from the user-facing chat flow.

## Safety boundaries

Helper permissions are enforced at the **configuration** layer, not at
the subprocess-CLI layer (we do not assume the OpenCode CLI supports
permission flags). In order of priority:

1. `.opencode/agent/*.md` frontmatter sets `edit/shell/webfetch: false`
   on every Helper agent.
2. `opencode.json` `permissions` block repeats the same constraints.
3. `backend/app/helper/runtime.py` exposes a fixed **operation
   allow-list**. Callers never pass free-form agent names. Only the
   `developer_fix` operation can reach Big guy, and it requires the
   explicit `developer=True` flag.
4. **`backend/app/helper/retry_loop.py`** wraps user-originated
   `generate_dashboard` and `patch_dashboard` in a bounded
   validate-then-retry loop. Schema errors and semantic contract
   violations become structured feedback injected into the next
   attempt (via `args["_prior_errors"]` / `args["_prior_feedback_message"]`).
   This layer NEVER weakens validation — it only gives a non-deterministic
   LLM a bounded budget to produce a correct answer. Review-loop
   dispatches do NOT route through the retry loop (Decision 2c in
   Phase 8) to prevent combinatorial LLM-call blow-up.
5. `backend/app/services/spec_validator.py` rejects unknown widget
   types, arbitrary HTML/JS, forbidden option keys, and forbidden
   fields regardless of the runtime source.
6. `backend/app/services/patch_service.py` only accepts typed
   operations from the `PatchSpec` schema.
7. The frontend renderer uses a fixed widget toolkit. Nothing in the
   spec is ever passed to `eval`, `new Function`, or
   `dangerouslySetInnerHTML`.
8. The `/api/developer/*` endpoints are gated by
   `HELPER_DASHBOARD_DEV_TOKEN`. They are the only path that ever
   triggers `developer_fix`.

See `docs/SECURITY_BOUNDARIES.md` for the rationale.

## Storage

Initial MVP uses local JSON files under `backend/app/storage/`. Each
artifact is one file named by its id. Schema is the Pydantic model.
Moving to SQLite/Postgres is a Big guy task once volume justifies it.

## Testing

Correctness is proved in three tiers. Only Tiers 0–2 are
authoritative. Tier 3 is a stress/acceptance test.

### Tier 0 — unit + integration (`tests/`)

- `tests/spec_validation/` — schema edge cases.
- `tests/patching/` — patch application correctness.
- `tests/prometheus_queries/` — client + validator behavior.
- `tests/browser_evaluation/` — Playwright harness + backend-down UX.
- `tests/security_boundaries/` — Helper cannot edit/shell; DashboardSpec
  cannot carry JS/HTML.
- `tests/helper_runtime/` — runtime modes, operation allow-list,
  subprocess safety, output-contract validation.
- `tests/review_loop/` — review loop + draft shadow fix.
- `tests/saved_dashboards/` — library CRUD + save-prompt flow.
- `tests/opencode_cli/` — CLI heuristic provider end-to-end.

Run: `python3 -m pytest tests/ -q`.

### Tier 1 — `scripts/tier1_cli_check.py`

CLI-only verification against the configured LLM. Fast, no
services started. Proves the subprocess protocol and each agent's
JSON output shape.

### Tier 2 — `scripts/tier2_backend_e2e.py`

In-process FastAPI (TestClient) drive of the full orchestrator
pipeline via real HTTP calls. The **primary reliable live
verification**. Covers create/patch/save/library/developer-gate.
Supports mock and opencode modes with review optionally on.

### Tier 3 — `scripts/live_full_demo.py`

Optional full UI stress demo. Starts uvicorn, `next start`, and
Playwright. Resource-heavy. Failures in this tier do NOT invalidate
the system's correctness proof — the script reports diagnostics and
points back to Tiers 0–2.
