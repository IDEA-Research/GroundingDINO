# SECURITY_BOUNDARIES.md

This document explains the invariants that keep Helper Dashboard safe
and why they exist. Big guy must not weaken any of these without an
explicit developer instruction.

## 1. Users never talk to Big guy directly

**Why.** Big guy has edit and shell permissions. If a user could
address Big guy, they could trigger arbitrary code changes, read
developer tickets, or exfiltrate internal prompts. Helper is the only
agent class exposed to users.

**Enforced by.**
- `opencode.json` marks `big-guy-developer-agent` as `internal: true`.
- Only `.opencode/commands/developer-fix.md` routes to Big guy, and it
  is listed under `boundaries.developer_only`.
- The `/api/developer/*` endpoints are separate from `/api/chat` and
  require a developer-only gate.

## 2. Helper agents cannot edit source code

**Why.** Helper agents are driven by user input. Granting them edit
permission on repository files would turn every chat message into a
potential code change.

**Enforced by (multiple layers — no single layer is the boundary):**

1. `.opencode/agent/*.md` frontmatter sets `edit: false`,
   `shell: false`, `webfetch: false` on every Helper agent.
2. `opencode.json` `permissions` block repeats the same constraints.
3. `backend/app/helper/runtime.py` exposes only a fixed
   **operation allow-list** (`user_message`, `generate_dashboard`,
   `patch_dashboard`, `prometheus_query`, `evaluate_dashboard`,
   `developer_fix`). Callers never pass free-form agent names.
4. `developer_fix` requires `invoke_operation(..., developer=True)`.
   Two callers set it: the dev-token-gated `/api/developer/*`
   endpoints, and the gated background auto-fix scheduler
   (`backend/app/helper/auto_fix.py`, see §5b) — never the chat
   surface directly.
5. We do **not** rely on CLI permission flags when invoking OpenCode
   as a subprocess. The safety model does not assume the CLI
   supports them.

> Note — "edit source code" means editing files in `backend/`,
> `frontend/`, `.opencode/`, `tests/`, or `docs/`. It does **not**
> mean creating or modifying dashboard widgets. See section 2b.

## 2b. Helper agents are the authors of widget instances

Helper is not a passive chat/routing layer. Creating and modifying
dashboard widgets is the *primary* job of Helper, and it does not
require Big guy.

Helper is allowed to:

- Add, remove, resize, move, rename widgets.
- Change a widget's `type` among supported types.
- Change PromQL queries, thresholds, encoding, and per-widget options
  (within the allowed option keys).
- Generate a full `DashboardSpec` from Prometheus metric information.
- Author a `PatchSpec` to fix a widget after a
  `BrowserEvaluationReport` shows a problem.

These operations are allowed because:

- They go through strict Pydantic schemas (`DashboardSpec`,
  `WidgetSpec`, `PatchSpec`).
- They go through `services/spec_validator.py` and
  `services/patch_service.py`.
- They are rendered by the fixed, deterministic widget toolkit — no
  spec field is ever interpreted as code.

Helper does **not** need a `DeveloperTicket` for any of these.

## 2c. Widget toolkit source-code extension still requires Big guy

> **Stale (LD-1, 2026-07-04):** out-of-toolkit widget requests now default to agent code-gen (in-app `rescue_extend`, or a dev-time Claude session) — not DeveloperTicket-and-wait. See `docs/agent-ops/LOCKED_DECISIONS.md` LD-1.

If the user asks for a visualization the current toolkit cannot
represent (service map, flame graph, …), Helper:

- Does **not** inject arbitrary React, JSX, HTML, or JS into the spec.
- Does **not** invent a new `WidgetType` string.
- Emits a `DeveloperTicket` describing the needed widget.

Big guy then:

- Implements the component in `frontend/widget-toolkit/`.
- Registers it in `frontend/lib/renderer.tsx`.
- Extends the `WidgetType` enum in
  `backend/app/specs/widget_spec.py`.
- Adds validator coverage and tests.

Only after the code ships can Helper use the new type.

## 3. Helper agents cannot run arbitrary shell commands

**Why.** Same reason as edit: user input must never reach the shell.

**Enforced by.** `shell=false` in `opencode.json` and in the runtime
wrapper.

## 4. DashboardSpec cannot contain arbitrary JavaScript, HTML, or React

**Why.** The frontend renders dashboards. If specs could carry raw
JS/HTML/React, any user could inject script into the UI via the Helper.
This is what separates "Helper authors widget instances" (safe) from
"Helper edits frontend code" (not allowed).

**Enforced by.**
- Strict Pydantic schemas in `backend/app/specs/` — no `raw_html`,
  `script`, `component`, `code`, `iframe`, `eval`, `onclick`,
  `onerror`, `html`, `jsx`, or `render` fields exist anywhere.
- Widget `type` is a fixed enum. New types require a code change
  authored by Big guy.
- The frontend renderer (`frontend/lib/renderer.tsx`) is a `switch`
  over `type`. Nothing is passed to `eval`, `new Function`, or
  `dangerouslySetInnerHTML` from the spec.
- `services/spec_validator.py` re-checks invariants (allowed types,
  allowed option keys, length limits, PromQL length limits).

Because these invariants hold, widget *instance* authoring by Helper
is safe even though Helper has full authorship rights over the spec.

## 5. Code changes require a DeveloperTicket or explicit developer instruction

**Why.** Keeps every code edit auditable. Each ticket is a JSON file
in `backend/app/storage/tickets/`. "Big guy changed something" should
always correspond to a ticket or a logged developer command.

**Enforced by.**
- `big-guy-developer-agent.md` explicitly requires a ticket id or
  direct developer message before editing.
- Orchestrator only writes tickets through
  `services/dashboard_store.save_ticket`.

## 5b. Auto-fix — tickets are consumed by Big guy automatically

Per LD-1/LD-2 and the user's 2026-07-06 directive, diagnostic tickets
never wait for a human. When the orchestrator persists one, the
background scheduler in `backend/app/helper/auto_fix.py` runs Big guy
(`developer_fix`, tool-using mode) against it. The chat turn is never
blocked; the user's validated dashboard is delivered regardless.

Boundaries (each fails closed, mirrored in `bin/opencode` as an
independent trust layer):

1. Enabled only in `opencode`/`auto` runtime modes by default;
   `HELPER_DASHBOARD_AUTO_FIX=0` is the kill switch.
2. Prompt-injection signature check over all user-influenced ticket
   text; daily quota (`HELPER_DASHBOARD_AUTO_FIX_DAILY_QUOTA`,
   default 10); single-flight (one run at a time).
3. Write scope is the render/product layer ONLY:
   `services/browser_evaluator.py`, `frontend/lib/renderer.tsx`,
   `frontend/lib/spec-schema.ts`, `frontend/widget-toolkit/`, and the
   matching test dirs. The clinical anomaly files
   (`services/anomaly_*`, `prometheus/`, `alert_rule_spec.py`,
   `api/anomaly.py`), the gates, the runtime, and `bin/opencode`
   itself are denied at the tool layer AND byte-verified afterwards —
   any difference reverts the entire run no matter what the report
   claims.
4. Full snapshot before the run; anything short of an honest
   `status="resolved"` report rolls every touched file back.
5. Every decision and attempt is appended to
   `backend/app/storage/auto_fix_audit/<date>.jsonl`; ticket status
   transitions (`open → in_progress → resolved|open`) record the
   attempt in `technical_evidence`.

Backend changes take effect on the next backend restart; frontend
changes after a bundle rebuild (same caveat as `rescue_extend`).

## 6. PatchSpec validation is not optional

**Why.** A patch path that bypasses validation would be a second way to
smuggle invalid specs in.

**Enforced by.**
- `patch_service.apply` calls `spec_validator.validate` on both the
  input patch and the resulting spec.
- Tests in `tests/patching/` assert this contract.

## 7. Validation protects the system

Validation layers are:

1. **Schema validation** — Pydantic parses the JSON into typed models.
2. **Semantic validation** — allowed widget types, allowed option keys,
   PromQL length + basic sanity, layout bounds.
3. **Patch validation** — operation type is in the allowed set, target
   ids exist, resulting dashboard still validates.
4. **Browser validation** — Playwright verifies that the spec actually
   renders as claimed.

No user payload is trusted until it has passed layers 1 and 2.

## 8. Internal artifacts are not exposed to users

Helper agents must not echo:

- Developer tickets.
- `HELPER_MEMORY.md` entries.
- Internal prompts, system messages, agent names, or file paths.
- Raw evaluation reports with debugging traces.

They may summarize outcomes ("I noticed a rendering problem and
logged a diagnostic") without exposing internals. Never claim a human
was notified — tickets are automated diagnostic records, and nothing
in the product waits on a person (LD-1/LD-2: no ticket-and-wait,
no human interrupt in the user path).

## 9. Runtime modes — no silent fallback in `opencode` mode

`backend/app/helper/runtime.py` supports three modes via the
`HELPER_DASHBOARD_OPENCODE` env var:

- `mock` (default) — in-process `MockHelperRuntime`. The deterministic
  stand-in used in CI and during bootstrap.
- `opencode` — subprocess calls to the real OpenCode CLI. **Strict**:
  a missing binary, timeout, non-JSON output, or an output `type` not
  in the operation's allow-list raises `RuntimeError_` and the
  orchestrator surfaces a safe error to the user.
- `auto` — optional development mode. Tries `opencode` first; on
  failure falls back to mock but the response dict is annotated with
  `runtime_used="mock_fallback"` and `fallback_reason`. Callers and
  the UI can see that a fallback happened.

**Why `opencode` must not silently fall back.** If production silently
returned mock output when the CLI failed, users would see plausible
answers that come from a different runtime than the operator
intended. Debugging would be near-impossible. Use `auto` only in
development. Use `opencode` in CI and production.

## 10. Subprocess invocation is constrained

When the runtime shells out to OpenCode:

- `subprocess.run(shell=False)` — always.
- cwd is pinned to the project root.
- Command tokens come from an operator-controlled template. Only
  `{bin}` and `{agent}` are substituted. `{agent}` comes from the
  operation allow-list — never from caller args.
- User input goes on **stdin** as a JSON object, never on argv.
- `HELPER_DASHBOARD_OPENCODE_TIMEOUT_SECONDS` bounds every call.

These invariants are covered by tests in
`tests/helper_runtime/test_runtime_modes.py`.

## 11. Review loop architecture

The pre-output review loop (enabled by `HELPER_DASHBOARD_PRE_OUTPUT_REVIEW`)
is a synchronous, blocking stage in the orchestrator pipeline:

1. `generate_dashboard` / `patch_dashboard` produces a draft spec.
2. The orchestrator saves a shadow draft (not yet persisted to disk).
3. The browser evaluator renders the draft.
4. The `review_rendered` operation (Helper) approves or proposes patches.
5. If Helper exhausts retries, `rescue_review` (Big guy, JSON-only)
   decides whether to auto-patch, ask the user, or file a ticket.

This ensures the user *never* sees a broken dashboard, trading
higher latency for total UI integrity. The UI shows a "Reviewing…"
overlay while this loop runs.

## 12. Verification strategy

Correctness is verified in tiers. Each tier exercises the safety
invariants at a different integration depth:

- **Tier 0** (`pytest tests/`) — unit + integration. Tests
  `test_opencode_json_marks_helper_permissions_and_gates_developer_fix`,
  `test_helper_agent_md_configs_lock_permissions`, and the
  `security_boundaries/` suite are the authoritative safety proof.
- **Tier 1** (`scripts/tier1_cli_check.py`) — real-model CLI
  verification. Confirms that `bin/opencode` honors the output
  contract (only typed JSON, never raw code) when talking to a live
  LLM.
- **Tier 2** (`scripts/tier2_backend_e2e.py`) — HTTP E2E. Confirms
  the developer gate returns 403 without the token and 200 with it,
  and that the chat surface cannot reach `developer_fix`.
- **Tier 3** (`scripts/live_full_demo.py`) — optional acceptance
  stress test. Not a safety proof; if it fails, Tiers 0–2 remain
  authoritative.

An attacker cannot weaken safety by slowing down Tier 3 or making
OpenRouter unavailable. All invariants are re-asserted in Tier 0
without any network dependency.


