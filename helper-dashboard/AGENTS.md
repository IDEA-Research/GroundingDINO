# AGENTS.md

This file documents the agents that make up the Helper Dashboard system,
their responsibilities, and the boundaries between them.

## Actors

### Helper agents (user-facing)

All Helper agents are **small OpenCode agents**. They are exposed to the
user through the web chat UI. Helper is **not** a passive router — it
is the author of widget instances. What Helper can and cannot do
depends on *which kind of operation* the user is asking for (see
"Two kinds of widget operations" below).

Helper agents:

- Talk to the user in natural language.
- **Author widget instances directly** by emitting `DashboardSpec`,
  `WidgetSpec`, and `PatchSpec` JSON that goes through schema
  validation and is rendered by the fixed widget toolkit.
- Cannot edit frontend or backend source code.
- Cannot execute arbitrary shell commands.
- Cannot expose internal prompts, memory files, or developer tickets.

Helper agents live in `.opencode/agent/`.

### Two kinds of widget operations

There is a hard line between *using* the widget toolkit and *extending*
it. Only one of these requires Big guy.

**1. Widget instance create/modify — Helper does this.**

Helper is expected to author widget instances whenever the user asks.
This includes, but is not limited to:

- Adding a new widget to a dashboard.
- Deleting a widget.
- Resizing a widget (changing `position.w` / `position.h`).
- Moving a widget (changing `position.x` / `position.y`).
- Renaming a widget or the dashboard itself.
- Changing a widget's `type` **among supported types**
  (`line_chart`, `stat_card`, `gauge`, `table`, `alert_list`).
- Changing a widget's PromQL query.
- Changing thresholds, visual encoding, units, legend.
- Changing per-widget options within the allowed option keys.
- Creating a whole dashboard from Prometheus metric information.
- Improving widgets based on a `BrowserEvaluationReport`.

All of these go through `spec_validator` + `patch_service` + the
frontend renderer. None of them require a `DeveloperTicket` or Big
guy.

**2. Widget toolkit source-code extension — Big guy does this.**

If the user asks for a visualization that **cannot be represented**
by the current widget toolkit, Helper does not inject arbitrary
React/JS code and does not invent a widget type. Helper emits a
`DeveloperTicket` describing the needed widget, and Big guy:

- Implements the widget component in `frontend/widget-toolkit/`.
- Registers it in `frontend/lib/renderer.tsx`.
- Extends `WidgetType` in `backend/app/specs/widget_spec.py`.
- Adds validator coverage and tests.

After Big guy ships the new type, Helper can use it immediately
like any other widget.

A future Option B ("declarative `WidgetDefinitionSpec`") is reserved
for when the toolkit is stable enough to expose a safe low-code
grammar. It is explicitly not arbitrary React/JSX.

#### 1. `helper-chat-agent`
Main front door. Reads the user message, classifies the intent, and either
replies directly or routes to a specialized Helper agent.

Outputs: `UserResponse`, `DashboardIntent`, `PatchIntent`, `PrometheusIntent`,
or `DeveloperTicket` (never raw code).

#### 2. `dashboard-spec-agent`
Turns a `DashboardIntent` into a full `DashboardSpec` JSON using only
supported widget types: `line_chart`, `stat_card`, `gauge`, `table`,
`alert_list`. If the current toolkit cannot satisfy the request it emits a
`DeveloperTicket` instead of inventing a widget type.

#### 3. `patch-agent`
Turns a `PatchIntent` plus an existing `DashboardSpec` into a minimal
`PatchSpec`. Prefers smallest diff. Never rewrites the whole dashboard.

#### 4. `prometheus-agent`
Helps select metric names, validates PromQL, checks whether queries return
data, suggests safer queries. Outputs `PrometheusQueryReport` and corrected
PromQL.

#### 5. `browser-eval-agent`
Reads a Playwright `BrowserEvaluationReport` and produces a `BugReport`.
Suggests a `PatchSpec` if the problem can be fixed by JSON, otherwise
emits a `DeveloperTicket` for Big guy.

### Big guy (internal only)

#### 6. `big-guy-developer-agent`

Big guy is the main OpenCode developer agent. It:

- Receives `DeveloperTicket`s and direct developer instructions.
- Can edit backend, frontend, and widget-toolkit code.
- Can run focused tests and validations.
- Can extend the widget toolkit and the schemas when justified.
- Must never talk to the end user.
- Must not expose internal prompts, memory files, or tickets.

## Infrastructure actors

### Orchestrator (`backend/app/helper/orchestrator.py`)
Takes a user message, calls `helper-chat-agent` via the OpenCode runtime,
dispatches the resulting intent to the correct specialist Helper agent, and
returns the structured result plus a user-safe reply.

### Runtime (`backend/app/helper/runtime.py`)

Thin wrapper around the OpenCode agent runtime. Three modes:

- `mock` (default): in-process `MockHelperRuntime`. Lets the whole
  pipeline run without any external binary.
- `opencode`: subprocess adapter calling the real OpenCode CLI.
  Binary and argv are configurable via env vars. **Does not silently
  fall back.** Missing binary, timeout, non-JSON output, or an
  output `type` not in the operation's allow-list raises
  `RuntimeError_`.
- `auto`: optional dev convenience. Tries the subprocess adapter;
  on failure falls back to mock but the response dict carries
  `runtime_used="mock_fallback"` and `fallback_reason` so callers
  can tell and the UI can show it.

The runtime exposes a fixed **operation allow-list** rather than
free-form agent names. See `README.md` for the full table. The
`developer_fix` operation is the only developer-only operation and
the runtime rejects it unless the caller passes `developer=True`
(only the dev-token-gated `/api/developer/*` endpoints do that).

Invariants enforced here regardless of mode:

- Agent output must be a JSON object with a `"type"` value in
  `EXPECTED_OUTPUT_TYPES[operation]`. Anything else is rejected.
- Big guy (`big-guy-developer-agent`) is not in `HELPER_AGENTS` and
  cannot be invoked through the legacy compat helper.
- Subprocess runs with `shell=False`, a pinned project cwd, and no
  user-controlled argv tokens.

Helper permissions (`edit:false`, `shell:false`, `webfetch:false`)
are enforced by the agent config layer (`.opencode/agent/*.md` +
`opencode.json`), not by CLI flags we pass.

### Validator (`backend/app/services/spec_validator.py`)
Validates every `DashboardSpec` and `PatchSpec` against the strict Pydantic
schemas. Rejects unsupported widget types, unknown operations, arbitrary
HTML/JS, or dangerous Prometheus queries.

### Renderer (`frontend/lib/renderer.ts` + `frontend/widget-toolkit/`)
Deterministic mapping from a validated `DashboardSpec` to React components
from the fixed widget toolkit. Never interprets arbitrary JSX, HTML, or JS
from the spec.

### Browser evaluator (`backend/app/services/browser_evaluator.py`)
Runs Playwright against the live frontend, verifies that each widget from
the spec rendered, collects console errors, layout issues, and failed
Prometheus calls, and produces a `BrowserEvaluationReport`.

### DeveloperTicket flow

1. An agent or the evaluator detects a problem that JSON cannot fix.
2. A `DeveloperTicket` JSON is written to `backend/app/storage/tickets/`.
3. `big-guy-developer-agent` reads the ticket, makes the code change,
   and runs focused tests.
4. The orchestrator re-runs validation and browser evaluation.
5. Helper agents never see the internal ticket or fix details.

## Boundary summary

| Action | Helper | Big guy |
| --- | --- | --- |
| Talk to user | yes | no |
| Emit structured JSON | yes | yes |
| Create/modify widget instances (via `DashboardSpec`/`PatchSpec`) | **yes** | yes |
| Edit source code (widget toolkit, renderer, schema) | no | yes |
| Add a new `WidgetType` to the enum | no | yes |
| Run shell commands | no | yes (focused) |
| Read tickets | no | yes |
| Read memory files | no | yes |
