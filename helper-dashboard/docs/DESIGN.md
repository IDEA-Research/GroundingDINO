# DESIGN.md

## Goal

Let a non-technical user produce a working Prometheus dashboard by chatting
with a Helper, with Big guy doing all the real engineering work in the
background when code changes are necessary.

## Product slice

A user lands on a page with three panels:

1. **Chat** (left) — they talk to Helper, ask for dashboards and edits.
2. **Preview** (center/right) — the current `DashboardSpec` renders live.
3. **Inspector** (bottom/right) — the raw JSON spec, validation status,
   and browser evaluation status.

Users only see Helper. They never see Big guy, prompts, or tickets.

## Core concepts

### DashboardSpec
Declarative JSON description of a dashboard. Includes title, layout,
widgets, variables, refresh interval. Fully validated by the backend.
Rendered by a fixed widget toolkit — **never** interpreted as code.

### WidgetSpec
One widget inside a dashboard. Constrained to a fixed set of types
(authoritative list: the `WidgetType` enum in
`backend/app/specs/widget_spec.py`, 9 types):
`line_chart`, `stat_card`, `gauge`, `table`, `alert_list`, `pie_chart`, `bar_chart`, `heatmap`, `decision_flow`. Adding a new
**type** is an agent code-gen job (LD-1). Adding or modifying widget **instances** of
existing types is a Helper job.

### Two kinds of widget operations

> **Stale (LD-1, 2026-07-04):** out-of-toolkit widget requests now default to agent code-gen (in-app `rescue_extend`, or a dev-time Claude session) — not DeveloperTicket-and-wait. See `docs/agent-ops/LOCKED_DECISIONS.md` LD-1.

This is the rule that keeps the product useful and the system safe.

- **Widget instance authoring** (create/modify/delete a widget using
  the existing toolkit) — **Helper's job**. Goes through schema
  validation, `PatchSpec` application, and re-validation. Does not
  need a `DeveloperTicket`, does not touch source code.
- **Widget toolkit extension** (a new kind of visualization that the
  toolkit can't represent) — **Big guy's job**. Triggered by a
  `DeveloperTicket`. Adds a new component, renderer case, schema
  enum entry, validator branch, and tests. Only then can Helper use
  the new type.

The safety boundary is *not* "Helper can't touch widgets". It is
"Helper can't edit source code and can't invent new widget types".

### PatchSpec
Minimal, structured patch to an existing dashboard. Operations are
typed (`add_widget`, `remove_widget`, `update_widget`, `update_dashboard`,
`reorder_widgets`). No raw JSON-path strings executed blindly.

### BugReport
Structured record of something wrong with a rendered dashboard. Comes
from the browser evaluator or from an automated validator. Can be
accompanied by a suggested `PatchSpec`.

### DeveloperTicket
The only channel into Big guy. Written by Helper agents or the evaluator
when the fix requires real code changes. Contains a summary, the
user-visible symptom, technical evidence, and a requested action. Stored
in `backend/app/storage/tickets/`.

## Why this shape

- **Safety**: JSON specs are easy to validate. React code is not.
- **Separation**: Helper is optimized for conversation; Big guy for code.
  Mixing them leads to Helper editing source or Big guy leaking internals.
- **Debuggability**: Every artifact (spec, patch, bug, ticket, evaluation)
  is a file on disk that can be replayed.
- **Incremental toolkit**: New widgets ship as new components; the Helper
  can only use what exists.

## Rendering model

```
DashboardSpec (validated)
   -> frontend/lib/renderer.ts
   -> <WidgetLayout>
        for each widget in spec.widgets:
          lookup widget.type in widget-toolkit
          pass widget.query + widget.options as props
```

The renderer is a `switch` over `widget.type`. Unknown types are ignored
and surfaced as validation errors in the inspector. No `eval`, no `new
Function`, no `dangerouslySetInnerHTML` on user-supplied content.

## Data sources

- Primary: Prometheus HTTP API (`/api/v1/query`, `/api/v1/query_range`).
- Fallback: deterministic mock generator in
  `backend/app/prometheus/client.py` when Prometheus is unreachable or
  the query fails validation. The mock is clearly marked so the UI can
  show a "using mock data" badge.

## Auto-debug loop

1. Spec arrives or is patched.
2. `spec_validator` validates shape.
3. `prometheus_validator` checks queries (optional, best-effort).
4. Frontend renders.
5. `browser_evaluator` runs Playwright and produces a
   `BrowserEvaluationReport`.
6. `browser-eval-agent` turns the report into either:
   - a `PatchSpec` (fix by JSON), or
   - a `DeveloperTicket` (fix by code).
7. Big guy handles the ticket; the loop restarts.

## Non-goals (initial)

- Multi-user auth.
- Dashboard versioning history.
- Native Grafana import/export.
- Live metric search via an embedded LLM.
- Arbitrary plugin loading.
