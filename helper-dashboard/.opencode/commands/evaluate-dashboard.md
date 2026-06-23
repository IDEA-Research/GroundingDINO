---
name: evaluate-dashboard
agent: browser-eval-agent
description: Interpret a Playwright BrowserEvaluationReport and produce a BugReport, PatchSpec, or DeveloperTicket.
---

# /evaluate-dashboard

Invoked by the orchestrator after
`services/browser_evaluator.py` finishes running Playwright against
the rendered dashboard.

## Arguments

- `report` (object, required): the `BrowserEvaluationReport` JSON.
- `dashboard` (object, required): the `DashboardSpec` that was
  rendered.

## Expected output

One JSON object, one of: `BugReport`, `PatchSpec`, or
`DeveloperTicket`. Shapes defined in
`.opencode/agent/browser-eval-agent.md`.
