---
name: generate-dashboard
agent: dashboard-spec-agent
description: Turn a DashboardIntent into a complete DashboardSpec JSON.
---

# /generate-dashboard

Invoked by the orchestrator after `helper-chat-agent` classifies the
user's request as a `DashboardIntent`.

## Arguments

- `intent` (object, required): the `DashboardIntent` payload.
- `catalog` (object, optional): Prometheus metric catalog.
- `previous_dashboard` (object, optional): the user's last spec, for
  stylistic continuity.

## Expected output

One JSON object:

- `DashboardSpec` — success.
- `DeveloperTicket` — the toolkit cannot satisfy the request.

Shapes defined in `.opencode/agent/dashboard-spec-agent.md`.
