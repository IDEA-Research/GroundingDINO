---
name: patch-dashboard
agent: patch-agent
description: Produce a minimal PatchSpec for an existing dashboard.
---

# /patch-dashboard

Invoked by the orchestrator after `helper-chat-agent` classifies the
user's request as a `PatchIntent`.

## Arguments

- `intent` (object, required): the `PatchIntent` payload.
- `dashboard` (object, required): the current validated `DashboardSpec`.
- `catalog` (object, optional): Prometheus metric catalog.

## Expected output

One JSON object:

- `PatchSpec`.

Shape defined in `.opencode/agent/patch-agent.md`.
