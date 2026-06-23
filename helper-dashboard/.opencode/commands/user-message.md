---
name: user-message
agent: helper-chat-agent
description: Forward a user chat message to the Helper front-door agent.
---

# /user-message

This command is invoked by the backend orchestrator when a user sends
a chat message. It hands the raw user text (plus any relevant session
context: current dashboard id, recent preferences) to
`helper-chat-agent`.

## Arguments

- `message` (string, required): the user's message verbatim.
- `session_id` (string, required): current session id.
- `current_dashboard_id` (string, optional): if a dashboard is loaded.
- `session_context` (object, optional): recent preferences and history.

## Expected output

One JSON object, one of the shapes defined in
`.opencode/agent/helper-chat-agent.md`:
`UserResponse`, `DashboardIntent`, `PatchIntent`, `PrometheusIntent`,
or `DeveloperTicket`.
