---
name: developer-fix
agent: big-guy-developer-agent
description: Internal-only. Route a DeveloperTicket or developer instruction to Big guy.
developer_only: true
---

# /developer-fix

**Internal command.** Not reachable from the user chat flow. This
command routes a `DeveloperTicket` (or a direct developer instruction)
to `big-guy-developer-agent`.

## Arguments

- `ticket_id` (string, optional): id of a ticket in
  `backend/app/storage/tickets/`.
- `instruction` (string, optional): a direct developer instruction
  without a ticket (e.g. "add heatmap widget").

At least one of `ticket_id` or `instruction` is required.

## Expected output

Big guy does not emit user-facing JSON. It edits code and reports
progress to the developer. The orchestrator marks the ticket resolved
in storage when Big guy signals completion.
