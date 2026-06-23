---
name: rescue-extend
description: Big guy in tool-using "extend the widget toolkit" mode. User-triggered code edit, gated by the five-layer M5 extend_gate.
agent: big-guy-developer-agent
operation: rescue_extend
---

# rescue-extend

This slash command is **not** intended for human invocation. The
backend orchestrator and review loop reach `rescue_extend` whenever:

  1. The specialist agent (or Big guy in rescue_review) identified
     that the failure was a missing widget type, AND
  2. The safety layers in `backend/app/helper/extend_gate.py` all
     pass:

       - `widget_type` matches `[a-z][a-z0-9_]{2,31}` and is not in
         the denylist (script, iframe, eval, exec, system, shell, …),
       - the original user message contains no prompt-injection
         signature (ignore previous, you are now X agent, drop table, …),
       - today's quota has not been exhausted
         (HELPER_DASHBOARD_AUTO_EXTEND_DAILY_QUOTA, default 50),
       - audit log write succeeds (best-effort, doesn't actually block).

The widget toolkit is a **cache of pre-built widgets** so the agent
codes less — it is NOT a fence on what can be built. Extend runs by
default; there is no opt-in flag.

When all safety gates pass, `bin/opencode` enters tool-using mode and
invokes Big guy with the OpenRouter function-calling protocol. The
tools and their allow-lists are defined in `bin/opencode` under
`# === BEGIN __TOOL_USE__ ===`.

See `big-guy-developer-agent.md` § Mode C for the actual agent
behavior contract.

## Operator notes

- Extend is **enabled by default**. No env flag required.
- To **raise/lower** the daily quota:
    `export HELPER_DASHBOARD_AUTO_EXTEND_DAILY_QUOTA=N`
- Audit log lives at:
    `backend/app/storage/extend_audit/<YYYY-MM-DD>.jsonl`
- To **revert** a failed extension manually (snapshot rollback runs
  automatically on failure; this is for the rare case where a
  resolved-but-broken extension shipped):
    `git diff HEAD~1` to see what Big guy added, then `git reset --hard HEAD~1`.

## Per-widget_type write allow-list

For `widget_type = <X>` (snake_case → `<XCamel>Widget` PascalCase):

    backend/app/specs/widget_spec.py
    backend/app/specs/widget_schema_doc.py
    frontend/lib/spec-schema.ts
    frontend/lib/renderer.tsx
    frontend/widget-toolkit/<XCamel>Widget.tsx
    tests/spec_validation/test_extend_<X>.py

Anything else is rejected by `_path_allowed_for_write` in
`bin/opencode`.

## Shell allow-list

    pytest, npm, node, npx, python3, python3.10

shell=False always; argv elements only; metacharacters rejected.
