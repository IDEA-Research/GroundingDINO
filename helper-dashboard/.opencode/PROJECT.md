# helper-dashboard

Chat-to-dashboard system for Prometheus, built on safe JSON specs.

## Command access boundary (enforced in the application, NOT here)

OpenCode config has no per-command ACL. This intent was previously in a
non-standard `boundaries` key in opencode.json and is recorded here:

- User-invokable: user-message, generate-dashboard, patch-dashboard, evaluate-dashboard
- Developer-only: developer-fix  (gate behind X-Developer-Token in /api/developer/*)

`big-guy-developer-agent` is the only agent with write access
(edit: allow, bash: ask) and is marked hidden: true.
