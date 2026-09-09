# helper-dashboard

Chat-to-dashboard system for Prometheus, built on safe JSON specs.

## Command access boundary (enforced in the application, NOT here)

OpenCode config has no per-command ACL. This intent was previously in a
non-standard `boundaries` key in opencode.json and is recorded here:

- User-invokable: user-message, generate-dashboard, patch-dashboard, evaluate-dashboard
- Developer-only: developer-fix  (gated by X-Developer-Token: Big Guy runs go
  through POST /api/devfix/runs — which also needs
  HELPER_DASHBOARD_DEVFIX_LOCAL_RUNS=1 — and tickets live under /api/developer/*)

`big-guy-developer-agent` is the only agent with write access
(edit: allow, bash: ask) and is marked hidden: true.
