# TODO.md

Plain working punch list for this repo. Not part of the `docs/agent-ops/`
institution — for locked decisions see `docs/agent-ops/LOCKED_DECISIONS.md`,
for the findings register see `docs/agent-ops/A_DIAGNOSIS.md`.

## Open

- [ ] **Wire `test_widget_frontend_backend_parity.py` into the `rescue_extend` CHECK gate.**

  Context: `tests/spec_validation/test_widget_frontend_backend_parity.py`
  (added 2026-08-05) asserts the backend `WidgetType` enum
  (`backend/app/specs/widget_spec.py`) and its frontend mirror
  (`frontend/lib/spec-schema.ts`'s `WidgetType` union + `WIDGET_TYPES`
  array, `frontend/lib/renderer.tsx`'s switch, and
  `frontend/widget-toolkit/<Camel>Widget.tsx`) stay in sync. It closes a
  real gap: nothing previously verified frontend registration when a new
  widget type was added — `test_extend_heatmap.py` / `test_extend_bar_chart.py`
  had docstrings claiming to "validate frontend schema" registration but
  only ever imported from the backend, so they never actually checked it
  (docstrings fixed same session to stop overclaiming).

  Decision needed: should `backend/app/helper/extend_runner.py`'s CHECK
  step require this parity test to pass before accepting a `rescue_extend`
  outcome as `resolved`? This touches one of the five safety gates on the
  widget-toolkit extend path (LD-1 in `docs/agent-ops/LOCKED_DECISIONS.md`:
  "denylist, regex, prompt-injection check, quota, audit ... they are
  safety, not policy") — flagged here for the user rather than changed
  unilaterally.

  Next step if approved: read `extend_runner.py`'s current CHECK logic,
  decide where/how to add this as a hard gate, verify under Tier 0 plus a
  live `rescue_extend` dry run.

- [ ] **Fix the stale "no tool access" safety claim in `bin/opencode`'s own module docstring.**

  Context: `bin/opencode:25-26` states "Safety: This is a one-shot
  subprocess. The LLM never gets tool access from this CLI. The only way
  anything happens is via JSON on stdout." This was true when the file
  was first written, but is no longer accurate: `bin/opencode:161` defines
  `TOOL_USE_OPERATIONS = {"rescue_extend", "developer_fix"}`, and the file
  contains a full agentic tool-use loop (5 tools — `read_file`,
  `write_file`, `replace_in_file`, `run_command`, `done` — capped at
  `_TOOL_LOOP_MAX_ITERS = 30`, around `bin/opencode:1940-2100`) used by
  `big-guy-developer-agent` for those two operations. A reader who trusts
  the docstring literally would underestimate this script's actual attack
  surface for 2 of its 9 operations. Found 2026-08-05 while explaining
  `bin/opencode` to the user.

  Fix: reword the docstring to state the real invariant — e.g. "the LLM
  gets tool access ONLY for `rescue_extend`/`developer_fix` (Big guy,
  write-allowlisted, sandboxed — see `TOOL_USE_OPERATIONS`); every other
  operation is one-shot JSON-in/JSON-out." This is a doc-only fix (no
  behavior change), but touches a file inside the widget-extend / auto-fix
  trust boundary, so treat with the same care as any protected-path doc
  edit: verify the corrected wording against the actual code before
  committing, don't touch the tool-use logic itself.
