# LESSONS.md — verified mistakes and the rules that prevent them

> Append-only, newest at top. Format and routing rules: `MAINTENANCE_PROTOCOL.md` §3.
> Only add an entry when a REAL mistake happened AND the fix is verified.

### 2026-07-03 — Don't trust git as ground truth for `frontend/lib/`
- What happened: INC4 widget wiring (`renderer.tsx`, `spec-schema.ts`) was "done" but
  absent from git; parent `.gitignore:21` pattern `lib/` silently ignores the dir. A
  session checking completeness via git would have re-implemented existing code.
- Root cause: Python-packaging ignore pattern in the root `.gitignore` collides with a
  frontend dir three levels down.
- Rule to follow now: for `helper-dashboard/frontend/lib/`, check the filesystem
  (`ls`, `cat`), never `git log/show`, until the user approves the `.gitignore` fix
  (`A_DIAGNOSIS.md` F4).
- Evidence: `git check-ignore -v frontend/lib/renderer.tsx` → `.gitignore:21:lib/`
  (that is the REPO-ROOT `.gitignore`, not `helper-dashboard/.gitignore` — git prints
  the source path relative to the repo root).

### 2026-07-03 — A number in memory can be read backwards; anchor semantics, not just values
- What happened: the anomaly threshold was at one point misread as
  `value < 0.20 × baseline` instead of the correct `value < 0.80 × baseline` (20%
  relative DROP). The memory entry now spells out both the formula and the misreading.
- Root cause: "20%" is ambiguous between "drops TO 20%" and "drops BY 20%".
- Rule to follow now: when recording thresholds, always write the full comparison
  expression plus a one-line plain-language gloss; when reading one, restate it in
  your plan and check against `LOCKED_DECISIONS.md` LD-3.
- Evidence: auto-memory `project_anomaly_detection_decisions` ("NOT `< 0.20×`").

### 2026-05-29 — Refusing out-of-toolkit widgets is wrong here
- What happened: a user request for a `sankey` widget was bounced with a
  DeveloperTicket-style "flagged for the team" refusal, per the written docs. The user
  explicitly corrected: the toolkit is a cache, not a cap — agents should code the
  missing widget.
- Root cause: docs encode an outdated policy; agent trusted docs over asking.
- Rule to follow now: out-of-toolkit widget requests default to code-gen
  (`LOCKED_DECISIONS.md` LD-1); docs saying otherwise are stale (`A_DIAGNOSIS.md` F1).
- Evidence: auto-memory `feedback_toolkit_philosophy`.
