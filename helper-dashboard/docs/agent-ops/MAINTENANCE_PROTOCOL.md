# MAINTENANCE_PROTOCOL.md — how future sessions safely update the institution files

> The institution = `CLAUDE.md` (both levels), `docs/agent-ops/*`, `.claude/agents/*`,
> and the auto-memory at
> `/home/cluster/.claude/projects/-etc-GroundDino-GroundingDINO-GroundingDINO/memory/`.
> This file tells you (a future session, probably Sonnet-level) what you may change on
> your own, what needs the user, where lessons go, and when to compact.

## 0. Universal rules

- **Backup before edit.** Before modifying any institution file:
  `cp <file> /etc/GroundDino/GroundingDINO/GroundingDINO/helper-dashboard/docs/agent-ops/backups/<name>.bak-$(date +%Y%m%d)`
  — that one absolute path is the backup dir for ALL institution files, including the
  repo-root CLAUDE.md and the auto-memory files. One backup per file per day is enough
  (don't stack `.bak-….bak-…`).
- **Additive over destructive.** Prefer appending a correction over rewriting a
  section. Never delete a rule you don't understand — mark it
  `[DISPUTED <date>: reason]` and ask the user.
- **One change, one reason.** Every institution edit states, in the changed file's
  change-log section, one line: date, what, why, session trigger. If the file has no
  change-log section, create one at the bottom — except `LESSONS.md` and
  `A_DIAGNOSIS.md`, where the dated entries / `[FIXED]` markers ARE the change log.
- **No invention.** You may record what happened; you may not invent new policy. New
  policy comes from the user (you may draft, they approve).

## 1. What you MAY change without asking (self-serve)

| Change | Where | Condition |
|---|---|---|
| Record a new lesson/pitfall | `LESSONS.md` (format below) | A real mistake happened in your session AND you verified the fix works |
| Fix a factually wrong path/command/name | any agent-ops file EXCEPT `LOCKED_DECISIONS.md` (§2 governs it entirely) | You verified the correct value by running/lsing it; note old→new in change log |
| Mark a rule as stale | any agent-ops file EXCEPT `LOCKED_DECISIONS.md` | Add `[STALE? <date>: evidence]` next to it; don't remove it |
| Update "current state" facts (branch, test counts, build status) | `LETTER_TO_FUTURE_SESSIONS.md` §Current state | You measured it this session |
| Add a memory entry | auto-memory dir | Follows the memory frontmatter format; also add the one-line pointer in `MEMORY.md` |
| Tighten a delegation template | `DELEGATION_TEMPLATES.md` | Only adding missing context/acceptance items, not removing constraints |

## 2. What REQUIRES explicit user approval first

- Anything in `LOCKED_DECISIONS.md` (add, edit, remove, "reinterpret").
- Any clinical-safety invariant, alert threshold, shadow-mode promotion, or the
  protected-paths list in `CLAUDE.md`.
- Deleting or git-untracking audit data (`anomaly_build_audit/` etc.) or session logs.
- Changing rule precedence, or which file owns a topic.
- Any `.gitignore` change, branch merge, force-push, or history rewrite.
- Deleting a memory file or a backup.
- Big-bang rewrites of README/AGENTS/docs (opportunistic single-section fixes are fine
  under §1).

Ask by presenting: the exact diff you propose, why, and what breaks if unchanged.
Where §1 and §2 both seem to apply, **§2 wins**.

## 3. Where lessons go (single funnel)

New file: `docs/agent-ops/LESSONS.md`. Append-only, newest at top. Entry format:

```
### YYYY-MM-DD — ⟪imperative one-liner, e.g. "Don't trust git for frontend/lib/"⟫
- What happened: ⟪1-3 lines, concrete⟫
- Root cause: ⟪1 line⟫
- Rule to follow now: ⟪1 line, checkable — a command, a path, a criterion⟫
- Evidence: ⟪path:line / command output / commit⟫
```

Routing guide — where does a given insight belong?
- Mistake made + fix verified → `LESSONS.md` (here).
- User states a preference/decision → draft entry for `LOCKED_DECISIONS.md`, get
  approval, then also add a short auto-memory entry pointing at it.
- Fact about the environment (paths, services, quirks) → `LETTER_TO_FUTURE_SESSIONS.md`
  §"Current state", or CLAUDE.md if every session needs it.
- In-app Helper/Big-guy product mistakes → `docs/HELPER_MEMORY.md` (existing file,
  keep its format).
- Anything only relevant to the current conversation → nowhere; let it die.

## 4. Compaction (when files get fat)

Token budgets (check with `wc -l` when you edit the file):
- `CLAUDE.md` (dashboard): hard cap **150 lines**. Overflow → move detail to the
  owning agent-ops file, leave a one-line pointer.
- `LESSONS.md`: at **300 lines**, compact: merge duplicate lessons, promote
  universally-true ones into CLAUDE.md (≤1 line each), archive the raw pre-compaction
  file to `docs/agent-ops/backups/`, and note the compaction in the change log.
- `LOCKED_DECISIONS.md`: no cap, but entries are terse; history detail belongs in the
  change log, not in the entry.
- Auto-memory `MEMORY.md` index: keep ≤20 lines; consolidate with the
  `consolidate-memory` skill when it exceeds that.
- `A_DIAGNOSIS.md` and `LETTER_TO_FUTURE_SESSIONS.md` are snapshots — do NOT grow them
  indefinitely. When a finding is fixed, mark it `[FIXED <date> — how]` in place.
  If more than half the findings are fixed, propose (to the user) writing a fresh
  diagnosis and archiving this one.

## 5. Verification after any institution edit

1. Re-read your own diff (`git diff -- <file>`). Check: no rule now contradicts
   another; no `⟪placeholder⟫` left unfilled; every path/command you added actually
   exists (run `ls` / `--help`).
2. The 60-second weak-reader test: could a session that has read NOTHING but this file
   misexecute the rule? If a rule needs context to be safe, inline the context or the
   pointer.
3. Commit institution changes on the current feature branch with message prefix
   `agent-ops:` so they're findable (`git log --oneline --grep agent-ops`), following
   the canonical commit policy in `CLAUDE.md` §Git rules (explicit paths only; if the
   user hasn't authorized commits this session, hand them the exact command instead).
   Never stage `docs/agent-ops/backups/` (may contain PII copies).

## 6. If this protocol itself seems wrong

Do not edit §2's list or §0 on your own authority — those are the load-bearing walls.
Everything else in this file may be improved under §1 rules (backup, change log,
verified facts only).

---

### Change log
- 2026-07-03: created by the Fable 5 institutional-review session.
- 2026-07-04: adversarial-review fixes: §1 now excludes LOCKED_DECISIONS.md (+"§2
  wins" tiebreak); §0 backup path made absolute; §0 change-log rule handles files
  without a change-log section; §5.3 defers to CLAUDE.md's canonical commit policy.
