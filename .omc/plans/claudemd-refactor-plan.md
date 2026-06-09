# CLAUDE.md Refactor Plan (Anthropic memory-guidance alignment)

**Date**: 2026-06-10
**Driver**: Align `/workspace/CLAUDE.md` with Anthropic's official CLAUDE.md guidance
(code.claude.com/docs/en/memory): keep build commands / conventions / project layout /
architecture / "always do X" rules; move **multi-step procedures** and **codebase-part-specific
rules** to `skill`/path-scoped `.claude/rules/`; strip outdated/historical asides; target <200 lines.

**Principle**: this is a **relocation**, not a deletion. Every fact that leaves the main file lands
in a path-scoped rule or a doc that auto-loads (rules) or is pointed to (docs). No information loss.

**Env check**: Claude Code 2.1.141 → path-scoped `.claude/rules/` (YAML `paths:` frontmatter) is
supported. `.claude/rules/` does not exist yet; `docs/dataset_reference.md` does not exist yet.

**Current**: 178 lines. **Target**: ~110–120 lines.

---

## Unit 0 — Add condensed coding principles (highest priority per user)

- **Why**: user-authored coding philosophy (Think-before-coding / Simplicity-first /
  Surgical-changes / Goal-driven). These are "always do X" rules → legitimate CLAUDE.md content.
  **Simplicity-first is flagged load-bearing** (user: "너가 불필요하게 복잡하게 구현하는 경향이 있어서").
- **Tension**: verbatim = ~40 lines, fights the <120-line goal AND violates Simplicity-first itself.
  → condense to ~14 lines (4 dense bullets), no meaning lost.
- **Action**:
  - Rename the existing `## Conventions` tail so working-style rules live under one umbrella:
    keep `### Implementation workflow (incremental + review-gated)` and add a sibling
    `### Coding principles` (the 4 condensed bullets) just above it.
  - De-duplicate: "Think before coding → plan first" overlaps existing "Plan the split first";
    phrase so they complement, not repeat.
- **Net**: +~14 lines (the one section we intentionally add; offset by Units 1–5).

## Unit 1 — Move baseline-clone audit checklist → `.claude/rules/baseline-clone.md`

- **Source**: CLAUDE.md lines 168–177 (`### Baseline / model-clone audit checklist`, the 8-point list).
- **Why**: it is a multi-step procedure only relevant when cloning/wiring a baseline — the canonical
  "move to a rule/skill" case in the docs.
- **Action**:
  - Create `.claude/rules/baseline-clone.md` with frontmatter
    `paths: ["src/baselines/**", "src/eval/**"]` and the 8-point checklist verbatim.
  - In CLAUDE.md, replace the section with a one-line pointer under Conventions:
    `- **Cloning a baseline?** See the 8-point silent-bug audit in `.claude/rules/baseline-clone.md` (auto-loads when editing `src/baselines/**` or `src/eval/**`).`
- **Net**: −9 lines in main file; checklist still auto-loads exactly when relevant.

## Unit 2 — Move fVLM internals → `.claude/rules/fvlm.md`

- **Source**: CLAUDE.md line 111 (fVLM adapter bullet) + line 112 (the giant organ-mask /
  preprocessing paragraph).
- **Why**: deep detail that only matters when touching `src/baselines/fvlm_*`.
- **Action**:
  - Create `.claude/rules/fvlm.md` with frontmatter
    `paths: ["src/baselines/fvlm_*.py"]` and both bullets verbatim (adapter contract + preprocessing).
  - In CLAUDE.md Architecture section, collapse to one bullet:
    `- **fVLM** is anatomy-aware (needs `(volume, organ_mask)`); adapter + TotalSegmentator-mask preprocessing details live in `.claude/rules/fvlm.md`.`
- **Net**: ~−10 lines; CT-CLIP bullet (110) stays (it's short and the primary backbone).

## Unit 3 — Compress Dataset reference → `docs/dataset_reference.md` + slim summary

- **Source**: CLAUDE.md lines 60–78.
- **Why**: exact byte counts, per-set provenance, deprecation history are *reference detail*, not
  per-session decision drivers. Docs are the right home; CLAUDE.md keeps only the traps + paths.
- **Action**:
  - Create `docs/dataset_reference.md` with the full current content (the 3 latent sets, full census,
    split provenance, etc.).
  - In CLAUDE.md keep a ~8-line block: the **doubled-path trap**, raw CT root, latent shape
    `[4,120,120,64]`, the canonical toy v2 path + proxy_test=1304, the storage/GPU conventions, and a
    pointer: `Full detail: docs/dataset_reference.md`.
- **Net**: ~−10 lines.

## Unit 4 — Strip historical / deprecation asides (state-only)

- **Source**: scattered — Win condition (81), Upper bound (88–89), Architecture (105), and any
  "removed 2026-06-09", "superseded because…", "Was … ⇒" parentheticals.
- **Why**: docs say remove outdated content; the changelog belongs in git/memory, and these asides
  spend context + dilute adherence. Keep **current state**, drop the **how-we-got-here**.
- **Action**: rewrite each to state only what is true now. Where a caveat is still load-bearing
  (e.g. "self-measured baselines pending re-measurement"), keep one short clause; cut the rest.
  Use `<!-- ... -->` only if a maintainer note is genuinely needed (HTML comments are stripped from context).
- **Net**: ~−8–12 lines, mostly from parentheticals.

## Unit 5 — Light trim of Common commands

- **Source**: CLAUDE.md lines 114–150.
- **Why**: keep — officially endorsed. But one-off smoke snippets (the inline `python -c` hydra-compose
  loop, redundant single-test examples) can be thinned.
- **Action**: keep the daily-driver commands (pytest, hydra `--cfg job`, train/eval, vlm3d_runner
  dry-run, env sanity); drop or shorten the verbose inline `python -c` blocks, pointing to
  `docs/vlm_baselines_runbook.md` where they already live.
- **Net**: ~−6 lines. (Lowest priority; skip if it hurts discoverability.)

---

## Order & gating
Execute one unit at a time, stop for review after each (per CLAUDE.md Implementation workflow rule):
1 → 2 → 3 → 4 → 5. Units 1–2 (create rules) are highest value + lowest risk. Unit 5 optional.

## Verification per unit
- After each move: `grep -n` the moved heading is gone from CLAUDE.md and present in the new file.
- After all units: `wc -l CLAUDE.md` (~110–120), and confirm `.claude/rules/*.md` frontmatter parses
  (valid YAML, `paths:` list). Optionally `/memory` to confirm rules are listed as loaded.

## Out of scope
- `.claude/CLAUDE.md` (OMC orchestration file) — untouched.
- Auto-memory files under `~/.claude/projects/.../memory/` — untouched.
- `[[wikilink]]` memory references in CLAUDE.md (Non-goals/Compute) — left as-is.
