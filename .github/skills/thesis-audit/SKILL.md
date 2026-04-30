---
name: thesis-audit
description: "Comprehensive PhD-level audit workflow for the GPU scheduling thesis. Use for: full project health check, pre-submission review, finding broken checkpoints, verifying experiment completeness, checking code quality, validating notebook coherence, verifying EN/TR parity. Runs 6 audit phases and produces a structured report."
argument-hint: "Optional: specific phase to audit (env, data, checkpoints, code, figures, notebooks) or 'full' for all"
---

# Thesis Audit Skill

## Overview

This skill conducts a rigorous, systematic audit of the entire thesis project across
6 phases. It is the equivalent of a PhD advisor's pre-submission review.

## When to Use

- Before any milestone commit or submission deadline
- After completing a new experiment (to verify checkpoint integrity)
- When tests fail and the root cause is unclear
- To verify EN/TR notebook parity after changes
- To check that no `os.path` or hardcoded paths have crept in

## Phase Overview

| Phase | Duration | What It Checks |
|-------|----------|---------------|
| 1. Environment | 5 min | deps importable, tests pass (11/11) |
| 2. Data | 5 min | raw CSV present, processed CSV has all columns |
| 3. Checkpoints | 5 min | all saved checkpoints have 5 metrics |
| 4. Code Quality | 5 min | no os.path, no hardcoded paths, seeds=42 |
| 5. Figures/Models | 2 min | inventory completeness, no empty files |
| 6. Notebooks | 10 min | EN/TR parity, imports, checkpoint guards |

## Procedure

### Step 1 — Load Context

Read these files before starting audits:
- [AGENTS.md](../../AGENTS.md) — quality gates and conventions
- [.github/context/results-summary.md](../context/results-summary.md) — expected state
- [.github/context/experiments-map.md](../context/experiments-map.md) — expected checkpoints

### Step 2 — Execute Each Phase

Follow the detailed commands in [.github/tasks/audit-project.md](../tasks/audit-project.md).

For each phase, mark status:
- ✅ PASS — criteria met
- ⚠️ WARNING — minor issue, doesn't block thesis
- ❌ FAIL — must fix before proceeding

### Step 3 — Generate Report

Use the report template from [.github/tasks/audit-project.md](../tasks/audit-project.md).

Populate with:
1. **Executive Summary** — 2-3 sentences on overall health
2. **Status Matrix** — one row per phase
3. **Critical Issues** — numbered, with file references and fix suggestions
4. **Minor Issues** — numbered, with optional fix suggestions
5. **Top 3 Next Actions** — prioritized by impact on thesis quality

### Step 4 — Remediate

For any ❌ FAIL:
- Fix-experiment issues → [.github/tasks/fix-experiment.md](../tasks/fix-experiment.md)
- Run-experiment gaps → [.github/tasks/run-experiment.md](../tasks/run-experiment.md)
- Code quality fixes → edit the offending `src/` file directly

### Step 5 — Re-verify

After any fix, re-run the failed phase command to confirm resolution.
Always end with: `python -m unittest discover tests -v` (must be 11/11).

---

## Report Template

```markdown
## Project Audit Report — {DATE}

### Executive Summary
[2-3 sentences: overall health, most critical issue, recommendation]

### Status Matrix
| Phase | Status | Finding |
|-------|--------|---------|
| Environment | ✅/⚠️/❌ | [detail] |
| Data Integrity | ✅/⚠️/❌ | [detail] |
| Checkpoints | ✅/⚠️/❌ | [detail] |
| Code Quality | ✅/⚠️/❌ | [detail] |
| Figures/Models | ✅/⚠️/❌ | [detail] |
| Notebooks | ✅/⚠️/❌ | [detail] |
| Academic Rigor | ✅/⚠️/❌ | [detail] |

### Critical Issues (must fix before submission)
1. [File/line ref] — [exact issue] — [recommended fix]

### Minor Issues (should fix)
1. ...

### Top 3 Next Actions (by impact on thesis quality)
1. [Highest impact] — [why] — [estimated effort]
```

---

## Research Guidance Principles

Apply these when reporting findings or advising on next steps:

1. **Correctness over novelty.** A correct, well-validated result on a narrow question
   beats a flashy but poorly-supported claim. The committee will probe every number.

2. **Limitations are strengths when acknowledged.** The 1,000-job scheduler sample is
   exactly the kind of limitation that must be stated prominently — not buried in a footnote.

3. **Ablation studies are mandatory.** "Sweep-line utilization features improve R²" is a
   claim that needs an ablation: train without those features and report the delta.

4. **DL underperforming trees is a result, not a failure.** Contextualize with the tabular
   data literature (Grinsztajn et al., 2022 — "Why tree-based models still outperform deep
   learning on tabular data").

5. **The 85.84% claim needs full-dataset validation.** The 1,000-job scheduler sample may
   not represent the full 100K trace distribution. Run on the full test set.

---

## Output Format

The audit skill returns a single, well-structured markdown report using the template above.
Copy it to a dated file (e.g., `docs/audit_2026-04-18.md`) if needed for records.
