---
description: "Review a specific Jupyter notebook for correctness, narrative quality, and EN/TR parity. Reports issues grouped by severity."
agent: "agent"
argument-hint: "Notebook phase number (00-05) or filename"
tools: [read, search]
---

Review the notebook specified by the user.

## What to Check

Read [.github/instructions/notebooks.instructions.md](.github/instructions/notebooks.instructions.md)
first for the full convention spec, then verify:

### Correctness (Critical)
- [ ] All imports come from `src/` — no logic duplicated inline
- [ ] Project root discovery uses `Path.cwd().parents[1]` or equivalent
- [ ] Config loaded via `load_paths_config()` — no hardcoded paths
- [ ] Train/test split is chronological (sort by `submit_time`, never `shuffle=True`)
- [ ] Checkpoint loading includes `if ckpt_path.exists()` guard
- [ ] All figures saved to `results/figures/` before `plt.show()`

### Execution Integrity
- [ ] No `NameError` traps: every variable is defined before it's used
- [ ] Cell execution order is top-to-bottom (no forward dependencies)
- [ ] Would pass Kernel → Restart & Run All without errors

### Narrative Quality
- [ ] Each major section has: What was done / What was found / Why it matters
- [ ] Results are quantitative (exact numbers, not "improved significantly")
- [ ] Conclusions connect to the thesis research questions

### EN/TR Parity (if both versions exist)
- [ ] Same number of code cells
- [ ] Same code content (byte-for-byte in code cells)
- [ ] Turkish narrative is a translation, not a summary

## Report Format

```markdown
## Notebook Review: {filename}

### Critical Issues (must fix — will cause errors or wrong results)
1. ...

### Narrative Issues (should fix — affects thesis quality)
1. ...

### Parity Issues (EN vs TR mismatch)
1. ...

### Suggestions (optional improvements)
1. ...

### Verdict: PASS / FAIL
```

## Context

Notebook phase map: [.github/instructions/notebooks.instructions.md](.github/instructions/notebooks.instructions.md)
