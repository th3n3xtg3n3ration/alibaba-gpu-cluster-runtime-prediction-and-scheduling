---
description: "Full PhD-level audit of the entire project — tests, data, checkpoints, code quality, notebooks, and academic rigor. Produces a structured report."
agent: "agent"
tools: [read, search, execute, todo]
---

Conduct a comprehensive audit of this thesis project as a rigorous PhD advisor would.

Read [.github/tasks/audit-project.md](.github/tasks/audit-project.md) for the
exact step-by-step procedure, then execute each phase in order.

## What to Audit

1. **Environment Integrity** — All unit tests pass (11/11), deps importable
2. **Data Integrity** — Raw CSV present and valid, processed CSV has all required columns
3. **Experiment Checkpoints** — Every saved checkpoint has all 5 metrics (mae, rmse, r2, mape, mdae)
4. **Code Quality** — No `os.path`, no hardcoded absolute paths, no scratch/ imports, seeds=42
5. **Figures & Models Inventory** — All exp_a figures present, models saved
6. **Notebook Coherence** — EN/TR parity, clean imports from src/, checkpoint existence checks

## Report Format

End with a structured report:

```markdown
## Audit Report — [DATE]

### Executive Summary
[2-3 sentences]

### Status Matrix
| Phase | Status | Critical Finding |
|-------|--------|-----------------|
| Environment | ✅/⚠️/❌ | ... |
| Data Integrity | ✅/⚠️/❌ | ... |
| Checkpoints | ✅/⚠️/❌ | ... |
| Code Quality | ✅/⚠️/❌ | ... |
| Figures/Models | ✅/⚠️/❌ | ... |
| Notebooks | ✅/⚠️/❌ | ... |

### Critical Issues
1. ...

### Minor Issues
1. ...

### Top 3 Next Actions
1. ...
```

Be direct. Do not soften findings. A missed issue here becomes a thesis rejection.
