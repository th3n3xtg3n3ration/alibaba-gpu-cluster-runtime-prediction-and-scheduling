# GitHub Copilot Instructions — GPU Scheduling Thesis

Full project guidelines live in [AGENTS.md](../AGENTS.md). This file adds VS Code–specific guidance.

---

## Agents Available in This Workspace

| Agent | Use When |
|-------|----------|
| `@phd-advisor` | Full project audits, academic rigor checks, experiment orchestration, thesis review |
| `@ml-experiment-agent` | Running or debugging a specific experiment (A–F) |
| `@thesis-reviewer` | Academic writing review, chapter structure, citation checks |
| `@code-reviewer` | Code quality, type hints, docstrings, refactoring suggestions |

---

## Slash Commands Available

| Command | Purpose |
|---------|---------|
| `/audit-project` | Comprehensive PhD-level project audit |
| `/run-experiment` | Execute a specific ML experiment (A–F) with checkpoint saving |
| `/evaluate-models` | Compare model performance across experiments |
| `/review-notebook` | Audit a single notebook for correctness and narrative |
| `/thesis-chapter-audit` | Review a chapter section for academic rigor |

---

## Quick Commands

```bash
# Run all tests
python -m unittest discover tests -v

# Full pipeline
bash scripts/run_all_experiments.sh

# PhD audit script
python scripts/phd_audit.py

# Check for hardcoded paths (quality gate)
grep -r "home\|Users\|/data/" src/ notebooks/ --include="*.py"
grep -rn "os\.path" src/ --include="*.py"
```

---

## Default Behavior Rules

1. Always read `configs/paths.yaml` before referencing any file path.
2. Load models from `results/models/` — never retrain unless explicitly asked.
3. Never overwrite `data/alibaba_cluster_trace/pai_job_no_estimate_100K.csv`.
4. After modifying any `src/` file, run: `python -m unittest discover tests -v`.
5. For all experiments, save checkpoints to `results/checkpoints/exp_{tag}_{model}.json`.
