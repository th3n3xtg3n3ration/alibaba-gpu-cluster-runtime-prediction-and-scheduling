---
name: notebook-reviewer
description: "Review a Jupyter notebook for correctness, narrative quality, execution integrity, and EN/TR bilingual parity. Use for: auditing a specific notebook, checking before thesis submission, verifying notebook can execute cleanly, finding NameError traps or hardcoded path issues."
argument-hint: "Notebook phase number (00-05) or notebook filename"
---

# Notebook Reviewer Skill

## Overview

This skill performs a thorough code and narrative review of thesis Jupyter notebooks,
applying both software engineering standards and academic writing quality checks.

## When to Use

- Before marking a notebook phase as "complete"
- After making significant changes to a notebook
- When a notebook fails Kernel → Restart & Run All
- To verify EN/TR parity after translating a section

## Pre-conditions

Before reviewing, load:
- [.github/instructions/notebooks.instructions.md](../instructions/notebooks.instructions.md) — full conventions
- The notebook file itself (EN version first, TR version second if checking parity)

## Procedure

### Step 1 — Identify Notebook

Locate the notebook from the phase number:

| Phase | EN File | TR File |
|-------|---------|---------|
| 00 | `notebooks/en/00_data_preparation.ipynb` | `notebooks/tr/00_veri_hazirlama.ipynb` |
| 01 | `notebooks/en/01_data_overview.ipynb` | `notebooks/tr/01_veri_ozeti.ipynb` |
| 02 | `notebooks/en/02_workload_analysis.ipynb` | `notebooks/tr/02_is_yuku_analizi.ipynb` |
| 03 | `notebooks/en/03_feature_engineering.ipynb` | `notebooks/tr/03_ozellik_muhendisligi.ipynb` |
| 04 | `notebooks/en/04_runtime_prediction_models.ipynb` | `notebooks/tr/04_calisma_zamani_tahmin_modelleri.ipynb` |
| 05 | `notebooks/en/05_scheduler_evaluation.ipynb` | `notebooks/tr/05_gorev_zamanlayici_degerlendirme.ipynb` |

### Step 2 — Correctness Checks (Critical)

For each code cell, verify:

```
□ Import Pattern
  - Uses sys.path.insert with PROJECT_ROOT = Path.cwd().parents[1]
  - Imports from src.* (not inline copies of src/ logic)

□ Paths
  - All paths built from PROJECT_ROOT / paths["..."]["..."]
  - No f-strings containing /home/, /Users/, or C:\

□ Data Splitting
  - df.sort_values("submit_time") before any split
  - No train_test_split(shuffle=True)

□ Checkpoint Guards
  - All checkpoint loads have: if ckpt_path.exists(): ... else: print("WARNING...")

□ Figure Saving
  - fig.savefig(...) called BEFORE plt.show()
  - Save path uses FIGURES_DIR / "filename.png"
```

### Step 3 — Execution Order

Check for forward reference issues:

```
□ Every variable used in a cell is defined in that cell or an earlier cell
□ No cell depends on side-effects from a cell that comes later
□ Entire notebook would survive Kernel → Restart & Run All
```

### Step 4 — Narrative Quality

For each markdown section:

```
□ Section explains: What was done
□ Section explains: What was found (with actual numbers)
□ Section explains: Why this matters to the thesis
□ No vague claims like "the model improved significantly"
□ All numeric claims match checkpoint values
```

### Step 5 — EN/TR Parity

If reviewing both language versions:

```python
# Quick code cell count parity check
import json

en_nb = json.loads(open(f"notebooks/en/0{N}_*.ipynb").read())
tr_nb = json.loads(open(f"notebooks/tr/0{N}_*.ipynb").read())

en_code_cells = [c for c in en_nb["cells"] if c["cell_type"] == "code"]
tr_code_cells = [c for c in tr_nb["cells"] if c["cell_type"] == "code"]

assert len(en_code_cells) == len(tr_code_cells), \
    f"Cell count mismatch: EN={len(en_code_cells)}, TR={len(tr_code_cells)}"
```

Also verify code content is byte-for-byte identical (not just similar).

### Step 6 — Report

Produce findings grouped by severity:

```markdown
## Notebook Review: {filename}

### CRITICAL (causes errors or wrong results)
1. Cell N: {issue description} → Fix: {exact fix}

### NARRATIVE (affects thesis quality)
1. Section X: {issue} → Fix: {suggestion}

### PARITY (EN/TR mismatch)
1. {specific difference}

### MINOR SUGGESTIONS
1. ...

### Verdict: ✅ PASS / ⚠️ REVISION NEEDED / ❌ FAIL
```
