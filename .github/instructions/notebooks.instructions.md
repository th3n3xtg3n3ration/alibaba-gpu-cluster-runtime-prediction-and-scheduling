---
description: "Use when creating, editing, reviewing, or executing Jupyter notebooks in notebooks/en/ or notebooks/tr/. Covers bilingual structure, import conventions, cell organization, variable naming, and narrative requirements for the thesis notebooks."
applyTo: "notebooks/**/*.ipynb"
---

# Jupyter Notebook Conventions — Thesis Notebooks

## Bilingual Structure Rule

Every notebook exists in two identical versions:

| English (canonical) | Turkish (translation) |
|--------------------|-----------------------|
| `notebooks/en/0N_*.ipynb` | `notebooks/tr/0N_*.ipynb` |

The EN version is canonical — logic changes happen there first, then mirrored to TR.
Turkish versions must have identical cell structure; only markdown narrative is translated.

## Notebook Phase Map

| File | Phase | Content |
|------|-------|---------|
| `00_data_preparation` | Setup | Load raw CSV, run sweep-line, save processed CSV |
| `01_data_overview` | EDA | Shape, types, null audit, summary stats |
| `02_workload_analysis` | Characterization | Distributions, arrival rates, heatmaps |
| `03_feature_engineering` | Features | Temporal, categorical, utilization features |
| `04_runtime_prediction_models` | ML | Experiments A–F, training, evaluation |
| `05_scheduler_evaluation` | Simulation | FIFO vs SJF-Oracle vs SJF-Pred |

## Standard Cell Order (each notebook)

```
[Markdown] Title + Author + Date
[Code]     %reload_ext autoreload / %autoreload 2
[Code]     Imports (std lib → third-party → src/)
[Code]     Project root setup + config loading
[Code]     ... section content ...
[Markdown] ## Conclusions / Key Findings
```

## Import Pattern in Notebooks

```python
import sys
from pathlib import Path

# Add project root to path so src/ is importable
PROJECT_ROOT = Path.cwd().parents[1]  # notebooks/en/../.. = project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_paths_config
from src.data_loading import load_processed_dataset
# etc.

paths = load_paths_config()
FIGURES_DIR = PROJECT_ROOT / paths["results"]["figures_dir"]
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
```

## Variable Naming in Notebooks

| Variable | Meaning |
|----------|---------|
| `df_raw` | Raw loaded DataFrame |
| `df` | Main working DataFrame (after feature engineering) |
| `X_train`, `X_test` | Feature matrices |
| `y_train`, `y_test` | Target vectors |
| `y_pred` | Model predictions on test set |
| `metrics_*` | Dict from `evaluate_regression()` |
| `fig, ax` | Matplotlib figure/axis pair |

## Figure Saving Pattern

```python
fig, ax = plt.subplots(figsize=(10, 6))
# ... plotting ...
fig.tight_layout()
fig.savefig(FIGURES_DIR / "exp_a_rf_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
```

Always save figures to disk before `plt.show()`.

## Cell Execution Order

- Notebooks must execute top-to-bottom without errors (Kernel → Restart & Run All).
- Never reference a variable before the cell that defines it.
- After reordering cells, always do a clean restart + run-all to verify.

## Narrative Requirements

Each major section must have a markdown cell with:
1. **What was done** — brief method description
2. **What was found** — key result or observation
3. **Why it matters** — connection to thesis argument

Turkish notebooks must translate all of the above, not just section headers.

## Checkpoint Loading Pattern (in notebooks)

```python
import json
ckpt_path = PROJECT_ROOT / paths["results"]["checkpoints_dir"] / "exp_a_xgb.json"
if ckpt_path.exists():
    with open(ckpt_path) as f:
        ckpt = json.load(f)
    print(f"Loaded checkpoint: MAE={ckpt['metrics']['mae']:.1f}s, R²={ckpt['metrics']['r2']:.3f}")
else:
    print("WARNING: Checkpoint not found — run training cells first.")
```

Never skip the existence check; missing checkpoints cause NameErrors later.
