# Alibaba GPU Scheduling Thesis — Agent Guidelines

This file is the authoritative guide for all AI agents and coding assistants working on this
repository. Read it fully before taking any action.

---

## Project At a Glance

**Title:** Runtime Prediction and Scheduling Optimization in Large-Scale GPU Clusters  
**Scope:** PhD thesis — workload characterization + multi-paradigm ML + discrete-event simulation  
**Dataset:** Alibaba PAI GPU Cluster Trace 2022 (~100K jobs, 7-day window)  
**Goal:** Demonstrate that SJF-Pred (ML-augmented scheduling) achieves ≥85% reduction in
average cluster waiting time vs FIFO baseline without additional hardware.

---

## Architecture

```
src/                     ← Production Python library (the only source of truth for logic)
├── config_utils.py      ← YAML loader: load_paths_config(), load_model_config(key)
├── data_loading.py      ← CSV readers, PathsConfig dataclass
├── feature_engineering.py  ← Temporal, categorical, sweep-line utilization features
├── tuning.py            ← RandomizedSearchCV with macOS stability patches
├── visualization.py     ← Matplotlib/Seaborn plots
├── analysis/            ← Workload statistics, arrival-rate analysis
├── models/              ← RF, XGB, LGBM, CNN, LSTM, Hybrid definitions + evaluation.py
└── simulation/          ← Single-queue + multi-node discrete-event schedulers

notebooks/en/            ← English Jupyter notebooks (6 phases, canonical)
notebooks/tr/            ← Turkish counterparts (same logic, translated narrative)
configs/                 ← paths.yaml, models.yaml (single source for all configuration)
data/                    ← Raw (alibaba_cluster_trace/) + Processed (processed/)
results/                 ← figures/, models/ (.joblib), checkpoints/ (.json), logs/
tests/                   ← unittest — 11 tests, 100% passing
docs/                    ← thesis_outline.md
scripts/                 ← run_all_experiments.sh, generate_all_figures.py, phd_audit.py
scratch/                 ← One-off patch scripts (do NOT import or modify without explicit request)
```

---

## Build & Test Commands

```bash
# Activate environment
conda activate gpu-scheduling

# Run all unit tests (must pass before any commit)
python -m unittest discover tests -v

# Run the full experiment pipeline
bash scripts/run_all_experiments.sh

# Generate all figures
python scripts/generate_all_figures.py

# Run PhD-level audit
python scripts/phd_audit.py

# Debug single experiment notebook
jupyter nbconvert --to notebook --execute notebooks/en/04_runtime_prediction_models.ipynb \
  --output notebooks/en/04_runtime_prediction_models.ipynb
```

---

## Code Conventions

### Python Standards
- **Version:** Python 3.10
- **Type hints:** `from __future__ import annotations` in every `src/` module
- **Paths:** Always `pathlib.Path` — never `os.path`, never hardcoded strings
- **Project root:** Discover dynamically: `Path(__file__).resolve().parents[N]`
- **Docstrings:** NumPy style — Parameters / Returns / Raises sections
- **Logging:** `logging.getLogger(__name__)` — module-level, never `print()` in `src/`

### Naming
| Element | Convention | Example |
|---------|-----------|---------|
| Functions/methods | `snake_case` | `compute_utilization()` |
| Classes | `PascalCase` | `RandomForestPredictor` |
| Constants | `ALL_CAPS` | `DEFAULT_SEED = 42` |
| Private helpers | `_underscore_prefix` | `_load_yaml_file()` |

### Config System
```python
from src.config_utils import load_paths_config, load_model_config

paths = load_paths_config()          # dict from configs/paths.yaml
rf_params = load_model_config("random_forest")  # hyperparameter grid
```
**Never** hardcode a data path or hyperparameter grid in a notebook or script.

### macOS Stability
`src/tuning.py` already sets `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` on Darwin.
Do **not** override these; do **not** increase `n_jobs` beyond 1 in the config without testing.

### Random Seeds
Always use seed `42` for reproducibility:
```python
import numpy as np, torch, random
np.random.seed(42); torch.manual_seed(42); random.seed(42)
```

---

## Experiment Map

| Tag | Models | Feature Mode | Status |
|-----|--------|--------------|--------|
| `exp_a` | RF, XGB, LGBM | Numeric only | ✅ Complete |
| `exp_b` | RF, XGB, LGBM, LGBM-Native | One-Hot Encoding | ⏳ Partial |
| `exp_c` | CNN, LSTM, Hybrid | Numeric sequence | 🔲 Planned |
| `exp_d` | CNN, LSTM, Hybrid | Categorical embedding | 🔲 Planned |
| `exp_e` | CNN, LSTM, Hybrid | Numeric + sequential | 🔲 Planned |
| `exp_f` | CNN, LSTM, Hybrid | Categorical + sequential | 🔲 Planned |

Checkpoints live at `results/checkpoints/exp_{tag}_{model}.json`.  
Backup copies at `results/checkpoints_backup_old/` (read-only, do not modify).

---

## Checkpoint Convention

```python
import json
from pathlib import Path

checkpoint = {
    "experiment": "exp_b",
    "model": "xgboost",
    "feature_mode": "one_hot",
    "metrics": {"mae": ..., "rmse": ..., "r2": ...},
    "best_params": {...},
    "train_size": ..., "test_size": ...,
    "timestamp": "2026-04-18T..."
}
Path("results/checkpoints/exp_b_xgb.json").write_text(json.dumps(checkpoint, indent=2))
```

---

## Quality Gates

Every experiment / PR must pass all of these before being considered done:

1. `python -m unittest discover tests -v` → 11/11 pass
2. Checkpoint JSON saved to `results/checkpoints/`
3. Associated figures saved to `results/figures/`
4. Notebook narrative updated (both EN and TR versions)
5. No hardcoded paths (run `grep -r "home\|Users\|/data/" src/ notebooks/`)
6. No `os.path` usage: `grep -r "os\.path" src/`

---

## Protected Files — Never Overwrite

| File | Reason |
|------|--------|
| `data/alibaba_cluster_trace/pai_job_no_estimate_100K.csv` | Primary dataset — irreplaceable |
| `results/checkpoints_backup_old/*.json` | Legacy safety net |
| `environment.yaml` | Pinned environment — change must be intentional |

---

## Sensitive Areas

- `scratch/` — 15+ one-off patch scripts. Never import from here; never run unless instructed.
- `results/checkpoints_backup_old/` — Read-only legacy backups.
- Any `.joblib` model file — Check the experiment tag matches before overwriting.

---

## See Also

- [Project Overview](.github/context/project-overview.md)
- [Experiment Map (detailed)](.github/context/experiments-map.md)
- [Data Schema](.github/context/data-schema.md)
- [Current Results Summary](.github/context/results-summary.md)
- [PhD Advisor Agent](.github/agents/phd-advisor.agent.md)
