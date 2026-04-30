---
name: experiment-runner
description: "Step-by-step workflow for running ML experiments A through F in the GPU scheduling thesis. Use for: executing a new experiment, resuming a partial run, completing missing checkpoints, retraining a model with updated hyperparameters. Handles tree-based (RF/XGB/LGBM) and deep learning (CNN/LSTM/Hybrid) variants."
argument-hint: "Experiment tag (exp_a–exp_f) and model (rf, xgb, lgbm, cnn, lstm, hybrid)"
---

# Experiment Runner Skill

## Overview

This skill executes one ML experiment in the GPU scheduling thesis pipeline, from
feature preparation through checkpoint saving and figure generation.

## When to Use

- Running experiment B–F (A is already complete)
- Resuming a partial experiment (`_partial.json` checkpoint exists)
- Re-running an experiment with updated hyperparameter grid
- Regenerating figures for a completed experiment

## Pre-conditions

Before starting, verify:
1. `conda activate gpu-scheduling` is active
2. `data/processed/100k_job_with_utilization.csv` exists
3. Target checkpoint does NOT exist (or user confirmed overwrite)

## Procedure

### Step 1 — Determine Feature Mode

| Experiment | Feature Mode | `build_feature_matrix` argument |
|-----------|--------------|--------------------------------|
| exp_a | Numeric only | `mode="numeric"` |
| exp_b | One-Hot Encoding | `mode="one_hot"` |
| exp_c | Numeric sequence | `mode="sequential"` |
| exp_d | Categorical embedding | `mode="categorical_seq"` |
| exp_e | Numeric + sequential | `mode="numeric_seq"` |
| exp_f | Categorical + sequential | `mode="categorical_seq_full"` |

### Step 2 — Load Config and Data

See [.github/tasks/run-experiment.md](../tasks/run-experiment.md), Steps 1–2.

### Step 3 — Validate Split

The test set must always be the last 20% by `submit_time`. Print confirmation:

```python
print(f"Train: {len(train_df):,} jobs  ({train_df.submit_time.min()} → {train_df.submit_time.max()})")
print(f"Test:  {len(test_df):,} jobs  ({test_df.submit_time.min()} → {test_df.submit_time.max()})")
assert train_df.submit_time.max() < test_df.submit_time.min(), "TEMPORAL LEAKAGE DETECTED"
```

### Step 4 — Hyperparameter Search

See [.github/tasks/run-experiment.md](../tasks/run-experiment.md), Step 3.

For DL models, use the `configs/models.yaml` DL section (CNN/LSTM/Hybrid keys).

### Step 5 — Evaluate & Checkpoint

See [.github/tasks/run-experiment.md](../tasks/run-experiment.md), Steps 4–6.

Checkpoint must contain all 5 metrics: mae, rmse, r2, mape, mdae.

### Step 6 — Figures

See [.github/tasks/run-experiment.md](../tasks/run-experiment.md), Step 7.

Required figures per experiment+model:
- `exp_{tag}_{model}_scatter.png`
- `exp_{tag}_{model}_residuals.png`
- `exp_{tag}_{model}_error_cdf.png`

### Step 7 — Post-run Validation

```bash
python -m unittest discover tests -v   # Must stay 11/11
```

## Resuming a Partial Run

If `_partial.json` exists in `results/checkpoints_backup_old/`:
1. Read the partial checkpoint to extract `best_params` (skip re-tuning)
2. Reconstruct test set with the same chronological split
3. Load or retrain model with `best_params` (use `fit()` directly, not the search)
4. Re-evaluate all 5 metrics
5. Save complete checkpoint (no `_partial` suffix)

## Output

A complete experiment produces:
- `results/checkpoints/exp_{tag}_{model}.json` — 5 metrics + params + timestamp
- `results/models/exp_{tag}_{model}.joblib` (or `.pt` for DL)
- 3 figures in `results/figures/`
- Updated notebook narrative (EN + TR)
