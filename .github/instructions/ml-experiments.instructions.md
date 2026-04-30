---
description: "Use when implementing, running, debugging, or reviewing ML experiments (exp_a through exp_f). Covers the full workflow: feature preparation, model training, hyperparameter tuning, evaluation, checkpoint saving, and figure generation for tree-based and deep learning models."
---

# ML Experiment Guidelines

## Experiment Lifecycle

Every experiment follows this exact sequence:

```
1. Load processed dataset  →  data/processed/100k_job_with_utilization.csv
2. Build feature matrix    →  src/feature_engineering.py
3. Train/test split        →  80/20, chronological (never shuffle for time-series)
4. Hyperparameter search   →  src/tuning.py (RandomizedSearchCV)
5. Evaluate                →  src/models/evaluation.evaluate_regression()
6. Save checkpoint         →  results/checkpoints/exp_{tag}_{model}.json
7. Save model              →  results/models/{tag}_{model}.joblib
8. Generate figures        →  results/figures/exp_{tag}_{model}_*.png
```

## Experiment Tags & Feature Modes

| Tag | Models | Feature Mode |
|-----|--------|--------------|
| `exp_a` | rf, xgb, lgbm | Numeric only (no encoding) |
| `exp_b` | rf, xgb, lgbm, lgbm_nat | One-Hot Encoding |
| `exp_c` | cnn, lstm, hybrid | Numeric sequence |
| `exp_d` | cnn, lstm, hybrid | Categorical embedding |
| `exp_e` | cnn, lstm, hybrid | Numeric + sequential |
| `exp_f` | cnn, lstm, hybrid | Categorical + sequential |

Checkpoint file name pattern: `results/checkpoints/exp_{tag}_{model}.json`  
Model file name pattern: `results/models/{tag}_{model}.joblib`

## Feature Matrix Construction

```python
from src.feature_engineering import build_feature_matrix
from src.data_loading import load_processed_dataset

df = load_processed_dataset()

# Numeric-only (exp_a)
X, y = build_feature_matrix(df, mode="numeric")

# One-hot (exp_b)
X, y = build_feature_matrix(df, mode="one_hot")

# Sequential reshape for DL (exp_c/e/f)
X, y = build_feature_matrix(df, mode="sequential")
```

## Train/Test Split — Chronological

```python
# Always sort by submit_time before splitting
df = df.sort_values("submit_time").reset_index(drop=True)
split_idx = int(len(df) * 0.80)
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
```

Never use `train_test_split(shuffle=True)` on this dataset — temporal leakage
invalidates results.

## Hyperparameter Tuning

```python
from src.tuning import run_hyperparameter_search
from src.config_utils import load_model_config

param_grid = load_model_config("xgboost")
best_model, best_params = run_hyperparameter_search(
    model_type="xgboost",
    X_train=X_train, y_train=y_train,
    param_grid=param_grid,
    n_iter=20,
    seed=42,
)
```

Config grids live in `configs/models.yaml`. Modify there — never inline in notebooks.

## Evaluation

```python
from src.models.evaluation import evaluate_regression

metrics = evaluate_regression(y_test, y_pred)
# Returns: {"mae": ..., "rmse": ..., "r2": ..., "mape": ..., "mdae": ...}
```

Always report all five metrics in checkpoints and notebook narratives.

## Checkpoint Integrity Rules

1. A checkpoint is only valid if it contains: `experiment`, `model`, `feature_mode`,
   `metrics` (all 5), `best_params`, `train_size`, `test_size`, `timestamp`.
2. Do not overwrite a completed checkpoint (status `complete`) unless explicitly requested.
3. Failed or partial runs → suffix the file `_partial.json` to distinguish.

## Figure Naming Convention

```
results/figures/
├── exp_a_rf_scatter.png          # True vs Predicted scatter
├── exp_a_rf_residuals.png        # Residuals histogram
├── exp_a_rf_error_cdf.png        # Error CDF (log scale)
├── exp_a_comparison_bar.png      # All models comparison bar chart
└── scheduler_comparison.png      # FIFO vs SJF-Oracle vs SJF-Pred
```

## DL Model Guidelines (exp_c–f)

- Input shape: `(batch, seq_len, features)` for LSTM/Hybrid
- Loss: `nn.MSELoss()` with log-scaled targets recommended
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Max epochs: 50 with early stopping (patience=5)
- Batch size: 256
- Always log epoch loss to `results/logs/exp_{tag}_{model}_train.log`

## Comparing Experiments

When comparing across experiments:
1. Use the same test set (held-out 20%, chronological)
2. Report relative improvement over FIFO baseline for scheduling metrics
3. Report absolute values (MAE in seconds, RMSE in seconds) for regression metrics
4. Never cherry-pick test subsets — report on the full held-out set
