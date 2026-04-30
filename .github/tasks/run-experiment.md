# Task: Run an ML Experiment (A–F)

This runbook describes the exact steps to execute one ML experiment in this thesis project.
Agents should follow these steps in order.

---

## Pre-flight Checks

```bash
# 1. Activate environment
conda activate gpu-scheduling

# 2. Verify tests pass
python -m unittest discover tests -v

# 3. Verify processed dataset exists
python -c "
from pathlib import Path
from src.config_utils import load_paths_config
paths = load_paths_config()
p = Path(paths['data']['processed_data_dir']) / '100k_job_with_utilization.csv'
assert p.exists(), f'Missing: {p}'
print('Dataset OK:', p)
"
```

If the processed dataset is missing, run `notebooks/en/00_data_preparation.ipynb` first.

---

## Step 1 — Load Config

```python
from src.config_utils import load_paths_config, load_model_config
from pathlib import Path

paths = load_paths_config()
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust as needed
```

## Step 2 — Load & Split Data

```python
from src.data_loading import load_processed_dataset
from src.feature_engineering import build_feature_matrix
import pandas as pd

df = load_processed_dataset()
df = df.sort_values("submit_time").reset_index(drop=True)
split_idx = int(len(df) * 0.80)
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

# Choose mode: "numeric" | "one_hot" | "sequential"
X_train, y_train = build_feature_matrix(train_df, mode="numeric")
X_test, y_test = build_feature_matrix(test_df, mode="numeric")
```

## Step 3 — Train with Hyperparameter Search

```python
from src.tuning import run_hyperparameter_search

param_grid = load_model_config("xgboost")  # "random_forest" | "lightgbm" | ...
best_model, best_params = run_hyperparameter_search(
    model_type="xgboost",
    X_train=X_train, y_train=y_train,
    param_grid=param_grid,
    n_iter=20, seed=42,
)
```

## Step 4 — Evaluate

```python
from src.models.evaluation import evaluate_regression

y_pred = best_model.predict(X_test)
metrics = evaluate_regression(y_test, y_pred)
print(metrics)
# Expected keys: mae, rmse, r2, mape, mdae
```

## Step 5 — Save Checkpoint

```python
import json
from datetime import datetime, timezone

checkpoint = {
    "experiment": "exp_a",            # CHANGE per experiment
    "model": "xgboost",
    "feature_mode": "numeric",
    "metrics": metrics,
    "best_params": best_params,
    "train_size": len(y_train),
    "test_size": len(y_test),
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
ckpt_dir = PROJECT_ROOT / paths["results"]["checkpoints_dir"]
ckpt_dir.mkdir(parents=True, exist_ok=True)
(ckpt_dir / "exp_a_xgb.json").write_text(json.dumps(checkpoint, indent=2))
```

## Step 6 — Save Model

```python
import joblib

models_dir = PROJECT_ROOT / paths["results"]["models_dir"]
models_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, models_dir / "exp_a_xgb_numeric.joblib")
```

## Step 7 — Generate Figures

```python
from src.visualization import (
    plot_true_vs_predicted, plot_residuals, plot_error_cdf
)

figs_dir = PROJECT_ROOT / paths["results"]["figures_dir"]
figs_dir.mkdir(parents=True, exist_ok=True)

plot_true_vs_predicted(y_test, y_pred, save_path=figs_dir / "exp_a_xgb_scatter.png")
plot_residuals(y_test, y_pred, save_path=figs_dir / "exp_a_xgb_residuals.png")
plot_error_cdf(y_test, y_pred, save_path=figs_dir / "exp_a_xgb_error_cdf.png")
```

## Step 8 — Update Notebook Narrative

1. Open `notebooks/en/04_runtime_prediction_models.ipynb`
2. Load the checkpoint in the corresponding cell
3. Update results table and narrative explanation
4. Run Kernel → Restart & Run All to verify clean execution
5. Mirror changes to `notebooks/tr/04_calisma_zamani_tahmin_modelleri.ipynb`

---

## Verification

After completing all steps:

```bash
python -m unittest discover tests -v   # Must still be 11/11 passing
ls results/checkpoints/               # exp_a_xgb.json must appear
ls results/models/                    # exp_a_xgb_numeric.joblib must appear
ls results/figures/exp_a_xgb*         # 3 figures must appear
```
