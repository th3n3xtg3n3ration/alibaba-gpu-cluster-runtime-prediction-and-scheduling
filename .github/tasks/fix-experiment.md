# Task: Fix or Resume a Partial Experiment

This runbook handles the case where an experiment checkpoint is missing, incomplete,
or shows a `_partial` suffix. Follow the steps below to diagnose and repair.

---

## Step 1 — Diagnose the Issue

```bash
python -c "
import json
from pathlib import Path
from src.config_utils import load_paths_config

paths = load_paths_config()
ckpt_dir = Path(paths['results']['checkpoints_dir'])
backup_dir = Path('results/checkpoints_backup_old')

print('=== CURRENT checkpoints ===')
for f in sorted(ckpt_dir.glob('*.json')):
    ckpt = json.loads(f.read_text())
    m = ckpt.get('metrics', {})
    print(f'  {f.name}: R²={m.get(\"r2\",\"?\")}  metrics_keys={list(m.keys())}')

print()
print('=== BACKUP checkpoints ===')
for f in sorted(backup_dir.glob('*.json')):
    print(f'  {f.name}')
"
```

---

## Step 2 — Choose Recovery Strategy

| Situation | Action |
|-----------|--------|
| Partial checkpoint (`_partial.json`) | Re-run from Step 3 (train from scratch or load partial state) |
| Checkpoint missing specific metrics | Reload model and re-evaluate |
| Model `.joblib` file missing | Retrain from scratch |
| Checkpoint claims complete but R²=0 or NaN | Data loading issue — recheck train/test split |

---

## Step 3 — Reload Existing Model and Re-evaluate (if model exists)

```python
import joblib, json
from pathlib import Path
from src.config_utils import load_paths_config
from src.data_loading import load_processed_dataset
from src.feature_engineering import build_feature_matrix
from src.models.evaluation import evaluate_regression
from datetime import datetime, timezone

paths = load_paths_config()
PROJECT_ROOT = Path(".")  # adjust

# Load model
model = joblib.load(PROJECT_ROOT / paths["results"]["models_dir"] / "xgb_numeric.joblib")

# Rebuild test set (MUST match original split)
df = load_processed_dataset().sort_values("submit_time").reset_index(drop=True)
split_idx = int(len(df) * 0.80)
test_df = df.iloc[split_idx:]
X_test, y_test = build_feature_matrix(test_df, mode="numeric")

# Evaluate
y_pred = model.predict(X_test)
metrics = evaluate_regression(y_test, y_pred)
print("Recovered metrics:", metrics)

# Save corrected checkpoint
ckpt_path = PROJECT_ROOT / paths["results"]["checkpoints_dir"] / "exp_a_xgb.json"
existing = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {}
existing.update({
    "metrics": metrics,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "repaired": True,
})
ckpt_path.write_text(json.dumps(existing, indent=2))
print(f"Saved repaired checkpoint: {ckpt_path}")
```

---

## Step 4 — Retrain from Scratch (if model is lost)

Follow [run-experiment.md](./run-experiment.md) from Step 1.

---

## Step 5 — Handle Partial LGBM Checkpoints

For `exp_b_lgbm_nat_rs_partial.json` in the backup directory:

```python
import json
from pathlib import Path

partial = json.loads(Path("results/checkpoints_backup_old/exp_b_lgbm_nat_rs_partial.json").read_text())
print("Partial checkpoint contents:")
print(json.dumps(partial, indent=2))
# Use best_params from partial if available to skip re-tuning
```

---

## Step 6 — Verify Fix

```bash
python -m unittest discover tests -v            # Still 11/11 passing
python -c "
import json
from pathlib import Path
ckpt = json.loads(Path('results/checkpoints/exp_a_xgb.json').read_text())
assert all(k in ckpt['metrics'] for k in ['mae','rmse','r2','mape','mdae']), 'Missing metrics'
print('Checkpoint complete:', ckpt['metrics'])
"
```
