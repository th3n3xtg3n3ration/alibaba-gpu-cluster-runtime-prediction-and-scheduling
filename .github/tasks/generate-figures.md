# Task: Generate All Figures

This runbook regenerates all thesis figures from saved models and checkpoints.
Use after completing new experiments or if figures are missing/corrupted.

---

## Prerequisites

```bash
conda activate gpu-scheduling

# Verify models exist
ls results/models/*.joblib   # Must have: rf_numeric, xgb_numeric, lgb_numeric

# Verify processed data exists
python -c "
from pathlib import Path
from src.config_utils import load_paths_config
paths = load_paths_config()
p = Path(paths['data']['processed_data_dir']) / '100k_job_with_utilization.csv'
assert p.exists(), f'Missing: {p}'
print('Data OK')
"
```

---

## Run the Full Figure Generator

```bash
# Generate all figures at once
python scripts/generate_all_figures.py

# Expected output directory: results/figures/
# Expected figure count: 40+
```

---

## Manually Regenerate Specific Figure Groups

### Workload Analysis Figures

```python
from src.analysis.workload_analysis import (
    plot_runtime_distribution,
    plot_arrival_heatmap,
    plot_resource_demand
)
from src.data_loading import load_processed_dataset
from src.config_utils import load_paths_config
from pathlib import Path

df = load_processed_dataset()
paths = load_paths_config()
figs_dir = Path(paths["results"]["figures_dir"])

plot_runtime_distribution(df, save_path=figs_dir / "workload_runtime_dist.png")
plot_arrival_heatmap(df, save_path=figs_dir / "workload_arrival_heatmap.png")
plot_resource_demand(df, save_path=figs_dir / "workload_resource_demand.png")
```

### Model Performance Figures

```python
import joblib, json
from src.models.evaluation import evaluate_regression
from src.visualization import plot_true_vs_predicted, plot_error_cdf
from src.feature_engineering import build_feature_matrix
from src.data_loading import load_processed_dataset

df = load_processed_dataset().sort_values("submit_time")
split_idx = int(len(df) * 0.80)
test_df = df.iloc[split_idx:]
X_test, y_test = build_feature_matrix(test_df, mode="numeric")

for model_name in ["rf_numeric", "xgb_numeric", "lgb_numeric"]:
    model = joblib.load(f"results/models/{model_name}.joblib")
    y_pred = model.predict(X_test)
    tag = model_name.replace("_numeric", "")
    plot_true_vs_predicted(y_test, y_pred,
        save_path=figs_dir / f"exp_a_{tag}_scatter.png")
    plot_error_cdf(y_test, y_pred,
        save_path=figs_dir / f"exp_a_{tag}_error_cdf.png")
```

### Scheduler Figures

```python
from src.simulation.scheduler_simulator import run_simulation
from src.visualization import plot_scheduler_comparison

results = {}
for policy in ["fifo", "sjf_oracle", "sjf_pred"]:
    results[policy] = run_simulation(df, policy=policy, model_path="results/models/xgb_numeric.joblib")

plot_scheduler_comparison(results, save_path=figs_dir / "scheduler_comparison.png")
```

---

## Figure Naming Convention

```
workload_runtime_dist.png           ← Runtime log-histogram
workload_arrival_heatmap.png        ← Hour × Day-of-week heatmap
workload_resource_demand.png        ← GPU/CPU demand over time

exp_a_rf_scatter.png               ← True vs predicted (log-log)
exp_a_rf_residuals.png             ← Residuals histogram
exp_a_rf_error_cdf.png             ← Error CDF
exp_a_xgb_scatter.png
...
exp_a_comparison_bar.png           ← All models bar chart (MAE, RMSE, R²)

scheduler_comparison.png           ← Wait time comparison bar chart
scheduler_cdf.png                  ← Wait time CDF per policy
```

---

## Quality Check After Generation

```bash
# Count figures
ls results/figures/*.png | wc -l     # Expect 40+

# Check no figure is empty (0 bytes)
find results/figures/ -name "*.png" -size 0 -exec echo "EMPTY: {}" \;

# Spot-check figure sizes (suspicious if < 5KB)
find results/figures/ -name "*.png" -size -5k -exec echo "SMALL: {}" \;
```
