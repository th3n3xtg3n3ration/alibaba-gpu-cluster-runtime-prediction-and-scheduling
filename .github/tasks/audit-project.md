# Task: Full Project Audit

This runbook defines the step-by-step process for conducting a comprehensive
PhD-level audit of the thesis project. Run this before any major milestone.

---

## Audit Phases

### Phase 1 — Environment Integrity (5 min)

```bash
# 1. Check conda environment
conda activate gpu-scheduling
python --version      # Expect: 3.10.x
python -c "import xgboost, lightgbm, torch, sklearn; print('Core deps OK')"

# 2. Run all unit tests
python -m unittest discover tests -v
# Expected: 11 tests, 0 failures, 0 errors

# 3. Check static types (optional but recommended)
pyright src/ --outputjson | python -c "
import json, sys
result = json.load(sys.stdin)
errs = result.get('generalDiagnostics', [])
print(f'Pyright: {len(errs)} diagnostics')
"
```

**Pass criteria:** All 11 tests green. Zero import errors.

---

### Phase 2 — Data Integrity (5 min)

```bash
python -c "
import pandas as pd
from pathlib import Path
from src.config_utils import load_paths_config

paths = load_paths_config()
raw = Path(paths['data']['raw_data_dir']) / paths['data']['main_sample_file']
proc = Path(paths['data']['processed_data_dir']) / '100k_job_with_utilization.csv'

df_raw = pd.read_csv(raw)
print(f'Raw: {len(df_raw):,} rows, {df_raw.shape[1]} cols')
assert len(df_raw) > 90_000, 'Raw data suspiciously small'

df_proc = pd.read_csv(proc)
print(f'Processed: {len(df_proc):,} rows, {df_proc.shape[1]} cols')
required_cols = ['submit_time', 'duration', 'num_gpu', 'cluster_load_gpu', 'active_job_count']
missing = [c for c in required_cols if c not in df_proc.columns]
assert not missing, f'Missing columns: {missing}'
print('Data integrity: PASS')
"
```

---

### Phase 3 — Experiment Checkpoint Audit (5 min)

```bash
python -c "
import json
from pathlib import Path
from src.config_utils import load_paths_config

paths = load_paths_config()
ckpt_dir = Path(paths['results']['checkpoints_dir'])
required_metrics = ['mae', 'rmse', 'r2', 'mape', 'mdae']

for f in sorted(ckpt_dir.glob('*.json')):
    ckpt = json.loads(f.read_text())
    missing = [m for m in required_metrics if m not in ckpt.get('metrics', {})]
    status = '❌ INCOMPLETE' if missing else '✅'
    print(f'{status}  {f.name}: R²={ckpt[\"metrics\"].get(\"r2\",\"?\"):.3f}  missing={missing}')
"
```

---

### Phase 4 — Code Quality Scan (5 min)

```bash
# Check for hardcoded absolute paths
echo '=== Hardcoded paths ==='
grep -rn "/home\|/Users\|C:\\\\Users" src/ notebooks/ --include="*.py" --include="*.ipynb" || echo 'None found'

# Check for os.path usage
echo '=== os.path usage ==='
grep -rn "os\.path\." src/ --include="*.py" || echo 'None found'

# Check random seeds
echo '=== Random seed audit ==='
grep -rn "random_state\|seed\|manual_seed" src/ --include="*.py" | head -20

# Check scratch/ isolation
echo '=== scratch/ imports ==='
grep -rn "from scratch\|import scratch" src/ notebooks/ --include="*.py" --include="*.ipynb" || echo 'None found (good)'
```

---

### Phase 5 — Figures & Models Inventory (2 min)

```bash
python -c "
from pathlib import Path
from src.config_utils import load_paths_config

paths = load_paths_config()
figs = list(Path(paths['results']['figures_dir']).glob('*.png'))
models = list(Path(paths['results']['models_dir']).glob('*.joblib'))

print(f'Figures: {len(figs)}')
for f in sorted(figs): print(f'  {f.name}')

print(f'Models: {len(models)}')
for m in sorted(models): print(f'  {m.name}')
"
```

---

### Phase 6 — Notebook Coherence (10 min)

Manually verify for each of the 6 notebooks:
1. EN and TR versions have the same number of code cells
2. All imports reference `src/` (no inline logic duplicating `src/`)
3. Figures are saved to `results/figures/` (not local notebook directory)
4. Checkpoint loading includes existence check
5. Kernel → Restart & Run All completes without errors

---

## Audit Report Template

```markdown
# Project Audit Report — {DATE}

## Executive Summary
{2-3 sentences: overall state, most critical issue, recommendation}

## Status Matrix
| Phase | Status | Notes |
|-------|--------|-------|
| Environment | ✅/⚠️/❌ | |
| Data Integrity | ✅/⚠️/❌ | |
| Checkpoints | ✅/⚠️/❌ | |
| Code Quality | ✅/⚠️/❌ | |
| Figures/Models | ✅/⚠️/❌ | |
| Notebooks | ✅/⚠️/❌ | |

## Critical Issues (must fix before thesis submission)
1. ...

## Minor Issues (should fix)
1. ...

## Next Priority Actions
1. ...
```
