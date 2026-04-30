---
description: "Use when writing, editing, or reviewing any Python source code in src/, scripts/, or tests/. Covers pathlib, type hints, docstrings, logging, config loading, random seeds, and macOS stability patterns specific to this GPU scheduling thesis project."
applyTo: "**/*.py"
---

# Python Code Standards — GPU Scheduling Thesis

## Imports & Module Structure

```python
from __future__ import annotations  # Required in every src/ module

import logging
from pathlib import Path
from typing import Any

# Third-party before local, separated by blank line
import numpy as np
import pandas as pd
from src.config_utils import load_paths_config
```

Never use `os.path` — always `pathlib.Path`.  
Never use `print()` in `src/` — use `logging.getLogger(__name__)`.

## Project Root Discovery

```python
# In src/ modules (2 levels up from src/module.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# In scripts/ (1 level up from scripts/script.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# In notebooks (manual or via config_utils)
from src.config_utils import load_paths_config
paths = load_paths_config()
```

## Config Loading Pattern

```python
from src.config_utils import load_paths_config, load_model_config

paths = load_paths_config()
raw_dir = PROJECT_ROOT / paths["data"]["raw_data_dir"]
checkpoints_dir = PROJECT_ROOT / paths["results"]["checkpoints_dir"]

model_cfg = load_model_config("xgboost")  # Returns hyperparameter grid dict
```

## Docstring Style (NumPy)

```python
def compute_utilization(jobs_df: pd.DataFrame, time_resolution: int = 60) -> pd.DataFrame:
    """Compute cluster resource utilization using a sweep-line algorithm.

    Parameters
    ----------
    jobs_df : pd.DataFrame
        DataFrame with columns: submit_time, duration, num_gpu, num_cpu.
    time_resolution : int, optional
        Granularity in seconds, by default 60.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by time with columns: load_gpu, load_cpu, active_jobs.

    Raises
    ------
    ValueError
        If `jobs_df` is empty or missing required columns.
    """
```

## Type Hints

```python
from __future__ import annotations
from typing import Optional, Union
import numpy.typing as npt

def train_model(
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    params: dict[str, Any],
    seed: int = 42,
) -> Any:
    ...
```

## Random Seeds — Always 42

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# For sklearn estimators, pass random_state=SEED
```

## macOS Stability

The following is already handled in `src/tuning.py`. Do not add it elsewhere:

```python
import os, platform
if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
```

Never set `n_jobs > 1` in sklearn calls on macOS without testing first.

## Checkpoint Save Pattern

```python
import json
from datetime import datetime, timezone

def save_checkpoint(tag: str, model_name: str, metrics: dict, params: dict) -> None:
    paths = load_paths_config()
    ckpt_dir = PROJECT_ROOT / paths["results"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": tag,
        "model": model_name,
        "metrics": metrics,
        "best_params": params,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path = ckpt_dir / f"{tag}_{model_name}.json"
    out_path.write_text(json.dumps(payload, indent=2))
```

## Error Handling

- Raise `ValueError` for invalid inputs at public API boundaries.
- Raise `FileNotFoundError` with descriptive messages when expected files are missing.
- Do not swallow exceptions silently with bare `except:`.

```python
csv_path = PROJECT_ROOT / paths["data"]["raw_data_dir"] / paths["data"]["main_sample_file"]
if not csv_path.exists():
    raise FileNotFoundError(
        f"[data_loading] Raw dataset not found at {csv_path}. "
        "Run the data preparation notebook first."
    )
```

## Naming Conventions

| Element | Pattern | Example |
|---------|---------|---------|
| Module-level functions | `snake_case` | `build_feature_matrix()` |
| Classes | `PascalCase` | `LGBMRuntimePredictor` |
| Constants | `ALL_CAPS` | `FEATURE_COLUMNS` |
| Private helpers | `_leading_underscore` | `_normalize_durations()` |
| Test methods | `test_<what>_<condition>` | `test_mae_zero_error_returns_zero()` |
