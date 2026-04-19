"""
tuning.py

Hyperparameter Tuning and Model Optimization

This module provides a unified framework for optimizing both classical Machine 
Learning (Random Forest, XGBoost, LightGBM) and Deep Learning (CNN, LSTM, Hybrid) 
models. It handles randomized and grid search, cross-validation, and early stopping.

Key Components:
  - run_ml_tuning: Main entry point for tree-based model optimization.
  - run_dl_randomsearch: Randomized search for PyTorch-based DL models.
  - EarlyStopping: Custom callback for DL training regularization.
  - finalize_and_evaluate_dl: Final refit and evaluation for the best DL architecture.
"""
from __future__ import annotations
import os
import platform
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union, List

# ── MacOS Threading Stability Patch ──────────────────────────────────────────
# Prevents 'OMP: Error #179: Function pthread_mutex_init failed' and 
# associated Segmentation Faults during joblib parallel execution.
if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    # Also limit other library thread pools inside workers
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
import lightgbm as lgb

# Progress bar support
import contextlib
import joblib
from tqdm.auto import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.evaluation import evaluate_regression

# Deep Learning Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.dl_runtime_predictor import RuntimePredictorCNN, RuntimePredictorLSTM, RuntimePredictorCNNLSTM
import copy
import time
import random
import itertools
import json

def get_default_device() -> str:
    """Automatically detect Apple Silicon (MPS), CUDA, or fallback to CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

# =====================================================================
# CHECKPOINT SYSTEM — Saves intermediate results to disk
# =====================================================================
_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "results" / "checkpoints"


def save_checkpoint(experiment_name: str, data: Dict[str, Any]) -> Path:
    """
    Save experiment results to disk so they survive kernel crashes.

    Parameters
    ----------
    experiment_name : str
        Identifier like 'exp_a_rf', 'exp_c_cnn', etc.
    data : dict
        Must contain JSON-serializable values (metrics, params).
        Model objects should be saved separately via joblib/torch.save.

    Returns
    -------
    Path
        Path to the saved checkpoint file.
    """
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _CHECKPOINT_DIR / f"{experiment_name}.json"

    # Convert numpy types to Python native for JSON
    clean = {}
    for k, v in data.items():
        if isinstance(v, dict):
            clean[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                        for kk, vv in v.items()}
        elif isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v)
        else:
            clean[k] = v

    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=str)

    print(f"  [Checkpoint] Saved → {path.name}")
    return path


def load_checkpoint(experiment_name: str) -> Optional[Dict[str, Any]]:
    """
    Load experiment results from disk.

    Returns None if checkpoint doesn't exist.
    """
    path = _CHECKPOINT_DIR / f"{experiment_name}.json"
    if not path.exists():
        return None

    with open(path, "r") as f:
        data = json.load(f)

    print(f"  [Checkpoint] Loaded ← {path.name}")
    return data


def load_all_checkpoints() -> Dict[str, Dict[str, Any]]:
    """
    Load ALL saved checkpoints. Used by the final summary cells.

    Returns
    -------
    dict
        {experiment_name: data_dict, ...}
    """
    if not _CHECKPOINT_DIR.exists():
        return {}

    results = {}
    for f in sorted(_CHECKPOINT_DIR.glob("*.json")):
        with open(f, "r") as fh:
            results[f.stem] = json.load(fh)

    print(f"  [Checkpoint] Loaded {len(results)} experiment results from disk.")
    return results


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def _run_search_with_progress(search, X, y, **fit_params):
    """Helper to run a search object with a tqdm progress bar."""
    # Estimate total fits
    if hasattr(search, "n_iter"):
        n_candidates = search.n_iter
    elif hasattr(search, "param_grid"):
        n_candidates = len(ParameterGrid(search.param_grid))
    else:
        n_candidates = 1
        
    cv = search.cv
    if hasattr(cv, "get_n_splits"):
        n_splits = cv.get_n_splits(X, y)
    else:
        try:
            n_splits = cv
        except (TypeError, AttributeError):
            n_splits = 3  # fallback default
            
    total_fits = n_candidates * n_splits
    desc = search.__class__.__name__

    print(f"Starting {desc} with {total_fits} fits...")
    with tqdm_joblib(tqdm(desc=desc, total=total_fits)) as _:
        # ── MacOS Stability Enforcement ──────────────────────────────────────
        # On Darwin (macOS), we force the 'threading' backend for search.fit.
        # This avoids the 'multiple OMP runtimes' conflict common with 'loky' (forking).
        if platform.system() == "Darwin":
            with joblib.parallel_backend("threading"):
                search.fit(X, y, **fit_params)
        else:
            search.fit(X, y, **fit_params)
        
    return search.best_estimator_, search.best_params_, float(-search.best_score_)



PROJECT_ROOT = Path(__file__).resolve().parent.parent
TUNING_CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"


def _load_tuning_config(config_path: Path = TUNING_CONFIG_PATH) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Tuning config not found at {config_path}")
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("tuning", {})


def _get_common(cfg: Dict[str, Any]) -> Dict[str, Any]:
    common = cfg.get("common", {})
    return {
        "cv": int(common.get("cv", 3)),
        "scoring": str(common.get("scoring", "neg_mean_absolute_error")),
        "n_jobs": int(common.get("n_jobs", -1)),
        "verbose": int(common.get("verbose", 1)),
        "random_state": int(common.get("random_state", 42)),
        "n_iter": int(common.get("n_iter", 30)),
    }


def _make_cv(common: Dict[str, Any]) -> KFold:
    return KFold(n_splits=common["cv"], shuffle=True, random_state=common["random_state"])


def get_param_distributions(model_key: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = _load_tuning_config() if cfg is None else cfg
    mapping = {
        "rf": "random_forest",
        "random_forest": "random_forest",
        "xgb": "xgboost",
        "xgboost": "xgboost",
        "lgbm": "lightgbm",
        "lightgbm": "lightgbm",
    }
    key = mapping.get(model_key)
    if key is None:
        raise ValueError("model_key must be one of: rf/xgb/lgbm")
    return cfg.get(key, {}) or {}


# ============================================
# IMPROVED: XGBoost with Early Stopping in CV
# ============================================
class XGBRegressorCV(xgb.XGBRegressor):
    """
    XGBoost wrapper that uses early stopping during CV.
    
    This ensures n_estimators search is realistic and prevents
    overfitting during hyperparameter tuning.
    """
    def fit(self, X, y, **fit_params):
        # Split X, y into train/validation
        from sklearn.model_selection import train_test_split
        import gc
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.15, random_state=self.random_state
        )
        
        # Add eval set for early stopping
        fit_params['eval_set'] = [(X_val, y_val)]
        fit_params['verbose'] = False # Disable verbose to reduce notebook output bloat
        
        # Call parent fit
        super().fit(X_tr, y_tr, **fit_params)
        
        # Explicit cleanup
        del X_tr, X_val, y_tr, y_val
        gc.collect()
        return self


class LGBMRegressorCV(lgb.LGBMRegressor):
    """
    LightGBM wrapper that uses early stopping during CV.
    """
    def fit(self, X, y, **fit_params):
        from sklearn.model_selection import train_test_split
        import gc
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.15, random_state=self.random_state
        )
        
        # Add eval set
        fit_params['eval_set'] = [(X_val, y_val)]
        fit_params['eval_metric'] = 'mae'
        
        # Early stopping callback
        if 'callbacks' not in fit_params:
            fit_params['callbacks'] = [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0) # Disable logging to reduce bloat
            ]
        
        super().fit(X_tr, y_tr, **fit_params)
        
        # Explicit cleanup
        del X_tr, X_val, y_tr, y_val
        gc.collect()
        return self


# ============================================
# Public API: RandomizedSearch (Improved)
# ============================================
def run_randomsearch_rf(X, y, n_iter: Optional[int] = None, random_state: Optional[int] = None):
    """
    Perform randomized hyperparameter search for Random Forest.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    n_iter : int, optional
        Number of parameter settings that are sampled.
    random_state : int, optional
        Seed used by the random number generator.

    Returns
    -------
    tuple
        (best_estimator, best_params, best_score)
    """

    cfg = _load_tuning_config()
    common = _get_common(cfg)
    param_dist = get_param_distributions("rf", cfg)

    rs = common["random_state"] if random_state is None else int(random_state)
    estimator = RandomForestRegressor(
        random_state=rs,
        n_jobs=1,  # Single-threaded tree building to avoid nested parallelism crash
    )
    
    cv = _make_cv(common)
    local_n_iter = n_iter if n_iter is not None else common["n_iter"]
    
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=local_n_iter,
        scoring=common["scoring"],
        cv=cv,
        random_state=rs,
        n_jobs=common["n_jobs"],  # Optimized core usage from config
        verbose=common["verbose"],
    )
    
    result = _run_search_with_progress(search, X, y)
    gc.collect()  # Free memory after search
    return result


def run_randomsearch_xgb(X, y, n_iter: Optional[int] = None, random_state: Optional[int] = None):
    """
    Perform randomized hyperparameter search for XGBoost.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    n_iter : int, optional
        Number of iterations for random search.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    tuple
        (best_estimator, best_params, best_score)
    """
    cfg = _load_tuning_config()
    common = _get_common(cfg)
    param_dist = get_param_distributions("xgb", cfg)

    rs = common["random_state"] if random_state is None else int(random_state)
    
    # Use CV wrapper with early stopping
    estimator = XGBRegressorCV(
        random_state=rs,
        n_jobs=1,  # Single-threaded to avoid OpenMP conflicts on macOS M1
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        early_stopping_rounds=30,
    )
    
    cv = _make_cv(common)
    local_n_iter = n_iter if n_iter is not None else common["n_iter"]
    
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=local_n_iter,
        scoring=common["scoring"],
        cv=cv,
        random_state=rs,
        n_jobs=common["n_jobs"], # Parallel search enabled
        verbose=2,
    )
    
    result = _run_search_with_progress(search, X, y)
    gc.collect()
    return result


def run_randomsearch_lgbm(X, y, n_iter: Optional[int] = None, random_state: Optional[int] = None, 
                          categorical_feature=None):
    """
    Perform randomized hyperparameter search for LightGBM.
    """
    cfg = _load_tuning_config()
    common = _get_common(cfg)
    param_dist = get_param_distributions("lgbm", cfg)

    rs = common["random_state"] if random_state is None else int(random_state)
    
    # Use CV wrapper with early stopping
    estimator = LGBMRegressorCV(
        random_state=rs,
        n_jobs=1,
        objective="regression_l1",
        verbose=-1,
    )
    
    cv = _make_cv(common)
    local_n_iter = n_iter if n_iter is not None else common["n_iter"]
    
    # For categorical features
    fit_params = {}
    if categorical_feature is not None:
        fit_params['categorical_feature'] = categorical_feature
    
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=local_n_iter,
        scoring=common["scoring"],
        cv=cv,
        random_state=rs,
        n_jobs=common["n_jobs"],
        verbose=2,
    )
    
    result = _run_search_with_progress(search, X, y, **fit_params)
    gc.collect()
    return result


# ============================================
# Public API: GridSearch
# ============================================
def run_gridsearch_rf(X, y, param_grid: Dict[str, Any], random_state: Optional[int] = None):
    cfg = _load_tuning_config()
    common = _get_common(cfg)

    rs = common["random_state"] if random_state is None else int(random_state)
    estimator = RandomForestRegressor(random_state=rs, n_jobs=1)
    
    cv = _make_cv(common)
    search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring=common["scoring"],
        cv=cv,
        n_jobs=common["n_jobs"], # Parallel core usage
        verbose=2,
    )
    
    result = _run_search_with_progress(search, X, y)
    gc.collect()
    return result


def run_gridsearch_xgb(X, y, param_grid: Dict[str, Any], random_state: Optional[int] = None):
    cfg = _load_tuning_config()
    common = _get_common(cfg)

    rs = common["random_state"] if random_state is None else int(random_state)
    estimator = XGBRegressorCV(
        random_state=rs,
        n_jobs=1,
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        early_stopping_rounds=30,
    )
    
    cv = _make_cv(common)
    search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring=common["scoring"],
        cv=cv,
        n_jobs=common["n_jobs"],
        verbose=2,
    )
    
    result = _run_search_with_progress(search, X, y)
    gc.collect()
    return result


def run_gridsearch_lgbm(X, y, param_grid: Dict[str, Any], random_state: Optional[int] = None,
                        categorical_feature=None):
    cfg = _load_tuning_config()
    common = _get_common(cfg)

    rs = common["random_state"] if random_state is None else int(random_state)
    estimator = LGBMRegressorCV(
        random_state=rs,
        n_jobs=1,
        objective="regression_l1",
        verbose=-1,
    )
    
    cv = _make_cv(common)
    
    fit_params = {}
    if categorical_feature is not None:
        fit_params['categorical_feature'] = categorical_feature
    
    search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring=common["scoring"],
        cv=cv,
        n_jobs=common["n_jobs"],
        verbose=2,
    )
    
    result = _run_search_with_progress(search, X, y, **fit_params)
    gc.collect()
    return result


def finalize_ml_model(
    model_name: str,
    best_params: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    random_state: int = 42,
    verbose: bool = True,
    categorical_feature: Optional[List[str]] = None
) -> Tuple[Any, Dict[str, float]]:
    """
    Final refit of the ML model with best parameters and evaluation on test set.

    Parameters
    ----------
    model_name : str
        'rf', 'xgb', or 'lgbm'.
    best_params : dict
        Hyperparameters found during tuning.
    X_train, y_train : array-like
        Full training data.
    X_test, y_test : array-like
        Test data for final metric reporting.
    random_state : int, default=42
        Seed for reproducibility.
    verbose : bool, default=True
        Whether to print status.

    Returns
    -------
    tuple
        (final_model, metrics)
    """
    start_time = time.time()
    
    cfg = _load_tuning_config()
    common = _get_common(cfg)
    safe_n_jobs = common.get("n_jobs", 1)
    
    if model_name == "rf":
        final_model = RandomForestRegressor(**best_params, random_state=random_state, n_jobs=safe_n_jobs)
        final_model.fit(X_train, y_train)
    else:
        # Split for internal validation to use early stopping for the final refit number of trees
        X_tr_split, X_val_split, y_tr_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.10, random_state=random_state
        )
        
        if model_name == "xgb":
            final_model = xgb.XGBRegressor(
                **best_params, 
                random_state=random_state, 
                n_jobs=safe_n_jobs,
                early_stopping_rounds=50
            )
            final_model.fit(
                X_tr_split, y_tr_split,
                eval_set=[(X_val_split, y_val_split)],
                verbose=False
            )
        else: # lgbm
            final_model = lgb.LGBMRegressor(
                **best_params, 
                random_state=random_state, 
                n_jobs=safe_n_jobs
            )
            final_model.fit(
                X_tr_split, y_tr_split,
                eval_set=[(X_val_split, y_val_split)],
                eval_metric="mae",
                callbacks=[lgb.early_stopping(50, verbose=False)],
                categorical_feature=categorical_feature
            )
    
    train_time = time.time() - start_time
    
    # Test Evaluation
    y_pred = final_model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)
    metrics['train_time'] = train_time
    
    if verbose:
        print(f"[{model_name.upper()}][FinalRefit] Completed in {train_time:.2f}s")
        print(f"[{model_name.upper()}][TestMetrics] MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.4f}")

    return final_model, metrics


# -----------------------------
# Narrow grid builder
# -----------------------------
def make_narrow_grid(
    model_name: str,
    best_params: Dict[str, Any],
    max_grid_size: int = 81,
) -> Dict[str, Any]:
    """Programmatic narrow grid around best_params.

    Rules:
    - No hard-coded "full grid": derive from best_params
    - Integer params: small deltas
    - Float params: multiplicative factors (0.5, 0.8, 1.0, 1.2, 1.5)
    - subsample/colsample clipped to [0.5, 1.0]
    - max_features / max_samples (if float) clipped to [0.1, 1.0]
    - min_samples_split>=2, min_samples_leaf>=1
    - XGB/LGBM gamma/min_split_gain>=0, reg_alpha/reg_lambda>=0
    - LGBM max_depth: if -1 -> [-1, 10, 20], else neighborhood
    - list values unique + sorted when possible
    - grid size capped by max_grid_size (shrinks less-important params first)
    """
    if model_name.lower() not in {"rf", "xgb", "lgbm", "cnn", "lstm", "hybrid"}:
        raise ValueError("model_name must be one of: rf/xgb/lgbm/cnn/lstm/hybrid")

    if not best_params:
        return {}

    def _uniq_sorted(values):
        uniq = []
        for v in values:
            if v not in uniq:
                uniq.append(v)
        try:
            return sorted(uniq)
        except TypeError:
            return uniq

    def _clip_05_10(v: float) -> float:
        return max(0.5, min(1.0, float(v)))

    def _clip_01_10(v: float) -> float:
        return max(0.1, min(1.0, float(v)))

    def _int_deltas(v: int) -> list[int]:
        v = int(v)
        if v >= 2000:
            steps = [-500, 0, 500]
        elif v >= 800:
            steps = [-200, 0, 200]
        elif v >= 300:
            steps = [-100, 0, 100]
        elif v >= 50:
            steps = [-10, 0, 10]
        elif v >= 10:
            steps = [-5, 0, 5]
        elif v >= 5:
            steps = [-2, 0, 2]
        else:
            steps = [-1, 0, 1]
        return [v + s for s in steps]

    def _narrow_int(name: str, v: Any) -> list[Any]:
        if v is None:
            return [None]
        vals = _int_deltas(int(v))

        if name == "min_samples_split":
            vals = [x for x in vals if x >= 2]
        if name == "min_samples_leaf":
            vals = [x for x in vals if x >= 1]
        if name in {
            "n_estimators", "num_leaves", "max_bin", "min_child_samples",
            "num_filters", "hidden_size", "lstm_hidden_size", "num_layers", "lstm_num_layers"
        }:
            vals = [x for x in vals if x >= 2]

        if name == "subsample_freq":
            vals = [x for x in vals if x >= 0]

        if name == "max_depth":
            # RF: can be None; XGB/LGBM: int >= 1 or -1 (LGBM handled specifically)
            vals = [x for x in vals if x >= 1]

        if name == "min_child_weight":
            vals = [x for x in vals if x >= 0]

        if name == "max_delta_step":
            vals = [x for x in vals if x >= 0]

        if name == "kernel_size":
            vals = [x for x in vals if x >= 3]

        return _uniq_sorted(vals) or [int(v)]

    def _narrow_float(name: str, v: Any) -> list[float]:
        v = float(v)
        factors = [0.5, 0.8, 1.0, 1.2, 1.5]
        vals = [v * f for f in factors]

        if name in {"subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode"}:
            vals = [_clip_05_10(x) for x in vals]

        if name in {"max_features", "max_samples"}:
            vals = [_clip_01_10(x) for x in vals]

        if name in {"reg_alpha", "reg_lambda", "gamma", "min_split_gain"}:
            vals = [max(0.0, float(x)) for x in vals]

        if name == "learning_rate":
            vals = [max(1e-6, float(x)) for x in vals]

        vals = [round(float(x), 6) for x in vals]
        return _uniq_sorted(vals) or [round(v, 6)]

    skip = {
        "random_state",
        "n_jobs",
        "objective",
        "eval_metric",
        "tree_method",
        "verbosity",
        "verbose",
        "device",
        "booster",
        "early_stopping_rounds",
    }

    if model_name == "rf":
        allowed = {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "max_samples",
            "criterion",
            "ccp_alpha",
        }
    elif model_name == "xgb":
        allowed = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "colsample_bylevel",
            "colsample_bynode",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "max_delta_step",
        }
    elif model_name == "lgbm":
        allowed = {
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "subsample",
            "subsample_freq",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "min_child_samples",
            "min_split_gain",
            "max_bin",
        }
    elif model_name.lower() == "cnn":
        allowed = {"num_filters", "kernel_size", "learning_rate", "batch_size"}
    elif model_name.lower() == "lstm":
        allowed = {"hidden_size", "num_layers", "learning_rate", "batch_size"}
    elif model_name.lower() == "hybrid":
        allowed = {"num_filters", "kernel_size", "lstm_hidden_size", "lstm_num_layers", "learning_rate", "batch_size"}
    else:  # fallback
        allowed = set(best_params.keys())

    grid: Dict[str, Any] = {}

    for name, v in best_params.items():
        if name in skip:
            continue
        if name not in allowed:
            continue

        # special case: LGBM max_depth
        if model_name == "lgbm" and name == "max_depth":
            try:
                iv = int(v)
            except (TypeError, ValueError):
                grid[name] = [v]
                continue
            if iv == -1:
                grid[name] = [-1, 10, 20]
            else:
                grid[name] = _narrow_int(name, iv)
            continue

        # RF max_depth may be None
        if model_name == "rf" and name == "max_depth" and v is None:
            grid[name] = [None]
            continue

        if isinstance(v, bool) or isinstance(v, str):
            grid[name] = [v]
        elif isinstance(v, (int, np.integer)):
            grid[name] = _narrow_int(name, int(v))
        elif isinstance(v, (float, np.float32, np.float64)):
            grid[name] = _narrow_float(name, float(v))
        else:
            grid[name] = [v]

    # Validations / clipping
        # RF: max_samples cannot be used when bootstrap=False
    if model_name == "rf":
        if best_params.get("bootstrap") is False and "max_samples" in grid:
                grid["max_samples"] = [None]
    if "min_samples_split" in grid:
        grid["min_samples_split"] = (
            [x for x in grid["min_samples_split"] if int(x) >= 2] or [int(best_params["min_samples_split"])]
        )
    if "min_samples_leaf" in grid:
        grid["min_samples_leaf"] = (
            [x for x in grid["min_samples_leaf"] if int(x) >= 1] or [int(best_params["min_samples_leaf"])]
        )
        
    for c in ["subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode"]:
        if c in grid:
            grid[c] = _uniq_sorted([_clip_05_10(x) for x in grid[c]])

    for c in ["max_features", "max_samples"]:
        if c in grid and any(isinstance(x, float) for x in grid[c]):
            grid[c] = _uniq_sorted([_clip_01_10(x) for x in grid[c]])

    if "reg_alpha" in grid:
        grid["reg_alpha"] = _uniq_sorted([max(0.0, float(x)) for x in grid["reg_alpha"]])
    if "reg_lambda" in grid:
        grid["reg_lambda"] = _uniq_sorted([max(0.0, float(x)) for x in grid["reg_lambda"]])

    if "gamma" in grid:
        grid["gamma"] = _uniq_sorted([max(0.0, float(x)) for x in grid["gamma"]])
    if "min_split_gain" in grid:
        grid["min_split_gain"] = _uniq_sorted([max(0.0, float(x)) for x in grid["min_split_gain"]])

    if "min_child_weight" in grid:
        grid["min_child_weight"] = (
            [x for x in grid["min_child_weight"] if float(x) >= 0.0] or [float(best_params["min_child_weight"])]
        )

    if "min_child_samples" in grid:
        grid["min_child_samples"] = (
            [x for x in grid["min_child_samples"] if int(x) >= 1] or [int(best_params["min_child_samples"])]
        )
    if "subsample_freq" in grid:
        grid["subsample_freq"] = (
            [x for x in grid["subsample_freq"] if int(x) >= 0] or [int(best_params["subsample_freq"])]
        )
    if "max_bin" in grid:
        grid["max_bin"] = (
            [x for x in grid["max_bin"] if int(x) >= 32] or [int(best_params["max_bin"])]
        )
    if "max_delta_step" in grid:
        grid["max_delta_step"] = (
            [x for x in grid["max_delta_step"] if int(x) >= 0] or [int(best_params["max_delta_step"])]
        )

    def _grid_size(g: Dict[str, Any]) -> int:
        size = 1
        for values in g.values():
            size *= max(1, len(values))
        return size

    # Prevent combinatorial explosions by shrinking secondary params first
    if max_grid_size is not None and int(max_grid_size) > 0:
        max_grid_size_int = int(max_grid_size)

        if model_name == "rf":
            shrink_order = [
                "ccp_alpha",
                "criterion",
                "max_samples",
                "bootstrap",
                "max_features",
                "min_samples_leaf",
                "min_samples_split",
                "max_depth",
                "n_estimators",
            ]
        elif model_name == "xgb":
            shrink_order = [
                "max_delta_step",
                "gamma",
                "min_child_weight",
                "colsample_bynode",
                "colsample_bylevel",
                "colsample_bytree",
                "subsample",
                "reg_lambda",
                "reg_alpha",
                "max_depth",
                "n_estimators",
                "learning_rate",
            ]
        elif model_name.lower() == "cnn":
            shrink_order = ["kernel_size", "num_filters", "learning_rate", "batch_size"]
        elif model_name.lower() == "lstm":
            shrink_order = ["num_layers", "hidden_size", "learning_rate", "batch_size"]
        elif model_name.lower() == "hybrid":
            shrink_order = ["lstm_num_layers", "lstm_hidden_size", "kernel_size", "num_filters", "learning_rate", "batch_size"]
        else:  # lgbm
            shrink_order = [
                "max_bin",
                "subsample_freq",
                "min_split_gain",
                "min_child_samples",
                "colsample_bytree",
                "subsample",
                "reg_lambda",
                "reg_alpha",
                "max_depth",
                "num_leaves",
                "n_estimators",
                "learning_rate",
            ]

        while _grid_size(grid) > max_grid_size_int:
            shrunk = False
            for param in shrink_order:
                if param in grid and len(grid[param]) > 1:
                    grid[param] = [best_params[param]]
                    shrunk = True
                    break
            if not shrunk:
                break

    return grid


# ---------------------------------------------------------------------
# Dedicated DL wrappers to mirror ML experience (birebir aynı yapı)
# ---------------------------------------------------------------------

def run_randomsearch_cnn(*args, **kwargs):
    kwargs['model_name'] = 'CNN'
    if 'search_space' not in kwargs:
        kwargs['search_space'] = load_dl_config('CNN')
    return run_dl_randomsearch(*args, **kwargs)

def run_gridsearch_cnn(*args, **kwargs):
    kwargs['model_name'] = 'CNN'
    return run_dl_gridsearch(*args, **kwargs)

def run_randomsearch_lstm(*args, **kwargs):
    kwargs['model_name'] = 'LSTM'
    if 'search_space' not in kwargs:
        kwargs['search_space'] = load_dl_config('LSTM')
    return run_dl_randomsearch(*args, **kwargs)

def run_gridsearch_lstm(*args, **kwargs):
    kwargs['model_name'] = 'LSTM'
    return run_dl_gridsearch(*args, **kwargs)

def run_randomsearch_hybrid(*args, **kwargs):
    kwargs['model_name'] = 'Hybrid'
    if 'search_space' not in kwargs:
        kwargs['search_space'] = load_dl_config('Hybrid')
    return run_dl_randomsearch(*args, **kwargs)

def run_gridsearch_hybrid(*args, **kwargs):
    kwargs['model_name'] = 'Hybrid'
    return run_dl_gridsearch(*args, **kwargs)

# =====================================================================
# DEEP LEARNING TUNING
# =====================================================================

def load_dl_config(model_name: str = None, config_path: str = "configs/models.yaml"):
    """
    Loads Deep Learning hyperparameter search spaces directly from the central YAML configuration.
    """
    full_path = Path(__file__).resolve().parent.parent / config_path
    
    with open(full_path, 'r') as f:
        config = yaml.safe_load(f)

    if model_name is None:
        return config    
    
    tuning_config = config.get('tuning', {})
    
    if model_name == 'CNN':
        return tuning_config.get('cnn', {})
    elif model_name == 'LSTM':
        return tuning_config.get('lstm', {})
    elif model_name == 'Hybrid':
        return tuning_config.get('cnn_lstm', {})
    else:
        raise ValueError(f"Unknown deep learning model: {model_name}")


class SequenceJobDataset(torch.utils.data.Dataset):
    """
    Sliding window dataset for Deep Learning models.
    Converts tabular rows into 3D sequential tensors (batch, seq_len, features).
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 10):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len
        
    def __len__(self):
        return max(0, len(self.features) - self.seq_len + 1)
        
    def __getitem__(self, idx):
        x_seq = self.features[idx : idx + self.seq_len]
        y_target = self.targets[idx + self.seq_len - 1]
        return x_seq, y_target


def prepare_dl_datasets(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    val_split: float = 0.2,
    random_state: int = 42,
    seq_len: int = 10,
):
    """
    Standardizes DL data preparation: Scaling, Sequence conversion, and Dataset creation.

    Automatically splits the training data into a train and a validation set so
    that Early Stopping can monitor generalisation performance rather than
    in-sample loss.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices.
    y_train, y_test : np.ndarray
        Target vectors.
    val_split : float, default=0.2
        Fraction of training samples reserved for validation.
    random_state : int, default=42
        Seed for the train/val split.
    seq_len : int, default=10
        Length of the sliding window for temporal prediction.

    Returns
    -------
    tuple
        (train_dataset, val_dataset, test_dataset, y_test_raw, scaler_x, scaler_y, input_features)
    """
    from torch.utils.data import random_split

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit & Transform
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled  = scaler_x.transform(X_test)

    # Target scaling (reshape for scaler)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1))

    # To handle testing for the very first item without losing sequence length,
    # we prepend the last (seq_len - 1) items of the training set to the test set.
    if seq_len > 1:
        prefix_x = X_train_scaled[-(seq_len - 1):]
        X_test_scaled_ext = np.vstack([prefix_x, X_test_scaled])
        
        # Prepend zeros for targets as well to maintain alignment
        prefix_y = np.zeros((seq_len - 1, 1)) 
        y_test_scaled_ext = np.vstack([prefix_y, y_test_scaled])
    else:
        X_test_scaled_ext = X_test_scaled
        y_test_scaled_ext = y_test_scaled

    full_dataset = SequenceJobDataset(X_train_scaled, y_train_scaled.flatten(), seq_len=seq_len)
    test_dataset = SequenceJobDataset(X_test_scaled_ext, y_test_scaled_ext.flatten(), seq_len=seq_len)

    # --- chronological train / val split -----------------------------------
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    # Strict chronological split (Temporal Integrity)
    # Validation data is the last chronological portion of the training data
    indices = list(range(n_total))
    train_dataset = torch.utils.data.Subset(full_dataset, indices[:n_train])
    val_dataset = torch.utils.data.Subset(full_dataset, indices[n_train:])
    # -----------------------------------------------------------------------

    input_features = X_train.shape[1]
    y_test_raw     = y_test.flatten()

    return train_dataset, val_dataset, test_dataset, y_test_raw, scaler_x, scaler_y, input_features

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=1e-4):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        self.val_loss_min = val_loss

def train_dl_model(model, train_loader, val_loader, criterion, optimizer, epochs=50, patience=5, device=None):
    """
    Trains a generic PyTorch model with Early Stopping and Learning Rate Scheduling.
    """
    # Reproducibility seed — ensures consistent results across runs
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if device is None:
        device = get_default_device()
        
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, verbose=False)
    
    # Academic Standard: Learning Rate Scheduler (Reduces LR when validation loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                loss = criterion(val_outputs.view(-1), y_val.view(-1))
                val_loss += loss.item() * X_val.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        # Track best weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        
        # Update scheduler
        scheduler.step(val_loss)
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"    [Early Stopping] Triggered at epoch {epoch+1}")
            break
            
    # Load best weights
    model.load_state_dict(best_model_wts)
    return model

def create_model_instance(model_name, input_features, params):
    dropout = params.get('dropout', 0.2)
    if model_name == 'CNN':
        return RuntimePredictorCNN(input_features, params['num_filters'], params.get('kernel_size', 1), dropout=dropout)
    elif model_name == 'LSTM':
        return RuntimePredictorLSTM(input_features, params['hidden_size'], params.get('num_layers', 1), dropout=dropout)
    elif model_name == 'Hybrid':
        return RuntimePredictorCNNLSTM(
            input_features, 
            params['num_filters'], 
            params.get('kernel_size', 1), 
            params['lstm_hidden_size'], 
            params.get('lstm_num_layers', 1),
            dropout=dropout
        )
    else:
        raise ValueError("Unsupported model architecture")

def run_dl_randomsearch(model_name, search_space, train_dataset, val_dataset, input_features, 
                        scaler_y, y_test_raw, test_dataset, num_trials=10, tuning_epochs=10, patience=5, device=None):
    """
    Perform randomized hyperparameter search for Deep Learning models.

    Parameters
    ----------
    model_name : str
        Architecture type ('CNN', 'LSTM', 'Hybrid').
    search_space : dict
        Hyperparameter search space with list values for sampled keys.
    train_dataset, val_dataset : torch.utils.data.Dataset
        Training and validation datasets.
    input_features : int
        Number of input features.
    scaler_y : sklearn.preprocessing.MinMaxScaler
        Scaler for inverse transformation of targets.
    y_test_raw : np.ndarray
        True unscaled target values for evaluation.
    test_dataset : torch.utils.data.Dataset
        Test features in format of Sequence dataset.
    num_trials : int, default=10
        Number of random parameter samples.
    tuning_epochs : int, default=10
        Epochs per trial for tuning.
    patience : int, default=5
        Early stopping patience.
    device : str, default=None
        Compute device ('cpu', 'cuda', or 'mps').

    Returns
    -------
    tuple
        (best_params, best_rmse)
    """
    if device is None:
        device = get_default_device()

    best_rmse = float('inf')
    best_params = None
    
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    
    for i in range(num_trials):
        # Only sample from values that are lists; others are treated as constant
        params = {}
        for k, v in search_space.items():
            if isinstance(v, list):
                params[k] = random.choice(v)
            else:
                params[k] = v
                
        print(f"  Random Trial {i+1}/{num_trials} -> trying: {params}")
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        model = create_model_instance(model_name, input_features, params)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Train with early stopping internally to tune
        model = train_dl_model(model, train_loader, val_loader, criterion, optimizer, epochs=tuning_epochs, patience=patience, device=device)
        
        # Evaluate on the validation set for unbiased hyperparameter selection
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
        current_rmse = np.sqrt(val_loss / len(val_dataset)) # Scaled RMSE for selection
            
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_params = params
            
    return best_params, best_rmse

def run_dl_gridsearch(model_name, grid_search_space, train_dataset, val_dataset, input_features, 
                      scaler_y, y_test_raw, test_dataset, tuning_epochs=10, patience=5, device=None):
    """
    Evaluates every combinatorial pair inside `grid_search_space` (Narrow Grid)
    Optimized with GPU acceleration.
    """
    if device is None:
        device = get_default_device()
    # Separate tunable params (lists) from constants
    tunable_keys = [k for k, v in grid_search_space.items() if isinstance(v, list)]
    tunable_values = [v for k, v in grid_search_space.items() if isinstance(v, list)]
    constants = {k: v for k, v in grid_search_space.items() if not isinstance(v, list)}
    
    grid_combinations = []
    for v in itertools.product(*tunable_values):
        p = dict(zip(tunable_keys, v))
        p.update(constants)
        grid_combinations.append(p)
    
    best_rmse = float('inf')
    best_params = None
    
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    
    for i, params in enumerate(grid_combinations):
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        model = create_model_instance(model_name, input_features, params)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Train with early stopping
        model = train_dl_model(model, train_loader, val_loader, criterion, optimizer, epochs=tuning_epochs, patience=patience, device=device)
        
        # Evaluate on the validation set for unbiased hyperparameter selection
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
        current_rmse = np.sqrt(val_loss / len(val_dataset)) # Scaled RMSE for selection
            
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_params = params
            
    return best_params, best_rmse

def finalize_dl_model(model_name, best_params, train_dataset, val_dataset, input_features, 
                     scaler_y, y_test_raw, test_dataset, final_epochs=50, patience=10, device=None):
    """
    Trains the final model configuration utilizing deep Early Stopping across all epochs and evaluates metrics.
    """
    if device is None:
        device = get_default_device()
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False) # Large batch for valid evaluation speed
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    
    model = create_model_instance(model_name, input_features, best_params)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    
    start_time = time.time()
    
    # Train robust final model with strictly more epochs and deeper early stopping patience setup
    final_model = train_dl_model(model, train_loader, val_loader, criterion, optimizer, epochs=final_epochs, patience=patience, device=device)
    
    train_time = time.time() - start_time
    
    final_model.eval()
    preds_scaled = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds = final_model(X_batch).cpu().numpy()
            preds_scaled.extend(preds)
            
    preds_scaled = np.array(preds_scaled)
    preds_unscaled = np.maximum(scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten(), 0)
        
    mae = mean_absolute_error(y_test_raw, preds_unscaled)
    rmse = np.sqrt(mean_squared_error(y_test_raw, preds_unscaled))
    r2 = r2_score(y_test_raw, preds_unscaled)
    
    return final_model, {'mae': mae, 'rmse': rmse, 'r2': r2, 'train_time': train_time}
