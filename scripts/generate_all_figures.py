#!/usr/bin/env python3
"""
generate_all_figures.py
=======================
Generates all 18 thesis figures. Mirrors the exact plotting code from all
6 EN notebooks (NB01-NB05); figures are saved to results/figures/.

Usage
-----
    python scripts/generate_all_figures.py --mode all
    python scripts/generate_all_figures.py --mode workload
    python scripts/generate_all_figures.py --mode features
    python scripts/generate_all_figures.py --mode models
    python scripts/generate_all_figures.py --mode scheduler
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import prepare_features_for_model
from src.simulation.multi_node_simulator import (
    MultiNodeClusterSimulator,
    provision_heterogeneous_gpu_cluster,
)
from src.simulation.scheduler_simulator import (
    FIFOScheduler,
    SJFPredScheduler,
    SJFScheduler,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "results" / "checkpoints"

# ---------------------------------------------------------------------------
# Global plot style (mirrors notebooks)
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "figure.figsize": (12, 6),
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "font.family": "sans-serif",
    }
)

_DPI = 300  # save DPI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved -> %s", path.name)


def _clean_figures(figures_dir: Path) -> None:
    """Remove all .png files except project_logo.png."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    for f in figures_dir.glob("*.png"):
        if f.name != "project_logo.png":
            f.unlink()
            logger.info("Removed stale figure: %s", f.name)


def _load_ckpt(name: str) -> dict:
    path = CHECKPOINTS_DIR / f"{name}.json"
    with path.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data() -> tuple:
    """Return (job_df, X_num, y_num, X_oh, y_oh, X_nat, y_nat)."""
    logger.info("Loading numeric feature set...")
    job_df, _, X_num, _, y_num, _, _ = prepare_features_for_model(
        dataset="main", feature_mode="numeric_only"
    )
    logger.info("Loading one-hot categorical feature set...")
    _, _, X_oh, _, y_oh, _, _ = prepare_features_for_model(
        dataset="main", feature_mode="with_categorical_onehot"
    )
    logger.info("Loading native categorical feature set...")
    _, _, X_nat, _, y_nat, _, _ = prepare_features_for_model(
        dataset="main", feature_mode="with_categorical_native"
    )
    return job_df, X_num, y_num, X_oh, y_oh, X_nat, y_nat


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_models() -> dict:
    """Load all saved models and scalers into a dict."""
    logger.info("Loading tree-based models...")
    models: dict = {}
    for key, fname in [
        ("rf_num",      "rf_numeric.joblib"),
        ("xgb_num",     "xgb_numeric.joblib"),
        ("lgb_num",     "lgbm_numeric.joblib"),
        ("rf_cat",      "rf_categorical.joblib"),
        ("xgb_cat",     "xgb_categorical.joblib"),
        ("lgb_cat_oh",  "lgbm_categorical.joblib"),
        ("lgb_cat_nat", "lgbm_categorical_native.joblib"),
    ]:
        p = MODELS_DIR / fname
        if p.exists():
            models[key] = joblib.load(p)
        else:
            logger.warning("Model missing: %s", fname)

    logger.info("Loading scalers...")
    for key, fname in [
        ("scaler_x_num", "lstm_scaler_x.joblib"),
        ("scaler_y_num", "lstm_scaler_y.joblib"),
        ("scaler_x_cat", "lstm_scaler_x_cat.joblib"),
        ("scaler_y_cat", "lstm_scaler_y_cat.joblib"),
    ]:
        p = MODELS_DIR / fname
        if p.exists():
            models[key] = joblib.load(p)
        else:
            logger.warning("Scaler missing: %s", fname)

    def _load_pt(fname: str):
        p = MODELS_DIR / fname
        if not p.exists():
            logger.warning("PyTorch model missing: %s", fname)
            return None
        m = torch.load(p, map_location="cpu", weights_only=False)
        m.eval()
        return m

    logger.info("Loading deep learning models...")
    dl_map = {
        "cnn_static_num":    "cnn_numeric.pth",
        "lstm_static_num":   "lstm_numeric.pth",
        "hybrid_static_num": "cnn_lstm_numeric.pth",
        "cnn_static_cat":    "cnn_categorical_pt.pth",
        "lstm_static_cat":   "lstm_categorical_pt.pth",
        "hybrid_static_cat": "cnn_lstm_categorical_pt.pth",
        "cnn_seq_num":       "cnn_numeric_seq.pth",
        "lstm_seq_num":      "lstm_numeric_seq.pth",
        "hybrid_seq_num":    "cnn_lstm_numeric_seq.pth",
        "cnn_seq_cat":       "cnn_categorical_seq.pth",
        "lstm_seq_cat":      "lstm_categorical_seq.pth",
        "hybrid_seq_cat":    "cnn_lstm_categorical_seq.pth",
    }
    for key, fname in dl_map.items():
        models[key] = _load_pt(fname)

    return models


# ---------------------------------------------------------------------------
# DL batch inference  (mirrors NB05 predict_in_batches)
# ---------------------------------------------------------------------------

def _predict_batches(
    model,
    x_2d: torch.Tensor,
    is_sequence: bool = True,
    seq_len: int = 10,
    batch_size: int = 8192,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    if model is None:
        return np.full(x_2d.shape[0], np.nan)
    model.eval()
    n = x_2d.shape[0]
    all_preds: list = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            if is_sequence:
                batch_X = []
                for idx in range(i, end):
                    start = max(0, idx - seq_len + 1)
                    seq = x_2d[start : idx + 1]
                    pad = seq_len - seq.shape[0]
                    if pad > 0:
                        padding = seq[0].unsqueeze(0).repeat(pad, 1)
                        seq = torch.cat([padding, seq], dim=0)
                    batch_X.append(seq.unsqueeze(0))
                bt = torch.cat(batch_X, dim=0).to(device)
                preds = model(bt)
            else:
                bt = x_2d[i:end].unsqueeze(1).to(device)
                preds = model(bt)
            all_preds.append(preds.cpu().numpy().flatten())
            if (i // batch_size) % 5 == 0:
                gc.collect()
    return np.concatenate(all_preds)


# ---------------------------------------------------------------------------
# NB01 -- Data Overview (4 figures)
# ---------------------------------------------------------------------------

def plot_nb01(job_df: pd.DataFrame, out_dir: Path) -> None:
    logger.info("[NB01] Generating workload overview figures...")

    # Figure 1 -- dual histogram: raw + log10
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(job_df["job_runtime"], bins=80, color="steelblue", edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("Runtime (seconds)")
    axes[0].set_ylabel("Number of Jobs")
    axes[0].set_title("Raw Runtime Distribution")
    log_runtimes = np.log10(job_df["job_runtime"] + 1)
    axes[1].hist(log_runtimes, bins=60, color="darkorange", edgecolor="white", linewidth=0.4)
    axes[1].set_xlabel("log10(Runtime + 1)")
    axes[1].set_ylabel("Number of Jobs")
    axes[1].set_title("Log10 Runtime Distribution")
    fig.suptitle("Job Runtime Distribution - Alibaba PAI 100K", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "nb01_runtime_hist_dual.png", out_dir)

    # Figure 2 -- CDF with median + P95
    runtimes_sorted = np.sort(job_df["job_runtime"].values)
    cdf = np.arange(1, len(runtimes_sorted) + 1) / len(runtimes_sorted)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(runtimes_sorted, cdf, lw=2, color="royalblue")
    ax.axvline(
        np.median(runtimes_sorted), color="crimson", ls="--",
        label=f"Median = {np.median(runtimes_sorted):.0f}s",
    )
    ax.axvline(
        np.percentile(runtimes_sorted, 95), color="orange", ls="--",
        label=f"P95 = {np.percentile(runtimes_sorted, 95):.0f}s",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (seconds, log scale)")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative Distribution of Job Runtimes")
    ax.legend()
    plt.tight_layout()
    _save(fig, "nb01_runtime_cdf.png", out_dir)

    # Figure 3 -- hourly arrival rate
    ts = job_df.set_index("arrival_time")["job_runtime"].resample("1h").count()
    t0 = ts.index[0]
    elapsed_days = (ts.index - t0).total_seconds() / 86400
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(elapsed_days, ts.values, lw=1.2, color="seagreen")
    ax.fill_between(elapsed_days, ts.values, alpha=0.15, color="seagreen")
    ax.set_xlabel("Elapsed Time (days)")
    ax.set_ylabel("Jobs per Hour")
    ax.set_title("Hourly Job Arrival Rate Over the Trace Duration")
    plt.tight_layout()
    _save(fig, "nb01_arrival_rate.png", out_dir)

    # Figure 4 -- GPU demand bar
    gpu_counts = job_df["gpu_demand"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    gpu_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("GPU Demand (per job)")
    ax.set_ylabel("Number of Jobs")
    ax.set_title("Distribution of GPU Demand per Job")
    plt.tight_layout()
    _save(fig, "nb01_gpu_demand_bar.png", out_dir)


# ---------------------------------------------------------------------------
# NB02 -- Workload Analysis (3 figures)
# ---------------------------------------------------------------------------

def plot_nb02(job_df: pd.DataFrame, out_dir: Path) -> None:
    logger.info("[NB02] Generating workload analysis figures...")

    # Figure 5 -- log10 histogram
    log_runtimes = np.log10(job_df["job_runtime"] + 1)
    hist_counts, bin_edges = np.histogram(log_runtimes, bins=60)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(bin_centres, hist_counts, width=np.diff(bin_edges), color="steelblue",
           edgecolor="white", linewidth=0.4)
    ax.set_xlabel("log10(Runtime + 1)")
    ax.set_ylabel("Number of Jobs")
    ax.set_title("Job Runtime Distribution (Log10 Scale)")
    plt.tight_layout()
    _save(fig, "nb02_runtime_hist_log.png", out_dir)

    # Figure 6 -- hourly arrival rate
    ts = job_df.set_index("arrival_time")["job_runtime"].resample("1h").count()
    t0 = ts.index[0]
    elapsed_days = (ts.index - t0).total_seconds() / 86400
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(elapsed_days, ts.values, lw=1.2, color="seagreen")
    ax.fill_between(elapsed_days, ts.values, alpha=0.15, color="seagreen")
    ax.set_xlabel("Elapsed Time (days)")
    ax.set_ylabel("Jobs per Hour")
    ax.set_title("Hourly Job Arrival Rate Over Trace Duration")
    plt.tight_layout()
    _save(fig, "nb02_arrival_rate.png", out_dir)

    # Figure 7 -- GPU vs runtime scatter (5k sample)
    sample = job_df.sample(n=min(5000, len(job_df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        sample["gpu_demand"], sample["job_runtime"],
        alpha=0.3, s=8, c=sample["gpu_demand"], cmap="plasma",
    )
    plt.colorbar(sc, ax=ax, label="GPU Demand")
    ax.set_yscale("log")
    ax.set_xlabel("GPU Demand")
    ax.set_ylabel("Job Runtime (s, log scale)")
    ax.set_title("GPU Demand vs Job Runtime (5,000-job sample)")
    plt.tight_layout()
    _save(fig, "nb02_gpu_vs_runtime_scatter.png", out_dir)


# ---------------------------------------------------------------------------
# NB03 -- Feature Engineering (2 figures)
# ---------------------------------------------------------------------------

def plot_nb03(job_df: pd.DataFrame, out_dir: Path) -> None:
    logger.info("[NB03] Generating feature engineering figures...")

    # Figure 8 -- cluster state over time (dual panel)
    if "active_job_count" not in job_df.columns or "cluster_load_gpu" not in job_df.columns:
        logger.warning("[NB03] Cluster-state columns not in job_df -- skipping nb03_cluster_state.png")
    else:
        days = job_df["arrival_sec"] / 86400
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        axes[0].plot(days, job_df["active_job_count"], color="steelblue", lw=0.8)
        axes[0].set_ylabel("Active Job Count")
        axes[0].set_title("Active Jobs Over Trace Duration")
        axes[1].plot(days, job_df["cluster_load_gpu"], color="darkorange", lw=0.8)
        axes[1].set_ylabel("Cluster GPU Load")
        axes[1].set_xlabel("Elapsed Time (days)")
        axes[1].set_title("GPU Cluster Load Over Trace Duration")
        fig.suptitle("Cluster State During Trace Period", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save(fig, "nb03_cluster_state.png", out_dir)

    # Figure 9 -- Pearson correlation heatmap
    numeric_features = [
        "arrival_sec", "gpu_demand", "num_inst", "num_cpu",
        "cluster_load_gpu", "active_job_count", "job_runtime",
    ]
    numeric_features = [c for c in numeric_features if c in job_df.columns]
    corr = job_df[numeric_features].corr()
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, square=True)
    ax.set_title("Pearson Correlation - Numeric Features vs job_runtime")
    plt.tight_layout()
    _save(fig, "nb03_correlation_heatmap.png", out_dir)


# ---------------------------------------------------------------------------
# NB04 -- Model Performance (5 figures)
# ---------------------------------------------------------------------------

def plot_nb04(models: dict, X_num: pd.DataFrame, y_num: pd.Series, out_dir: Path) -> None:
    logger.info("[NB04] Generating model performance figures...")

    def _m(ckpt_name: str) -> dict:
        ck = _load_ckpt(ckpt_name)
        return ck["metrics"]

    all_data = [
        # Exp A -- ML Numeric
        {"Experiment": "Exp A", "Track": "ML Numeric",    "Model": "Random Forest",    "mae": _m("exp_a_rf")["mae"],       "rmse": _m("exp_a_rf")["rmse"],       "r2": _m("exp_a_rf")["r2"]},
        {"Experiment": "Exp A", "Track": "ML Numeric",    "Model": "XGBoost",          "mae": _m("exp_a_xgb")["mae"],      "rmse": _m("exp_a_xgb")["rmse"],      "r2": _m("exp_a_xgb")["r2"]},
        {"Experiment": "Exp A", "Track": "ML Numeric",    "Model": "LightGBM",         "mae": _m("exp_a_lgbm")["mae"],     "rmse": _m("exp_a_lgbm")["rmse"],     "r2": _m("exp_a_lgbm")["r2"]},
        # Exp B -- ML Categorical
        {"Experiment": "Exp B", "Track": "ML Categorical","Model": "RF (One-Hot)",     "mae": _m("exp_b_rf_oh")["mae"],    "rmse": _m("exp_b_rf_oh")["rmse"],    "r2": _m("exp_b_rf_oh")["r2"]},
        {"Experiment": "Exp B", "Track": "ML Categorical","Model": "XGB (One-Hot)",    "mae": _m("exp_b_xgb_oh")["mae"],   "rmse": _m("exp_b_xgb_oh")["rmse"],   "r2": _m("exp_b_xgb_oh")["r2"]},
        {"Experiment": "Exp B", "Track": "ML Categorical","Model": "LGBM (One-Hot)",   "mae": _m("exp_b_lgbm_oh")["mae"],  "rmse": _m("exp_b_lgbm_oh")["rmse"],  "r2": _m("exp_b_lgbm_oh")["r2"]},
        {"Experiment": "Exp B", "Track": "ML Categorical","Model": "LGBM (Native)",    "mae": _m("exp_b_lgbm_nat")["mae"], "rmse": _m("exp_b_lgbm_nat")["rmse"], "r2": _m("exp_b_lgbm_nat")["r2"]},
        # Exp C -- DL Numeric Static
        {"Experiment": "Exp C", "Track": "DL Numeric",    "Model": "CNN",              "mae": _m("exp_c_cnn")["mae"],      "rmse": _m("exp_c_cnn")["rmse"],      "r2": _m("exp_c_cnn")["r2"]},
        {"Experiment": "Exp C", "Track": "DL Numeric",    "Model": "LSTM",             "mae": _m("exp_c_lstm")["mae"],     "rmse": _m("exp_c_lstm")["rmse"],     "r2": _m("exp_c_lstm")["r2"]},
        {"Experiment": "Exp C", "Track": "DL Numeric",    "Model": "CNN-LSTM Hybrid",  "mae": _m("exp_c_hybrid")["mae"],   "rmse": _m("exp_c_hybrid")["rmse"],   "r2": _m("exp_c_hybrid")["r2"]},
        # Exp D -- DL Categorical Static
        {"Experiment": "Exp D", "Track": "DL Categorical","Model": "CNN (Cat)",        "mae": _m("exp_d_cnn")["mae"],      "rmse": _m("exp_d_cnn")["rmse"],      "r2": _m("exp_d_cnn")["r2"]},
        {"Experiment": "Exp D", "Track": "DL Categorical","Model": "LSTM (Cat)",       "mae": _m("exp_d_lstm")["mae"],     "rmse": _m("exp_d_lstm")["rmse"],     "r2": _m("exp_d_lstm")["r2"]},
        {"Experiment": "Exp D", "Track": "DL Categorical","Model": "Hybrid (Cat)",     "mae": _m("exp_d_hybrid")["mae"],   "rmse": _m("exp_d_hybrid")["rmse"],   "r2": _m("exp_d_hybrid")["r2"]},
        # Exp E -- DL Numeric Sequence
        {"Experiment": "Exp E", "Track": "DL Num Seq",    "Model": "CNN (Seq)",        "mae": _m("exp_e_cnn")["mae"],      "rmse": _m("exp_e_cnn")["rmse"],      "r2": _m("exp_e_cnn")["r2"]},
        {"Experiment": "Exp E", "Track": "DL Num Seq",    "Model": "LSTM (Seq)",       "mae": _m("exp_e_lstm")["mae"],     "rmse": _m("exp_e_lstm")["rmse"],     "r2": _m("exp_e_lstm")["r2"]},
        {"Experiment": "Exp E", "Track": "DL Num Seq",    "Model": "Hybrid (Seq)",     "mae": _m("exp_e_hybrid")["mae"],   "rmse": _m("exp_e_hybrid")["rmse"],   "r2": _m("exp_e_hybrid")["r2"]},
        # Exp F -- DL Categorical Sequence
        {"Experiment": "Exp F", "Track": "DL Cat Seq",    "Model": "CNN (Cat Seq)",    "mae": _m("exp_f_cnn")["mae"],      "rmse": _m("exp_f_cnn")["rmse"],      "r2": _m("exp_f_cnn")["r2"]},
        {"Experiment": "Exp F", "Track": "DL Cat Seq",    "Model": "LSTM (Cat Seq)",   "mae": _m("exp_f_lstm")["mae"],     "rmse": _m("exp_f_lstm")["rmse"],     "r2": _m("exp_f_lstm")["r2"]},
        {"Experiment": "Exp F", "Track": "DL Cat Seq",    "Model": "Hybrid (Cat Seq)", "mae": _m("exp_f_hybrid")["mae"],   "rmse": _m("exp_f_hybrid")["rmse"],   "r2": _m("exp_f_hybrid")["r2"]},
    ]

    df_all = pd.DataFrame(all_data)
    df_all = df_all.rename(columns={"mae": "MAE (s)", "rmse": "RMSE (s)", "r2": "R2"})

    # -- Figure 10 -- Comparative performance barplot
    df_sorted_mae = df_all.sort_values("MAE (s)", ascending=True)
    df_sorted_r2  = df_all.sort_values("R2", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sns.barplot(data=df_sorted_mae, x="MAE (s)", y="Model", hue="Track", ax=axes[0], dodge=False)
    axes[0].set_title("Test MAE by Model - Lower is Better")
    axes[0].legend(loc="lower right", fontsize=8)

    sns.barplot(data=df_sorted_r2, x="R2", y="Model", hue="Track", ax=axes[1], dodge=False)
    axes[1].set_title("Test R2 by Model - Higher is Better")
    axes[1].legend(loc="lower right", fontsize=8)

    fig.suptitle("Comparative Performance Analysis: All Models & Features",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "nb04_model_comparison.png", out_dir)

    # -- Figure 11 -- Feature importance (RF / XGB / LGBM Numeric)
    feature_names = X_num.columns.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (label, model_key) in zip(
        axes,
        [("Random Forest", "rf_num"), ("XGBoost", "xgb_num"), ("LightGBM", "lgb_num")],
    ):
        mdl = models.get(model_key)
        if mdl is None:
            ax.set_title(f"{label} - model not loaded")
            continue
        importances = mdl.feature_importances_
        top_idx = np.argsort(importances)[-10:][::-1]
        ax.barh(
            [feature_names[i] for i in top_idx],
            importances[top_idx],
            color=sns.color_palette("muted", 10),
        )
        ax.invert_yaxis()
        ax.set_title(label)
        ax.set_xlabel("Importance")

    fig.suptitle("Feature Importance - Top 10 Features by Model (Numeric-Only)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "nb04_feature_importance.png", out_dir)

    # -- Figure 12 -- Predicted vs Actual scatter
    y_arr = y_num.values
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(y_arr), size=min(3000, len(y_arr)), replace=False)
    X_samp = X_num.iloc[sample_idx]
    y_samp = y_arr[sample_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (label, model_key) in zip(
        axes,
        [("Random Forest", "rf_num"), ("XGBoost", "xgb_num"), ("LightGBM", "lgb_num")],
    ):
        mdl = models.get(model_key)
        if mdl is None:
            ax.set_title(f"{label} - model not loaded")
            continue
        y_pred = mdl.predict(X_samp)
        ax.scatter(y_samp, y_pred, alpha=0.25, s=6)
        max_val = max(y_samp.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Ideal")
        ax.set_xlabel("Actual Runtime (s)")
        ax.set_ylabel("Predicted Runtime (s)")
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle("Predicted vs Actual Runtime - Tree-Based Models (sample n=3,000)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "nb04_predicted_vs_actual.png", out_dir)

    # -- Figure 13 -- Residual distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (label, model_key) in zip(
        axes,
        [("Random Forest", "rf_num"), ("XGBoost", "xgb_num"), ("LightGBM", "lgb_num")],
    ):
        mdl = models.get(model_key)
        if mdl is None:
            ax.set_title(f"{label} - model not loaded")
            continue
        y_pred_full = mdl.predict(X_num)
        residuals = y_arr - y_pred_full
        ax.hist(residuals, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axvline(residuals.mean(), color="gold", lw=2, label=f"Mean={residuals.mean():.0f}s")
        ax.set_xlabel("Residual (Actual - Predicted) (s)")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle("Residual Distribution - Models (lower spread & zero-centred = better)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "nb04_residual_dist.png", out_dir)

    # -- Figure 14 -- DL MAE summary bar
    dl_summary = pd.DataFrame([
        {"Experiment": "Exp C", "Architecture": "CNN",             "mae": _m("exp_c_cnn")["mae"],    "rmse": _m("exp_c_cnn")["rmse"],    "r2": _m("exp_c_cnn")["r2"]},
        {"Experiment": "Exp C", "Architecture": "LSTM",            "mae": _m("exp_c_lstm")["mae"],   "rmse": _m("exp_c_lstm")["rmse"],   "r2": _m("exp_c_lstm")["r2"]},
        {"Experiment": "Exp C", "Architecture": "CNN-LSTM Hybrid", "mae": _m("exp_c_hybrid")["mae"], "rmse": _m("exp_c_hybrid")["rmse"], "r2": _m("exp_c_hybrid")["r2"]},
        {"Experiment": "Exp D", "Architecture": "CNN (Cat)",        "mae": _m("exp_d_cnn")["mae"],    "rmse": _m("exp_d_cnn")["rmse"],    "r2": _m("exp_d_cnn")["r2"]},
        {"Experiment": "Exp D", "Architecture": "LSTM (Cat)",       "mae": _m("exp_d_lstm")["mae"],   "rmse": _m("exp_d_lstm")["rmse"],   "r2": _m("exp_d_lstm")["r2"]},
        {"Experiment": "Exp D", "Architecture": "Hybrid (Cat)",     "mae": _m("exp_d_hybrid")["mae"], "rmse": _m("exp_d_hybrid")["rmse"], "r2": _m("exp_d_hybrid")["r2"]},
        {"Experiment": "Exp E", "Architecture": "CNN (Seq)",        "mae": _m("exp_e_cnn")["mae"],    "rmse": _m("exp_e_cnn")["rmse"],    "r2": _m("exp_e_cnn")["r2"]},
        {"Experiment": "Exp E", "Architecture": "LSTM (Seq)",       "mae": _m("exp_e_lstm")["mae"],   "rmse": _m("exp_e_lstm")["rmse"],   "r2": _m("exp_e_lstm")["r2"]},
        {"Experiment": "Exp E", "Architecture": "Hybrid (Seq)",     "mae": _m("exp_e_hybrid")["mae"], "rmse": _m("exp_e_hybrid")["rmse"], "r2": _m("exp_e_hybrid")["r2"]},
        {"Experiment": "Exp F", "Architecture": "CNN (Cat Seq)",    "mae": _m("exp_f_cnn")["mae"],    "rmse": _m("exp_f_cnn")["rmse"],    "r2": _m("exp_f_cnn")["r2"]},
        {"Experiment": "Exp F", "Architecture": "LSTM (Cat Seq)",   "mae": _m("exp_f_lstm")["mae"],   "rmse": _m("exp_f_lstm")["rmse"],   "r2": _m("exp_f_lstm")["r2"]},
        {"Experiment": "Exp F", "Architecture": "Hybrid (Cat Seq)", "mae": _m("exp_f_hybrid")["mae"], "rmse": _m("exp_f_hybrid")["rmse"], "r2": _m("exp_f_hybrid")["r2"]},
    ])

    fig, ax = plt.subplots(figsize=(16, 5))
    x = range(len(dl_summary))
    ax.bar(x, dl_summary["mae"], color=sns.color_palette("muted", len(dl_summary)), edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        dl_summary["Experiment"] + "\n" + dl_summary["Architecture"],
        rotation=15, ha="right",
    )
    ax.set_ylabel("Test MAE (seconds)")
    ax.set_title("Deep Learning Models - Final Test MAE (Experiments C, D, E & F)")
    for i, row in dl_summary.iterrows():
        ax.text(i, row["mae"] + 50, f'{row["mae"]:.0f}', ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    _save(fig, "nb04_dl_mae_summary.png", out_dir)


# ---------------------------------------------------------------------------
# NB05 -- Scheduler Evaluation (4 figures)
# ---------------------------------------------------------------------------

_PRED_COL_MAP = {
    "SJF-RF (Numeric)":                    "pred_rf_num",
    "SJF-LGBM (Numeric)":                  "pred_lgb_num",
    "SJF-XGBoost (Numeric)":               "pred_xgb_num",
    "SJF-RF (Categorical)":                "pred_rf_cat",
    "SJF-LGBM (Categorical)":              "pred_lgb_cat",
    "SJF-XGBoost (Categorical)":           "pred_xgb_cat",
    "SJF-CNN (Numeric)":                   "pred_cnn",
    "SJF-LSTM (Numeric)":                  "pred_lstm",
    "SJF-CNN-LSTM (Numeric)":              "pred_cnn_lstm",
    "SJF-CNN (Categorical)":               "pred_cnn_cat",
    "SJF-LSTM (Categorical)":              "pred_lstm_cat",
    "SJF-CNN-LSTM (Categorical)":          "pred_cnn_lstm_cat",
    "SJF-CNN (Numeric Sequence)":          "pred_cnn_num_seq",
    "SJF-LSTM (Numeric Sequence)":         "pred_lstm_num_seq",
    "SJF-CNN-LSTM (Numeric Sequence)":     "pred_hybrid_num_seq",
    "SJF-CNN (Categorical Sequence)":      "pred_cnn_cat_seq",
    "SJF-LSTM (Categorical Sequence)":     "pred_lstm_cat_seq",
    "SJF-CNN-LSTM (Categorical Sequence)": "pred_hybrid_cat_seq",
}

_POLICIES = [
    "FIFO",
    "SJF-Oracle",
    "SJF-RF (Numeric)",
    "SJF-LGBM (Numeric)",
    "SJF-XGBoost (Numeric)",
    "SJF-RF (Categorical)",
    "SJF-LGBM (Categorical)",
    "SJF-XGBoost (Categorical)",
    "SJF-CNN (Numeric)",
    "SJF-LSTM (Numeric)",
    "SJF-CNN-LSTM (Numeric)",
    "SJF-CNN (Categorical)",
    "SJF-LSTM (Categorical)",
    "SJF-CNN-LSTM (Categorical)",
    "SJF-CNN (Numeric Sequence)",
    "SJF-LSTM (Numeric Sequence)",
    "SJF-CNN-LSTM (Numeric Sequence)",
    "SJF-CNN (Categorical Sequence)",
    "SJF-LSTM (Categorical Sequence)",
    "SJF-CNN-LSTM (Categorical Sequence)",
]


def _run_policy(policy_name: str, jobs: pd.DataFrame) -> pd.DataFrame:
    """Run one scheduling policy on a job workload (mirrors NB05 run_policy helper)."""
    jobs = jobs.copy()
    if policy_name == "FIFO":
        scheduler = FIFOScheduler()
    elif policy_name == "SJF-Oracle":
        scheduler = SJFScheduler()
    else:
        pred_col = _PRED_COL_MAP[policy_name]
        jobs["predicted_runtime"] = jobs[pred_col]
        scheduler = SJFPredScheduler()

    machines = provision_heterogeneous_gpu_cluster(n_high=2, n_mid=4, n_cpu=2)
    sim = MultiNodeClusterSimulator(scheduler, machines)
    results = sim.run(jobs)
    results["policy"] = policy_name

    pred_col = _PRED_COL_MAP.get(policy_name)
    if pred_col and pred_col in jobs.columns:
        results = results.merge(
            jobs[["job_id", pred_col, "runtime"]].rename(columns={"runtime": "true_runtime"}),
            on="job_id",
            how="left",
        )
    return results


def _build_sim_jobs(
    job_df: pd.DataFrame,
    models: dict,
    X_num: pd.DataFrame,
    y_num: pd.Series,
    X_oh: pd.DataFrame,
    y_oh: pd.Series,
    X_nat: pd.DataFrame,
    y_nat: pd.Series,
) -> pd.DataFrame:
    """Attach all ML/DL predictions to sim_jobs (mirrors NB05 Steps 3-4)."""
    device = torch.device("cpu")
    torch.set_num_threads(1)

    sim_indices = X_num.index
    sim_jobs = job_df.loc[sim_indices].copy()

    # Tree predictions
    logger.info("[NB05] Tree-model predictions...")
    for col, mkey, X in [
        ("pred_rf_num",  "rf_num",      X_num),
        ("pred_xgb_num", "xgb_num",     X_num),
        ("pred_lgb_num", "lgb_num",     X_num),
        ("pred_rf_cat",  "rf_cat",      X_oh),
        ("pred_xgb_cat", "xgb_cat",     X_oh),
        ("pred_lgb_cat", "lgb_cat_nat", X_nat),
    ]:
        mdl = models.get(mkey)
        sim_jobs[col] = mdl.predict(X) if mdl is not None else np.nan

    # Scale features for DL
    scx_num = models.get("scaler_x_num")
    scy_num = models.get("scaler_y_num")
    scx_cat = models.get("scaler_x_cat")
    scy_cat = models.get("scaler_y_cat")

    if scx_num and scy_num:
        logger.info("[NB05] Scaling numeric features...")
        X_num_scaled = scx_num.transform(X_num)
        t_num = torch.tensor(X_num_scaled, dtype=torch.float32)
    else:
        t_num = None

    if scx_cat and scy_cat:
        logger.info("[NB05] Scaling categorical features...")
        X_cat_scaled = scx_cat.transform(X_oh)
        t_cat = torch.tensor(X_cat_scaled, dtype=torch.float32)
    else:
        t_cat = None

    # Static DL predictions (Experiments C & D)
    logger.info("[NB05] Static DL predictions (Exp C & D)...")
    for col, mkey, t, scy in [
        ("pred_cnn",          "cnn_static_num",    t_num, scy_num),
        ("pred_lstm",         "lstm_static_num",   t_num, scy_num),
        ("pred_cnn_lstm",     "hybrid_static_num", t_num, scy_num),
        ("pred_cnn_cat",      "cnn_static_cat",    t_cat, scy_cat),
        ("pred_lstm_cat",     "lstm_static_cat",   t_cat, scy_cat),
        ("pred_cnn_lstm_cat", "hybrid_static_cat", t_cat, scy_cat),
    ]:
        mdl = models.get(mkey)
        if mdl is not None and t is not None:
            raw = _predict_batches(mdl, t, is_sequence=False, device=device)
            sim_jobs[col] = scy.inverse_transform(raw.reshape(-1, 1)).flatten()
        else:
            sim_jobs[col] = np.nan

    # Sequence DL predictions (Experiments E & F)
    logger.info("[NB05] Sequence DL predictions (Exp E & F)...")
    seq_len = 10
    for col, mkey, t, scy in [
        ("pred_cnn_num_seq",    "cnn_seq_num",    t_num, scy_num),
        ("pred_lstm_num_seq",   "lstm_seq_num",   t_num, scy_num),
        ("pred_hybrid_num_seq", "hybrid_seq_num", t_num, scy_num),
        ("pred_cnn_cat_seq",    "cnn_seq_cat",    t_cat, scy_cat),
        ("pred_lstm_cat_seq",   "lstm_seq_cat",   t_cat, scy_cat),
        ("pred_hybrid_cat_seq", "hybrid_seq_cat", t_cat, scy_cat),
    ]:
        mdl = models.get(mkey)
        if mdl is not None and t is not None:
            raw = _predict_batches(mdl, t, is_sequence=True, seq_len=seq_len, device=device)
            sim_jobs[col] = scy.inverse_transform(raw.reshape(-1, 1)).flatten()
        else:
            sim_jobs[col] = np.nan
        gc.collect()

    # Normalise time axis (mirrors NB05 Step 4)
    sim_jobs = sim_jobs.sort_values("arrival_time").reset_index(drop=True)
    t0 = sim_jobs["arrival_time"].min()
    LOAD_FACTOR = 0.4
    sim_jobs["submit_time"] = (sim_jobs["arrival_time"] - t0).dt.total_seconds() * LOAD_FACTOR
    sim_jobs["runtime"]     = sim_jobs["job_runtime"]

    n_pred_cols = sum(1 for c in sim_jobs.columns if c.startswith("pred_"))
    logger.info("[NB05] sim_jobs ready: %d jobs, %d prediction columns", len(sim_jobs), n_pred_cols)
    return sim_jobs


def plot_nb05(
    models: dict,
    job_df: pd.DataFrame,
    X_num: pd.DataFrame,
    y_num: pd.Series,
    X_oh: pd.DataFrame,
    y_oh: pd.Series,
    X_nat: pd.DataFrame,
    y_nat: pd.Series,
    out_dir: Path,
) -> None:
    logger.info("[NB05] Building simulation job table...")
    sim_jobs = _build_sim_jobs(job_df, models, X_num, y_num, X_oh, y_oh, X_nat, y_nat)

    # Run all policies
    results_list = []
    for policy in _POLICIES:
        logger.info("[NB05] Running policy: %s", policy)
        try:
            res = _run_policy(policy, deepcopy(sim_jobs))
            results_list.append(res)
        except Exception as exc:
            logger.warning("[NB05] Policy %s failed: %s", policy, exc)

    all_results = pd.concat(results_list, ignore_index=True)
    logger.info("[NB05] Simulation complete: %d job-policy records", len(all_results))

    # Build eval_df
    summary_rows = []
    for policy in all_results["policy"].unique():
        df_p = all_results[all_results["policy"] == policy].copy()
        mean_wait    = df_p["waiting_time"].mean()
        median_wait  = df_p["waiting_time"].median()
        p95_wait     = df_p["waiting_time"].quantile(0.95)
        mean_jct     = df_p["turnaround_time"].mean()
        max_jct      = df_p["turnaround_time"].max()
        mean_slowdown = df_p["slowdown"].mean()
        pred_col = _PRED_COL_MAP.get(policy)
        mae = rmse = r2 = None
        if pred_col and pred_col in df_p.columns and "true_runtime" in df_p.columns:
            y_true = df_p["true_runtime"].dropna()
            y_pred = df_p[pred_col].loc[y_true.index]
            mae  = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2   = r2_score(y_true, y_pred)
        summary_rows.append({
            "Policy / Architecture": policy,
            "Mean Wait (s)":   mean_wait,
            "Median Wait (s)": median_wait,
            "P95 Wait (s)":    p95_wait,
            "Mean JCT (s)":    mean_jct,
            "Max JCT (s)":     max_jct,
            "Slowdown":        mean_slowdown,
            "Model MAE (s)":   mae,
            "Model RMSE (s)":  rmse,
            "Model R2":        r2,
        })

    eval_df = pd.DataFrame(summary_rows)
    fifo_jct = eval_df.loc[eval_df["Policy / Architecture"] == "FIFO", "Mean JCT (s)"].values[0]
    eval_df["JCT Improvement %"] = (
        (fifo_jct - eval_df["Mean JCT (s)"]) / fifo_jct * 100
    ).round(2)
    eval_sorted = eval_df.sort_values("Mean JCT (s)").reset_index(drop=True)

    # -- Figure 15 -- Scheduler JCT dual barh
    plot_df = eval_sorted.sort_values("Mean JCT (s)", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    palette1 = sns.color_palette("mako", n_colors=len(plot_df))
    bars1 = axes[0].barh(
        plot_df["Policy / Architecture"], plot_df["Mean JCT (s)"],
        color=palette1, edgecolor="white",
    )
    axes[0].set_xlabel("Mean JCT (seconds)")
    axes[0].set_title("Mean Job Completion Time by Policy\n(Lower is Better)")
    axes[0].invert_yaxis()
    for bar in bars1:
        axes[0].text(
            bar.get_width() + 0.01 * plot_df["Mean JCT (s)"].max(),
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.0f}s", va="center", fontsize=8,
        )

    plot_df2 = plot_df[plot_df["Policy / Architecture"] != "FIFO"].copy()
    palette2 = sns.color_palette("flare", n_colors=len(plot_df2))
    bars2 = axes[1].barh(
        plot_df2["Policy / Architecture"], plot_df2["JCT Improvement %"],
        color=palette2, edgecolor="white",
    )
    axes[1].set_xlabel("JCT Improvement over FIFO (%)")
    axes[1].set_title("JCT Improvement over FIFO Baseline\n(Higher is Better)")
    axes[1].invert_yaxis()
    for bar in bars2:
        axes[1].text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}%", va="center", fontsize=8,
        )

    fig.suptitle("Scheduler Evaluation - Multi-Node GPU Cluster (200 GPUs)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "nb05_scheduler_jct.png", out_dir)

    # -- Figure 16 -- Wait time CDF
    fig, ax = plt.subplots(figsize=(14, 7))
    palette_cdf = sns.color_palette("tab20", n_colors=len(all_results["policy"].unique()))
    for color, (policy, df_p) in zip(palette_cdf, all_results.groupby("policy")):
        wait_sorted = np.sort(df_p["waiting_time"].values)
        cdf = np.arange(1, len(wait_sorted) + 1) / len(wait_sorted)
        lw = 2.5 if policy in ("FIFO", "SJF-Oracle") else 1.2
        ls = "--" if policy == "SJF-Oracle" else "-"
        ax.plot(wait_sorted, cdf, lw=lw, ls=ls, color=color, label=policy, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlim(left=1)
    ax.set_xlabel("Waiting Time (seconds)")
    ax.set_ylabel("CDF (Fraction of Jobs)")
    ax.set_title("Wait Time CDF - All Scheduling Policies\n(Upper-Left = Better)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    _save(fig, "nb05_wait_time_cdf.png", out_dir)

    # -- Figure 17 -- Slowdown boxplot
    policy_order = (
        all_results.groupby("policy")["slowdown"]
        .median()
        .sort_values()
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.boxplot(
        data=all_results,
        x="policy", y="slowdown",
        order=policy_order,
        palette="muted",
        whis=1.5,
        flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Slowdown (log scale)")
    ax.set_title("Job Slowdown Distribution by Scheduling Policy - Lower is Better")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    _save(fig, "nb05_slowdown_boxplot.png", out_dir)

    # -- Figure 18 -- Improvement heatmap
    fifo_vals = all_results[all_results["policy"] == "FIFO"].agg(
        {"waiting_time": "mean", "turnaround_time": "mean", "slowdown": "mean"}
    )
    heatmap_rows = []
    for policy, df_p in all_results.groupby("policy"):
        if policy == "FIFO":
            continue
        heatmap_rows.append({
            "Policy": policy,
            "Wait down %": (fifo_vals["waiting_time"]    - df_p["waiting_time"].mean())    / fifo_vals["waiting_time"]    * 100,
            "JCT down %":  (fifo_vals["turnaround_time"] - df_p["turnaround_time"].mean()) / fifo_vals["turnaround_time"] * 100,
            "Slowdown down %": (fifo_vals["slowdown"]    - df_p["slowdown"].mean())         / fifo_vals["slowdown"]        * 100,
        })
    hm_df = pd.DataFrame(heatmap_rows).set_index("Policy").sort_values("JCT down %", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        hm_df, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn",
        center=0, linewidths=0.5,
        cbar_kws={"label": "Improvement over FIFO (%)"},
    )
    ax.set_title("Scheduling Policy Improvement over FIFO Baseline\n(% reduction - higher = better)")
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    _save(fig, "nb05_improvement_heatmap.png", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all 18 thesis figures from saved models and data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["all", "workload", "features", "models", "scheduler"],
        default="all",
        help="Which set of figures to generate (default: all).",
    )
    parser.add_argument(
        "--out",
        default="results/figures",
        help="Output directory for figures (default: results/figures).",
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out
    _clean_figures(out_dir)

    mode = args.mode

    # --- shared data load (used by multiple modes) -------------------------
    job_df = X_num = y_num = X_oh = y_oh = X_nat = y_nat = None
    models_dict = None

    def _ensure_data():
        nonlocal job_df, X_num, y_num, X_oh, y_oh, X_nat, y_nat
        if job_df is None:
            job_df, X_num, y_num, X_oh, y_oh, X_nat, y_nat = _load_data()

    def _ensure_models():
        nonlocal models_dict
        if models_dict is None:
            models_dict = _load_models()

    if mode in ("all", "workload"):
        logger.info("=== Workload figures (NB01 + NB02) ===")
        _ensure_data()
        plot_nb01(job_df, out_dir)
        plot_nb02(job_df, out_dir)

    if mode in ("all", "features"):
        logger.info("=== Feature engineering figures (NB03) ===")
        _ensure_data()
        plot_nb03(job_df, out_dir)

    if mode in ("all", "models"):
        logger.info("=== Model performance figures (NB04) ===")
        _ensure_data()
        _ensure_models()
        plot_nb04(models_dict, X_num, y_num, out_dir)

    if mode in ("all", "scheduler"):
        logger.info("=== Scheduler evaluation figures (NB05) ===")
        _ensure_data()
        _ensure_models()
        plot_nb05(models_dict, job_df, X_num, y_num, X_oh, y_oh, X_nat, y_nat, out_dir)

    logger.info("All requested figures saved to: %s", out_dir)


if __name__ == "__main__":
    main()
