"""
visualization.py

Project Visualization and Reporting Library

This module provides a centralized collection of plotting functions for workload
characterization, model performance evaluation, and scheduler simulation results.
It decouples plotting logic from execution scripts, ensuring reproducibility and
consistent figure styling across the thesis.

Key Components
--------------
plot_workload_summary
    Comprehensive workload characterization plots (histogram, CDF, arrivals, heatmap).
plot_regression_analysis
    Regression scatter and error CDF plots for a trained model.
plot_scheduler_comparison
    Bar plots comparing scheduling policy performance metrics.
"""

from __future__ import annotations

from pathlib import Path
import math
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


__all__ = [
    "plot_workload_summary",
    "plot_regression_analysis",
    "plot_scheduler_comparison",
]


# ---------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------


def _ensure_dir(path: Path | str) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------
# Workload characterization plots
# ---------------------------------------------------------------------


def plot_workload_summary(job_df: pd.DataFrame, output_dir: Path | str) -> None:
    """
    Generate all workload characterization plots and save to disk.

    Creates up to four figures:

    1. Job runtime histogram (log10 scale).
    2. Job runtime CDF (log x-axis).
    3. Hourly job arrival rate over the trace duration.
    4. Day-of-week × hour-of-day arrival heatmap.

    Parameters
    ----------
    job_df : pd.DataFrame
        Job-level table containing at least:

        - ``job_runtime`` (float): job duration in seconds.
        - ``arrival_time`` (datetime, optional): for arrival-rate plots.
    output_dir : Path or str
        Directory where PNG figures are written (created if absent).

    Returns
    -------
    None
    """
    out_path = _ensure_dir(output_dir)
    runtimes = job_df["job_runtime"].astype(float).values

    # 1. Runtime histogram
    plt.figure(figsize=(10, 6))
    plt.hist(np.log10(runtimes + 1), bins=50, color="skyblue", edgecolor="black")
    plt.xlabel("log₁₀(Runtime + 1)")
    plt.ylabel("Frequency")
    plt.title("Job Runtime Distribution (Log Scale)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "runtime_dist_hist.png", dpi=300)
    plt.close()

    # 2. Runtime CDF
    runtimes_sorted = np.sort(runtimes)
    cdf = np.arange(1, len(runtimes_sorted) + 1) / len(runtimes_sorted)
    plt.figure(figsize=(10, 6))
    plt.plot(runtimes_sorted, cdf, color="darkblue", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Runtime (seconds)")
    plt.ylabel("CDF")
    plt.title("Cumulative Distribution of Runtimes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "runtime_cdf.png", dpi=300)
    plt.close()

    if "arrival_time" not in job_df.columns:
        return

    # 3. Hourly arrival rate
    df_time = job_df.set_index("arrival_time")
    arrival_rate = df_time.resample("1h").size()
    hours = (arrival_rate.index - arrival_rate.index.min()).total_seconds() / 3600.0
    days = hours / 24.0

    plt.figure(figsize=(12, 6))
    plt.plot(days, arrival_rate.values, color="forestgreen")
    plt.xlabel("Elapsed Time (days)")
    plt.ylabel("Jobs per Hour")
    plt.title("Hourly Job Arrival Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "arrival_rate_hourly.png", dpi=300)
    plt.close()

    # 4. Heatmap: day-of-week × hour-of-day
    df_heat = job_df.copy()
    df_heat["hour"] = df_heat["arrival_time"].dt.hour
    df_heat["dow"] = df_heat["arrival_time"].dt.dayofweek
    heatmap = df_heat.groupby(["dow", "hour"]).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap, aspect="auto", cmap="YlGnBu")
    plt.colorbar(label="Job Count")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week (0=Mon)")
    plt.title("Arrival Heatmap — Day × Hour")
    plt.tight_layout()
    plt.savefig(out_path / "arrival_heatmap.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Model performance plots
# ---------------------------------------------------------------------


def plot_regression_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Path | str,
) -> None:
    """
    Generate standardized regression diagnostic plots for one model.

    Creates two figures:

    1. Scatter plot of true vs. predicted runtimes.
    2. CDF of absolute prediction errors (log x-axis).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth runtime values.
    y_pred : np.ndarray
        Model-predicted runtime values.
    model_name : str
        Human-readable name used in titles and output filenames.
    output_dir : Path or str
        Directory where PNG figures are written (created if absent).

    Returns
    -------
    None
    """
    out_path = _ensure_dir(output_dir)
    suffix = model_name.lower().replace(" ", "_")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Scatter: True vs. Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect Prediction")
    plt.xlabel("True Runtime (s)")
    plt.ylabel("Predicted Runtime (s)")
    plt.title(f"True vs. Predicted — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / f"scatter_{suffix}.png", dpi=300)
    plt.close()

    # Error CDF
    errors = np.abs(y_true - y_pred)
    errors_sorted = np.sort(errors)
    cdf = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)

    plt.figure(figsize=(10, 6))
    plt.plot(errors_sorted, cdf, color="red", lw=2)
    plt.xscale("log")
    plt.xlabel("|Prediction Error| (s)")
    plt.ylabel("CDF")
    plt.title(f"Error CDF — {model_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / f"error_cdf_{suffix}.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Scheduler comparison plots
# ---------------------------------------------------------------------


def plot_scheduler_comparison(
    results_df: pd.DataFrame,
    output_dir: Path | str,
) -> None:
    """
    Compare scheduling policies across standard performance metrics.

    Generates one bar chart per metric:

    - ``waiting_time``    (mean per policy)
    - ``turnaround_time`` (mean per policy)
    - ``slowdown``        (mean per policy)

    Parameters
    ----------
    results_df : pd.DataFrame
        Combined results from multiple scheduler runs. Must contain:

        - ``policy``          (str): scheduler name (e.g. ``"FIFO"``, ``"SJF"``).
        - ``waiting_time``    (float): per-job queue wait in seconds.
        - ``turnaround_time`` (float): per-job turnaround in seconds.
        - ``slowdown``        (float): per-job slowdown ratio.
    output_dir : Path or str
        Directory where PNG figures are written (created if absent).

    Returns
    -------
    None
    """
    out_path = _ensure_dir(output_dir)
    metrics = ["waiting_time", "turnaround_time", "slowdown"]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        results_df.groupby("policy")[metric].mean().sort_values().plot(
            kind="bar", color="plum"
        )
        plt.ylabel(f"Mean {metric.replace('_', ' ').title()}")
        plt.title(f"Scheduler Comparison: {metric.replace('_', ' ').title()}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_path / f"sched_compare_{metric}.png", dpi=300)
        plt.close()
