"""
workload_analysis.py

Workload Analysis Utilities

This module provides helper functions to analyze the GPU job workload based on
the Alibaba PAI 100K job sample. It builds on top of:

- :mod:`src.data_loading`       (for raw CSV loading)
- :mod:`src.feature_engineering` (for job-level feature table)

Key Components
--------------
load_prepared_job_table
    Convenience wrapper returning the full engineered job table.
compute_basic_stats
    Compute scalar workload statistics.
compute_runtime_histogram
    Histogram data for job runtimes.
compute_arrival_rate_series
    Time-indexed job arrival counts at a chosen frequency.
summarize_workload
    One-row DataFrame suitable for LaTeX / Markdown tables.

These functions are meant to be called from Jupyter notebooks to generate
tables and figures for the thesis.
"""

from __future__ import annotations

from typing import Dict, Tuple, Literal

import numpy as np
import pandas as pd

from src.feature_engineering import prepare_features_for_model


__all__ = [
    "load_prepared_job_table",
    "compute_basic_stats",
    "compute_runtime_histogram",
    "compute_arrival_rate_series",
    "summarize_workload",
]


# -------------------------------------------------------------------
# High-level loader
# -------------------------------------------------------------------


def load_prepared_job_table(
    dataset: Literal["main", "baseline"] = "main",
    time_unit: str = "s",
) -> pd.DataFrame:
    """
    Load and prepare the job-level table with engineered features.

    Convenience wrapper around :func:`~src.feature_engineering.prepare_features_for_model`
    that discards the train/test splits and returns only the full job table.

    Parameters
    ----------
    dataset : {"main", "baseline"}, default ``"main"``
        Dataset key forwarded to :func:`~src.feature_engineering.prepare_features_for_model`.

        - ``"main"``     : primary thesis dataset (no estimate).
        - ``"baseline"`` : optional dataset with Alibaba estimates.
    time_unit : str, default ``"s"``
        Unit of ``submit_time``; for the PAI trace this is typically seconds.

    Returns
    -------
    pd.DataFrame
        Job-level table with features such as:

        - ``job_id``, ``arrival_time``, ``arrival_sec``
        - ``job_runtime``, ``gpu_demand``
        - ``user``, ``gpu_type``
        - ``hour_of_day``, ``day_of_week``
        - ``cluster_load_cpu``, ``cluster_load_gpu``, ``active_job_count``
    """
    job_df, _, _, _, _, _, _ = prepare_features_for_model(
        dataset=dataset,
        time_unit=time_unit,
    )
    return job_df


# -------------------------------------------------------------------
# Basic statistics
# -------------------------------------------------------------------


def compute_basic_stats(job_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute basic workload statistics for the given job table.

    Parameters
    ----------
    job_df : pd.DataFrame
        Job-level table with at least:

        - ``job_runtime`` (float): job duration in seconds.
        - ``gpu_demand``  (int):   GPUs requested.
        - ``arrival_time`` (datetime): submission timestamp.

    Returns
    -------
    dict
        Dictionary with key indicators:

        - ``num_jobs``          : total number of jobs.
        - ``mean_runtime``      : mean job runtime (s).
        - ``median_runtime``    : median job runtime (s).
        - ``p95_runtime``       : 95th-percentile runtime (s).
        - ``p99_runtime``       : 99th-percentile runtime (s).
        - ``mean_gpu_demand``   : mean GPUs per job.
        - ``max_gpu_demand``    : maximum GPUs in a single job.
        - ``time_span_hours``   : total trace duration (hours).

    Raises
    ------
    ValueError
        If any required column is absent.
    """
    for col in ("job_runtime", "gpu_demand", "arrival_time"):
        if col not in job_df.columns:
            raise ValueError(f"job_df must contain '{col}' column.")

    runtimes = job_df["job_runtime"].values
    gpu_demand = job_df["gpu_demand"].values

    t0 = job_df["arrival_time"].min()
    t1 = job_df["arrival_time"].max()
    time_span_hours = float((t1 - t0).total_seconds() / 3600.0)

    return {
        "num_jobs": len(job_df),
        "mean_runtime": float(np.mean(runtimes)),
        "median_runtime": float(np.median(runtimes)),
        "p95_runtime": float(np.percentile(runtimes, 95)),
        "p99_runtime": float(np.percentile(runtimes, 99)),
        "mean_gpu_demand": float(np.mean(gpu_demand)),
        "max_gpu_demand": int(np.max(gpu_demand)),
        "time_span_hours": time_span_hours,
    }


# -------------------------------------------------------------------
# Histogram data
# -------------------------------------------------------------------


def compute_runtime_histogram(
    job_df: pd.DataFrame,
    bins: int = 50,
    log_scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram data for job runtimes.

    This function does **not** plot directly; it returns counts and bin edges
    suitable for use in matplotlib or seaborn.

    Parameters
    ----------
    job_df : pd.DataFrame
        Job-level table with ``job_runtime``.
    bins : int, default 50
        Number of histogram bins.
    log_scale : bool, default False
        If ``True``, compute the histogram on ``log10(runtime + 1)``.

    Returns
    -------
    hist : np.ndarray
        Counts per bin.
    bin_edges : np.ndarray
        Edges of the bins (length = ``bins + 1``).

    Raises
    ------
    ValueError
        If ``job_runtime`` is absent from ``job_df``.
    """
    if "job_runtime" not in job_df.columns:
        raise ValueError("job_df must contain 'job_runtime' column.")

    x = job_df["job_runtime"].astype(float).values
    if log_scale:
        x = np.log10(x + 1.0)

    return np.histogram(x, bins=bins)


# -------------------------------------------------------------------
# Arrival rate analysis
# -------------------------------------------------------------------


def compute_arrival_rate_series(
    job_df: pd.DataFrame,
    freq: str = "1h",
) -> pd.Series:
    """
    Compute arrival rate over time as a pandas Series.

    Parameters
    ----------
    job_df : pd.DataFrame
        Must contain the column ``arrival_time`` (datetime or Unix timestamp).
    freq : str, default ``"1h"``
        Resampling frequency, e.g. ``"1h"`` (hourly), ``"30min"``, ``"1D"``.

        .. note::
            Use lower-case aliases (``"1h"`` not ``"1H"``) for compatibility
            with pandas ≥ 2.2.

    Returns
    -------
    pd.Series
        Timedelta-indexed series where each point is the number of jobs
        submitted during the interval. The index is normalized so that
        ``t = 0`` corresponds to the start of the trace.

    Raises
    ------
    ValueError
        If ``arrival_time`` column is absent.
    """
    if "arrival_time" not in job_df.columns:
        raise ValueError("job_df must contain 'arrival_time' column.")

    df = job_df.copy()

    if not np.issubdtype(df["arrival_time"].dtype, np.datetime64):
        df["arrival_time"] = pd.to_datetime(df["arrival_time"], unit="s")

    t0 = df["arrival_time"].min()
    df["rel_time"] = df["arrival_time"] - t0

    df = df.set_index("rel_time")
    return df["job_id"].resample(freq).count()


# -------------------------------------------------------------------
# Workload summary for tables
# -------------------------------------------------------------------


def summarize_workload(job_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a one-row summary table of key workload statistics.

    Parameters
    ----------
    job_df : pd.DataFrame
        Job-level table (same requirements as :func:`compute_basic_stats`).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns:

        - ``num_jobs``
        - ``mean_runtime_sec``, ``median_runtime_sec``
        - ``p95_runtime_sec``, ``p99_runtime_sec``
        - ``mean_gpu_demand``, ``max_gpu_demand``
        - ``time_span_hours``

        Suitable for LaTeX / Markdown export via ``df.to_latex()`` or ``df.to_markdown()``.
    """
    stats = compute_basic_stats(job_df)

    return pd.DataFrame(
        {
            "num_jobs": [stats["num_jobs"]],
            "mean_runtime_sec": [stats["mean_runtime"]],
            "median_runtime_sec": [stats["median_runtime"]],
            "p95_runtime_sec": [stats["p95_runtime"]],
            "p99_runtime_sec": [stats["p99_runtime"]],
            "mean_gpu_demand": [stats["mean_gpu_demand"]],
            "max_gpu_demand": [stats["max_gpu_demand"]],
            "time_span_hours": [stats["time_span_hours"]],
        }
    )