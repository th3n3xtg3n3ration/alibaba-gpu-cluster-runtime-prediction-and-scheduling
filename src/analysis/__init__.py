"""
src.analysis

Workload Characterization and Statistical Analysis Package

This sub-package provides utility functions for analyzing the GPU job workload
derived from the Alibaba PAI 100K trace. It builds on top of
:mod:`src.feature_engineering` and :mod:`src.data_loading`.

Public API
----------
load_prepared_job_table
    Load and return the full engineered job table (no train/test split).
compute_basic_stats
    Compute scalar workload statistics (runtimes, GPU demand, time span).
compute_runtime_histogram
    Return histogram counts and bin edges for job runtimes.
compute_arrival_rate_series
    Compute a time-indexed arrival-rate series at arbitrary frequency.
summarize_workload
    Build a one-row summary DataFrame suitable for LaTeX/Markdown tables.

These functions are called from:

- ``notebooks/02_workload_analysis.ipynb``
- ``notebooks/04_runtime_prediction_models.ipynb``
"""

from .workload_analysis import (
    load_prepared_job_table,
    compute_basic_stats,
    compute_runtime_histogram,
    compute_arrival_rate_series,
    summarize_workload,
)

__all__ = [
    "load_prepared_job_table",
    "compute_basic_stats",
    "compute_runtime_histogram",
    "compute_arrival_rate_series",
    "summarize_workload",
]