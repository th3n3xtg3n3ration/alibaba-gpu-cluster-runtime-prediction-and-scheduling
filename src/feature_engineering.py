"""
feature_engineering.py

Feature Engineering and Transformation Pipeline

This module implements the core transformation logic for converting raw Alibaba PAI
job traces into high-quality feature matrices. It handles temporal extraction,
categorical encoding, and derived cluster utilization metrics (offered load).

Key Components
--------------
build_job_table_from_sample
    Canonical normalization of raw CSV data.
add_temporal_features
    Extraction of cyclical time features (hour, day).
add_categorical_features
    Conversion of string columns to categorical dtype.
add_cluster_utilization_features
    Sweep-line algorithm for offered load tracking.
build_feature_matrix
    Train/test split builder for regression models.
prepare_features_for_model
    High-level entry point for model-ready data preparation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.data_loading import load_sample
from src.config_utils import load_paths_config


__all__ = [
    "build_job_table_from_sample",
    "add_temporal_features",
    "add_categorical_features",
    "add_cluster_utilization_features",
    "build_feature_matrix",
    "prepare_features_for_model",
]

# Absolute path of the processed data directory (used as fallback in prepare_features_for_model)
_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


# ---------------------------------------------------------------------
# Core transformations
# ---------------------------------------------------------------------


def build_job_table_from_sample(
    df: pd.DataFrame,
    time_unit: str = "s",
) -> pd.DataFrame:
    """
    Perform canonical normalization and cleaning of the raw job dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe loaded from CSV.
    time_unit : str, default ``"s"``
        Unit of the ``submit_time`` timestamps (passed to
        :func:`pandas.to_datetime` ``unit`` argument).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with proper column types and basic time offsets.

    Raises
    ------
    ValueError
        If any required column is missing from ``df``.
    """
    # ------------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------------
    required_cols = [
        "job_id",
        "submit_time",
        "duration",
        "num_gpu",
        "user",
        "gpu_type",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input DataFrame.")

    # ------------------------------------------------------------------
    # Single validity filter
    # ------------------------------------------------------------------
    job = df[
        (df["duration"] > 0) &
        (df["num_gpu"] > 0)
    ].copy()

    # ------------------------------------------------------------------
    # Convert submit_time to datetime
    # ------------------------------------------------------------------
    if np.issubdtype(job["submit_time"].dtype, np.number):
        job["arrival_time"] = pd.to_datetime(job["submit_time"], unit=time_unit)
    else:
        job["arrival_time"] = pd.to_datetime(job["submit_time"])

    # ------------------------------------------------------------------
    # Normalize core fields
    # ------------------------------------------------------------------
    job["job_runtime"] = job["duration"].astype(float)
    job["gpu_demand"] = job["num_gpu"].astype(int)

    # ------------------------------------------------------------------
    # Relative arrival time (seconds since first job)
    # ------------------------------------------------------------------
    t0 = job["arrival_time"].min()
    job["arrival_sec"] = (job["arrival_time"] - t0).dt.total_seconds()

    # ------------------------------------------------------------------
    # Select final columns (only those present in the source data)
    # ------------------------------------------------------------------
    keep_cols = [
        "job_id",
        "arrival_time",
        "arrival_sec",
        "job_runtime",
        "gpu_demand",
        "user",
        "gpu_type",
        "num_inst",
        "num_cpu",
    ]
    keep_cols = [c for c in keep_cols if c in job.columns]

    return job[keep_cols].reset_index(drop=True)


def add_temporal_features(job: pd.DataFrame) -> pd.DataFrame:
    """
    Extract cyclical and categorical temporal features from submit time.

    Parameters
    ----------
    job : pd.DataFrame
        Input dataframe containing the ``arrival_time`` column (datetime).

    Returns
    -------
    pd.DataFrame
        Copy of ``job`` with two new integer columns:

        - ``hour_of_day``  : Hour of submit time [0, 23].
        - ``day_of_week``  : Weekday of submit time [0=Mon, 6=Sun].

    Raises
    ------
    ValueError
        If ``arrival_time`` column is absent.
    """
    if "arrival_time" not in job.columns:
        raise ValueError("Column 'arrival_time' is required for temporal features.")

    df = job.copy()
    df["hour_of_day"] = df["arrival_time"].dt.hour
    df["day_of_week"] = df["arrival_time"].dt.dayofweek
    return df


def add_categorical_features(job: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ``user`` and ``gpu_type`` columns to :class:`pandas.CategoricalDtype`.

    This does *not* perform one-hot encoding; tree-based models (e.g. LightGBM
    with native categorical support) can consume categorical features directly.

    Parameters
    ----------
    job : pd.DataFrame
        Input dataframe possibly containing ``user`` and/or ``gpu_type`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of ``job`` with categorical dtype applied where applicable.
    """
    df = job.copy()
    for col in ("user", "gpu_type"):
        if col in df.columns and df[col].dtype.name != "category":
            df[col] = df[col].fillna("unknown").astype("category")
    return df


def add_cluster_utilization_features(
    job: pd.DataFrame,
    time_window: int = 300,
) -> pd.DataFrame:
    """
    Calculate cluster-level offered load (CPU/GPU) at each job's arrival time.

    Implements an **O(N log N) sweep-line** algorithm: arrival and departure
    events are created for each job, sorted by time, and cumulatively summed
    to track the system state at each instant.

    The resulting features represent the *background* load seen by each job
    (i.e. total cluster demand excluding the job itself).

    Parameters
    ----------
    job : pd.DataFrame
        Input dataframe containing at least:

        - ``arrival_sec``  (float): relative arrival time in seconds.
        - ``job_runtime``  (float): job duration in seconds.
        - ``gpu_demand``   (int):   number of GPUs requested.
        - ``num_cpu``      (int, optional): number of CPUs requested.
    time_window : int, default 300
        Reserved for future windowed-averaging variants; currently unused.

    Returns
    -------
    pd.DataFrame
        Input dataframe augmented with:

        - ``cluster_load_cpu``  : background CPU demand at arrival.
        - ``cluster_load_gpu``  : background GPU demand at arrival.
        - ``active_job_count``  : number of concurrently running jobs at arrival.
    """
    df = job.copy().sort_values("arrival_sec")

    # Guard: num_cpu may not exist in all dataset variants
    has_cpu = "num_cpu" in df.columns
    cpu_vals = df["num_cpu"] if has_cpu else pd.Series(0, index=df.index)

    # ------------------------------------------------------------------
    # Build events: (time, delta_cpu, delta_gpu, delta_count)
    # ------------------------------------------------------------------
    starts = pd.DataFrame({
        "time": df["arrival_sec"].values,
        "d_cpu": cpu_vals.values,
        "d_gpu": df["gpu_demand"].values,
        "change_count": 1,
    })

    ends = pd.DataFrame({
        "time": (df["arrival_sec"] + df["job_runtime"]).values,
        "d_cpu": -cpu_vals.values,
        "d_gpu": -df["gpu_demand"].values,
        "change_count": -1,
    })

    events = pd.concat([starts, ends], ignore_index=True).sort_values("time")

    # ------------------------------------------------------------------
    # Sweep-line: group simultaneous events, then cumulative sum
    # ------------------------------------------------------------------
    state_at_time = (
        events.groupby("time")[["d_cpu", "d_gpu", "change_count"]]
        .sum()
        .reset_index()
    )
    state_at_time["load_cpu"] = state_at_time["d_cpu"].cumsum()
    state_at_time["load_gpu"] = state_at_time["d_gpu"].cumsum()
    state_at_time["active_jobs"] = state_at_time["change_count"].cumsum()

    # ------------------------------------------------------------------
    # Map back: merge_asof gives state AT or BEFORE each arrival
    # ------------------------------------------------------------------
    df = pd.merge_asof(
        df,
        state_at_time[["time", "load_cpu", "load_gpu", "active_jobs"]],
        left_on="arrival_sec",
        right_on="time",
        direction="backward",
    )

    # The load at arrival_sec already includes this job's start event;
    # subtract to get the background load *excluding* this job.
    df["cluster_load_cpu"] = df["load_cpu"] - cpu_vals.values
    df["cluster_load_gpu"] = df["load_gpu"] - df["gpu_demand"]
    df["active_job_count"] = df["active_jobs"] - 1

    # Fill NaN for the very first job (no earlier events)
    fill_cols = [
        "cluster_load_cpu", "cluster_load_gpu", "active_job_count",
        "load_cpu", "load_gpu", "active_jobs",
    ]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df.drop(columns=["time", "load_cpu", "load_gpu", "active_jobs"], inplace=True, errors="ignore")

    return df


# ---------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------


def build_feature_matrix(
    job: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_col: str = "job_runtime",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build train/test splits for regression models.

    Parameters
    ----------
    job : pd.DataFrame
        Preprocessed job-level table with features.
    numeric_cols : list of str
        Names of numerical feature columns.
    categorical_cols : list of str
        Names of categorical feature columns.
    target_col : str, default ``"job_runtime"``
        Name of the regression target column.
    test_size : float, default 0.2
        Fraction of samples reserved for the test set.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    y_train : np.ndarray
    y_test : np.ndarray

    Raises
    ------
    ValueError
        If any specified column is absent from ``job``.
    """
    for col in numeric_cols + categorical_cols + [target_col]:
        if col not in job.columns:
            raise ValueError(f"Column '{col}' not found in job table.")

    X = job[numeric_cols + categorical_cols].copy()
    y = job[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------
# High-level helper for the whole pipeline
# ---------------------------------------------------------------------


def prepare_features_for_model(
    dataset: Literal["main", "baseline"] = "main",
    time_unit: str = "s",
    test_size: float = 0.2,
    random_state: int = 42,
    feature_mode: Literal[
        "numeric_only", "with_categorical_native", "with_categorical_onehot", "with_categorical"
    ] = "numeric_only",
    use_processed: bool = False,
    shuffle: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    High-level helper: raw CSV → train/test feature matrices.

    Parameters
    ----------
    dataset : {"main", "baseline"}, default ``"main"``
        Source dataset key (forwarded to :func:`load_sample`).
    time_unit : str, default ``"s"``
        Unit of ``submit_time`` timestamps.
    test_size : float, default 0.2
        Fraction of samples reserved for the test set.
    random_state : int, default 42
        Random seed for reproducibility.
    feature_mode : {"numeric_only", "with_categorical_native", "with_categorical_onehot"}
        Controls how categorical columns are handled:

        - ``"numeric_only"``              : drop categoricals.
        - ``"with_categorical_native"``   : keep as pandas Categorical (LightGBM-ready).
        - ``"with_categorical_onehot"``   : one-hot encode categoricals.
    use_processed : bool, default False
        If ``True``, load the pre-processed utilization CSV produced by
        notebook 00 instead of running the full pipeline.

    Returns
    -------
    job_df : pd.DataFrame
        Full job-level table with all engineered features.
    X_train, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_test : np.ndarray
        Target arrays.
    numeric_cols : list of str
    categorical_cols : list of str
    """
    if use_processed:
        paths = load_paths_config()
        root = Path(__file__).resolve().parents[1]
        processed_path = root / paths["data"]["processed_data_dir"] / paths["data"]["processed_full_file"]

        if not processed_path.exists():
            raise FileNotFoundError(
                f"Processed file not found at {processed_path}. Run notebook 00 first."
            )
        job_df = pd.read_csv(processed_path)
        if "arrival_time" in job_df.columns:
            job_df["arrival_time"] = pd.to_datetime(job_df["arrival_time"])
    else:
        raw_df = load_sample(which=dataset)
        job_df = build_job_table_from_sample(raw_df, time_unit=time_unit)
        job_df = add_temporal_features(job_df)
        job_df = add_categorical_features(job_df)
        job_df = add_cluster_utilization_features(job_df)

    # Restore categorical dtype when loading from CSV (CSV drops metadata)
    if feature_mode.startswith("with_categorical"):
        job_df = add_categorical_features(job_df)

    # ------------------------------------------------------------------
    # Default feature sets
    # ------------------------------------------------------------------
    numeric_cols = [
        "gpu_demand",
        "arrival_sec",
        "num_inst",
        "num_cpu",
        "hour_of_day",
        "day_of_week",
        "cluster_load_cpu",
        "cluster_load_gpu",
        "active_job_count",
    ]
    numeric_cols = [c for c in numeric_cols if c in job_df.columns]

    categorical_cols: List[str] = []
    if "user" in job_df.columns:
        categorical_cols.append("user")
    if "gpu_type" in job_df.columns:
        categorical_cols.append("gpu_type")

    # ------------------------------------------------------------------
    # Shared index split (same rows for all feature_mode variants)
    # ------------------------------------------------------------------
    idx_train, idx_test = train_test_split(
        job_df.index,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    idx_train = pd.Index(idx_train).sort_values()
    idx_test = pd.Index(idx_test).sort_values()

    X_full = job_df[numeric_cols + categorical_cols].copy()
    y_full = job_df["job_runtime"].copy()

    X_train_full = X_full.loc[idx_train].copy()
    X_test_full = X_full.loc[idx_test].copy()
    y_train = y_full.loc[idx_train].to_numpy()
    y_test = y_full.loc[idx_test].to_numpy()

    # ------------------------------------------------------------------
    # Feature-mode-specific processing
    # ------------------------------------------------------------------
    if feature_mode == "numeric_only":
        X_train = X_train_full[numeric_cols].copy()
        X_test = X_test_full[numeric_cols].copy()

    elif feature_mode == "with_categorical_native":
        X_train = X_train_full.copy()
        X_test = X_test_full.copy()
        for col in categorical_cols:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")

    elif feature_mode == "with_categorical_onehot":
        if categorical_cols:
            # Support both scikit-learn ≥ 1.2 (sparse_output) and older (sparse)
            try:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

            train_cat = encoder.fit_transform(X_train_full[categorical_cols])
            test_cat = encoder.transform(X_test_full[categorical_cols])

            if hasattr(train_cat, "toarray"):
                train_cat = train_cat.toarray()
            if hasattr(test_cat, "toarray"):
                test_cat = test_cat.toarray()

            cat_names = encoder.get_feature_names_out(categorical_cols)
            train_cat_df = pd.DataFrame(train_cat, columns=cat_names, index=X_train_full.index)
            test_cat_df = pd.DataFrame(test_cat, columns=cat_names, index=X_test_full.index)

            X_train = pd.concat([X_train_full[numeric_cols], train_cat_df], axis=1)
            X_test = pd.concat([X_test_full[numeric_cols], test_cat_df], axis=1)
        else:
            X_train = X_train_full[numeric_cols].copy()
            X_test = X_test_full[numeric_cols].copy()

    elif feature_mode == "with_categorical":
        # Returns all numeric AND categorical columns without any encoding
        # This is strictly designed for LightGBM native categorical support
        feature_cols = numeric_cols + categorical_cols
        X_train = X_train_full[feature_cols].copy()
        X_test = X_test_full[feature_cols].copy()

    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode!r}")

    return job_df, X_train, X_test, y_train, y_test, numeric_cols, categorical_cols