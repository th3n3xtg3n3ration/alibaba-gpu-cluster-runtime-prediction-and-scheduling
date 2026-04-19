"""
data_loading.py

Data Loading Utilities

This module provides high-level functions for loading the Alibaba PAI 100K job
samples. It supports loading the primary thesis dataset, baseline estimates, and
pre-processed utilization datasets, utilizing the path configurations defined in
``configs/paths.yaml``.

Key Components
--------------
PathsConfig
    Dataclass for managing file and directory paths.
load_main_sample
    Loads the primary 100K job dataset.
load_baseline_estimate
    Loads optional baseline estimates for comparison.
load_processed_full
    Loads the feature-engineered utilization dataset.
load_sample
    Generic dispatcher controlled by a simple flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import pandas as pd
import yaml


__all__ = [
    "PathsConfig",
    "load_main_sample",
    "load_baseline_estimate",
    "load_processed_full",
    "load_sample",
]

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.yaml"


# ---------------------------------------------------------------------
# Paths configuration
# ---------------------------------------------------------------------


@dataclass
class PathsConfig:
    """
    Centralized holder for all data-related filesystem paths.

    Parameters
    ----------
    raw_data_dir : Path
        Directory containing raw CSV files.
    processed_data_dir : Path
        Directory for feature-engineered / processed CSVs.
    cache_dir : Path
        Directory for temporary cache files.
    main_sample_file : str
        Filename of the primary 100K job sample CSV.
    baseline_estimate_file : str, optional
        Filename of the optional Alibaba baseline estimate CSV.
    processed_full_file : str, optional
        Filename of the feature-engineered utilization CSV.
    """

    raw_data_dir: Path
    processed_data_dir: Path
    cache_dir: Path
    main_sample_file: str
    baseline_estimate_file: Optional[str] = None
    processed_full_file: str = "100k_job_with_utilization_full.csv"

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_PATHS_CONFIG) -> "PathsConfig":
        """
        Load ``paths.yaml`` and construct a :class:`PathsConfig` object.

        Parameters
        ----------
        path : Path or str, optional
            Location of the YAML config file. Defaults to
            ``configs/paths.yaml`` relative to the project root.

        Returns
        -------
        PathsConfig

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"paths.yaml not found at: {path}")

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        data_cfg: Dict = cfg.get("data", {})

        raw_dir = PROJECT_ROOT / data_cfg.get("raw_data_dir", "data/alibaba_cluster_trace/")
        processed_dir = PROJECT_ROOT / data_cfg.get("processed_data_dir", "data/processed/")
        cache_dir = PROJECT_ROOT / data_cfg.get("cache_dir", "data/cache/")

        main_sample_file = data_cfg.get("main_sample_file", "pai_job_no_estimate_100K.csv")
        baseline_estimate_file = data_cfg.get("baseline_estimate_file", None)
        processed_full_file = data_cfg.get(
            "processed_full_file", "100k_job_with_utilization_full.csv"
        )

        return cls(
            raw_data_dir=raw_dir,
            processed_data_dir=processed_dir,
            cache_dir=cache_dir,
            main_sample_file=main_sample_file,
            baseline_estimate_file=baseline_estimate_file,
            processed_full_file=processed_full_file,
        )


# ---------------------------------------------------------------------
# Core CSV loader
# ---------------------------------------------------------------------


def _load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file with a clear error message on failure.

    Parameters
    ----------
    path : Path
        Full path to the CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the given path.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------


def load_main_sample(paths_cfg: Optional[PathsConfig] = None) -> pd.DataFrame:
    """
    Load the thesis primary dataset.

    Parameters
    ----------
    paths_cfg : PathsConfig, optional
        Configuration object containing data paths. If ``None``, loads
        from the default YAML at ``configs/paths.yaml``.

    Returns
    -------
    pd.DataFrame
        Job-level table containing the 100K job sample with true durations.
    """
    if paths_cfg is None:
        paths_cfg = PathsConfig.from_yaml()

    csv_path = paths_cfg.raw_data_dir / paths_cfg.main_sample_file
    logger.info("[data_loading] Loading MAIN sample from: %s", csv_path)
    return _load_csv(csv_path)


def load_baseline_estimate(paths_cfg: Optional[PathsConfig] = None) -> pd.DataFrame:
    """
    Load the optional baseline dataset containing Alibaba's duration estimates.

    Parameters
    ----------
    paths_cfg : PathsConfig, optional
        Configuration object containing data paths.

    Returns
    -------
    pd.DataFrame
        Dataset containing Alibaba's historical runtime estimates for comparison.

    Raises
    ------
    ValueError
        If ``baseline_estimate_file`` is not configured in ``paths.yaml``.
    """
    if paths_cfg is None:
        paths_cfg = PathsConfig.from_yaml()

    if not paths_cfg.baseline_estimate_file:
        raise ValueError("No 'baseline_estimate_file' configured in paths.yaml")

    csv_path = paths_cfg.raw_data_dir / paths_cfg.baseline_estimate_file
    logger.info("[data_loading] Loading BASELINE estimate sample from: %s", csv_path)
    return _load_csv(csv_path)


def load_processed_full(paths_cfg: Optional[PathsConfig] = None) -> pd.DataFrame:
    """
    Load the feature-engineered dataset with cluster utilization features.

    The filename is read from ``configs/paths.yaml`` under
    ``data.processed_full_file`` (default: ``100k_job_with_utilization_full.csv``).

    Parameters
    ----------
    paths_cfg : PathsConfig, optional
        Configuration object containing data paths.

    Returns
    -------
    pd.DataFrame
        Processed dataset with utilization features.
    """
    if paths_cfg is None:
        paths_cfg = PathsConfig.from_yaml()

    csv_path = paths_cfg.processed_data_dir / paths_cfg.processed_full_file
    logger.info("[data_loading] Loading PROCESSED FULL sample from: %s", csv_path)
    return _load_csv(csv_path)


def load_sample(
    which: Literal["main", "baseline", "main_utilization"] = "main",
    paths_cfg: Optional[PathsConfig] = None,
) -> pd.DataFrame:
    """
    Generic loader dispatched by a simple flag.

    Parameters
    ----------
    which : {"main", "baseline", "main_utilization"}
        - ``"main"``             : thesis primary dataset (raw 100K).
        - ``"baseline"``         : Alibaba's estimate dataset (if configured).
        - ``"main_utilization"`` : processed full dataset with utilization features.
    paths_cfg : PathsConfig, optional
        Configuration object. If ``None``, loads from default YAML.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If ``which`` is not a recognized key.
    """
    if which == "main":
        return load_main_sample(paths_cfg)
    elif which == "baseline":
        return load_baseline_estimate(paths_cfg)
    elif which == "main_utilization":
        return load_processed_full(paths_cfg)
    else:
        raise ValueError(f"Unknown dataset key: {which!r}. Choose from 'main', 'baseline', 'main_utilization'.")