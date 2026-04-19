"""
config_utils.py

Configuration Loading Utilities

This module provides a robust interface for loading YAML configuration files
from the ``configs/`` directory. It dynamically handles both centralized
(``models.yaml``) and legacy individual model configuration files.

Key Components
--------------
load_paths_config
    Accesses global directory and file path settings.
load_model_config
    Retrieves model-specific hyperparameters and training settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


__all__ = [
    "load_paths_config",
    "load_model_config",
]

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------
# Low-level YAML loader
# ---------------------------------------------------------------------


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and return its content as a dict.

    Parameters
    ----------
    path : Path
        Full path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML content. Returns an empty dict if the file is empty.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg if cfg is not None else {}


# ---------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------


def load_paths_config() -> Dict[str, Any]:
    """
    Load the global paths configuration from ``configs/paths.yaml``.

    Returns
    -------
    dict
        Dictionary with at least the ``data``, ``results``, and
        ``notebooks`` sections.

        Example::

            cfg = load_paths_config()
            raw_dir = cfg["data"]["raw_data_dir"]
    """
    path = CONFIG_DIR / "paths.yaml"
    return _load_yaml_file(path)


def load_model_config(model_key: str) -> Dict[str, Any]:
    """
    Load model configuration for a given model key.

    The function first tries ``configs/models.yaml``::

        models:
          lgbm: { ... }
          xgb:  { ... }
          rf:   { ... }

    If that fails, it falls back to an individual file::

        configs/model_<model_key>.yaml
        e.g., configs/model_lgbm.yaml

    Parameters
    ----------
    model_key : str
        Model key, e.g. ``"lgbm"``, ``"xgb"``, ``"rf"``.

    Returns
    -------
    dict
        Configuration dictionary for the requested model.

        Typical sections:

        - ``model``
        - ``hyperparameters``
        - ``training``
        - ``features``

    Raises
    ------
    FileNotFoundError
        If neither ``models.yaml`` nor ``model_<key>.yaml`` contains the
        configuration for the requested key.
    """
    # 1) Try combined models.yaml
    models_yaml = CONFIG_DIR / "models.yaml"
    if models_yaml.exists():
        cfg = _load_yaml_file(models_yaml)
        models_block = cfg.get("models", {})
        if model_key in models_block:
            return models_block[model_key]

    # 2) Fallback: individual file configs/model_<key>.yaml
    single_path = CONFIG_DIR / f"model_{model_key}.yaml"
    if single_path.exists():
        return _load_yaml_file(single_path)

    # 3) Nothing found
    raise FileNotFoundError(
        f"No configuration found for model key '{model_key}'. "
        f"Checked: {models_yaml} and {single_path}"
    )
