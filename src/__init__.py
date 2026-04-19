"""
src

Alibaba GPU Cluster Runtime Prediction and Scheduling — Core Package

This package provides a modular, production-quality library for:
  - Loading and normalizing Alibaba PAI GPU cluster trace data
  - Feature engineering (temporal, categorical, utilization sweep-line)
  - Runtime prediction models (LightGBM, XGBoost, RandomForest, DL)
  - Hyperparameter tuning (random search / grid search)
  - Discrete-event scheduling simulation (FIFO, SJF-Oracle, SJF-Pred)
  - Visualization and statistical analysis utilities

Sub-packages
------------
analysis    : Workload characterization and statistical summaries.
models      : ML and DL runtime prediction model implementations.
simulation  : Scheduler simulation framework.

Modules
-------
data_loading        : CSV loaders for raw and processed datasets.
feature_engineering : Feature transformation pipeline.
config_utils        : YAML configuration loader utilities.
visualization       : Matplotlib-based plotting library.
tuning              : Hyperparameter search framework.
"""

from __future__ import annotations

__all__ = [
    "data_loading",
    "feature_engineering",
    "config_utils",
    "visualization",
    "tuning",
    "analysis",
    "models",
    "simulation",
]
