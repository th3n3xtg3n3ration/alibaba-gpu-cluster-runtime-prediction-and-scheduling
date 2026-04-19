"""
evaluation.py

Centralized Regression Evaluation Metrics

This module provides the standard evaluation function used across all runtime
prediction experiments in the thesis. All metrics are computed in a single pass
for consistency and to avoid ad-hoc per-notebook implementations.

Key Components
--------------
evaluate_regression
    Returns a dictionary of MAE, RMSE, R², MAPE, and MdAE.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


__all__ = ["evaluate_regression"]


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute standard regression metrics for model evaluation.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth (correct) target runtime values.
    y_pred : array-like of shape (n_samples,)
        Model-predicted runtime values.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``mae``   : Mean Absolute Error (seconds).
        - ``rmse``  : Root Mean Squared Error (seconds).
        - ``r2``    : Coefficient of Determination R².
        - ``mape``  : Mean Absolute Percentage Error (fraction, not %).
          Only computed over samples where ``y_true > 0`` to avoid
          division by zero.
        - ``mdae``  : Median Absolute Error (seconds).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mdae = float(median_absolute_error(y_true, y_pred))

    # MAPE: guard against zero-runtime jobs (would produce inf / nan)
    nonzero_mask = y_true > 0
    if nonzero_mask.any():
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
        )
    else:
        mape = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "mdae": mdae,
    }
