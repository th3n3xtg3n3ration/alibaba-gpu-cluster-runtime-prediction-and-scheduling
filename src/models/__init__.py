"""
src.models

Runtime Prediction Model Package

This sub-package provides clean, consistent wrappers around LightGBM, XGBoost,
RandomForest, and PyTorch deep learning architectures for GPU job runtime prediction.

All models expose the same ``fit / predict / save / load`` interface so they can be
used interchangeably in experiment notebooks and tuning scripts.

Public API
----------
LightGBMPredictor
    LightGBM-based regressor with early stopping support.
XGBPredictor
    XGBoost-based regressor with early stopping support.
RandomForestPredictor
    scikit-learn RandomForest regressor.

Deep learning architectures (see :mod:`src.models.dl_runtime_predictor`):

RuntimePredictorCNN
    1D-Convolutional network for tabular feature extraction.
RuntimePredictorLSTM
    LSTM recurrent network.
RuntimePredictorCNNLSTM
    Hybrid CNN + LSTM architecture.

Evaluation utility (see :mod:`src.models.evaluation`):

evaluate_regression
    Compute MAE, RMSE, R², MAPE, and MdAE for any prediction array.

These models are trained in:
    ``notebooks/04_runtime_prediction_models.ipynb``

and used for scheduling in:
    ``notebooks/05_scheduler_evaluation.ipynb``
"""

from .lgb_runtime_predictor import LightGBMPredictor
from .xgb_runtime_predictor import XGBPredictor
from .rf_runtime_predictor import RandomForestPredictor
from .evaluation import evaluate_regression

# Deep learning models require PyTorch — import conditionally so the package
# remains usable in CPU-only / non-DL environments.
try:
    from .dl_runtime_predictor import (
        RuntimePredictorCNN,
        RuntimePredictorLSTM,
        RuntimePredictorCNNLSTM,
    )
    _DL_AVAILABLE = True
except ModuleNotFoundError:
    # torch not installed — DL classes unavailable but rest of package works.
    _DL_AVAILABLE = False

__all__ = [
    # ML tree models (always available)
    "LightGBMPredictor",
    "XGBPredictor",
    "RandomForestPredictor",
    # Evaluation utility (always available)
    "evaluate_regression",
    # Deep learning models (available when torch is installed)
    "RuntimePredictorCNN",
    "RuntimePredictorLSTM",
    "RuntimePredictorCNNLSTM",
]