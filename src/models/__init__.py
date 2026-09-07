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

Evaluation utilities (see :mod:`src.models.evaluation`):

evaluate_regression
    Compute MAE, RMSE, R², MAPE, and MdAE for any prediction array.
is_degenerate_prediction
    Read the prediction-spread fields that ``evaluate_regression`` records and
    report whether the model collapsed to a constant output. Exported next to
    the function that writes those fields so a results table can consult it
    without reaching past the package API -- the missing half of the guard was
    what let a constant predictor be ranked as an ordinary model.
is_near_constant_prediction
    The weaker, non-excluding finding beside it: a model that does rank, but
    out of only a handful of distinct values. Kept apart from the exclusion so
    a coarse-but-working predictor is not reported as one that ranks nothing.
prediction_ranks_nothing
    The rule both of those and :class:`~src.simulation.SJFPredScheduler`'s
    refusal are written in terms of, so the modelling and scheduling chapters
    cannot reach opposite verdicts about the same model.

These models are trained in:
    ``notebooks/04_runtime_prediction_models.ipynb``

and used for scheduling in:
    ``notebooks/05_scheduler_evaluation.ipynb``
"""

from .lgb_runtime_predictor import LightGBMPredictor
from .xgb_runtime_predictor import XGBPredictor
from .rf_runtime_predictor import RandomForestPredictor
from .evaluation import (
    evaluate_regression,
    is_degenerate_prediction,
    is_near_constant_prediction,
    prediction_ranks_nothing,
)

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
    # Evaluation utilities (always available)
    "evaluate_regression",
    "is_degenerate_prediction",
    "is_near_constant_prediction",
    "prediction_ranks_nothing",
    # Deep learning models (available when torch is installed)
    "RuntimePredictorCNN",
    "RuntimePredictorLSTM",
    "RuntimePredictorCNNLSTM",
]