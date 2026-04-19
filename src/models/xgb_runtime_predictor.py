"""
xgb_runtime_predictor.py

XGBoost Runtime Predictor

This module implements an XGBoost-based regressor for GPU job runtime
prediction. XGBoost is a strong gradient boosting baseline and is
commonly used in production systems.

Typical usage:

    from src.models import XGBPredictor

    model = XGBPredictor(params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model.save("results/models/xgb_runtime_model.pkl")
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import joblib
import numpy as np
import xgboost as xgb


__all__ = ["XGBPredictor"]


class XGBPredictor:
    """
    XGBoost regressor for GPU job runtime prediction.

    Parameters
    ----------
    params : dict
        Hyperparameters for XGBoost, typically loaded from
        configs/model_xgb.yaml or configs/models.yaml.

        Example:
        params = {
            "objective": "reg:squarederror",
            "max_depth": 8,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "nthread": -1,
            "tree_method": "hist",
        }

    Attributes
    ----------
    params : dict
        Stored hyperparameters.
    model : xgboost.Booster or None
        Trained XGBoost model after fit().
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model: Optional[xgb.Booster] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: Optional[int] = 50,
        verbose_eval: int = 50,
    ) -> "XGBPredictor":
        """
        Train the XGBoost model.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training targets.
        X_val : np.ndarray, optional
            Validation features for early stopping.
        y_val : np.ndarray, optional
            Validation targets.
        num_boost_round : int
            Maximum number of boosting rounds.
        early_stopping_rounds : int or None
            Early stopping patience.
        verbose_eval : int
            Evaluation print frequency.

        Returns
        -------
        self : XGBPredictor
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)

        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "valid"))

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict runtimes for a set of feature vectors.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted runtimes.
        """
        if self.model is None:
            raise ValueError("XGBoost model has not been trained yet.")

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, file_path: str) -> None:
        """
        Save the trained XGBoost model to disk.

        Parameters
        ----------
        file_path : str
            File path for the saved model.
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        joblib.dump(self.model, file_path)

    def load(self, file_path: str) -> "XGBPredictor":
        """
        Load a trained XGBoost model from disk.

        Parameters
        ----------
        file_path : str
            Path to a previously saved model file.

        Returns
        -------
        self : XGBPredictor
        """
        self.model = joblib.load(file_path)
        return self