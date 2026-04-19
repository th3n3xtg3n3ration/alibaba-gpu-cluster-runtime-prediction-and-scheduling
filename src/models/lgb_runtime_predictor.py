"""
lightgbm_runtime_predictor.py

LightGBM Runtime Predictor

This module implements a clean wrapper around LightGBM for GPU job runtime
prediction. It supports training with early stopping (LightGBM 4.x style),
prediction, and model persistence.

Typical usage:

    from src.models import LightGBMPredictor

    model = LightGBMPredictor(params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model.save("results/models/lgbm_runtime_model.pkl")
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import joblib
import lightgbm as lgb


__all__ = ["LightGBMPredictor"]
import numpy as np


class LightGBMPredictor:
    """
    LightGBM-based regressor for GPU job runtime prediction.

    Parameters
    ----------
    params : dict
        Hyperparameters for LightGBM, typically loaded from
        configs/model_lgbm.yaml or configs/models.yaml.

        Example
        -------
        params = {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": 64,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 3,
            "verbose": -1,
        }

    Attributes
    ----------
    params : dict
        Stored hyperparameters.
    model : lightgbm.Booster or None
        Trained LightGBM booster after calling fit().
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model: Optional[lgb.Booster] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        num_boost_round: int = 2000,
        early_stopping_rounds: Optional[int] = 100,
        verbose_eval: int = 100,
    ) -> "LightGBMPredictor":
        """
        Train a LightGBM model on the provided training data.

        This implementation is compatible with recent LightGBM versions
        (>= 4.x), where early stopping is configured via callbacks instead
        of the legacy `early_stopping_rounds` argument in `lgb.train`.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training targets (true runtimes).
        X_val : np.ndarray, optional
            Validation features for early stopping (optional).
        y_val : np.ndarray, optional
            Validation targets for early stopping (optional).
        num_boost_round : int
            Maximum number of boosting iterations.
        early_stopping_rounds : int or None
            Early stopping patience. If None, no early stopping is used.
        verbose_eval : int
            Print evaluation results every `verbose_eval` iterations.

        Returns
        -------
        self : LightGBMPredictor
        """
        # Construct the training dataset
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        # Optional validation set for early stopping
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val)
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Configure callbacks (LightGBM >= 4.x uses callbacks for early stopping)
        callbacks = []

        if early_stopping_rounds is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    verbose=True,
                )
            )

        if verbose_eval is not None and verbose_eval > 0:
            callbacks.append(
                lgb.log_evaluation(period=verbose_eval)
            )

        # Train the model
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
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
            raise ValueError("LightGBM model has not been trained yet.")

        # Use best_iteration if available (early stopping)
        num_iteration = getattr(self.model, "best_iteration", None)
        return self.model.predict(X, num_iteration=num_iteration)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, file_path: str) -> None:
        """
        Save the trained LightGBM model to disk.

        Parameters
        ----------
        file_path : str
            File path for the saved model (e.g., .pkl).
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        joblib.dump(self.model, file_path)

    def load(self, file_path: str) -> "LightGBMPredictor":
        """
        Load a trained LightGBM model from disk.

        Parameters
        ----------
        file_path : str
            Path to a previously saved model file.

        Returns
        -------
        self : LightGBMPredictor
        """
        self.model = joblib.load(file_path)
        return self