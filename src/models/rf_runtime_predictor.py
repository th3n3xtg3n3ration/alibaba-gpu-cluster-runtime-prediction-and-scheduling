"""
rf_runtime_predictor.py

RandomForest Runtime Predictor

This module implements a :class:`sklearn.ensemble.RandomForestRegressor`-based
model for GPU job runtime prediction. RandomForest serves as a strong classical
baseline against gradient-boosted and deep learning models.

Typical usage::

    from src.models import RandomForestPredictor

    model = RandomForestPredictor(params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model.save("results/models/rf_runtime_model.pkl")
"""

from __future__ import annotations

from typing import Dict, Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


__all__ = ["RandomForestPredictor"]


class RandomForestPredictor:
    """
    RandomForest regression model for GPU job runtime prediction.

    Parameters
    ----------
    params : dict
        Hyperparameters for RandomForestRegressor, typically loaded from
        configs/model_rf.yaml or configs/models.yaml.

        Example::

            params = {
                "n_estimators": 300,
                "max_depth": 20,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",  # "auto" removed in scikit-learn >= 1.1
                "n_jobs": -1,
                "bootstrap": True,
            }

    Attributes
    ----------
    params : dict
        Stored hyperparameters.
    model : RandomForestRegressor
        Underlying scikit-learn model.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = RandomForestRegressor(**params)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestPredictor":
        """
        Train the RandomForest model.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training targets.

        Returns
        -------
        self : RandomForestPredictor
        """
        self.model.fit(X_train, y_train)
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
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, file_path: str) -> None:
        """
        Save the trained RandomForest model to disk.

        Parameters
        ----------
        file_path : str
            File path for the saved model.
        """
        joblib.dump(self.model, file_path)

    def load(self, file_path: str) -> "RandomForestPredictor":
        """
        Load a trained RandomForest model from disk.

        Parameters
        ----------
        file_path : str
            Path to a previously saved model file.

        Returns
        -------
        self : RandomForestPredictor
        """
        self.model = joblib.load(file_path)
        return self