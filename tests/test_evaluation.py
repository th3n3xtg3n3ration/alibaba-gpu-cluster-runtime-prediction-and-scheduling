"""
tests/test_evaluation.py

Unit tests for src.models.evaluation.

Provides verification for regression metrics (MAE, RMSE, R2, MAPE, MdAE) 
including edge cases like division by zero.
"""
import unittest
import numpy as np
from src.models.evaluation import evaluate_regression

class TestEvaluation(unittest.TestCase):
    def test_evaluate_regression_basic(self):
        """Test standard regression metrics calculation."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 30.0])
        
        metrics = evaluate_regression(y_true, y_pred)
        
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)
        self.assertIn("mape", metrics)
        self.assertIn("mdae", metrics)
        
        # MAE = (1 + 1 + 0) / 3 = 0.666...
        self.assertAlmostEqual(metrics["mae"], 2/3)
        # MdAE = median([1, 1, 0]) = 1.0
        self.assertEqual(metrics["mdae"], 1.0)

    def test_evaluate_regression_mape_zeros(self):
        """Test division by zero guard in MAPE calculation."""
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([5.0, 11.0])
        
        metrics = evaluate_regression(y_true, y_pred)
        # MAPE should only be calculated for y_true > 0: |(10-11)/10| = 0.1
        self.assertAlmostEqual(metrics["mape"], 0.1)

    def test_evaluate_regression_all_zeros(self):
        """Test handling of MAPE when all ground truth values are zero."""
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        
        metrics = evaluate_regression(y_true, y_pred)
        self.assertTrue(np.isnan(metrics["mape"]))

if __name__ == "__main__":
    unittest.main()
