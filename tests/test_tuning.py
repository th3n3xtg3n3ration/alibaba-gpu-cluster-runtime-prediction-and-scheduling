"""
Unit tests for src.tuning.

Verifies chronological validation splitting and checkpoint metadata enrichment.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.tuning import chronological_train_validation_split, save_checkpoint


class TestTuning(unittest.TestCase):
    def test_chronological_train_validation_split_preserves_order(self):
        """Validation split should use the trailing rows without shuffling."""
        X = pd.DataFrame({"value": [0, 1, 2, 3, 4]})
        y = np.array([10, 11, 12, 13, 14])

        X_train, X_val, y_train, y_val = chronological_train_validation_split(
            X, y, validation_size=0.4
        )

        self.assertEqual(X_train["value"].tolist(), [0, 1, 2])
        self.assertEqual(X_val["value"].tolist(), [3, 4])
        self.assertEqual(y_train.tolist(), [10, 11, 12])
        self.assertEqual(y_val.tolist(), [13, 14])

    def test_save_checkpoint_enriches_required_metadata(self):
        """Checkpoint saves should include required experiment metadata fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            with patch("src.tuning._CHECKPOINT_DIR", checkpoint_dir):
                checkpoint_path = save_checkpoint(
                    "exp_b_xgb_oh",
                    {
                        "metrics": {
                            "mae": np.float64(1.5),
                            "rmse": np.float64(2.5),
                            "r2": np.float64(0.5),
                            "mape": np.float64(0.1),
                            "mdae": np.float64(1.0),
                        },
                        "best_params": {"max_depth": np.int64(4)},
                        "train_size": 6,
                        "test_size": 2,
                    },
                )

            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["experiment"], "exp_b")
            self.assertEqual(payload["model"], "xgb_oh")
            self.assertEqual(payload["feature_mode"], "with_categorical_onehot")
            self.assertEqual(payload["train_size"], 6)
            self.assertEqual(payload["test_size"], 2)
            self.assertEqual(payload["status"], "complete")
            self.assertIn("timestamp", payload)
            self.assertEqual(payload["best_params"]["max_depth"], 4)
            self.assertEqual(
                sorted(payload["metrics"].keys()),
                ["mae", "mape", "mdae", "r2", "rmse"],
            )


if __name__ == "__main__":
    unittest.main()
