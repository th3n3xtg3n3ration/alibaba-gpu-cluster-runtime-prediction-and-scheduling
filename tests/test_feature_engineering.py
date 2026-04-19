"""
tests/test_feature_engineering.py

Unit tests for src.feature_engineering.

Tests cover job table construction, temporal feature extraction, 
and validation of required columns.
"""
import unittest
import pandas as pd
import numpy as np
from src.feature_engineering import build_job_table_from_sample, add_temporal_features

class TestFeatureEngineering(unittest.TestCase):
    def test_build_job_table_logic(self):
        """Test successful construction of the job table from raw trace sample."""
        # Minimal valid input
        data = {
            "job_id": [1, 2],
            "submit_time": [1000, 2000],
            "duration": [10.0, 20.0],
            "num_gpu": [1, 2],
            "user": ["alice", "bob"],
            "gpu_type": ["T4", "V100"]
        }
        df = pd.DataFrame(data)
        
        result = build_job_table_from_sample(df, time_unit="s")
        
        self.assertEqual(len(result), 2)
        self.assertIn("job_runtime", result.columns)
        self.assertIn("gpu_demand", result.columns)
        self.assertEqual(result["job_runtime"].iloc[0], 10.0)
        self.assertEqual(result["gpu_demand"].iloc[1], 2)

    def test_build_job_table_missing_cols(self):
        """Test that missing required columns raise a ValueError."""
        data = {"job_id": [1]}
        df = pd.DataFrame(data)
        with self.assertRaisesRegex(ValueError, "Required column"):
            build_job_table_from_sample(df)

    def test_add_temporal_features(self):
        """Test extraction of hour and day features from arrival timestamps."""
        df = pd.DataFrame({
            "arrival_time": [pd.to_datetime("2024-01-01 10:00:00")]
        })
        result = add_temporal_features(df)
        self.assertEqual(result["hour_of_day"].iloc[0], 10)
        self.assertIn("day_of_week", result.columns)

if __name__ == "__main__":
    unittest.main()
