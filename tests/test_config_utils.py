"""
tests/test_config_utils.py

Unit tests for src.config_utils.

Tests exercise the real configs/paths.yaml and configs/models.yaml files
that ship with the project.
"""
import unittest
from pathlib import Path
from src.config_utils import load_paths_config, load_model_config

class TestConfigUtils(unittest.TestCase):
    def test_load_paths_config(self):
        """Test that paths.yaml loads and contains expected top-level keys."""
        cfg = load_paths_config()
        self.assertIn("data", cfg)
        self.assertIn("results", cfg)
        self.assertIn("notebooks", cfg)

    def test_load_model_config_valid(self):
        """Test loading a confirmed model key from models.yaml."""
        # "lgbm" is a standard key in models.yaml
        cfg = load_model_config("lgbm")
        self.assertIn("model", cfg)
        self.assertIn("hyperparameters", cfg)

    def test_load_model_config_invalid(self):
        """Test that a non-existent model key raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_model_config("non_existent_model_key_123")

if __name__ == "__main__":
    unittest.main()
