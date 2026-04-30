---
description: "Use when writing, editing, running, or debugging unit tests in tests/. Covers test structure, assertions, mock patterns, and the 11-test quality gate for the GPU scheduling thesis project."
applyTo: "tests/**/*.py"
---

# Testing Guidelines — GPU Scheduling Thesis

## Framework & Location

- **Framework:** Python `unittest` (standard library — no pytest dependency)
- **Location:** `tests/` directory
- **Discovery:** `python -m unittest discover tests -v`
- **Quality gate:** All 11 tests must pass before any experiment is considered complete.

## Test File Map

| File | Module Tested | # Tests |
|------|--------------|---------|
| `test_config_utils.py` | `src/config_utils` | 2 |
| `test_feature_engineering.py` | `src/feature_engineering` | 2 |
| `test_evaluation.py` | `src/models/evaluation` | 3 |
| `test_simulation.py` | `src/simulation/scheduler_simulator` | 2 |

New tests go in the relevant existing file. Create a new file only when testing a new module.

## Test Boilerplate

```python
import unittest
import numpy as np
import pandas as pd


class TestEvaluationMetrics(unittest.TestCase):

    def setUp(self) -> None:
        """Set up fixtures reused across tests."""
        self.y_true = np.array([100.0, 200.0, 300.0])
        self.y_pred = np.array([110.0, 190.0, 310.0])

    def test_mae_correct_computation(self) -> None:
        """MAE should equal mean absolute difference."""
        from src.models.evaluation import evaluate_regression
        metrics = evaluate_regression(self.y_true, self.y_pred)
        self.assertAlmostEqual(metrics["mae"], 10.0, places=2)

    def test_mae_perfect_prediction_returns_zero(self) -> None:
        from src.models.evaluation import evaluate_regression
        metrics = evaluate_regression(self.y_true, self.y_true)
        self.assertEqual(metrics["mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
```

## What to Test

**DO test:**
- Correct outputs for known inputs (unit tests with manually computed expected values)
- Edge cases: empty DataFrames, zero-duration jobs, single-job schedulers
- Error handling: `ValueError` when required columns are missing
- Ordering guarantees: SJF produces shorter-first queue order

**DO NOT test:**
- Model accuracy (non-deterministic, depends on data)
- Exact figure contents
- Config file existence (integration concern, not unit)
- Private helper functions directly (`_*` functions)

## Assertions

Prefer specific assertions over `assertTrue`:

| Instead of | Use |
|-----------|-----|
| `assertTrue(a == b)` | `assertEqual(a, b)` |
| `assertTrue(abs(a-b) < 0.01)` | `assertAlmostEqual(a, b, places=2)` |
| `assertTrue(len(x) == 0)` | `assertEqual(len(x), 0)` |
| `assertTrue(x is None)` | `assertIsNone(x)` |

## Handling File Paths in Tests

Use `unittest.mock.patch` or temp directories — never expect specific absolute paths:

```python
import tempfile
from pathlib import Path

class TestCheckpointSaving(unittest.TestCase):
    def test_checkpoint_writes_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "exp_test_rf.json"
            # call the function under test with out path
            self.assertTrue(out.exists())
```

## Running Tests

```bash
# All tests with verbose output
python -m unittest discover tests -v

# Single file
python -m unittest tests.test_evaluation -v

# Single test method
python -m unittest tests.test_evaluation.TestEvaluationMetrics.test_mae_correct_computation
```
