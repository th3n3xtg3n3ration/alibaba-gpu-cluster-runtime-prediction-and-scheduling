"""
tests/test_evaluation.py

Unit tests for src.models.evaluation.

Provides verification for regression metrics (MAE, RMSE, R2, MAPE, MdAE)
including edge cases like division by zero, and for the collapse guard that
reads the prediction-spread fields those metrics carry.

The guard is the half that was missing. ``evaluate_regression`` recorded
``pred_std`` and ``pred_unique_frac`` into every checkpoint and every results
table while nothing looked at them, so CNN-LSTM (Numeric Sequence) -- one
constant, 4128.124023 s, for all 16,437 test jobs -- was ranked on its
perfectly ordinary MAE alongside models that had learned something. Writing
evidence is not reading it, so both directions are pinned here: the warning
fires where the collapse happens, and it stays silent for a model that ranks.

The last class pins the guard against the other half of the same rule, the
scheduling simulator's refusal. Nothing compared the two, and they drifted into
opposite verdicts on one model; the rule now lives in one function and this is
what keeps a change to either side from being invisible.
"""
import unittest
import warnings

import numpy as np
import pandas as pd

from src.models.evaluation import (
    evaluate_regression,
    is_degenerate_prediction,
    is_near_constant_prediction,
)
from src.simulation.scheduler_simulator import DegeneratePredictionError, SJFPredScheduler

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

    # Both MAPE tests below carry a third sample purely so the arrays can show
    # three distinct predictions. Two-element fixtures genuinely trip the
    # collapse rule -- two or fewer distinct outputs cannot order jobs at any
    # sample size -- and the resulting warning is correct, but it has nothing
    # to do with the division-by-zero guard these two are about. The collapse
    # behaviour is asserted on its own below rather than silenced here.

    def test_evaluate_regression_mape_zeros(self):
        """Test division by zero guard in MAPE calculation."""
        y_true = np.array([0.0, 10.0, 20.0])
        y_pred = np.array([5.0, 11.0, 22.0])

        metrics = evaluate_regression(y_true, y_pred)
        # MAPE should only be calculated for y_true > 0:
        # mean(|(10-11)/10|, |(20-22)/20|) = 0.1 -> 10.0%
        # (mape is reported on a 0-100 percentage scale, not as a raw fraction)
        self.assertAlmostEqual(metrics["mape"], 10.0)

    def test_evaluate_regression_all_zeros(self):
        """Test handling of MAPE when all ground truth values are zero."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        metrics = evaluate_regression(y_true, y_pred)
        self.assertTrue(np.isnan(metrics["mape"]))


class TestCollapsedPredictionsAreStatedWhereTheyHappen(unittest.TestCase):
    """The collapse verdict is taken on the evidence, not left for a table.

    ``pred_std`` and ``pred_unique_frac`` were already in the metrics dict when
    the constant predictor was published; what was missing was anything reading
    them. So these assert the reading, in both directions -- a guard that only
    ever fires is as useless as one that never does, and this one has to stay
    quiet for the working models whose numbers the thesis reports.
    """

    def test_a_constant_prediction_warns_at_the_point_it_is_measured(self):
        # The real failure, at its real value: one output for every test job.
        # MAE 6717 s and R2 -0.01 sit mid-pack among the working models, so
        # nothing else in the returned dict separates this from a model that
        # learned something.
        y_true = np.linspace(10.0, 20000.0, 100)
        y_pred = np.full(100, 4128.124023)

        with self.assertWarns(UserWarning) as caught:
            metrics = evaluate_regression(y_true, y_pred)

        self.assertIn("collapsed", str(caught.warning))
        # The metrics still come back: the caller decides what to do with a
        # collapsed model, and losing the numbers would only hide the reason.
        self.assertEqual(metrics["n_predictions"], 100)
        self.assertTrue(is_degenerate_prediction(metrics))

    def test_a_prediction_that_ranks_is_left_alone(self):
        # A guard that warns about every model would be worked around within a
        # notebook and stop meaning anything.
        rng = np.random.default_rng(0)
        y_true = np.linspace(10.0, 20000.0, 100)
        y_pred = y_true + rng.normal(0.0, 500.0, 100)

        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            metrics = evaluate_regression(y_true, y_pred)

        self.assertEqual([str(w.message) for w in raised], [])
        self.assertFalse(is_degenerate_prediction(metrics))


class TestNoEvidenceIsNotACleanBillOfHealth(unittest.TestCase):
    """``is_degenerate_prediction`` is three-valued, and the third state matters.

    Every deep-learning checkpoint currently on disk was written before this
    module recorded the spread fields, so it carries nothing to judge. Answering
    ``False`` there would print a clean verdict for exactly the model the
    scheduling notebook refuses to simulate with.
    """

    def test_a_metrics_dict_with_no_spread_evidence_is_unknown(self):
        self.assertIsNone(is_degenerate_prediction({"mae": 1.0}))

    def test_a_stale_checkpoints_error_metrics_alone_decide_nothing(self):
        # The shape of a pre-guard checkpoint: full error metrics, no spread.
        stale = {"mae": 6717.0, "rmse": 11268.0, "r2": -0.01, "mape": 4200.0,
                 "mdae": 3210.0}
        self.assertIsNone(is_degenerate_prediction(stale))

    def test_a_collapsed_saved_seed_condemns_a_healthy_average(self):
        # A multi-seed refit averages every numeric metric, so pred_unique_frac
        # is a mean over seeds while '_seed0' belongs to the one network written
        # to disk and replayed by the scheduling notebook. Two healthy seeds out
        # of three leave a mean that looks fine; the saved network is the one
        # that has to rank jobs.
        multi_seed = {
            "mae": 6717.0,
            "pred_unique_frac": 0.66,
            "n_predictions": 16437,
            "pred_unique_frac_seed0": 1.0 / 16437,
            "n_predictions_seed0": 16437,
        }
        self.assertTrue(is_degenerate_prediction(multi_seed))

    def test_a_healthy_saved_seed_is_not_condemned_by_the_average(self):
        healthy = {
            "mae": 3820.0,
            "pred_unique_frac": 0.98,
            "n_predictions": 16437,
            "pred_unique_frac_seed0": 0.97,
            "n_predictions_seed0": 16437,
        }
        self.assertFalse(is_degenerate_prediction(healthy))

    def test_a_seed_that_is_neither_the_mean_nor_seed_zero_still_counts(self):
        # The two verdicts above are blind to the middle of a three-seed refit.
        # This is that gap at its real values: seeds 0 and 2 predict normally,
        # seed 1 collapses to one distinct value in 16,437, and the aggregate
        # finalize_dl_model returns therefore shows a mean fraction of 0.67 --
        # ordinary -- beside a seed0 fraction of 1.0. Both look clean, while
        # the reported MAE/RMSE/R2 are means that average a constant predictor
        # in with two working ones and so describe no model that exists.
        collapsed_middle_seed = {
            "mae": 6717.0,
            "pred_unique_frac": 0.666687,
            "n_predictions": 16437,
            "pred_unique_frac_seed0": 1.0,
            "n_predictions_seed0": 16437,
            "pred_unique_frac_per_seed": [1.0, 6.083835249741437e-05, 1.0],
            "n_predictions_per_seed": [16437, 16437, 16437],
        }
        self.assertTrue(is_degenerate_prediction(collapsed_middle_seed))

    def test_a_refit_healthy_on_every_seed_keeps_its_place(self):
        # The same keys on a refit where nothing collapsed. Without this the
        # rule above could be satisfied by condemning every multi-seed entry in
        # the results table, which is the entire deep-learning half of it.
        healthy_every_seed = {
            "mae": 3820.0,
            "pred_unique_frac": 1.0,
            "n_predictions": 16437,
            "pred_unique_frac_seed0": 1.0,
            "n_predictions_seed0": 16437,
            "pred_unique_frac_per_seed": [1.0, 1.0, 1.0],
            "n_predictions_per_seed": [16437, 16437, 16437],
        }
        self.assertFalse(is_degenerate_prediction(healthy_every_seed))

    def test_per_seed_lists_carrying_nothing_to_judge_stay_unknown(self):
        # finalize_dl_model writes None into these lists for a run whose
        # metrics dict predates the spread fields, so an entry that is present
        # but empty of evidence must reach the "unknown" state rather than
        # being counted as a clean seed.
        no_evidence = {
            "mae": 6717.0,
            "pred_unique_frac_per_seed": [None, None],
            "n_predictions_per_seed": [None, None],
        }
        self.assertIsNone(is_degenerate_prediction(no_evidence))


class TestTheExclusionRuleIsTheSameOnBothSides(unittest.TestCase):
    """One rule, or two chapters that contradict each other about one model.

    The metrics-side verdict and SJFPredScheduler's refusal are read by
    consecutive chapters -- notebook 04's "Constant output?" column and
    notebook 05's EXCLUDED POLICIES block. While they held separate thresholds
    they disagreed: Exp A LightGBM (Numeric) is a one-tree refit emitting 15
    distinct values over 16,437 test jobs (fraction 0.0009), which the metrics
    side condemned as EXCLUDED while the simulator scheduled it and reported a
    20.80% JCT improvement for it.
    """

    def _simulator_refuses(self, predictions) -> bool:
        jobs = pd.DataFrame({
            "job_id": range(len(predictions)),
            "submit_time": 0.0,
            "runtime": np.linspace(10.0, 20000.0, len(predictions)),
            "predicted_runtime": predictions,
        })
        try:
            SJFPredScheduler().validate_workload(jobs)
        except DegeneratePredictionError:
            return True
        return False

    def _columns(self):
        n = 16437
        rng = np.random.default_rng(0)
        # (name, predictions) -- the real collapse, the real coarse-but-ranking
        # refit, the two-bucket edge, and a model that ranks normally.
        return [
            ("constant", np.full(n, 4128.124023)),
            ("fifteen buckets", np.repeat(np.linspace(500.0, 9000.0, 15), n // 15 + 1)[:n]),
            ("two buckets", np.where(np.arange(n) % 2 == 0, 10.0, 20.0)),
            ("ranks", np.linspace(10.0, 20000.0, n) + rng.normal(0.0, 500.0, n)),
        ]

    def test_neither_side_excludes_what_the_other_schedules(self):
        y_true = np.linspace(10.0, 20000.0, 16437)
        for name, predictions in self._columns():
            with self.subTest(column=name):
                metrics = evaluate_regression(y_true, predictions)
                self.assertEqual(
                    bool(is_degenerate_prediction(metrics)),
                    self._simulator_refuses(predictions),
                    "the metrics table and the simulator must reach the same "
                    "verdict on one prediction column",
                )

    def test_only_a_constant_column_is_excluded(self):
        y_true = np.linspace(10.0, 20000.0, 16437)
        verdicts = {
            name: is_degenerate_prediction(evaluate_regression(y_true, predictions))
            for name, predictions in self._columns()
        }
        self.assertTrue(verdicts["constant"])
        # 15 buckets order 16,437 jobs; the simulation measures a real
        # improvement from exactly this column, so excluding it would delete a
        # result the next chapter reports.
        self.assertFalse(verdicts["fifteen buckets"])
        self.assertFalse(verdicts["two buckets"])
        self.assertFalse(verdicts["ranks"])

    def test_a_coarse_predictor_is_reported_without_being_excluded(self):
        y_true = np.linspace(10.0, 20000.0, 16437)
        coarse = dict(self._columns())["fifteen buckets"]
        metrics = evaluate_regression(y_true, coarse)
        self.assertTrue(is_near_constant_prediction(metrics))
        self.assertFalse(is_degenerate_prediction(metrics))
        # A constant predictor is the other finding, not this one.
        constant = evaluate_regression(y_true, np.full(16437, 4128.124023))
        self.assertFalse(is_near_constant_prediction(constant))
        # And a model that ranks normally trips neither flag, or the column
        # stops meaning anything.
        healthy = evaluate_regression(y_true, dict(self._columns())["ranks"])
        self.assertFalse(is_near_constant_prediction(healthy))


if __name__ == "__main__":
    unittest.main()
