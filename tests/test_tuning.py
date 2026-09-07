"""
Unit tests for src.tuning.

Verifies chronological validation splitting and checkpoint metadata enrichment.
"""
import json
import tempfile
import warnings
from contextlib import redirect_stdout
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.tuning import (
    EarlyStopping,
    chronological_train_validation_split,
    make_narrow_grid,
    save_checkpoint,
)


class TestMakeNarrowGrid(unittest.TestCase):
    """Regression tests for code_bugs-3 / modeling-5 (DL narrow-grid construction)."""

    def test_dropout_is_carried_through_not_dropped(self):
        """dropout was previously absent from every DL model's `allowed` set,
        so random search's chosen value never reached the grid or final
        refit -- both silently trained at the framework default (0.2)
        regardless of what random search had actually picked.
        """
        for model_name, best_params in [
            ("cnn", {"num_filters": 128, "kernel_size": 3, "learning_rate": 0.001,
                     "batch_size": 512, "dropout": 0.3}),
            ("lstm", {"hidden_size": 128, "num_layers": 2, "learning_rate": 0.001,
                      "batch_size": 512, "dropout": 0.3}),
            ("hybrid", {"num_filters": 128, "kernel_size": 3, "lstm_hidden_size": 128,
                        "lstm_num_layers": 2, "learning_rate": 0.001, "batch_size": 512,
                        "dropout": 0.3}),
        ]:
            grid = make_narrow_grid(model_name, best_params)
            self.assertIn("dropout", grid, f"{model_name}: dropout missing from narrow grid")
            self.assertEqual(grid["dropout"], [0.3], f"{model_name}: dropout not carried as-is")

    def test_single_layer_recurrent_stack_survives_narrowing(self):
        """A single-layer LSTM/hybrid is a valid, commonly-best
        configuration; filtering the narrow grid's num_layers/
        lstm_num_layers at >=2 silently excluded 1 even when random search
        had selected it.
        """
        lstm_grid = make_narrow_grid("lstm", {
            "hidden_size": 128, "num_layers": 1, "learning_rate": 0.001,
            "batch_size": 512, "dropout": 0.2,
        })
        self.assertIn(1, lstm_grid["num_layers"])

        hybrid_grid = make_narrow_grid("hybrid", {
            "num_filters": 128, "kernel_size": 3, "lstm_hidden_size": 128,
            "lstm_num_layers": 1, "learning_rate": 0.001, "batch_size": 512,
            "dropout": 0.2,
        })
        self.assertIn(1, hybrid_grid["lstm_num_layers"])

    def test_sampling_fraction_below_the_old_floor_survives_narrowing(self):
        """A narrow grid must always contain the value it is centred on.

        The fraction params were clipped to [0.5, 1.0], but the search space in
        configs/models.yaml offers colsample_bytree down to 0.3 for both
        boosters. When random search picked 0.3, every multiplicative variant
        (0.15 ... 0.45) clipped up to 0.5 and the grid degenerated to the single
        value [0.5] -- so the refinement stage could only return 0.5, a point
        that had never been scored against the 0.3 it was supposed to refine,
        and best_params reported it anyway.
        """
        for model_name in ("xgb", "lgbm"):
            # colsample_bytree alone, so the max_grid_size shrink (which
            # legitimately collapses secondary params to their winning value)
            # cannot mask what the clipping did.
            grid = make_narrow_grid(model_name, {"colsample_bytree": 0.3})
            self.assertIn(
                0.3, grid["colsample_bytree"],
                f"{model_name}: the winning fraction was clipped out of its own grid",
            )
            self.assertGreater(
                len(grid["colsample_bytree"]), 1,
                f"{model_name}: clipping collapsed the grid to a single point, so "
                "the refinement round cannot score anything against 0.3",
            )
            self.assertTrue(
                all(0.0 < v <= 1.0 for v in grid["colsample_bytree"]),
                f"{model_name}: fractions must stay inside the library-legal (0, 1]",
            )

    def test_a_fraction_the_search_can_reach_is_never_replaced_by_the_old_floor(self):
        """Even when the grid is shrunk to one point, that point must be the
        value random search actually selected -- not 0.5, which the old floor
        substituted in whenever the winner was below it.
        """
        grid = make_narrow_grid(
            "xgb",
            {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6,
             "colsample_bytree": 0.3, "subsample": 0.4},
        )
        self.assertEqual(grid["colsample_bytree"], [0.3])
        self.assertEqual(grid["subsample"], [0.4])


class TestFinalizeMlModelDataBudget(unittest.TestCase):
    """Regression tests for code_bugs-6: XGB/LGBM final refits must use
    100% of X_train (matching RF's data budget), with the tree count set
    to whatever early stopping found on the internal validation split --
    not the internal split's model returned directly, which would leave
    the last ~10% of chronologically-ordered training rows never
    contributing a single gradient update.
    """

    def _make_data(self, n=300, n_train=240):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": np.arange(n, dtype=float), "b": rng.normal(size=n)})
        y = 2 * X["a"] + 5 * X["b"] + rng.normal(scale=2.0, size=n)
        return (
            X.iloc[:n_train], X.iloc[n_train:],
            y.iloc[:n_train].to_numpy(), y.iloc[n_train:].to_numpy(),
        )

    def _fit_row_counts(self, estimator_class, model_name, best_params):
        """Row count handed to every ``.fit`` call made during finalize_ml_model.

        The two assertions above compare model.n_estimators against
        metrics['n_estimators_effective'], but both sides are the same
        best_n_estimators variable, so they say nothing about which rows the
        final estimator was fit on -- the very thing this class is named for.
        Spying on fit is the only way to observe the data budget directly.
        """
        from src.tuning import finalize_ml_model

        X_train, X_test, y_train, y_test = self._make_data()
        seen_rows = []
        real_fit = estimator_class.fit

        def spy(self, X, y, *args, **kwargs):
            seen_rows.append(len(X))
            return real_fit(self, X, y, *args, **kwargs)

        with patch.object(estimator_class, "fit", spy):
            model, metrics = finalize_ml_model(
                model_name, best_params, X_train, y_train, X_test, y_test, verbose=False,
            )
        return seen_rows, len(X_train), model, metrics

    def test_xgb_final_refit_sees_every_training_row(self):
        """The LAST fit -- the one that produces the returned model -- must get
        100% of X_train. The first fit is the early-stopping search on the
        chronological 90% split and is allowed to be short; the returned model
        must not be.
        """
        import xgboost as xgb

        best_params = {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3}
        seen_rows, n_train, model, metrics = self._fit_row_counts(
            xgb.XGBRegressor, "xgb", best_params,
        )
        self.assertEqual(len(seen_rows), 2, "expected an early-stopping fit then a final refit")
        self.assertLess(seen_rows[0], n_train, "the search fit is the one on the 90% split")
        self.assertEqual(
            seen_rows[-1], n_train,
            "the final refit must see every training row; the chronologically "
            "last 10% is the closest-to-test-period data and RF's bagging gets it",
        )
        # The booster is the artefact that carries the trees, so read the count
        # off it rather than off the constructor argument the metric was copied
        # from -- otherwise both sides of the comparison are the same variable.
        self.assertEqual(
            model.get_booster().num_boosted_rounds(), metrics["n_estimators_effective"]
        )
        self.assertEqual(model.n_estimators, metrics["n_estimators_effective"])
        # The objective is a constructor argument no search grid sets, so it is
        # never in best_params and has to be named at every construction site;
        # the search and the refit silently minimising different losses is how
        # LightGBM's half of this pair broke. Squared error here is deliberate
        # and asymmetric with LightGBM's L1 (see _XGB_OBJECTIVE in src/tuning.py
        # for the measured reason), so it is pinned in both places it has to
        # agree: the refit's actual gradient, and the train_loss every
        # checkpoint and results table carries. Read the first off the RETURNED
        # estimator, not off _XGB_OBJECTIVE, or both sides of the check are the
        # same variable.
        self.assertEqual(model.get_params()["objective"], "reg:squarederror")
        self.assertEqual(metrics["train_loss"], "reg:squarederror")
        # The search budget belongs in the checkpoint beside the effective
        # count: best_params['n_estimators'] is the only tree count that ever
        # reached the thesis table, and a table printing it alone describes a
        # search rather than a model.
        self.assertEqual(metrics["n_estimators_searched"], best_params["n_estimators"])

    def test_lgbm_final_refit_sees_every_training_row(self):
        """Same data-budget guarantee for LightGBM (see the XGB test above)."""
        import lightgbm as lgb

        best_params = {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 15}
        seen_rows, n_train, model, metrics = self._fit_row_counts(
            lgb.LGBMRegressor, "lgbm", best_params,
        )
        self.assertEqual(len(seen_rows), 2, "expected an early-stopping fit then a final refit")
        self.assertLess(seen_rows[0], n_train)
        self.assertEqual(
            seen_rows[-1], n_train,
            "the final refit must see every training row, matching RF's budget",
        )
        self.assertEqual(model.booster_.num_trees(), metrics["n_estimators_effective"])
        self.assertEqual(model.n_estimators, metrics["n_estimators_effective"])
        # The other half of the pair the XGB test above asserts: without the
        # objective set here the refit falls back to LightGBM's default L2, and
        # the asymmetry is only visible when both sides are pinned to the
        # returned estimator.
        self.assertEqual(model.get_params()["objective"], "regression_l1")
        self.assertEqual(metrics["train_loss"], "regression_l1")
        self.assertEqual(metrics["n_estimators_searched"], best_params["n_estimators"])

    def test_rf_is_fit_once_on_the_whole_training_set(self):
        """RF is the reference budget the other two families are compared
        against, so it must stay a single fit over 100% of X_train -- if it
        ever grew an internal holdout the comparison above would go quiet
        while both sides moved together.
        """
        from sklearn.ensemble import RandomForestRegressor

        seen_rows, n_train, _, _ = self._fit_row_counts(
            RandomForestRegressor, "rf", {"n_estimators": 20},
        )
        self.assertEqual(seen_rows, [n_train])

    def test_rf_has_no_effective_count(self):
        """RF has no early-stopping concept and always sees 100% of
        X_train, so it should not report n_estimators_effective at all.
        """
        from src.tuning import finalize_ml_model
        X_train, X_test, y_train, y_test = self._make_data()
        _, metrics = finalize_ml_model(
            "rf", {"n_estimators": 50}, X_train, y_train, X_test, y_test, verbose=False,
        )
        self.assertNotIn("n_estimators_effective", metrics)
        self.assertNotIn("n_estimators_searched", metrics)

    def test_rf_reports_the_loss_it_is_actually_ranked_against(self):
        """The third family's train_loss, and the reason the column exists.

        Both boosters minimise MAE; RandomForest minimises squared error and is
        then bolded in the same MAE table. A row without this column cannot tell
        a reader which comparison is controlled for loss -- and because the
        criterion is itself searchable in configs/models.yaml, the value has to
        come from best_params rather than from the family name.
        """
        from src.tuning import finalize_ml_model
        X_train, X_test, y_train, y_test = self._make_data()

        _, defaulted = finalize_ml_model(
            "rf", {"n_estimators": 20}, X_train, y_train, X_test, y_test, verbose=False,
        )
        self.assertEqual(defaulted["train_loss"], "squared_error")

        _, searched = finalize_ml_model(
            "rf", {"n_estimators": 20, "criterion": "absolute_error"},
            X_train, y_train, X_test, y_test, verbose=False,
        )
        self.assertEqual(searched["train_loss"], "absolute_error")


class TestEnsembleCollapseWarning(unittest.TestCase):
    """Regression tests for the early-stopping collapse warning.

    Exp A LightGBM reached the thesis's best Experiment A MAE with a refit of
    exactly ONE tree, while the hyperparameter table printed
    best_params['n_estimators'] = 1300. Nothing else in the pipeline can see
    that: is_degenerate_prediction (called by evaluate_regression) judges
    prediction-value collapse, and one L1 tree with dozens of leaves still emits
    dozens of distinct values, so it passes. This threshold is the only thing
    separating "the tuned ensemble its best_params describe" from "a statement
    about the internal 10% chronological holdout", so it is exercised at the
    boundary rather than at a comfortable distance from it.
    """

    def _warns(self, effective, searched):
        from src.tuning import _warn_if_ensemble_collapsed

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_ensemble_collapsed("lgbm", effective, searched)
        return [w for w in caught if issubclass(w.category, UserWarning)]

    # (n_estimators_effective, best_params['n_estimators']) exactly as the six
    # boosted checkpoints in results/checkpoints/ hold them. Both lists matter:
    # a warning that fires on the four ordinary refits is noise the reader
    # learns to skip, which is how the one-tree model went unremarked.
    REAL_COLLAPSED = [(1, 1300), (7, 600)]
    REAL_HEALTHY = [(19, 600), (54, 1300), (63, 1300), (140, 400)]

    def test_the_real_collapsed_refits_warn(self):
        for effective, searched in self.REAL_COLLAPSED:
            with self.subTest(effective=effective, searched=searched):
                caught = self._warns(effective, searched)
                self.assertTrue(
                    caught, f"{effective} tree(s) of {searched} must be reported",
                )
                # Both numbers have to reach the message, and in their own
                # phrases: the whole point is that the table prints only the
                # second one. Matched in context because a bare "1" is a
                # substring of "1300" and would assert nothing.
                message = str(caught[0].message)
                self.assertIn(f"early-stopped at {effective} tree(s)", message)
                self.assertIn(f"search budget of {searched}", message)

    def test_the_real_healthy_refits_stay_quiet(self):
        for effective, searched in self.REAL_HEALTHY:
            with self.subTest(effective=effective, searched=searched):
                self.assertFalse(
                    self._warns(effective, searched),
                    f"{effective} tree(s) of {searched} is an ordinary early "
                    "stop, not a collapse",
                )

    def test_the_boundary_sits_exactly_at_the_absolute_floor(self):
        """With a 300-tree budget the 1%-of-budget rule falls below the absolute
        floor of 10, so 10 is the last count that warns and 11 the first that
        does not. Asserted from both sides: a ``>=`` where the code has ``>``
        moves the line by one, and none of the six real pairs above lands close
        enough to notice.
        """
        self.assertTrue(self._warns(10, 300))
        self.assertFalse(self._warns(11, 300))

    def test_the_budget_proportional_floor_rises_above_the_absolute_one(self):
        """A 1300-tree search makes 13 the floor, so a count that would be fine
        against a small budget is a collapse against a large one -- 13 warns
        where 19 (of 600) did not.
        """
        self.assertTrue(self._warns(13, 1300))
        self.assertFalse(self._warns(14, 1300))

    def test_an_unset_search_budget_still_gets_the_absolute_floor(self):
        """The six checkpoints above were all written before
        n_estimators_searched existed, and _searched_n_estimators returns None
        for any best_params that never set n_estimators. A missing budget must
        weaken the check to the absolute floor, not switch it off.
        """
        caught = self._warns(1, None)
        self.assertTrue(caught)
        self.assertIn("unset", str(caught[0].message))
        self.assertFalse(self._warns(11, None))

    def test_the_finalizer_actually_consults_it_with_both_counts(self):
        """Everything above tests a function nothing has shown is still wired
        in. The pair finalize_ml_model hands it is the whole point: the count
        early stopping arrived at, judged against the search budget the
        hyperparameter table prints in its place.
        """
        import src.tuning as tuning_module

        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": np.arange(300, dtype=float), "b": rng.normal(size=300)})
        y = 2 * X["a"] + rng.normal(scale=2.0, size=300)
        seen = []

        with patch.object(tuning_module, "_warn_if_ensemble_collapsed",
                          lambda *args: seen.append(args)):
            _, metrics = tuning_module.finalize_ml_model(
                "xgb", {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
                X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:], verbose=False,
            )

        self.assertEqual(
            seen,
            [("xgb", metrics["n_estimators_effective"], metrics["n_estimators_searched"])],
        )


class TestGridConvergence(unittest.TestCase):
    """Regression tests for modeling-11: a narrow grid can only return values
    it offered, so a winner sitting on the grid's edge means the search never
    converged -- that has to be detected, widened once, and reported."""

    def test_boundary_detection_flags_edges_and_ignores_interior(self):
        from src.tuning import grid_boundary_params

        grid = {"n_estimators": [200, 300, 400], "learning_rate": [0.01, 0.05, 0.1]}
        # n_estimators at the low edge, learning_rate interior
        self.assertEqual(
            grid_boundary_params(grid, {"n_estimators": 200, "learning_rate": 0.05}),
            ["n_estimators"],
        )
        # both interior -> converged
        self.assertEqual(
            grid_boundary_params(grid, {"n_estimators": 300, "learning_rate": 0.05}),
            [],
        )
        # both at edges (one low, one high)
        self.assertEqual(
            sorted(grid_boundary_params(grid, {"n_estimators": 400, "learning_rate": 0.01})),
            ["learning_rate", "n_estimators"],
        )

    def test_single_value_and_non_comparable_params_are_not_boundaries(self):
        from src.tuning import grid_boundary_params

        grid = {"bootstrap": [True], "max_features": ["sqrt", "log2"], "max_depth": [5, 10]}
        found = grid_boundary_params(
            grid, {"bootstrap": True, "max_features": "sqrt", "max_depth": 5}
        )
        # 'bootstrap' has one value; 'max_features' values are strings whose
        # min/max ordering is meaningless for convergence. Only max_depth counts.
        self.assertEqual(found, ["max_depth"])

    def test_iterative_search_widens_when_winner_is_at_the_edge(self):
        from src.tuning import run_gridsearch_iterative

        calls = []

        def fake_search(X, y, param_grid=None, **kwargs):
            calls.append(param_grid)
            # Always return the lowest n_estimators offered -> always at the edge,
            # so the loop must stop at max_rounds rather than spin forever.
            best = {"n_estimators": min(param_grid["n_estimators"])}
            return object(), best, -1.0

        _, best, _, diag = run_gridsearch_iterative(
            "rf", fake_search, None, None, {"n_estimators": 300}, max_rounds=2, verbose=False
        )
        self.assertEqual(len(calls), 2, "a boundary winner must trigger exactly one widening")
        self.assertEqual(diag["grid_rounds"], 2)
        self.assertIn("n_estimators", diag["params_at_boundary"])

    def test_iterative_search_stops_immediately_when_winner_is_interior(self):
        from src.tuning import run_gridsearch_iterative

        calls = []

        def fake_search(X, y, param_grid=None, **kwargs):
            calls.append(param_grid)
            vals = sorted(param_grid["n_estimators"])
            return object(), {"n_estimators": vals[len(vals) // 2]}, -1.0

        _, _, _, diag = run_gridsearch_iterative(
            "rf", fake_search, None, None, {"n_estimators": 300}, max_rounds=3, verbose=False
        )
        self.assertEqual(len(calls), 1, "an interior winner needs no second round")
        self.assertEqual(diag["grid_rounds"], 1)
        self.assertEqual(diag["params_at_boundary"], [])


class TestFinalizeDlModelSaveAllSeeds(unittest.TestCase):
    """Regression test for robustness-4: finalize_dl_model previously kept
    only the first seed's model on disk, so a downstream simulation could
    never replay anything but seed0 even when multiple seeds were trained
    and averaged. save_all_seeds_to must write every seed's model, not
    just the returned one.
    """

    # Seed VALUES that are not also seed indices. Notebook 04 trains
    # DL_SEEDS = [42, 1337, 2024] and formats the path with the seed value, so
    # a template filled with an index would silently write different files than
    # the ones notebook 05 loads; seeds [0, 1, 2] hide that because value and
    # index coincide.
    SEEDS = [42, 1337, 2024]

    def _train_three_seeds(self, tmp):
        from src.tuning import finalize_dl_model, prepare_dl_datasets

        rng = np.random.default_rng(0)
        n = 80
        X = rng.normal(size=(n, 3)).astype(np.float32)
        y = (X[:, 0] * 2 + X[:, 1]).astype(np.float32)
        X_train, X_test = X[:60], X[60:]
        y_train, y_test = y[:60], y[60:]

        (train_dataset, val_dataset, test_dataset, y_test_raw,
         _scaler_x, scaler_y, input_features) = prepare_dl_datasets(
            X_train, X_test, y_train, y_test, seq_len=1,
        )

        path_template = str(Path(tmp) / "lstm_seed{seed}.pth")
        model, metrics = finalize_dl_model(
            "LSTM",
            {"hidden_size": 4, "num_layers": 1, "dropout": 0.1,
             "batch_size": 16, "learning_rate": 0.01},
            train_dataset, val_dataset, input_features,
            scaler_y, y_test_raw, test_dataset,
            final_epochs=1, patience=1,
            seeds=self.SEEDS, save_all_seeds_to=path_template,
        )
        return model, metrics, path_template

    def test_every_seed_model_is_saved_to_its_own_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, metrics, path_template = self._train_three_seeds(tmp)
            for seed in self.SEEDS:
                self.assertTrue(
                    Path(path_template.format(seed=seed)).exists(),
                    f"seed {seed}'s model was not saved to disk",
                )
            self.assertEqual(metrics["n_seeds"], 3)
            self.assertEqual(metrics["seeds"], self.SEEDS)

    def test_the_aggregate_carries_every_seeds_spread_evidence(self):
        """A mean over three seeds cannot show that one of them collapsed.

        ``is_degenerate_prediction`` judges the mean, the ``*_seed0`` values and
        each entry of these two lists. Without them a middle seed predicting a
        single constant leaves no trace any reader could judge -- two healthy
        seeds out of three keep the mean fraction ordinary, seed0 is one of the
        healthy ones, and the collapsed run's error metrics are averaged into
        the headline MAE the results table prints. Positionally aligned with
        ``seeds``, because the exclusion notice names the seed.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _, metrics, _ = self._train_three_seeds(tmp)
            per_seed = metrics["pred_unique_frac_per_seed"]
            counts = metrics["n_predictions_per_seed"]
            self.assertEqual(len(per_seed), len(self.SEEDS))
            self.assertEqual(len(counts), len(self.SEEDS))
            # Position 0 belongs to the network actually written to disk, so
            # the list is anchored to seeds rather than to some other order.
            self.assertEqual(per_seed[0], metrics["pred_unique_frac_seed0"])
            self.assertEqual(counts[0], metrics["n_predictions_seed0"])
            for seed, frac, n in zip(self.SEEDS, per_seed, counts):
                self.assertIsNotNone(frac, f"seed {seed} contributed no spread evidence")
                self.assertGreater(n, 0, f"seed {seed} recorded no predictions")

    def test_each_seed_path_holds_a_different_network(self):
        """Three files existing is not three seeds.

        Notebook 05 loads these paths to report the seed spread of the
        scheduling results. If the loop ever writes the first seed's network to
        every path -- one wrong variable -- the files are all present, n_seeds
        is still 3, the reported spread collapses to zero, and the existence
        check above stays green while the robustness claim it backs is
        fabricated. The artefacts themselves have to be shown to differ.
        """
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            model, _, path_template = self._train_three_seeds(tmp)
            # weights_only=False: finalize_dl_model saves whole nn.Module
            # objects, not state dicts.
            weights = [
                next(torch.load(path_template.format(seed=s), weights_only=False).parameters())
                .detach()
                for s in self.SEEDS
            ]
            for i in range(len(weights)):
                for j in range(i + 1, len(weights)):
                    self.assertFalse(
                        torch.equal(weights[i], weights[j]),
                        f"seeds {self.SEEDS[i]} and {self.SEEDS[j]} were saved as "
                        "the same network, so the reported seed spread is not real",
                    )
            # ...and the returned model -- the one whose score is recorded under
            # the *_seed0 keys -- must be the first seed's, not some other one.
            self.assertTrue(torch.equal(weights[0], next(model.parameters()).detach()))


class TestEarlyStopping(unittest.TestCase):
    """Regression tests for code_bugs-2: delta must be a relative threshold."""

    def test_small_improvement_at_small_loss_scale_is_not_penalized(self):
        """A real improvement that is smaller than the fixed 1e-4 the old
        absolute-delta implementation used must still count as progress once
        the loss itself is already at a comparably small scale -- exactly the
        regime DL final refits operate in on a MinMax-scaled target.
        """
        es = EarlyStopping(patience=3, delta=1e-4)
        es(val_loss=0.0005, model=None)
        self.assertEqual(es.counter, 0)

        # Absolute improvement is only 5e-5 (< the old fixed delta of 1e-4),
        # but it is a genuine 10% relative gain over the current best.
        es(val_loss=0.00045, model=None)

        self.assertEqual(
            es.counter, 0,
            "a 10% relative improvement must reset the patience counter, "
            "not be swallowed by a fixed absolute threshold",
        )
        self.assertAlmostEqual(es.best_score, -0.00045)

    def test_stops_after_patience_when_truly_flat(self):
        """A loss that stops improving at all should still trigger early
        stopping after `patience` non-improving epochs, regardless of scale.
        """
        es = EarlyStopping(patience=2, delta=1e-4)
        es(val_loss=0.001, model=None)
        es(val_loss=0.001, model=None)
        self.assertFalse(es.early_stop)
        es(val_loss=0.001, model=None)
        self.assertTrue(es.early_stop)


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

            with patch("src.tuning._CHECKPOINT_DIR", checkpoint_dir), redirect_stdout(io.StringIO()):
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
