"""
tests/test_regression_guards.py

Regression tests for defects that silently corrupted published results.

Each test here corresponds to a bug that produced plausible-looking but wrong
numbers, survived a full pipeline run, and was caught only by manual audit.
The existing suite passed throughout. These tests exist so that none of them
can return unnoticed.
"""
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd

from src.feature_engineering import build_job_table_from_sample
from src.simulation import (
    ClusterSimulator,
    FIFOScheduler,
    Machine,
    MultiNodeClusterSimulator,
    SJFPredScheduler,
    provision_heterogeneous_gpu_cluster,
)
from src.simulation.multi_node_simulator import _gpu_request
from src.simulation.scheduler_simulator import DegeneratePredictionError


class TestFractionalGpuDemand(unittest.TestCase):
    """
    Alibaba PAI supports GPU sharing, so num_gpu is legitimately fractional.
    Casting it to int truncated every sub-1.0 request to zero, erasing the GPU
    demand of 52.5% of the trace and the feature's predictive signal with it.
    """

    def _table(self):
        return build_job_table_from_sample(
            pd.DataFrame({
                "job_id": [1, 2, 3, 4],
                "submit_time": [1000, 2000, 3000, 4000],
                "duration": [10.0, 20.0, 30.0, 40.0],
                "num_gpu": [0.01, 0.5, 1.0, 2.0],
                "user": ["a", "b", "c", "d"],
                "gpu_type": ["T4", "V100", "T4", "V100"],
            }),
            time_unit="s",
        )

    def test_fractional_requests_survive(self):
        gpu = self._table()["gpu_demand"]
        self.assertEqual(gpu.tolist(), [0.01, 0.5, 1.0, 2.0])

    def test_no_request_is_truncated_to_zero(self):
        gpu = self._table()["gpu_demand"]
        self.assertEqual((gpu == 0).sum(), 0)

    def test_dtype_is_float(self):
        self.assertTrue(
            np.issubdtype(self._table()["gpu_demand"].dtype, np.floating)
        )


class TestSimulatorEnforcesGpuLimit(unittest.TestCase):
    """
    The job tables built by src.feature_engineering name the GPU column
    ``gpu_demand``; the simulator read only ``num_gpu``. The lookup returned
    0.0 for every job, so no job ever consumed a GPU and the cluster behaved as
    if it had infinite capacity. Every scheduling result in the thesis came
    from that regime.
    """

    def _run(self, gpu_column: str):
        jobs = pd.DataFrame({
            "job_id": [1, 2],
            "submit_time": [0.0, 0.0],
            "runtime": [10.0, 10.0],
            "num_cpu": [1.0, 1.0],
            gpu_column: [1.0, 1.0],
        })
        machines = [Machine(machine_id=0, cpu_capacity=64.0, gpu_capacity=1.0)]
        return MultiNodeClusterSimulator(FIFOScheduler(), machines).run(jobs)

    def test_gpu_demand_column_constrains_the_cluster(self):
        # One GPU, two 1-GPU jobs: they must run in sequence, not together.
        res = self._run("gpu_demand").sort_values("job_id")
        self.assertEqual(res["start_time"].tolist(), [0.0, 10.0])

    def test_num_gpu_column_constrains_the_cluster(self):
        res = self._run("num_gpu").sort_values("job_id")
        self.assertEqual(res["start_time"].tolist(), [0.0, 10.0])

    def test_gpu_request_reads_both_column_names(self):
        self.assertEqual(_gpu_request(pd.Series({"gpu_demand": 0.5})), 0.5)
        self.assertEqual(_gpu_request(pd.Series({"num_gpu": 2.0})), 2.0)

    def test_nan_request_cannot_disable_accounting(self):
        # float('nan') compares False against every capacity check, which would
        # make can_fit() accept the job and then poison gpu_used permanently.
        self.assertEqual(_gpu_request(pd.Series({"gpu_demand": float("nan")})), 0.0)

    def test_every_job_appears_exactly_once_in_results(self):
        res = self._run("gpu_demand")
        self.assertEqual(len(res), 2)
        self.assertEqual(res["job_id"].nunique(), 2)


class TestChronologicalSplitGuard(unittest.TestCase):
    """
    Experiments A and B were trained on a randomised split. The reported R2 of
    0.27 could not be reproduced by the committed code, and the thesis
    explicitly rejects randomised k-fold in Section 4.4.1. Nothing in the code
    objected at the time.
    """

    def test_shuffled_split_warns(self):
        from src.feature_engineering import prepare_features_for_model

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prepare_features_for_model(feature_mode="numeric_only", shuffle=True)

        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(
            any("NON-chronological" in m for m in messages),
            "A randomised split must announce itself.",
        )

    def test_default_split_is_strictly_chronological(self):
        from src.feature_engineering import prepare_features_for_model

        job_df, X_train, X_test, _, _, _, _ = prepare_features_for_model(
            feature_mode="numeric_only"
        )
        latest_train = job_df.loc[X_train.index, "arrival_sec"].max()
        earliest_test = job_df.loc[X_test.index, "arrival_sec"].min()
        self.assertGreater(earliest_test, latest_train)


class TestDeepLearningReproducibility(unittest.TestCase):
    """
    torch.manual_seed was called inside train_dl_model, after the model had
    already been constructed, so weight initialisation was governed by leftover
    RNG state and the twelve DL results were not reproducible.

    The defect lives in the ORDER of two statements at three call sites in
    src/tuning.py, not in seed_everything itself, so it has to be checked
    there: the test below calls the two functions itself, in the right order,
    and therefore proves a property of PyTorch rather than of this repository.
    Moving seed_everything() back after create_model_instance() in
    _finalize_dl_single or run_dl_randomsearch left it green.
    TestDeepLearningSeedingAtTheCallSite is the guard that actually watches the
    production ordering; this one is kept only for what it does cover, which is
    that seed_everything seeds torch at all.
    """

    def test_seed_everything_makes_construction_deterministic(self):
        try:
            import torch
            from src.tuning import create_model_instance, seed_everything
        except ModuleNotFoundError:
            self.skipTest("PyTorch not installed")

        params = {"num_filters": 8, "kernel_size": 1, "dropout": 0.2}
        seed_everything()
        first = next(create_model_instance("CNN", 9, params).parameters()).detach().clone()
        seed_everything()
        second = next(create_model_instance("CNN", 9, params).parameters()).detach().clone()
        self.assertTrue(torch.equal(first, second))


class TestDeepLearningSeedingAtTheCallSite(unittest.TestCase):
    """
    The same defect, watched where it actually happened: in the functions that
    fit and save every published DL artifact.

    PyTorch draws initial weights from the global RNG when the module is
    constructed, so the guarantee is entirely about ordering, seed_everything
    must run before create_model_instance, at every call site, on every seed.
    Nothing about the models themselves is asserted here; the spy records
    torch.initial_seed() at the moment of construction, which is the one
    observable that distinguishes "seeded first" from "seeded afterwards".
    """

    #: Deliberately not DL_SEED. The global RNG is left at some value by every
    #: preceding test, and if that value were the seed under test, seeding
    #: after construction would record the right number by accident. The setUp
    #: below pins the leftover state to a value no call site ever requests, so
    #: a misordered call site records THAT instead.
    LEFTOVER_SEED = 999_983

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("PyTorch not installed")
        import torch

        torch.manual_seed(self.LEFTOVER_SEED)

    def _seeds_seen_at_construction(self, call):
        """Run ``call`` and return torch's seed as of each model construction."""
        import torch

        from src import tuning as T

        real = T.create_model_instance
        seen = []

        def spy(*args, **kwargs):
            seen.append(torch.initial_seed())
            return real(*args, **kwargs)

        # Patched on the module, because both call sites resolve the name
        # through the module globals at call time.
        with mock.patch.object(T, "create_model_instance", spy):
            call()
        return seen

    def _tiny_datasets(self):
        from src.tuning import prepare_dl_datasets

        rng = np.random.default_rng(0)
        n = 60
        X = rng.normal(size=(n, 3)).astype(np.float32)
        y = (X[:, 0] * 2 + X[:, 1]).astype(np.float32)
        return prepare_dl_datasets(X[:45], X[45:], y[:45], y[45:], seq_len=1)

    def test_final_refit_seeds_before_it_builds_the_model(self):
        from src.tuning import finalize_dl_model

        (train_ds, val_ds, test_ds, y_test_raw,
         _scaler_x, scaler_y, input_features) = self._tiny_datasets()
        seeds = [1337, 2024]

        seen = self._seeds_seen_at_construction(lambda: finalize_dl_model(
            "LSTM",
            {"hidden_size": 4, "num_layers": 1, "dropout": 0.1,
             "batch_size": 16, "learning_rate": 0.01},
            train_ds, val_ds, input_features, scaler_y, y_test_raw, test_ds,
            final_epochs=1, patience=1, seeds=seeds,
        ))

        # One construction per seed, each under the seed it was asked for. If
        # the seeding moved after construction, the first model would carry
        # LEFTOVER_SEED and every later one the PREVIOUS seed.
        self.assertEqual(seen, seeds)

    def test_random_search_seeds_before_it_builds_each_trial(self):
        from src.tuning import DL_SEED, run_dl_randomsearch

        (train_ds, val_ds, test_ds, y_test_raw,
         _scaler_x, scaler_y, input_features) = self._tiny_datasets()

        seen = self._seeds_seen_at_construction(lambda: run_dl_randomsearch(
            "LSTM",
            {"hidden_size": [4], "num_layers": 1, "dropout": 0.1,
             "batch_size": 16, "learning_rate": 0.01},
            train_ds, val_ds, input_features, scaler_y, y_test_raw, test_ds,
            num_trials=2, tuning_epochs=1, patience=1,
        ))

        # Every trial must start from the same weights, so that trials differ
        # by hyperparameters alone and not by initialisation luck.
        self.assertEqual(seen, [DL_SEED, DL_SEED])

    def test_grid_search_seeds_before_it_builds_each_point(self):
        from src.tuning import DL_SEED, run_dl_gridsearch

        (train_ds, val_ds, test_ds, y_test_raw,
         _scaler_x, scaler_y, input_features) = self._tiny_datasets()

        seen = self._seeds_seen_at_construction(lambda: run_dl_gridsearch(
            "LSTM",
            {"hidden_size": [4, 8], "num_layers": [1], "dropout": [0.1],
             "batch_size": [16], "learning_rate": [0.01]},
            train_ds, val_ds, input_features, scaler_y, y_test_raw, test_ds,
            tuning_epochs=1, patience=1,
        ))

        self.assertEqual(seen, [DL_SEED, DL_SEED])


class TestEasyBackfilling(unittest.TestCase):
    """
    EASY backfilling must fill idle capacity without ever delaying the job it
    reserved for. Both halves matter: a backfiller that never fires is useless,
    and one that pushes back the reservation is not EASY at all.
    """

    def _jobs(self):
        # One 2-GPU machine. Job 2 needs both GPUs, so it is blocked behind
        # job 1. Job 3 is small and short enough to fit in the gap.
        return pd.DataFrame({
            "job_id":      [1, 2, 3],
            "submit_time": [0.0, 1.0, 2.0],
            "runtime":     [100.0, 50.0, 10.0],
            "num_cpu":     [1.0, 1.0, 1.0],
            "gpu_demand":  [1.0, 2.0, 1.0],
        })

    def _run(self, backfill, estimate_col="runtime"):
        machines = [Machine(machine_id=0, cpu_capacity=64.0, gpu_capacity=2.0)]
        sim = MultiNodeClusterSimulator(
            FIFOScheduler(), machines, backfill=backfill, estimate_col=estimate_col
        )
        res = sim.run(self._jobs()).set_index("job_id").sort_index()
        return sim, res

    def test_disabled_backfill_leaves_the_gap_idle(self):
        _, res = self._run(backfill=False)
        self.assertEqual(res.loc[3, "start_time"], 150.0)

    def test_enabled_backfill_uses_the_gap(self):
        sim, res = self._run(backfill=True)
        self.assertEqual(res.loc[3, "start_time"], 2.0)
        self.assertEqual(sim.backfilled_jobs, 1)

    def test_reservation_is_never_delayed(self):
        # Job 2 holds the reservation; backfilling job 3 must not push it back.
        _, without = self._run(backfill=False)
        _, with_bf = self._run(backfill=True)
        self.assertEqual(without.loc[2, "start_time"], with_bf.loc[2, "start_time"])

    def test_job_too_long_for_the_window_is_not_backfilled(self):
        jobs = self._jobs()
        jobs.loc[2, "runtime"] = 500.0   # outlives the reservation window
        machines = [Machine(machine_id=0, cpu_capacity=64.0, gpu_capacity=2.0)]
        sim = MultiNodeClusterSimulator(FIFOScheduler(), machines, backfill=True)
        sim.run(jobs)
        self.assertEqual(sim.backfilled_jobs, 0)

    def test_estimate_column_drives_the_window(self):
        # The window is built from the scheduler's estimate, not ground truth.
        # A job that truly runs long but is *estimated* short gets backfilled,
        # which is precisely how an inaccurate predictor degrades a real EASY
        # scheduler.
        jobs = self._jobs()
        jobs.loc[2, "runtime"] = 500.0
        jobs["predicted_runtime"] = [100.0, 50.0, 10.0]
        machines = [Machine(machine_id=0, cpu_capacity=64.0, gpu_capacity=2.0)]
        sim = MultiNodeClusterSimulator(
            FIFOScheduler(), machines, backfill=True, estimate_col="predicted_runtime"
        )
        sim.run(jobs)
        self.assertEqual(sim.backfilled_jobs, 1)

    def test_backfill_off_reproduces_the_original_schedule(self):
        # Guards against the backfill work perturbing published results.
        _, res = self._run(backfill=False)
        self.assertEqual(res["start_time"].tolist(), [0.0, 100.0, 150.0])


class TestClusterProvisioningDefault(unittest.TestCase):
    """
    GPU-less nodes silently absorbed a quarter of the provisioned cluster while
    never running a single job, because every job in the trace requests at least
    a fraction of a GPU. The default must not reintroduce that dead capacity.
    """

    def test_default_provisions_no_gpu_less_nodes(self):
        machines = provision_heterogeneous_gpu_cluster(n_high=1, n_mid=1)
        self.assertEqual([m for m in machines if m.gpu_capacity == 0], [])

    def test_gpu_less_node_cannot_admit_a_fractional_gpu_job(self):
        # 0.01 GPU is the smallest request in the trace; it must not "fit" on a
        # node with no GPU at all.
        node = Machine(machine_id=0, cpu_capacity=64.0, gpu_capacity=0.0)
        self.assertFalse(node.can_fit(job_cpu=1.0, job_gpu=0.01))


class TestDegeneratePredictorIsRefused(unittest.TestCase):
    """
    A predictor that emits one value for every job turns SJF-Pred into FIFO:
    idxmin on a constant column returns the first row of the ready queue, and
    the queue is held in arrival order. A shipped checkpoint did exactly that
    (4128.124023 for all 16,437 test jobs) and its run matched the FIFO
    baseline in every digit of every metric while still being reported as a
    distinct ML scheduler with its own MAE and R2. Nothing in the pipeline
    noticed, so the check has to be in the run itself.
    """

    def _jobs(self, predictions):
        return pd.DataFrame({
            "job_id": [1, 2, 3],
            "submit_time": [0.0, 0.0, 0.0],
            "runtime": [30.0, 10.0, 20.0],
            "num_cpu": [1.0, 1.0, 1.0],
            "gpu_demand": [1.0, 1.0, 1.0],
            "predicted_runtime": predictions,
        })

    def _machines(self):
        return [Machine(machine_id=0, cpu_capacity=8.0, gpu_capacity=1.0)]

    def test_multinode_run_refuses_a_constant_prediction(self):
        with self.assertRaisesRegex(ValueError, "constant"):
            MultiNodeClusterSimulator(SJFPredScheduler(), self._machines()).run(
                self._jobs([4128.124023] * 3)
            )

    def test_single_node_run_refuses_a_constant_prediction(self):
        # Both simulators replay these policies in notebook 05, so both gates
        # have to be closed; one open door is enough to publish the number.
        with self.assertRaisesRegex(ValueError, "constant"):
            ClusterSimulator(SJFPredScheduler()).run(self._jobs([4128.124023] * 3))

    def test_the_refusal_has_its_own_exception_type(self):
        # The caller replaying 28 policies has to catch THIS refusal and record
        # the policy as excluded, while every other ValueError a run can raise
        # is a wiring or provisioning bug that must still stop the run. A bare
        # ValueError here would make `except ValueError: continue` in notebook
        # 05 swallow those too, and a broken pipeline would be filed as a
        # reportable exclusion.
        with self.assertRaises(DegeneratePredictionError):
            MultiNodeClusterSimulator(SJFPredScheduler(), self._machines()).run(
                self._jobs([4128.124023] * 3)
            )
        self.assertTrue(issubclass(DegeneratePredictionError, ValueError))

    def test_the_refusal_carries_what_a_results_table_needs(self):
        # The refused row has to say which policy, on what column, and what the
        # single value was, that is what makes it a reported result rather
        # than a silent skip.
        with self.assertRaises(DegeneratePredictionError) as raised:
            MultiNodeClusterSimulator(
                SJFPredScheduler("SJF-CNN-LSTM (Numeric Sequence)"), self._machines()
            ).run(self._jobs([4128.124023] * 3))

        refusal = raised.exception
        self.assertEqual(refusal.policy, "SJF-CNN-LSTM (Numeric Sequence)")
        self.assertEqual(refusal.column, "predicted_runtime")
        self.assertAlmostEqual(refusal.value, 4128.124023)
        self.assertEqual(refusal.n_jobs, 3)
        # Every prediction-driven policy shares one class, so a message naming
        # only the class would not say which of them was refused.
        self.assertIn("SJF-CNN-LSTM (Numeric Sequence)", str(refusal))

    def test_the_single_node_refusal_carries_the_same_payload(self):
        with self.assertRaises(DegeneratePredictionError) as raised:
            ClusterSimulator(SJFPredScheduler("SJF-CNN (Numeric)")).run(
                self._jobs([4128.124023] * 3)
            )
        self.assertEqual(raised.exception.policy, "SJF-CNN (Numeric)")

    def test_a_broken_pipeline_is_not_reported_as_a_refused_policy(self):
        # A missing prediction column and an all-non-finite one are wiring
        # bugs. If either raised DegeneratePredictionError, notebook 05's
        # handler would record "predictions collapsed to a constant" for a
        # policy whose predictions never arrived at all, and the comparison
        # would continue past a pipeline that is not producing predictions.
        machines = self._machines()
        missing = self._jobs([1.0, 2.0, 3.0]).drop(columns=["predicted_runtime"])
        for jobs in (missing, self._jobs([float("nan")] * 3)):
            with self.assertRaises(ValueError) as raised:
                MultiNodeClusterSimulator(SJFPredScheduler(), machines).run(jobs)
            self.assertNotIsInstance(raised.exception, DegeneratePredictionError)

    def test_the_refusal_happens_before_any_simulation_state_is_touched(self):
        # It is a refusal, not a failure partway through. The check runs ahead
        # of the state reset at the top of run(), so a refused run leaves the
        # simulator exactly as it was: no half-populated utilization history
        # that a caller could integrate and report.
        sim = MultiNodeClusterSimulator(SJFPredScheduler(), self._machines())
        sim.run(self._jobs([30.0, 10.0, 20.0]))
        history_before = list(sim.utilization_history)
        self.assertTrue(history_before, "the accepted run should have recorded a history")

        with self.assertRaises(DegeneratePredictionError):
            sim.run(self._jobs([4128.124023] * 3))
        self.assertEqual(list(sim.utilization_history), history_before)

    def test_a_prediction_that_actually_ranks_is_accepted(self):
        res = MultiNodeClusterSimulator(SJFPredScheduler(), self._machines()).run(
            self._jobs([30.0, 10.0, 20.0])
        )
        # Shortest predicted first: 2, then 3, then 1.
        self.assertEqual(
            res.sort_values("start_time", kind="mergesort")["job_id"].tolist(), [2, 3, 1]
        )

    def test_missing_prediction_column_is_refused(self):
        jobs = self._jobs([1.0, 2.0, 3.0]).drop(columns=["predicted_runtime"])
        with self.assertRaises(ValueError):
            MultiNodeClusterSimulator(SJFPredScheduler(), self._machines()).run(jobs)

    def test_all_non_finite_predictions_are_refused(self):
        with self.assertRaises(ValueError):
            MultiNodeClusterSimulator(SJFPredScheduler(), self._machines()).run(
                self._jobs([float("nan")] * 3)
            )

    def test_a_single_job_workload_is_not_treated_as_degenerate(self):
        # One job is trivially "constant" and still perfectly schedulable;
        # rejecting it would break every single-job replay for no reason.
        jobs = self._jobs([5.0, 5.0, 5.0]).iloc[:1]
        res = MultiNodeClusterSimulator(SJFPredScheduler(), self._machines()).run(jobs)
        self.assertEqual(len(res), 1)

    def test_fifo_is_unaffected_by_the_check(self):
        # FIFO ranks on arrival order, which is always well defined, so a
        # constant prediction column must not stop a FIFO baseline run.
        res = MultiNodeClusterSimulator(FIFOScheduler(), self._machines()).run(
            self._jobs([4128.124023] * 3)
        )
        self.assertEqual(len(res), 3)


class TestUtilizationHistoryClosesAtMakespan(unittest.TestCase):
    """
    Snapshots are taken at the top of the event loop, before the clock
    advances, so the loop used to exit one event short: the last recorded time
    was the second-to-last event, never the final completion. Consumers
    integrate this history as a left Riemann sum, which then dropped the drain
    interval entirely, and the tail of a run is exactly when the cluster
    empties, so time-weighted utilization came out too high (0.8804 instead of
    0.8765 on a backfilled 800-job run, 2-3% on shorter replays).
    """

    def _run(self):
        rng = np.random.default_rng(3)
        n = 40
        jobs = pd.DataFrame({
            "job_id": range(1, n + 1),
            "submit_time": np.sort(rng.uniform(0, 50, n)).round(2),
            "runtime": rng.uniform(5, 40, n).round(2),
            "num_cpu": rng.integers(1, 4, n).astype(float),
            "gpu_demand": rng.integers(1, 3, n).astype(float),
        })
        machines = [Machine(machine_id=i, cpu_capacity=16.0, gpu_capacity=4.0) for i in range(2)]
        sim = MultiNodeClusterSimulator(FIFOScheduler(), machines)
        return sim, jobs, sim.run(jobs), machines

    def test_history_reaches_the_makespan(self):
        sim, _, res, _ = self._run()
        history = pd.DataFrame(sim.utilization_history)
        self.assertAlmostEqual(
            history["time"].iloc[-1], res["completion_time"].max(),
            msg="the history stops before the cluster finishes draining",
        )

    def test_left_riemann_sum_reproduces_the_analytic_integral(self):
        """The terminating snapshot is what makes the history integrable: with
        it, the left sum equals sum(runtime * gpu) / (makespan * capacity)
        exactly. Without it the last interval is missing and the last value is
        never weighted at all.
        """
        sim, jobs, res, machines = self._run()
        history = pd.DataFrame(sim.utilization_history)
        makespan = res["completion_time"].max()

        times = history["time"].to_numpy()
        util = history["gpu_util"].to_numpy()
        left_sum = float((util[:-1] * np.diff(times)).sum() / makespan)

        gpu_by_job = jobs.set_index("job_id")["gpu_demand"]
        occupancy = (res["completion_time"] - res["start_time"]).to_numpy() * \
            gpu_by_job.loc[res["job_id"]].to_numpy()
        analytic = float(occupancy.sum() / (makespan * sum(m.gpu_capacity for m in machines)))

        self.assertAlmostEqual(left_sum, analytic, places=9)

    def test_the_closing_snapshot_carries_no_duration(self):
        # It exists to bound the last interval, so it must land exactly on the
        # makespan rather than extend the timeline past it.
        sim, _, res, _ = self._run()
        history = pd.DataFrame(sim.utilization_history)
        self.assertLessEqual(history["time"].max(), res["completion_time"].max() + 1e-9)


# Last statement in the file, deliberately. It used to sit in the middle, above
# TestEasyBackfilling, so `python -m tests.test_regression_guards` collected
# only the classes defined before it and reported OK over the backfilling,
# provisioning, degenerate-predictor and utilization-integral guards without
# running any of them. tests/test_suite_integrity.py keeps it here.
if __name__ == "__main__":
    unittest.main()
