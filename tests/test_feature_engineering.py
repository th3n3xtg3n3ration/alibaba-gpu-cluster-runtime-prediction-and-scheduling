"""
tests/test_feature_engineering.py

Unit tests for src.feature_engineering.

Tests cover job table construction, temporal feature extraction, 
and validation of required columns.
"""
import unittest
import pandas as pd
from src.feature_engineering import (
    build_job_table_from_sample,
    add_temporal_features,
    prepare_features_for_model,
)

class TestFeatureEngineering(unittest.TestCase):
    def test_with_categorical_is_the_same_train_restricted_matrix_as_native(self):
        """'with_categorical' is the name the notebooks pass; it must carry the
        leakage-6 mitigation, not merely sit next to it.

        The two names used to be two branches. They returned the same columns
        (X_full is already numeric + categorical, so slicing it changed
        nothing), so when the train-only category restriction was added to
        'with_categorical_native' the branches diverged in silence: every
        reported categorical model went through the unrestricted
        'with_categorical' branch, its train dtype carried all 681 users
        (103 of which occur in no training row) and all 5 gpu_types, and the
        test below -- which only ever asked for the other name -- passed.
        Asserting the two are element-wise identical is what makes that
        divergence impossible to reintroduce.
        """
        _, X_train_alias, X_test_alias, _, _, _, _ = prepare_features_for_model(
            dataset="main", time_unit="s", test_size=0.20, random_state=42,
            feature_mode="with_categorical",
        )
        _, X_train_nat, X_test_nat, _, _, _, _ = prepare_features_for_model(
            dataset="main", time_unit="s", test_size=0.20, random_state=42,
            feature_mode="with_categorical_native",
        )

        self.assertTrue(
            X_train_alias.equals(X_train_nat),
            "'with_categorical' must build the same train matrix as "
            "'with_categorical_native', category lists included",
        )
        self.assertTrue(X_test_alias.equals(X_test_nat))

        # Spelled out rather than left to .equals(): the failure mode is
        # specifically a train dtype that still lists test-only categories, so
        # the restriction itself is asserted under the notebooks' own name.
        for col in ("user", "gpu_type"):
            train_categories = set(X_train_alias[col].cat.categories)
            self.assertEqual(train_categories, set(X_train_alias[col].dropna().unique()))
            self.assertTrue(set(X_test_alias[col].dropna().unique()) <= train_categories)
        self.assertGreater(int(X_test_alias["user"].isna().sum()), 0)

    def test_prepare_features_runs_the_sweepline_over_cpu_only_jobs_too(self):
        """The pipeline function -- not just its helpers -- must count CPU-only
        jobs in the cluster-load features.

        test_cpu_only_jobs_are_kept_for_cluster_load_but_dropped_for_modelling
        below calls build_job_table_from_sample / add_cluster_utilization_features
        by hand, so it stays green even when prepare_features_for_model (the
        only entry point notebooks 04 and 05 use) drops include_cpu_only=True
        and goes back to sweeping GPU jobs alone. That regression already cost
        the thesis a headline number once (RF Numeric MAE 6796.98 s), so the
        assertion has to be made against what the pipeline actually returns.
        """
        from src.data_loading import load_sample
        from src.feature_engineering import (
            add_categorical_features,
            add_cluster_utilization_features,
            add_temporal_features as _atf,
        )

        raw = load_sample(which="main")

        def _reference(include_cpu_only):
            df = build_job_table_from_sample(
                raw, time_unit="s", include_cpu_only=include_cpu_only
            )
            df = add_cluster_utilization_features(add_categorical_features(_atf(df)))
            df = df[df["gpu_demand"] > 0].copy().reset_index(drop=True)
            return df.sort_values("arrival_sec", kind="mergesort").reset_index(drop=True)

        swept_over_all_jobs = _reference(True)
        swept_over_gpu_jobs_only = _reference(False)

        load_cols = ["cluster_load_cpu", "active_job_count"]
        # Guards against a vacuous pass: if the two references ever agreed, the
        # comparison below would hold no matter which one the pipeline used.
        for col in load_cols:
            self.assertGreater(
                int((swept_over_all_jobs[col].to_numpy()
                     != swept_over_gpu_jobs_only[col].to_numpy()).sum()),
                0,
                f"{col} must differ between the two sweeps for this test to mean anything",
            )

        job_df, _, _, _, _, _, _ = prepare_features_for_model(feature_mode="numeric_only")

        # CPU-only jobs are counted, never modelled.
        self.assertTrue((job_df["gpu_demand"] > 0).all())
        self.assertEqual(len(job_df), len(swept_over_all_jobs))

        for col in load_cols:
            self.assertTrue(
                (job_df[col].to_numpy() == swept_over_all_jobs[col].to_numpy()).all(),
                f"{col} does not match a sweep-line run over every job in the trace",
            )

    def test_native_categorical_categories_come_from_train_only(self):
        """Regression test for leakage-6: with feature_mode=
        'with_categorical_native', X_train/X_test are sliced from a job_df
        whose categorical dtype was assigned before the chronological
        split, so a plain `.astype("category")` is a no-op that keeps every
        category seen anywhere in the full dataset -- including ones that
        appear only in the test partition and that the model never trained
        on a single row of. Train's category list must be derived from
        train alone, and a test-only category must map to NaN rather than
        silently being carried as a "known" category.
        """
        _, X_train, X_test, _, _, _, cat_cols = prepare_features_for_model(
            dataset="main", time_unit="s", test_size=0.20, random_state=42,
            feature_mode="with_categorical_native",
        )
        self.assertIn("user", cat_cols)
        train_categories = set(X_train["user"].cat.categories)
        test_categories = set(X_test["user"].dropna().unique())

        # Every category test actually uses must have come from train...
        self.assertTrue(test_categories <= train_categories)
        # ...and there must be at least one train-unseen user in the raw
        # test column that this mapping turned into NaN (this trace has
        # test-only users; if this ever becomes 0 the fixture/assumption
        # needs revisiting, not a silent pass).
        self.assertGreater(int(X_test["user"].isna().sum()), 0)
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

    def test_day_of_week_counts_trace_days_rather_than_calendar_weekdays(self):
        """The column's name invites exactly the implementation it must not have.

        ``dt.dayofweek`` was the original committed version and was corrected in
        4e45f74 without a test, so the leaking form is one plausible edit away
        -- including a well-meant "make the code match the name" change. Only a
        fixture spanning more than a week tells the two apart: over 8 days the
        counter keeps climbing while the calendar weekday wraps, and it is that
        wrap the trace cannot afford. The split is chronological, so trace-day 7
        appears only in test and trace-day 0 only in train; folding them onto
        one value means the model is scored on a value it "learned" eight days
        earlier.
        """
        df = pd.DataFrame({"arrival_time": pd.to_datetime([
            "1970-01-01 00:00:00",   # trace day 0 -- a Thursday
            "1970-01-04 12:00:00",   # trace day 3
            "1970-01-08 06:00:00",   # trace day 7 -- Thursday again
        ])})

        result = add_temporal_features(df)

        # dt.dayofweek would give [3, 6, 3] here.
        self.assertEqual(list(result["day_of_week"]), [0, 3, 7])
        self.assertNotEqual(
            result["day_of_week"].iloc[0], result["day_of_week"].iloc[2],
            "trace-day 0 and trace-day 7 must not share a value: with the "
            "chronological split every occurrence of that value in training "
            "comes from the first day and every one in test from the eighth",
        )
        self.assertEqual(list(result["hour_of_day"]), [0, 12, 6])

    def test_day_of_week_is_counted_from_the_first_arrival(self):
        """Zero is the trace's own start, not midnight of whatever date the
        release happens to carry -- the public trace does not disclose its
        collection date, so an absolute origin would be meaningless.
        """
        df = pd.DataFrame({"arrival_time": pd.to_datetime([
            "1970-01-08 06:00:00",   # rows deliberately out of order
            "1970-01-03 00:00:00",
        ])})
        result = add_temporal_features(df)
        self.assertEqual(list(result["day_of_week"]), [5, 0])

    def test_the_real_trace_spans_eight_distinct_day_values(self):
        """The 8-value range is what makes the collision real rather than
        theoretical: the trace runs ~7.7 days, so a calendar weekday would put
        trace-day 7 back on trace-day 0's value. Pinned on the actual data
        because a fixture cannot show that this trace reaches day 7 at all.
        """
        job_df, X_train, X_test, *_ = prepare_features_for_model(
            dataset="main", time_unit="s", test_size=0.20, random_state=42,
            feature_mode="numeric_only",
        )
        self.assertEqual(int(job_df["day_of_week"].min()), 0)
        self.assertEqual(int(job_df["day_of_week"].max()), 7)
        # And the two sides of the split really do straddle the wrap point.
        self.assertIn(0, set(X_train["day_of_week"]))
        self.assertIn(7, set(X_test["day_of_week"]))

    def test_cpu_only_jobs_are_kept_for_cluster_load_but_dropped_for_modelling(self):
        """Regression test for leakage-4 / robustness-11.

        The sweep-line features describe the background load a job arrives
        into, so CPU-only jobs (num_gpu == 0) -- which occupy real CPU
        capacity on the same machines -- must be counted there, even though
        they are not themselves modelled. Previously they were filtered out
        before the sweep-line ran, making cluster_load_cpu / active_job_count
        describe GPU-job traffic rather than cluster load.
        """
        from src.feature_engineering import (
            add_cluster_utilization_features,
            add_temporal_features as _atf,
        )

        raw = pd.DataFrame({
            "job_id": [1, 2, 3],
            "submit_time": [0, 10, 20],
            "duration": [1000, 1000, 1000],
            "num_gpu": [1.0, 0.0, 1.0],   # job 2 is CPU-only
            "num_cpu": [4.0, 8.0, 4.0],
            "user": ["u", "u", "u"],
            "gpu_type": ["T4", "CPU", "T4"],
        })

        gpu_only = build_job_table_from_sample(raw, time_unit="s")
        self.assertEqual(len(gpu_only), 2, "CPU-only job must be dropped by default")

        with_cpu = build_job_table_from_sample(raw, time_unit="s", include_cpu_only=True)
        self.assertEqual(len(with_cpu), 3, "include_cpu_only=True must keep the CPU-only job")

        load_gpu_only = add_cluster_utilization_features(_atf(gpu_only))
        load_with_cpu = add_cluster_utilization_features(_atf(with_cpu))
        load_with_cpu = load_with_cpu[load_with_cpu["gpu_demand"] > 0]

        # The third job arrives while both earlier jobs are still running, so
        # counting the CPU-only one raises both the CPU load and the count.
        last_without = load_gpu_only.iloc[-1]
        last_with = load_with_cpu.iloc[-1]
        self.assertGreater(last_with["cluster_load_cpu"], last_without["cluster_load_cpu"])
        self.assertGreater(last_with["active_job_count"], last_without["active_job_count"])
        # ...but not the GPU load: a CPU-only job requests no GPU.
        self.assertEqual(last_with["cluster_load_gpu"], last_without["cluster_load_gpu"])


class TestTiedArrivalsKeepTheOrderTheTraceGaveThem(unittest.TestCase):
    """Row order out of the feature pipeline is the trace's own, ties included.

    Two sorts decide that order -- the one in add_cluster_utilization_features
    and the one in prepare_features_for_model -- and both have to be stable.
    Neither changes a single feature value (merge_asof keys on arrival_sec
    alone, and every per-job value is identical either way), so the permutation
    is invisible to every accuracy assertion in this suite: with
    kind="mergesort" dropped from add_cluster_utilization_features the whole
    suite stayed green while 11,115 of the 82,184 rows the pipeline returns
    changed place.

    What that order feeds is not invisible. The sequence models window over
    consecutive rows, and notebook 05 uses row position as the simulator's
    arrival tie-break while asserting the order is the trace's own -- so a
    re-permutation moves jobs between windows and changes which of two
    same-second arrivals the scheduler sees first.

    The references below are built WITHOUT calling either function, which is the
    point: the sweep-line tests above build theirs by calling
    add_cluster_utilization_features, so they inherit whatever permutation it
    produced and agree with it whichever sort it used.
    """

    # 24 rows in ties of 3. numpy's introsort only reaches quicksort above 16
    # elements -- below that it uses insertion sort, which is stable -- so a
    # smaller fixture would come back in input order under either sort and
    # would assert nothing at all.
    TIED_ROWS = 24
    JOBS_PER_ARRIVAL = 3

    def _tied_arrivals(self):
        n = self.TIED_ROWS
        return pd.DataFrame({
            "job_id": list(range(n)),
            # Jobs 0-2 all arrive at t=0, jobs 3-5 at t=10, and so on, listed in
            # the order the trace lists them.
            "arrival_sec": [float(i // self.JOBS_PER_ARRIVAL * 10) for i in range(n)],
            "job_runtime": [100.0] * n,
            "gpu_demand": [1.0] * n,
            "num_cpu": [4.0] * n,
        })

    def test_the_sweepline_returns_tied_rows_in_the_order_it_was_given_them(self):
        from src.feature_engineering import add_cluster_utilization_features

        job = self._tied_arrivals()
        expected = job["job_id"].tolist()

        # Guards against a vacuous pass: the default sort has to actually
        # disagree with the input here, or the assertion below is satisfied by
        # an unstable sort too. If a pandas/numpy change ever makes these equal
        # the fixture needs growing -- the assertion is not the part to drop.
        self.assertNotEqual(
            job.sort_values("arrival_sec")["job_id"].tolist(), expected,
            "this fixture no longer tells a stable sort apart from the default "
            "one, so it can no longer guard the sweep-line's row order",
        )

        result = add_cluster_utilization_features(job)
        self.assertEqual(
            result["job_id"].tolist(), expected,
            "jobs sharing an arrival_sec must come back in the order they were "
            "passed in, not in the order the sort implementation put them",
        )

    def test_the_sweepline_keeps_the_real_traces_own_row_order(self):
        """Pinned on the trace as well as on a fixture: 35,478 of its 100,000
        rows share an arrival_sec with another job, which is far past the size
        at which an unstable sort agrees with its input by accident.
        """
        from src.data_loading import load_sample
        from src.feature_engineering import add_cluster_utilization_features

        raw = load_sample(which="main")
        job = build_job_table_from_sample(raw, time_unit="s", include_cpu_only=True)

        # The trace ships sorted, so its own row order IS the stable sort's
        # output -- which is what makes "the order is the trace's own" a
        # statement about this frame rather than about the sort.
        self.assertTrue(job["arrival_sec"].is_monotonic_increasing)
        expected = job["job_id"].tolist()
        unstable = job.sort_values("arrival_sec")["job_id"].tolist()
        self.assertGreater(
            sum(a != b for a, b in zip(expected, unstable)), 0,
            "the trace no longer has ties an unstable sort moves, so this test "
            "would pass under either sort",
        )

        result = add_cluster_utilization_features(add_temporal_features(job))
        self.assertEqual(result["job_id"].tolist(), expected)

    def test_the_pipeline_returns_the_modelled_jobs_in_trace_order(self):
        """The end of the chain, against a reference neither sort touched.

        Only prepare_features_for_model's own sort is covered elsewhere (by the
        sweep-line comparison above, whose reference sorts stably itself); this
        asserts the composition, so reverting either site alone fails here.
        """
        from src.data_loading import load_sample

        raw = load_sample(which="main")
        table = build_job_table_from_sample(raw, time_unit="s", include_cpu_only=True)
        # CPU-only jobs are swept over but not modelled, so the pipeline's rows
        # are the GPU jobs in the order the trace lists them.
        gpu_jobs = table[table["gpu_demand"] > 0]
        expected = gpu_jobs["job_id"].tolist()
        unstable = gpu_jobs.sort_values("arrival_sec")["job_id"].tolist()
        self.assertGreater(
            sum(a != b for a, b in zip(expected, unstable)), 0,
            "the modelled rows no longer carry ties an unstable sort moves",
        )

        job_df, _, _, y_train, y_test, _, _ = prepare_features_for_model(
            feature_mode="numeric_only",
        )

        self.assertEqual(
            job_df["job_id"].tolist(), expected,
            "the pipeline's row order must be the trace's own: the split is "
            "positional, so a permutation moves jobs between train and test as "
            "well as between the sequence models' windows",
        )
        self.assertEqual(len(y_train) + len(y_test), len(expected))


if __name__ == "__main__":
    unittest.main()
