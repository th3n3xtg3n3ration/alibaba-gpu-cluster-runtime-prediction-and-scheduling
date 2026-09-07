"""Regression tests for the checkpoint currency guard (audit blocker).

Also covers the model-artifact half of the same guard
(``model_artifact_is_current`` / ``record_model_artifact``) and the provenance
snapshot both halves are judged on. That machinery was committed with a message
claiming "Tests cover both: sidecar absent/present/stale, and provenance carried
forward on an unchanged re-save" while the commit touched no test file at all,
it is the only thing standing between the reported metrics and a model artifact
fit by different code, and until now nothing would have noticed its removal.

These are ``unittest.TestCase`` classes rather than pytest module functions on
purpose. The only test command this repository documents and runs is
``python -m unittest discover tests`` (README.md, scripts/run_all_experiments.sh),
which collects TestCase subclasses and nothing else: written as pytest
functions, every test in this file was invisible to the project's own QA gate,
so both currency guards could be deleted from src/tuning.py and the documented
gate still printed "All unit tests passed successfully!". tests/
test_suite_integrity.py keeps the two collections from drifting apart again.
"""
import ast
import json
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import tuning as T

_REPO_ROOT = Path(__file__).resolve().parents[1]


class _ProvenanceCase(unittest.TestCase):
    """Shared fixture: a throwaway checkpoint directory and a way to fake the
    environment a result was produced under.

    Every test here writes checkpoints and sidecars, so ``_CHECKPOINT_DIR`` is
    redirected into a temporary directory first, a stray write into
    results/checkpoints/ would corrupt the very provenance these tests exist to
    protect.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)
        # Captured before any patching, so _source_matches_again restores the
        # real snapshot rather than whatever the last pretence installed.
        self._real_provenance = T._compute_provenance
        self.ckpt_dir = self.tmp_path / "checkpoints"
        self.ckpt_dir.mkdir()
        self._patch("_CHECKPOINT_DIR", self.ckpt_dir)

        # load_checkpoint records its refusals in a module-level set that
        # save_checkpoint then consumes. It is process-wide state, so a refusal
        # left behind by one test would license another test's save to restamp
        # itself and make the carry-forward assertions pass for the wrong
        # reason.
        previous = set(T._RECOMPUTE_REQUESTED)
        T._RECOMPUTE_REQUESTED.clear()
        self.addCleanup(lambda: (T._RECOMPUTE_REQUESTED.clear(),
                                 T._RECOMPUTE_REQUESTED.update(previous)))

        # The two globals _note_model_fit writes are process-wide in the same
        # way, and record_model_artifact falls back to them whenever the caller
        # passes no model object. Any real fit earlier in the process, the
        # finalize_* tests below, or another test module in the same discovery
        # run, leaves a snapshot of its own tree behind, and a sidecar written
        # afterwards would be stamped from that leftover instead of from the
        # fit under test. Every currency assertion here would then hold for the
        # wrong reason: with _note_model_fit removed from finalize_ml_model
        # entirely, the leftover still carries the pre-edit tree and
        # test_resaving_a_model_fitted_before_a_source_edit_does_not_launder_it
        # below stays green. Patched rather than merely cleared so
        # _note_model_fit's rebinding during a test is undone too.
        self._patch("_PROVENANCE_AT_LAST_FIT", None)
        self._patch("_FIT_PROVENANCE", weakref.WeakKeyDictionary())

    def _patch(self, name, value):
        patcher = mock.patch.object(T, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _pretend_source_changed(self, name="src/feature_engineering.py",
                                digest="CHANGED"):
        """Make the current environment report a different source tree.

        ``digest`` distinguishes one edit from the next, for the sequences that
        need the source to change, come back, and then change again.
        """
        real = self._real_provenance
        self._patch(
            "_compute_provenance",
            lambda: {**real(), "src_sha256": {name: digest}},
        )

    def _source_matches_again(self):
        """Undo the pretence: the edit was reverted, or the branch switched
        back. Restores the real snapshot rather than the previously patched
        one, so this is the inverse of any number of stacked changes.
        """
        self._patch("_compute_provenance", self._real_provenance)

    def _read(self, experiment_name):
        return json.loads((self.ckpt_dir / f"{experiment_name}.json").read_text())


class TestCheckpointCurrency(_ProvenanceCase):

    def test_missing_checkpoint_is_not_current(self):
        self.assertIs(T.checkpoint_is_current("exp_a_rf"), False)

    def test_freshly_saved_checkpoint_is_current(self):
        T.save_checkpoint("exp_a_rf", {"metrics": {"mae": 1.0}, "best_params": {"a": 1}})
        self.assertIs(T.checkpoint_is_current("exp_a_rf"), True)
        self.assertIsNotNone(T.load_checkpoint("exp_a_rf"))

    def test_source_change_makes_checkpoint_stale(self):
        T.save_checkpoint("exp_a_rf", {"metrics": {"mae": 1.0}, "best_params": {"a": 1}})
        self._pretend_source_changed()
        self.assertIs(T.checkpoint_is_current("exp_a_rf"), False)
        # This is the defect the guard exists for: the caller must not receive a
        # result computed by different source code.
        self.assertIsNone(T.load_checkpoint("exp_a_rf"))
        # ...but it stays readable for inspection.
        self.assertEqual(
            T.load_checkpoint("exp_a_rf", allow_stale=True)["metrics"]["mae"], 1.0
        )

    def test_stale_checkpoints_are_excluded_from_load_all(self):
        T.save_checkpoint("exp_a_rf", {"metrics": {"mae": 1.0}, "best_params": {}})
        self._pretend_source_changed("src/tuning.py")
        T.save_checkpoint("exp_a_xgb", {"metrics": {"mae": 2.0}, "best_params": {}})
        loaded = T.load_all_checkpoints()
        self.assertEqual(
            set(loaded), {"exp_a_xgb"},
            "only the checkpoint matching current source may be returned",
        )

    def test_unchanged_resave_does_not_refresh_stale_provenance(self):
        """A re-save of an unrecomputed result must not relabel it as current.

        This is what previously hid the staleness: the cell loaded the old
        metrics and wrote them straight back, restamping the provenance each
        run.
        """
        T.save_checkpoint("exp_a_rf", {"metrics": {"mae": 1.0}, "best_params": {"a": 1}})
        original = self._read("exp_a_rf")["provenance"]
        self._pretend_source_changed()
        T.save_checkpoint("exp_a_rf", {"metrics": {"mae": 1.0}, "best_params": {"a": 1}})
        self.assertEqual(self._read("exp_a_rf")["provenance"], original)
        self.assertIs(T.checkpoint_is_current("exp_a_rf"), False)

    def test_a_relabelled_best_params_is_still_the_same_computation(self):
        """Only the metrics decide whether a re-save recorded a new computation.

        The non-learned baselines put a prose description in best_params
        ('estimator': 'single global training median, no grouping'), and the
        Turkish notebook writes it translated. best_params used to take part in
        the same-result test, so merely opening the TR notebook made that
        language switch look like a fresh run: exp_b_constant_median,
        exp_b_constant_zero and exp_b_profile_median were restamped with the
        current commit, source hashes and timestamp although nothing was
        recomputed. A human-readable label is not evidence about what produced
        a number.
        """
        T.save_checkpoint(
            "exp_b_constant_median",
            {"metrics": {"mae": 1.0, "train_time": 0.25},
             "best_params": {"estimator": "single global training median, no grouping"}},
        )
        before = self._read("exp_b_constant_median")

        self._pretend_source_changed("src/tuning.py")
        T.save_checkpoint(
            "exp_b_constant_median",
            {"metrics": {"mae": 1.0, "train_time": 0.25},
             "best_params": {"estimator": "gruplamasız tek global eğitim medyanı"}},
        )

        after = self._read("exp_b_constant_median")
        self.assertEqual(after["provenance"], before["provenance"])
        self.assertEqual(after["timestamp"], before["timestamp"])
        self.assertIs(T.checkpoint_is_current("exp_b_constant_median"), False)

    def test_a_recomputed_result_is_stamped_afresh(self):
        """The other half of the carry-forward branch: different metrics mean
        the result was recomputed, so it must take the current provenance and a
        new timestamp. Carrying the old one forward here would make a freshly
        trained model look stale and send load_checkpoint down the retrain path
        forever.
        """
        T.save_checkpoint("exp_a_rf", {"metrics": {"mae": 1.0}, "best_params": {"a": 1}})
        before = self._read("exp_a_rf")

        self._pretend_source_changed("src/tuning.py")
        T.save_checkpoint("exp_a_rf", {"metrics": {"mae": 0.9}, "best_params": {"a": 1}})

        after = self._read("exp_a_rf")
        self.assertNotEqual(after["provenance"], before["provenance"])
        self.assertNotEqual(after["timestamp"], before["timestamp"])
        self.assertEqual(after["provenance"]["src_sha256"], {"src/tuning.py": "CHANGED"})
        self.assertIs(T.checkpoint_is_current("exp_a_rf"), True)


class TestRecomputationLicence(_ProvenanceCase):
    """Identical metrics are not evidence that nothing was recomputed.

    Eight non-learned baselines hard-code ``train_time = 0.0`` and three more
    store no train_time at all, so a genuine refit of a deterministic baseline
    reproduces its metrics block bit for bit. Reading that as "loaded, not
    computed" froze the OLD provenance onto a freshly computed result, which
    checkpoint_is_current can then never certify and load_all_checkpoints drops
    for good, the Experiment B and summary tables stop being rebuildable from
    disk. Which of the two happened is known only to the caller, or to
    load_checkpoint's own refusal, and this is the machinery that carries that
    knowledge from one to the other.
    """

    def _save_then_change_source(self):
        T.save_checkpoint("exp_b_constant_median",
                          {"metrics": {"mae": 1.0, "train_time": 0.0}, "best_params": {}})
        before = self._read("exp_b_constant_median")
        self._pretend_source_changed("src/tuning.py")
        return before

    def _resave_same_metrics(self, **kwargs):
        T.save_checkpoint("exp_b_constant_median",
                          {"metrics": {"mae": 1.0, "train_time": 0.0}, "best_params": {}},
                          **kwargs)
        return self._read("exp_b_constant_median")

    def test_a_refused_load_licenses_the_next_save_to_restamp(self):
        before = self._save_then_change_source()
        # The refusal IS the message "recompute this": the caller was sent down
        # its retrain path, so whatever it writes back was computed just now.
        self.assertIsNone(T.load_checkpoint("exp_b_constant_median"))

        after = self._resave_same_metrics()
        self.assertNotEqual(after["provenance"], before["provenance"])
        self.assertNotEqual(after["timestamp"], before["timestamp"])
        self.assertIs(T.checkpoint_is_current("exp_b_constant_median"), True)

    def test_the_licence_is_consumed_by_the_save_that_answers_it(self):
        # A save cell re-run later in the same kernel, old numbers still in
        # memory, source edited since, must fall back to the conservative
        # carry-forward instead of relabelling them as freshly produced.
        self._save_then_change_source()
        T.load_checkpoint("exp_b_constant_median")
        first = self._resave_same_metrics()

        second = self._resave_same_metrics()
        self.assertEqual(second["provenance"], first["provenance"])
        self.assertEqual(second["timestamp"], first["timestamp"])

    def test_a_successful_load_withdraws_an_earlier_refusal(self):
        """The refusal has to be a STALE one, on a file that exists.

        Written the obvious way, refuse because the file is missing, save,
        then load, the save in the middle consumes the licence before the
        successful load ever runs, so load_checkpoint's own withdrawal fires
        against an already-empty set and the assertions below hold whether that
        line is there or not. The licence must still be outstanding when the
        load succeeds, which means nothing may write in between.
        """
        T.save_checkpoint("exp_a_rf", {"metrics": {"mae": 1.0}, "best_params": {}})
        first = self._read("exp_a_rf")

        # Refused for staleness: the caller was sent down its retrain path and
        # the licence to restamp is now outstanding.
        self._pretend_source_changed("src/tuning.py")
        self.assertIsNone(T.load_checkpoint("exp_a_rf"))

        # The source comes back (a reverted edit, a branch switched back), so
        # this load succeeds and the caller holds numbers that DID come from
        # disk. The outstanding licence no longer describes what it will write.
        self._source_matches_again()
        self.assertIsNotNone(T.load_checkpoint("exp_a_rf"))

        # Source changes again and the caller writes those loaded numbers back.
        # With the licence still standing they would be restamped with the
        # current source hash, a stale result relabelled as freshly produced,
        # which also suppresses the mismatch warning from the next run onwards.
        self._pretend_source_changed("src/tuning.py", digest="CHANGED-AGAIN")
        T.save_checkpoint("exp_a_rf", {"metrics": {"mae": 1.0}, "best_params": {}})

        after = self._read("exp_a_rf")
        self.assertEqual(after["provenance"], first["provenance"])
        self.assertEqual(after["timestamp"], first["timestamp"])
        self.assertIs(T.checkpoint_is_current("exp_a_rf"), False)

    def test_an_explicit_flag_covers_a_write_the_inference_cannot_see(self):
        # Notebook 04's nbae02 writes three Alibaba-estimate checkpoints after
        # nbae01 loaded only the first of them, so the loading cell and the
        # writing cell are not the same cell and the inference has no link
        # between them to follow.
        before = self._save_then_change_source()
        after = self._resave_same_metrics(recomputed=True)
        self.assertNotEqual(after["provenance"], before["provenance"])
        self.assertIs(T.checkpoint_is_current("exp_b_constant_median"), True)

    def test_an_explicit_false_keeps_the_original_stamp(self):
        before = self._save_then_change_source()
        self.assertIsNone(T.load_checkpoint("exp_b_constant_median"))
        # The caller says outright that it did not recompute, which must beat
        # the refusal-based inference rather than be overridden by it.
        after = self._resave_same_metrics(recomputed=False)
        self.assertEqual(after["provenance"], before["provenance"])
        self.assertEqual(after["timestamp"], before["timestamp"])


# ---------------------------------------------------------------------
# Model artifacts: the .joblib / .pth files notebook 05 replays.
#
# The save cells used to read `elif dest.exists(): skip`, so an artifact once
# written was never refreshed, which is how the trained models came to predate
# a feature-engineering fix while the metrics printed beside them had been
# recomputed, and notebook 05 simulated a pre-fix model against a post-fix test
# set. The sidecar is what makes that detectable.
# ---------------------------------------------------------------------


class TestModelArtifactCurrency(_ProvenanceCase):

    def test_missing_model_artifact_is_not_current(self):
        self.assertIs(T.model_artifact_is_current(self.tmp_path / "rf.joblib"), False)

    def test_artifact_without_a_sidecar_is_not_current(self):
        """Artifacts written before the sidecar existed must be refit once, not
        trusted on the strength of the file merely being there.
        """
        dest = self.tmp_path / "rf.joblib"
        dest.write_bytes(b"pretend-model")
        self.assertIs(T.model_artifact_is_current(dest), False)

    def test_recorded_sidecar_makes_the_artifact_current(self):
        dest = self.tmp_path / "rf.joblib"
        dest.write_bytes(b"pretend-model")
        sidecar = T.record_model_artifact(dest)

        # The name is the contract between the two functions:
        # model_artifact_is_current looks for exactly this path and silently
        # reports "not current" if the writer ever puts it somewhere else.
        self.assertEqual(sidecar, self.tmp_path / "rf.joblib.provenance.json")
        self.assertEqual(
            json.loads(sidecar.read_text())["src_sha256"],
            T._compute_provenance()["src_sha256"],
        )
        self.assertIs(T.model_artifact_is_current(dest), True)

    def test_source_change_makes_the_model_artifact_stale(self):
        dest = self.tmp_path / "rf.joblib"
        dest.write_bytes(b"pretend-model")
        T.record_model_artifact(dest)

        self._pretend_source_changed()
        self.assertIs(T.model_artifact_is_current(dest), False)

    def test_resaving_a_model_fitted_before_a_source_edit_does_not_launder_it(self):
        """A sidecar must describe the FIT, not the write.

        The test above only edits the source after both writes, so it passes
        just as well when ``record_model_artifact`` samples the tree standing at
        write time. This is the sequence that tells the two apart, and it is the
        one the notebooks actually run: the save is a separate cell from the
        training cell precisely so it can be re-run on its own, with the trained
        model still bound in the kernel. Stamping the write then certified
        pre-edit weights as current while ``save_checkpoint``'s carry-forward
        correctly left the metric beside them stale, and that artifact sailed
        through notebook 05's stale-artifact gate, which exists to exclude
        exactly it, reinstating the pre-fix-model / post-fix-test-set pairing.

        Executed rather than parsed: a real ``finalize_ml_model`` call is the
        only thing that puts a genuine fit-time snapshot on record.
        """
        import joblib

        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": np.arange(120, dtype=float), "b": rng.normal(size=120)})
        y = 2 * X["a"] + rng.normal(scale=2.0, size=120)
        best_params = {"n_estimators": 5}
        dest = self.tmp_path / "rf.joblib"

        model, metrics = T.finalize_ml_model(
            "rf", best_params, X.iloc[:100], y.iloc[:100], X.iloc[100:], y.iloc[100:],
            verbose=False,
        )
        T.save_checkpoint("exp_a_rf", {"metrics": metrics, "best_params": best_params})
        joblib.dump(model, dest)
        T.record_model_artifact(dest, model)
        self.assertIs(T.checkpoint_is_current("exp_a_rf"), True)
        self.assertIs(T.model_artifact_is_current(dest), True)

        self._pretend_source_changed()

        # Only the save cell is re-run: same object, same numbers, edited tree.
        T.save_checkpoint("exp_a_rf", {"metrics": metrics, "best_params": best_params})
        joblib.dump(model, dest)
        T.record_model_artifact(dest, model)

        self.assertIs(T.checkpoint_is_current("exp_a_rf"), False)
        self.assertIs(
            T.model_artifact_is_current(dest), False,
            "the model predates the edit, so it must go stale with the metric "
            "it was scored against, a re-save cannot launder it into a fit "
            "under the current tree",
        )

    def _fit_through_this_module(self):
        """A real ``finalize_ml_model`` fit, plus the data it was fitted on.

        Only a genuine call puts a genuine fit-time snapshot on record; a
        hand-made stand-in would let these tests pass against a mechanism that
        never records anything.
        """
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": np.arange(120, dtype=float), "b": rng.normal(size=120)})
        y = 2 * X["a"] + rng.normal(scale=2.0, size=120)
        model, _metrics = T.finalize_ml_model(
            "rf", {"n_estimators": 5}, X.iloc[:100], y.iloc[:100],
            X.iloc[100:], y.iloc[100:], verbose=False,
        )
        return model, X, y

    def test_losing_the_module_state_between_fit_and_re_save_does_not_launder_it(self):
        """The fit record has to outlive this module's own globals.

        The test above never disturbs them, so it passes just as well when the
        record lives only in module state: the fallback to
        ``_PROVENANCE_AT_LAST_FIT`` still holds the pre-edit snapshot and the
        sidecar comes out stale for a reason the notebooks do not have.
        Notebook 04's import cell (cd02) ends in
        ``importlib.reload(src.tuning)``, and re-executing the module body used
        to hand a blank registry to precisely the run that reloads to pick a
        source edit up, after which ``record_model_artifact`` fell back to the
        tree standing at the write and certified pre-edit weights as current.
        Emptying the two globals stands in for that reload: an actual
        ``importlib.reload`` would also discard ``_pretend_source_changed``'s
        patch, which is the pretence under test.
        """
        import joblib

        model, _X, _y = self._fit_through_this_module()
        dest = self.tmp_path / "rf.joblib"
        joblib.dump(model, dest)
        T.record_model_artifact(dest, model)
        self.assertIs(T.model_artifact_is_current(dest), True)

        self._pretend_source_changed()
        T._FIT_PROVENANCE = weakref.WeakKeyDictionary()
        T._PROVENANCE_AT_LAST_FIT = None

        # Only the save cell is re-run, in a kernel whose module state no
        # longer remembers the fit: same object, edited tree.
        joblib.dump(model, dest)
        T.record_model_artifact(dest, model)

        self.assertIs(
            T.model_artifact_is_current(dest), False,
            "the record rides on the model object, so a reloaded module must "
            "not turn pre-edit weights into a fit under the current tree",
        )

    def test_a_model_this_process_did_not_fit_borrows_no_other_fit(self):
        """A model handed over with no fit on record is refit, not certified.

        Falling back to the last fit in the kernel is right for the notebooks'
        non-learned baselines, which are computed in the notebook and passed as
        a path alone, but applied to a model object it stamps whatever was
        trained most recently onto something else entirely, and the sidecar then
        vouches for a tree this process never saw that model fitted under.
        """
        import joblib
        from sklearn.ensemble import RandomForestRegressor

        _fitted_here, X, y = self._fit_through_this_module()
        # A current snapshot is now on record, i.e. there is something in reach
        # for the artifact below to be wrongly stamped with.
        self.assertIsNotNone(T._PROVENANCE_AT_LAST_FIT)

        foreign = RandomForestRegressor(n_estimators=3, random_state=0).fit(X, y)
        dest = self.tmp_path / "foreign.joblib"
        joblib.dump(foreign, dest)
        sidecar = T.record_model_artifact(dest, foreign)

        stored = json.loads(sidecar.read_text())
        self.assertNotIn(
            "src_sha256", stored,
            "a sidecar that cannot name the tree the model was fitted under "
            "must not carry source hashes that say it can",
        )
        self.assertIs(T.model_artifact_is_current(dest), False)

    def test_unreadable_sidecar_is_not_current(self):
        """A truncated sidecar must read as "refit", never as "current", the
        failure has to fall on the safe side.
        """
        dest = self.tmp_path / "rf.joblib"
        dest.write_bytes(b"pretend-model")
        (self.tmp_path / "rf.joblib.provenance.json").write_text("{not json")
        self.assertIs(T.model_artifact_is_current(dest), False)

    def test_artifact_and_checkpoint_agree_on_what_stale_means(self):
        """A metric and the model beside it must never disagree about staleness
       , that disagreement is exactly what produced a per-bucket table pairing
        a pre-fix model with a post-fix test set.
        """
        dest = self.tmp_path / "rf.joblib"
        dest.write_bytes(b"pretend-model")
        T.record_model_artifact(dest)
        T.save_checkpoint("exp_a_rf", {"metrics": {"mae": 1.0}, "best_params": {}})
        self.assertIs(T.model_artifact_is_current(dest), True)
        self.assertIs(T.checkpoint_is_current("exp_a_rf"), True)

        self._pretend_source_changed("src/tuning.py")
        self.assertIs(T.model_artifact_is_current(dest), False)
        self.assertIs(T.checkpoint_is_current("exp_a_rf"), False)


class TestEverySeedCheckpointIsRecorded(_ProvenanceCase):
    """The one artifact writer that is not a notebook save cell.

    ``finalize_dl_model`` writes the per-seed LSTM checkpoints itself, from a
    ``{seed}`` template the notebook only passes in, and it wrote them with no
    sidecar. Notebook 05's multi-seed robustness cell gates exactly those paths,
    so the gate was unsatisfiable: the files exist, which rules out its "absent
    is a legitimate skip" branch, and re-running notebook 04, the remedy the
    refusal prints, produced the same unstamped files again. That cell is the
    only evidence behind the claim that the single-seed ranking is robust to
    initialization.

    Executed rather than parsed: a gate is only proved satisfiable by something
    actually satisfying it.
    """

    def setUp(self):
        super().setUp()
        # The real one trains a network for 50 epochs; the branch under test is
        # the save that follows it, which does not care what the model is.
        self._patch(
            "_finalize_dl_single",
            lambda *args, **kwargs: ({"weights": args[-1]}, {"mae": 1.0 + args[-1]}),
        )

    def _train(self, seeds):
        template = str(self.tmp_path / "lstm_categorical_pt_seed{seed}.pth")
        T.finalize_dl_model(
            "LSTM", {}, None, None, None, None, None, None,
            device="cpu", seeds=seeds, save_all_seeds_to=template,
        )
        return template

    def test_every_seed_checkpoint_can_satisfy_the_gate_that_reads_it(self):
        seeds = (42, 1337, 2024)
        template = self._train(seeds)
        for seed in seeds:
            path = Path(template.format(seed=seed))
            self.assertTrue(path.exists(), f"seed {seed} was not written at all")
            self.assertIs(
                T.model_artifact_is_current(path), True,
                "notebook 05 refuses a per-seed checkpoint without a sidecar, "
                "and re-running notebook 04 is the only remedy it names, so a "
                "sidecar missing here can never be repaired",
            )

    def test_a_source_change_still_makes_them_stale(self):
        # Otherwise the assertion above would pass on any file at all.
        template = self._train((42,))
        self._pretend_source_changed()
        self.assertIs(
            T.model_artifact_is_current(Path(template.format(seed=42))), False
        )


# ---------------------------------------------------------------------
# What the provenance snapshot actually covers.
# ---------------------------------------------------------------------


class TestProvenanceCoverage(_ProvenanceCase):

    def test_every_tracked_source_file_really_exists(self):
        """A path that does not resolve hashes to None on both sides and
        therefore compares equal forever, so a typo or a rename turns a tracked
        file into a permanently "matching" one without any visible symptom.
        """
        for name in T._PROVENANCE_SRC_FILES:
            self.assertTrue(
                (_REPO_ROOT / name).exists(),
                f"{name} is tracked for provenance but does not exist",
            )
            self.assertIsNotNone(T._file_sha256(_REPO_ROOT / name))

    def test_provenance_tracks_every_source_the_training_path_imports(self):
        """No new src dependency may join the training path untracked.

        The list used to hold only feature_engineering.py and tuning.py, so the
        DL architectures, the metric definitions and the loader that resolves
        the training CSV could all change while model_artifact_is_current()
        kept returning True. This walks the import graph instead of restating
        the list, so the next module pulled in has to be tracked deliberately.

        src/config_utils.py is the one dependency still outside the list: it
        decides which YAML is read and how, so a change there can move the
        numbers even though the YAMLs themselves are hashed. Named here rather
        than silently tolerated, the assertion is a subset, so closing that
        hole keeps this passing while opening a new one does not.
        """
        known_untracked = {"src/config_utils.py"}

        def _src_imports(rel_path):
            tree = ast.parse((_REPO_ROOT / rel_path).read_text(encoding="utf-8"))
            found = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."):
                    found.add(node.module)
                elif isinstance(node, ast.Import):
                    found.update(a.name for a in node.names if a.name.startswith("src."))
            return found

        seen, queue = set(), ["src/tuning.py", "src/feature_engineering.py"]
        while queue:
            rel = queue.pop()
            if rel in seen:
                continue
            seen.add(rel)
            for module in _src_imports(rel):
                queue.append(module.replace(".", "/") + ".py")

        untracked = seen - set(T._PROVENANCE_SRC_FILES)
        self.assertLessEqual(
            untracked, known_untracked,
            f"these modules determine the numbers but are not fingerprinted: "
            f"{sorted(untracked - known_untracked)}",
        )

    def test_data_fingerprint_covers_the_trace_the_run_reads(self):
        """data_sha256 must fingerprint the raw trace, not the processed CSV.

        Every prepare_features_for_model call in the modelling notebooks leaves
        use_processed=False, so the processed utilization CSV is written by
        notebook 00 and read by no training cell. Hashing it fired a
        data-change warning on 26 of 31 checkpoints whose numbers it cannot
        influence while a swap of the raw trace went unreported, a guard that
        cries wolf about an unread file teaches the reader to skip the warning
        block entirely.
        """
        paths = T._training_input_files()
        self.assertTrue(paths, "no training input file was resolved from paths.yaml")
        self.assertTrue(all("raw" in n or "trace" in n for n in paths), sorted(paths))
        self.assertFalse(any("processed" in n for n in paths), sorted(paths))

        fingerprint = T._compute_provenance()["data_sha256"]
        self.assertIsInstance(fingerprint, dict)
        self.assertEqual(set(fingerprint), set(paths))
        self.assertTrue(all(v is not None for v in fingerprint.values()))

    def test_a_changed_trace_is_reported_per_file(self):
        stored = T._compute_provenance()
        changed = {**stored, "data_sha256": {**stored["data_sha256"]}}
        victim = sorted(changed["data_sha256"])[0]
        changed["data_sha256"][victim] = "SWAPPED"

        mismatches = T._provenance_mismatches(stored, changed)
        self.assertTrue(any(victim in line for line in mismatches), mismatches)

    def test_a_pre_mapping_checkpoint_does_not_crash_the_mismatch_report(self):
        """Checkpoints written before data_sha256 became a mapping stored a
        single processed-CSV hash there. Those files are still on disk, and
        reporting on them must produce a mismatch line, not a TypeError that
        takes the whole load path down.
        """
        current = T._compute_provenance()
        legacy = {**current, "data_sha256": "a-single-processed-csv-hash"}
        mismatches = T._provenance_mismatches(legacy, current)
        self.assertTrue(mismatches)
        self.assertTrue(all(isinstance(line, str) for line in mismatches))


class TestTheDataHalfIsEnforcedNotOnlyReported(_ProvenanceCase):
    """A changed raw trace must make a result stale, not merely warn.

    The data half was excluded from the currency decision on the grounds that
    data_sha256 hashed a processed file the training path never reads. That
    stopped being true once the fingerprint was repointed at the raw trace
    load_sample opens, and until the predicate followed, a re-sampled or
    regenerated trace left every checkpoint and artifact certified current
    behind nothing but a printed warning.
    """

    def _pretend_trace_changed(self):
        real = T._compute_provenance

        def fake():
            snapshot = real()
            data = dict(snapshot["data_sha256"])
            data[sorted(data)[0]] = "SWAPPED"
            return {**snapshot, "data_sha256": data}

        self._patch("_compute_provenance", fake)

    def test_a_swapped_trace_makes_a_checkpoint_stale(self):
        T.save_checkpoint("exp_a_rf", {"metrics": {"mae": 1.0}, "best_params": {}})
        self.assertIs(T.checkpoint_is_current("exp_a_rf"), True)
        self._pretend_trace_changed()
        self.assertIs(T.checkpoint_is_current("exp_a_rf"), False)

    def test_a_swapped_trace_makes_a_model_artifact_stale(self):
        dest = self.tmp_path / "rf.joblib"
        dest.write_bytes(b"pretend-model")
        T.record_model_artifact(dest)
        self._pretend_trace_changed()
        self.assertIs(T.model_artifact_is_current(dest), False)

    def test_a_trace_this_environment_cannot_hash_is_not_judged(self):
        """baseline_estimate_file is optional: notebook 04 skips the Alibaba
        rows when it is missing. Reading "the reader never downloaded it" as a
        data change would discard all 31 checkpoints over a file that
        contributed to three of them.
        """
        T.save_checkpoint("exp_a_rf", {"metrics": {"mae": 1.0}, "best_params": {}})
        stored = self._read("exp_a_rf")["provenance"]
        self.assertNotIn("never_downloaded.csv", stored["data_sha256"])

        real = T._compute_provenance
        self._patch(
            "_compute_provenance",
            lambda: {**real(),
                     "data_sha256": {**real()["data_sha256"], "never_downloaded.csv": None}},
        )
        self.assertIs(T.checkpoint_is_current("exp_a_rf"), True)

    def test_a_pre_mapping_data_fingerprint_is_not_current(self):
        # A single processed-CSV hash says nothing about the raw trace, so it
        # is unverifiable and must fall on the "refit" side rather than pass
        # the check by not being comparable.
        stored = {**T._compute_provenance(), "data_sha256": "a-single-processed-csv-hash"}
        self.assertIs(T._provenance_is_current(stored), False)


class TestTrainingNotebooksAreFingerprinted(_ProvenanceCase):
    """Most of what decides a checkpoint's numbers lives in notebook 04's cells,
    not in src/: the train/test fraction, DL_SEEDS, the search and final-refit
    budgets, seq_len, the ablation feature groups and the whole definition of
    the non-learned baselines. Hashing only src/ let all of those change while
    both guards still answered "yes, computed by the source code in the tree
    now", so the training cells took their ``if ckpt:`` branch and reported
    numbers produced under the previous settings.
    """

    def _notebook(self, cells):
        path = self.tmp_path / "nb.ipynb"
        path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
        return path

    def _code(self, source, execution_count=1, outputs=()):
        return {"cell_type": "code", "source": source,
                "execution_count": execution_count, "outputs": list(outputs)}

    def test_both_training_notebooks_are_hashed(self):
        src_hashes = T._compute_provenance()["src_sha256"]
        for name in T._PROVENANCE_NOTEBOOK_FILES:
            self.assertTrue((_REPO_ROOT / name).exists(),
                            f"{name} is fingerprinted but does not exist")
            self.assertIn(name, src_hashes)
            self.assertIsNotNone(src_hashes[name],
                                 f"{name} hashed to None, which compares equal forever")
        # EN and TR both write the same checkpoint files, so a setting changed
        # in one of them alone still has to invalidate the results.
        self.assertEqual(len(T._PROVENANCE_NOTEBOOK_FILES), 2)

    def test_the_scheduling_notebooks_are_deliberately_not_hashed(self):
        """Notebook 05 trains nothing and writes no checkpoint, so it cannot
        change a checkpoint's numbers. Hashing it would invalidate every
        training result whenever a scheduling cell is edited, the cry-wolf
        failure the guard is meant to avoid.
        """
        self.assertFalse(
            [n for n in T._PROVENANCE_NOTEBOOK_FILES if "05_" in Path(n).name],
            "a notebook that writes no checkpoint must not invalidate one",
        )

    def test_a_changed_training_notebook_makes_a_checkpoint_stale(self):
        T.save_checkpoint("exp_a_rf", {"metrics": {"mae": 1.0}, "best_params": {}})
        self.assertIs(T.checkpoint_is_current("exp_a_rf"), True)
        self._patch("_notebook_code_sha256", lambda path: "EDITED")
        self.assertIs(T.checkpoint_is_current("exp_a_rf"), False)

    def test_only_code_cells_are_hashed(self):
        """Hashing the .ipynb bytes would fold in stored outputs and execution
        counts, so merely running the notebook, or clearing its outputs, or
        editing prose, would invalidate every checkpoint it just wrote, none
        of which changes a number.
        """
        before = T._notebook_code_sha256(self._notebook([
            {"cell_type": "markdown", "source": ["# Runtime prediction\n"]},
            self._code(["seq_len = 10\n"], execution_count=None),
        ]))
        after = T._notebook_code_sha256(self._notebook([
            {"cell_type": "markdown", "source": ["# Completely rewritten prose\n"]},
            self._code(["seq_len = 10\n"], execution_count=42,
                       outputs=[{"output_type": "stream", "text": ["done\n"]}]),
        ]))
        self.assertEqual(before, after)

    def test_a_changed_code_cell_changes_the_hash(self):
        before = T._notebook_code_sha256(self._notebook([self._code(["seq_len = 10\n"])]))
        after = T._notebook_code_sha256(self._notebook([self._code(["seq_len = 20\n"])]))
        self.assertNotEqual(before, after)

    def test_moving_a_line_between_adjacent_cells_changes_the_hash(self):
        # Without the separator between cells the concatenation is identical
        # and the hash cannot see the move.
        before = T._notebook_code_sha256(self._notebook([
            self._code(["a = 1\n"]), self._code(["b = 2\n"]),
        ]))
        after = T._notebook_code_sha256(self._notebook([
            self._code(["a = 1\n", "b = 2\n"]), self._code([]),
        ]))
        self.assertNotEqual(before, after)

    def test_an_unreadable_notebook_hashes_to_none(self):
        self.assertIsNone(T._notebook_code_sha256(self.tmp_path / "absent.ipynb"))
        broken = self.tmp_path / "broken.ipynb"
        broken.write_text("{not json", encoding="utf-8")
        self.assertIsNone(T._notebook_code_sha256(broken))


if __name__ == "__main__":
    unittest.main()
