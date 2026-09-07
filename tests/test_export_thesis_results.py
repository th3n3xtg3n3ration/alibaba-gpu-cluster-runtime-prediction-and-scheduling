"""
Regression test for figures_tables-14: export_thesis_results.py wrote
positionally-named PNG/HTML files (e.g. "NB05_32GPU-Figure07.png") but never
removed a notebook's previously-exported files before writing new ones. A
notebook that used to produce N figures and now produces fewer than N left
the extra, stale files sitting in the export directory looking exactly like
output from the current run.
"""
import base64
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import tuning as T

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_thesis_results.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_thesis_results", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_attr(case, target, name, value):
    """Rebind ``target.name`` for the duration of one test."""
    patcher = mock.patch.object(target, name, value)
    patcher.start()
    case.addCleanup(patcher.stop)


# The gate judges the model files the notebooks themselves load, read out of
# their source, so a fixture notebook has to spell one the way notebook 04 does.
_MODEL_ARTIFACT = "rf_model.joblib"
_SAVE_CELL_SOURCE = (f'joblib.dump(model, MODEL_DIR / "{_MODEL_ARTIFACT}")\n',
                     "plt.show()\n")


def _record_current_model_inputs(model_dir, checkpoint="exp_a_rf",
                                 artifact=_MODEL_ARTIFACT):
    """Write one checkpoint and one model artifact, both certified current.

    Written through ``src.tuning``'s own recording functions rather than by
    hand: the gate under test asks that module's predicates, so a hand-built
    record would keep certifying itself even if the two halves drifted apart.
    The artifact's bytes are never read, only the sidecar beside it is.
    """
    T.save_checkpoint(checkpoint, {"metrics": {"mae": 1.0}}, recomputed=True)
    path = model_dir / artifact
    path.write_bytes(b"stand-in for a fitted model")
    T.record_model_artifact(path)
    return checkpoint, artifact


def _freeze_provenance(case):
    """Hold the source-tree snapshot still for the duration of one test.

    The verdict has to depend on the records the test writes, not on whether an
    unrelated file in src/ was saved between the moment a record was written and
    the moment the gate read it back, which really does happen while the suite
    runs and would make a passing gate look broken. Everything the gate itself
    uses stays real: the sidecars, the checkpoints and both currency predicates.
    """
    snapshot = T._compute_provenance()
    _patch_attr(case, T, "_compute_provenance", lambda: snapshot)


def _pretend_source_changed(case):
    """Make the tree report a source hash no stored record can carry.

    This is what an edit to src/feature_engineering.py after the last training
    run looks like to ``_provenance_is_current``: the numbers on disk are the
    record of a run this source tree would no longer reproduce.
    """
    real = T._compute_provenance
    _patch_attr(case, T, "_compute_provenance",
                lambda: {**real(), "src_sha256": {"src/feature_engineering.py": "CHANGED"}})


class TestCleanStaleExports(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()
        self._tmp = tempfile.TemporaryDirectory()
        # Redirect the module's output directories to a throwaway location
        # so this test never touches the real results/figures/thesis_export/.
        self.module.PNG_DIR = Path(self._tmp.name) / "png"
        self.module.HTML_DIR = Path(self._tmp.name) / "html"
        self.module.PNG_DIR.mkdir(parents=True)
        self.module.HTML_DIR.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_removes_stale_higher_numbered_figure(self):
        # Simulate a previous run that produced 7 figures for NB05_32GPU.
        for i in range(1, 8):
            (self.module.PNG_DIR / f"NB05_32GPU-Figure{i:02d}.png").write_bytes(b"x")
        # This run only produces 6 (one figure cell was removed upstream).
        self.module._clean_stale_exports("NB05_32GPU")
        remaining = sorted(self.module.PNG_DIR.glob("NB05_32GPU-Figure*.png"))
        self.assertEqual(remaining, [], "stale exports must all be cleared before re-extraction")

    def test_does_not_touch_a_different_prefixs_files(self):
        (self.module.PNG_DIR / "NB05_32GPU-Figure01.png").write_bytes(b"x")
        (self.module.PNG_DIR / "NB05_256GPU-Figure01.png").write_bytes(b"x")
        self.module._clean_stale_exports("NB05_32GPU")
        self.assertFalse((self.module.PNG_DIR / "NB05_32GPU-Figure01.png").exists())
        self.assertTrue((self.module.PNG_DIR / "NB05_256GPU-Figure01.png").exists())

    def test_clears_stale_html_tables_too(self):
        (self.module.HTML_DIR / "NB04_Table09.html").write_text("<table></table>")
        self.module._clean_stale_exports("NB04")
        remaining = sorted(self.module.HTML_DIR.glob("NB04_Table*.html"))
        self.assertEqual(remaining, [])


# A 1x1 PNG, base64-encoded exactly as a notebook stores an image output.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)
# A second, visibly different 1x1 PNG. The two languages export under identical
# filenames, so telling one run's output from the other's takes distinct bytes.
_OTHER_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhQGAWpr"
    "eKAAAAABJRU5ErkJggg=="
)


def _figure_cell(cell_id, execution_count, extra_outputs=(), png_b64=_PNG_B64,
                 source=("plt.show()",)):
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": execution_count,
        # Only the model-provenance gate reads a cell's source (it takes the
        # artifact names out of it); every other gate judges the outputs.
        "source": list(source),
        "outputs": [{"output_type": "display_data", "data": {"image/png": png_b64}},
                    *extra_outputs],
    }


class TestStoredOutputAudit(unittest.TestCase):
    """Regression tests for the trust gate on a notebook's stored outputs.

    The extraction loop reads whatever a cell has stored and writes it out as a
    result of the current run. Two kinds of output pass that reading while being
    nothing of the sort, and neither is visible to the figure-count gate:
    matplotlib flushes the half-drawn figure after a traceback, so a crashed
    cell contributes a picture indistinguishable from a valid one; and a cell
    that never ran keeps outputs from an earlier revision of the notebook, which
    then HELP the expected count pass instead of tripping it. Both had already
    reached thesis/latex/figures with exit status 0.
    """

    def setUp(self):
        self.module = _load_module()
        self._tmp = tempfile.TemporaryDirectory()
        self.module.PNG_DIR = Path(self._tmp.name) / "png"
        self.module.HTML_DIR = Path(self._tmp.name) / "html"
        self.module.PNG_DIR.mkdir(parents=True)
        self.module.HTML_DIR.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_clean_top_to_bottom_run_is_accepted(self):
        nb = {"cells": [
            {"cell_type": "markdown", "id": "md1", "source": ["# title"]},
            _figure_cell("c1", 1),
            {"cell_type": "code", "id": "c2", "execution_count": 2, "outputs": []},
            _figure_cell("c3", 3),
        ]}
        self.assertEqual(self.module._audit_stored_outputs(nb), [])

    def test_a_cell_that_raised_is_reported(self):
        nb = {"cells": [_figure_cell(
            "c1", 1,
            extra_outputs=[{"output_type": "error", "ename": "LinAlgError",
                            "evalue": "SVD did not converge", "traceback": []}],
        )]}
        problems = self.module._audit_stored_outputs(nb)
        self.assertEqual(len(problems), 1)
        self.assertIn("LinAlgError", problems[0])

    def test_a_cell_that_never_ran_is_reported(self):
        # Cell d7a6e287 of the 256-GPU notebook 05 is exactly this shape: no
        # execution count, one leftover image, already exported once.
        nb = {"cells": [_figure_cell("d7a6e287", None)]}
        problems = self.module._audit_stored_outputs(nb)
        self.assertEqual(len(problems), 1)
        self.assertIn("d7a6e287", problems[0])

    def test_an_unexecuted_cell_with_no_outputs_is_fine(self):
        # Nothing stale can be extracted from a cell that stored nothing, and
        # notebooks are routinely committed with trailing cells not yet run.
        nb = {"cells": [{"cell_type": "code", "id": "c1",
                         "execution_count": None, "outputs": []}]}
        self.assertEqual(self.module._audit_stored_outputs(nb), [])

    def test_outputs_from_two_kernel_sessions_are_reported(self):
        # The second cell's output was produced before the first cell's, so the
        # stored outputs are a mixture of runs whatever their order on screen.
        nb = {"cells": [_figure_cell("c1", 7), _figure_cell("c2", 3)]}
        problems = self.module._audit_stored_outputs(nb)
        self.assertEqual(len(problems), 1)
        self.assertIn("c2", problems[0])

    def test_a_gap_in_the_execution_sequence_is_not_reported(self):
        # A gap only shows some cell was skipped or cleared; every output still
        # present did come from the same increasing run, and a skipped figure
        # cell already surfaces as a count mismatch in _sync_thesis_figures.
        nb = {"cells": [_figure_cell("c1", 2), _figure_cell("c2", 9)]}
        self.assertEqual(self.module._audit_stored_outputs(nb), [])

    def test_extraction_refuses_an_untrusted_notebook(self):
        nb = {"cells": [_figure_cell(
            "c1", 1,
            extra_outputs=[{"output_type": "error", "ename": "LinAlgError",
                            "evalue": "", "traceback": []}],
        )]}
        with self.assertRaises(self.module.UntrustedNotebookOutputs):
            self.module.extract_from_nb_dict(nb, "NB05_32GPU", True, True, {})

    def test_nothing_is_written_before_the_refusal(self):
        """The refusal has to come before the first file, or a rejected
        notebook still leaves a half-written export behind for the thesis sync
        to pick up.
        """
        nb = {"cells": [
            _figure_cell("c1", 1),                       # would extract cleanly
            _figure_cell("c2", None),                    # never ran
        ]}
        thesis_buffer = {}
        with self.assertRaises(self.module.UntrustedNotebookOutputs):
            self.module.extract_from_nb_dict(nb, "NB05_32GPU", True, True, thesis_buffer)

        self.assertEqual(list(self.module.PNG_DIR.iterdir()), [])
        self.assertEqual(list(self.module.HTML_DIR.iterdir()), [])
        self.assertEqual(thesis_buffer, {})

    def test_every_problem_is_carried_not_just_the_first(self):
        nb = {"cells": [
            _figure_cell("c1", 1, extra_outputs=[
                {"output_type": "error", "ename": "ValueError", "evalue": "", "traceback": []}]),
            _figure_cell("c2", None),
        ]}
        with self.assertRaises(self.module.UntrustedNotebookOutputs) as raised:
            self.module.extract_from_nb_dict(nb, "NB05_32GPU", True, True, {})
        self.assertEqual(len(raised.exception.problems), 2)


def _table_cell(cell_id, execution_count, label):
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": execution_count,
        "source": ["display(df)"],
        "outputs": [{"output_type": "display_data",
                     "data": {"text/html": f"<table><tr><td>{label}</td></tr></table>"}}],
    }


class TestExportDirectoriesAreSeparatedByLanguage(unittest.TestCase):
    """The two languages produce the SAME positional filenames.

    NOTEBOOKS_TR reuses the prefixes NB01…NB05_256GPU, and _clean_stale_exports
    deletes by prefix with no language filter, so while both wrote into one
    directory a `--lang tr` run deleted the English figures and replaced the
    English tables with Turkish ones under identical names, silently, and with
    exit status 0. It was not even a like-for-like swap: the table position
    index differs between the mirrors, so one filename changed from one
    benchmark table to a different one.
    """

    def setUp(self):
        self.module = _load_module()

    def test_the_two_languages_do_not_share_a_directory(self):
        en_png, en_html = self.module._export_dirs("en")
        tr_png, tr_html = self.module._export_dirs("tr")
        self.assertNotEqual(en_png, tr_png)
        self.assertNotEqual(en_html, tr_html)

    def test_english_keeps_the_documented_git_tracked_paths(self):
        # That set is the record the thesis numbers were transcribed from, so
        # it must not move to a new location.
        en_png, en_html = self.module._export_dirs("en")
        self.assertEqual(en_png, self.module.EXPORT_DIR / "png")
        self.assertEqual(en_html, self.module.EXPORT_DIR / "html")

    def test_neither_language_tree_contains_the_other(self):
        en_png, _ = self.module._export_dirs("en")
        tr_png, _ = self.module._export_dirs("tr")
        # A stale-export sweep for one language must not be able to reach the
        # other's files.
        self.assertNotIn(en_png, tr_png.parents)
        self.assertNotIn(tr_png, en_png.parents)


class TestTableNamesArePinnedToTheirCell(unittest.TestCase):
    """Figures are protected against a shifted position by THESIS_FIGURE_MAP
    and EXPECTED_FIGURE_COUNT; tables had nothing. A notebook whose middle
    cells were re-run, or the other language's mirror, renumbers every table
    after the first gap, so one filename silently came to hold a different
    table between two exports.
    """

    def setUp(self):
        self.module = _load_module()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.module.PNG_DIR = Path(self._tmp.name) / "png"
        self.module.HTML_DIR = Path(self._tmp.name) / "html"
        self.module.PNG_DIR.mkdir(parents=True)
        self.module.HTML_DIR.mkdir(parents=True)

    def _export(self, cells):
        self.module.extract_from_nb_dict({"cells": cells}, "NB04", False, True, None)
        return sorted(p.name for p in self.module.HTML_DIR.glob("*.html"))

    def test_the_producing_cell_is_part_of_the_filename(self):
        names = self._export([_table_cell("dl-comparison", 1, "DL comparison")])
        self.assertEqual(names, ["NB04_Table01_dl-comparison.html"])

    def test_a_table_that_shifts_position_lands_under_a_new_name(self):
        # A stale reference then fails to resolve, instead of quietly
        # resolving to the wrong table.
        first = self._export([_table_cell("dl-comparison", 1, "DL comparison")])
        for stale in self.module.HTML_DIR.glob("*.html"):
            stale.unlink()
        shifted = self._export([
            _table_cell("new-table", 1, "Something else"),
            _table_cell("dl-comparison", 2, "DL comparison"),
        ])
        self.assertNotIn(first[0], shifted)
        self.assertIn("NB04_Table02_dl-comparison.html", shifted)

    def test_two_tables_from_one_cell_stay_distinct(self):
        # The position stays in front of the id, so the directory still sorts
        # in notebook order and one cell displaying two tables keeps both.
        cell = _table_cell("both", 1, "first")
        cell["outputs"].append({
            "output_type": "display_data",
            "data": {"text/html": "<table><tr><td>second</td></tr></table>"},
        })
        self.assertEqual(
            self._export([cell]),
            ["NB04_Table01_both.html", "NB04_Table02_both.html"],
        )

    def test_a_cell_id_that_is_not_filename_safe_is_sanitised(self):
        # nbformat restricts ids to [A-Za-z0-9-_], but the notebook is read as
        # plain JSON, so a hand-edited file could carry a path separator.
        names = self._export([_table_cell("../../etc/passwd", 1, "x")])
        self.assertEqual(len(names), 1)
        self.assertNotIn("/", names[0])
        self.assertTrue(names[0].startswith("NB04_Table01_"))

    def test_a_cell_without_an_id_still_exports(self):
        cell = _table_cell("x", 1, "x")
        del cell["id"]
        self.assertEqual(self._export([cell]), ["NB04_Table01.html"])

    def test_stale_tables_from_the_old_naming_are_cleared(self):
        # Otherwise they linger forever under names no run writes any more.
        (self.module.HTML_DIR / "NB04_Table01.html").write_text("<table></table>")
        (self.module.HTML_DIR / "NB04_Table01_oldcell.html").write_text("<table></table>")
        self.module._clean_stale_exports("NB04")
        self.assertEqual(list(self.module.HTML_DIR.glob("NB04_Table*.html")), [])


class TestEveryRefusalFailsTheRun(unittest.TestCase):
    """A refusal has to reach run_pipeline and end the run non-zero.

    run_all_experiments.sh runs this script under `set -euo pipefail` and
    prints "PIPELINE COMPLETED SUCCESSFULLY" on exit 0. A figure-count mismatch
    used to return 0 files with nothing but a warning buried among dozens of
    [PNG]/[HTML] success lines, so the thesis kept the previous run's images
    under a green banner.
    """

    def setUp(self):
        self.module = _load_module()

    def test_a_count_mismatch_returns_a_reason(self):
        reason = self.module._check_figure_count("NB01", produced=5, sync_thesis=True)
        self.assertIsNotNone(reason)
        self.assertIn("5", reason)
        self.assertIn(str(self.module.EXPECTED_FIGURE_COUNT["NB01"]), reason)

    def test_a_matching_count_returns_no_reason(self):
        expected = self.module.EXPECTED_FIGURE_COUNT["NB01"]
        self.assertIsNone(
            self.module._check_figure_count("NB01", expected, sync_thesis=True)
        )

    def test_an_unmapped_prefix_is_not_refused(self):
        self.assertIsNone(self.module._check_figure_count("NB99", 3, sync_thesis=True))

    def test_the_turkish_run_is_checked_too(self):
        # The export filenames are positional in the same way and the Turkish
        # export is an artifact set in its own right; only the wording changes,
        # because nothing is copied into the thesis on a TR run.
        reason = self.module._check_figure_count("NB01", 5, sync_thesis=False)
        self.assertIsNotNone(reason)
        self.assertNotIn("tez", reason.lower())

    def test_a_missing_mapped_position_returns_a_reason_not_a_silent_zero(self):
        written, reason = self.module._sync_thesis_figures("NB01", thesis_buffer={})
        self.assertEqual(written, 0)
        self.assertIsNotNone(
            reason,
            "returning 0 silently was how a run that left thesis/latex/figures "
            "untouched still exited 0",
        )

    def test_a_prefix_with_no_mapped_figures_is_not_a_refusal(self):
        self.assertEqual(self.module._sync_thesis_figures("NB99", {}), (0, None))


class _PipelineCase(unittest.TestCase):
    """Shared harness for driving run_pipeline over a throwaway notebook tree.

    Never the repository's own tree: the pipeline deletes and rewrites
    everything under the export directories.

    ``PREFIX`` says which notebook the single fixture notebook stands in for,
    because the pipeline does not treat them alike, NB01 passes the gates that
    judge a notebook against itself, while NB04 and both NB05 variants are
    additionally judged against the model provenance records (see
    ``TestModelDerivedNotebooksAreJudgedOnProvenance``).
    """

    PREFIX = "NB01"

    def setUp(self):
        self.module = _load_module()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)

        self.module.BASE_DIR = self.base
        self.module.EXPORT_DIR = self.base / "results" / "figures" / "thesis_export"
        self.module.THESIS_FIG_DIR = self.base / "thesis" / "latex" / "figures"
        # One notebook per language, one mapped figure, so a single missing or
        # extra figure is the whole difference between a run and a refusal.
        self.module.NOTEBOOKS_EN = [("nb_en.ipynb", self.PREFIX, True, False)]
        self.module.NOTEBOOKS_TR = [("nb_tr.ipynb", self.PREFIX, True, False)]
        self.thesis_figure = f"{self.PREFIX.lower()}-fig01.png"
        self.export_name = f"{self.PREFIX}-Figure01.png"
        self.stale_export_name = f"{self.PREFIX}-Figure02.png"
        self.png_dir, _ = self.module._export_dirs("en")
        self.module.THESIS_FIGURE_MAP = {
            (self.PREFIX, 1): ("figcell", self.thesis_figure)
        }
        self.module.EXPECTED_FIGURE_COUNT = {self.PREFIX: 1}

    def _write_notebook(self, lang, name, cells):
        nb_dir = self.base / "notebooks" / lang
        nb_dir.mkdir(parents=True, exist_ok=True)
        (nb_dir / name).write_text(json.dumps({"cells": cells}), encoding="utf-8")

    def _run(self, **kwargs):
        """Run the pipeline with its console output captured and returned."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.module.run_pipeline(**kwargs)
        self.output = buffer.getvalue()
        return self.output

    def _refuse(self, **kwargs):
        """Run the pipeline expecting it to end non-zero; returns the reason.

        The console output is kept in ``self.output`` too: the exception lists
        one line per refused notebook, while which record or figure was at
        fault is in the warning block printed above it.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(RuntimeError) as raised:
                self.module.run_pipeline(**kwargs)
        self.output = buffer.getvalue()
        return str(raised.exception)

    def _exported(self, lang="en"):
        png_dir, _ = self.module._export_dirs(lang)
        return sorted(p.name for p in png_dir.iterdir()) if png_dir.exists() else []


class TestRunPipelineAssemblesTheGuards(_PipelineCase):
    """The gates above, driven through the entry point that has to act on them.

    Every piece was covered in isolation and the assembly was not, so each of
    these passed the whole suite: `if refusals:` turned into `if False:`, both
    `refusals.append` calls after the two returned reasons deleted, the
    untrusted-outputs handler's append deleted, and `_export_dirs(lang)`
    replaced by `_export_dirs("en")`. Each one restores exactly the failure the
    class above is named after, run_all_experiments.sh printing "PIPELINE
    COMPLETED SUCCESSFULLY" over an export that skipped a notebook, or over
    English thesis figures replaced by Turkish ones.
    """

    #, the run that must succeed ------------------------------------------
    # Without it every assertion below is satisfiable by a pipeline that
    # refuses everything, which is the deadlock shape this repository has
    # already shipped once.

    def test_a_complete_run_exports_and_updates_the_thesis(self):
        self._write_notebook("en", "nb_en.ipynb", [_figure_cell("figcell", 1)])
        self._run()
        self.assertEqual(self._exported(), [self.export_name])
        self.assertTrue((self.module.THESIS_FIG_DIR / self.thesis_figure).exists())

    #, and the four ways it must not ---------------------------------------

    def test_a_figure_count_mismatch_ends_the_run(self):
        self._write_notebook("en", "nb_en.ipynb",
                             [_figure_cell("figcell", 1), _figure_cell("extra", 2)])
        message = self._refuse()
        self.assertIn("nb_en.ipynb", message)
        self.assertEqual(
            self._exported(), [],
            "the positions no longer line up with the names, so nothing may be "
            "flushed after a run that has already failed",
        )
        self.assertFalse((self.module.THESIS_FIG_DIR / self.thesis_figure).exists())

    def test_a_refusal_leaves_the_previous_exports_in_place(self):
        """A refused run must not empty the record it tells the reader to fix.

        The deletion of a notebook's previous exports used to run before
        extraction, so every refusal above removed the git-tracked export set
        and wrote nothing back: a default run took out ten PNGs and eight NB04
        tables while printing a message telling the reader to compare against
        them. The other refusal tests all start from an empty export directory,
        where "nothing written" and "everything deleted" look identical, which
        is why this shipped.
        """
        previous = self.png_dir / self.export_name
        previous.parent.mkdir(parents=True, exist_ok=True)
        previous.write_bytes(b"previous export")

        self._write_notebook("en", "nb_en.ipynb",
                             [_figure_cell("figcell", 1), _figure_cell("extra", 2)])
        self._refuse()

        self.assertTrue(
            previous.exists(),
            "a refused notebook must leave its previous exports alone; deleting "
            "them destroys the very record the refusal asks the reader to check",
        )
        self.assertEqual(previous.read_bytes(), b"previous export")

    def test_a_successful_run_still_clears_the_previous_exports(self):
        """...but the deletion must still happen when the replacement is ready.

        Export filenames are positional, so a run producing fewer figures than
        the last one would otherwise leave stale higher-numbered files behind
         Deferring the deletion must not lose that.
        """
        stale = self.png_dir / self.stale_export_name
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"left over from a longer run")

        self._write_notebook("en", "nb_en.ipynb", [_figure_cell("figcell", 1)])
        self._run()

        self.assertFalse(stale.exists(), "a successful run must still clear stale positions")
        self.assertEqual(self._exported(), [self.export_name])

    def test_a_figure_from_the_wrong_cell_ends_the_run(self):
        # The count still matches, so only the pinned cell id catches this,
        # and a mislabeled thesis figure is worse than a stale one.
        self._write_notebook("en", "nb_en.ipynb", [_figure_cell("moved", 1)])
        message = self._refuse()
        self.assertIn("nb_en.ipynb", message)
        self.assertFalse((self.module.THESIS_FIG_DIR / self.thesis_figure).exists())

    def test_an_untrusted_notebook_ends_the_run(self):
        self._write_notebook("en", "nb_en.ipynb", [_figure_cell(
            "figcell", 1,
            extra_outputs=[{"output_type": "error", "ename": "LinAlgError",
                            "evalue": "SVD did not converge", "traceback": []}],
        )])
        message = self._refuse()
        self.assertIn("nb_en.ipynb", message)
        self.assertEqual(self._exported(), [])

    def test_a_notebook_with_no_stored_outputs_ends_the_run(self):
        # It contributed nothing, so the run did not refresh it, a gap, not a
        # success with a warning in it.
        self._write_notebook("en", "nb_en.ipynb", [
            {"cell_type": "code", "id": "figcell", "execution_count": None,
             "outputs": []},
        ])
        message = self._refuse()
        self.assertIn("nb_en.ipynb", message)

    #, the two languages ---------------------------------------------------

    def test_a_turkish_run_leaves_the_english_export_untouched(self):
        """Both languages produce the same positional filenames.

        While they shared a directory, a ``--lang tr`` run deleted the English
        figures and replaced the English tables with Turkish ones under
        identical names, silently and with exit status 0.
        """
        self._write_notebook("en", "nb_en.ipynb", [_figure_cell("figcell", 1)])
        self._run()
        english = self.module._export_dirs("en")[0] / self.export_name
        before = english.read_bytes()

        # Distinct bytes, or an overwrite would be invisible.
        self._write_notebook("tr", "nb_tr.ipynb",
                             [_figure_cell("trcell", 1, png_b64=_OTHER_PNG_B64)])
        self._run(lang="tr")

        self.assertEqual(
            english.read_bytes(), before,
            "the English export is the record the thesis numbers were "
            "transcribed from; a Turkish run must not be able to reach it",
        )
        turkish = self.module._export_dirs("tr")[0] / self.export_name
        self.assertEqual(turkish.read_bytes(), base64.b64decode(_OTHER_PNG_B64))

    def test_a_turkish_run_never_writes_thesis_figures(self):
        # The thesis is English, and THESIS_FIGURE_MAP's pinned ids are the
        # English notebooks', the Turkish mirrors carry different ones.
        self._write_notebook("tr", "nb_tr.ipynb", [_figure_cell("trcell", 1)])
        self._run(lang="tr")
        self.assertFalse((self.module.THESIS_FIG_DIR / self.thesis_figure).exists())


class TestModelDerivedNotebooksAreJudgedOnProvenance(_PipelineCase):
    """The provenance gate on the notebooks whose numbers come from models.

    ``_audit_stored_outputs`` judges a notebook against itself, so it cannot see
    a SOURCE change made after the run: on the default no-``--execute`` path the
    script harvested whatever notebook 04/05 happened to have stored, produced
    under any earlier source tree, and copied the mapped figures straight into
    thesis/latex/figures. ``_stale_model_inputs`` closes that last hop by asking
    the same predicate the notebooks' own ``if ckpt:`` branches use.

    Nothing covered it. With ``MODEL_DERIVED_PREFIXES = frozenset()``, the
    behaviour before the gate existed, the whole suite stayed green while a run
    rebuilt notebook 04's five mapped thesis figures from outputs produced under
    a superseded source tree, and run_all_experiments.sh reported PIPELINE
    COMPLETED SUCCESSFULLY over them.

    Driven through run_pipeline, on the same throwaway tree as the guards above:
    a gate that decides correctly and is then not acted on stops nothing, and
    the entry point is also the one part of this that does not change shape as
    the refusal is reworded or resplit internally.

    The verdict is left to ``src.tuning``'s own predicates rather than to
    stand-ins, because agreeing with those predicates IS the property under
    test: a fake would keep certifying records long after the export and the
    notebooks had stopped asking the same question.
    """

    PREFIX = "NB04"

    def setUp(self):
        super().setUp()
        self.model_dir = self.base / "results" / "models"
        self.model_dir.mkdir(parents=True)
        ckpt_dir = self.base / "results" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        # Redirected before anything is written: a stray write into the real
        # results/checkpoints/ would corrupt the very provenance records the
        # thesis numbers are certified by.
        _patch_attr(self, T, "_CHECKPOINT_DIR", ckpt_dir)
        self.module.MODEL_DIR = self.model_dir
        _freeze_provenance(self)
        # _stale_model_inputs puts BASE_DIR, here a temporary directory, on
        # sys.path so it can import src.tuning; the entry would otherwise
        # outlive the directory it names.
        saved = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), saved))
        # Both mirrors, because there is only one results/models and the export
        # reads the model filenames out of the notebooks' own source. The
        # outputs are clean, complete and produced by the pinned cell, so the
        # provenance records are the only thing left that can decide the run.
        for lang, nb_file in (("en", "nb_en.ipynb"), ("tr", "nb_tr.ipynb")):
            self._write_notebook(
                lang, nb_file,
                [_figure_cell("figcell", 1, source=_SAVE_CELL_SOURCE)],
            )

    #, the run that must succeed ------------------------------------------

    def test_a_current_record_lets_the_notebook_through(self):
        # Without this every refusal below is satisfiable by a gate that refuses
        # everything, which would leave the thesis figures unbuildable.
        _record_current_model_inputs(self.model_dir)
        self._run()
        self.assertEqual(self._exported(), [self.export_name])
        self.assertTrue((self.module.THESIS_FIG_DIR / self.thesis_figure).exists())

    def test_a_leftover_file_no_notebook_loads_does_not_end_the_run(self):
        # results/models also accumulates files no current run writes, from
        # earlier revisions; demanding a provenance record from those would be a
        # gate nothing could satisfy.
        _record_current_model_inputs(self.model_dir)
        (self.model_dir / "leftover_from_an_old_revision.joblib").write_bytes(b"x")
        self._run()
        self.assertTrue((self.module.THESIS_FIG_DIR / self.thesis_figure).exists())

    #, and the ways it must not --------------------------------------------

    def test_a_stamped_artifact_the_scan_did_not_name_is_still_judged(self):
        # Which files get judged is read out of the notebooks' own source, so
        # any save cell that stops spelling MODEL_DIR / "<name>" the way
        # _loaded_model_artifacts parses it, a variable, a helper, a rename,
        # would drop that file out of the judged set while its provenance record
        # sits right beside it, and the gate would narrow itself in silence.
        # The sidecar is the floor: a stamped file stays judged whatever the
        # scan managed to name, so the set can only ever widen.
        for lang, nb_file in (("en", "nb_en.ipynb"), ("tr", "nb_tr.ipynb")):
            self._write_notebook(lang, nb_file, [_figure_cell(
                "figcell", 1,
                source=('joblib.dump(model, MODEL_DIR / "a_name_the_scan_reads.joblib")\n',
                        "plt.show()\n"))])
        _, artifact = _record_current_model_inputs(self.model_dir)
        _pretend_source_changed(self)

        self._refuse()
        self.assertIn(
            artifact, self.output,
            "the stale model file carries a provenance record and was not named "
            "by the scan, so nothing but the sidecar floor keeps it judged; "
            "without it the run refuses on the checkpoint alone and the file "
            "the thesis figures were fitted from is never mentioned",
        )
        self.assertFalse((self.module.THESIS_FIG_DIR / self.thesis_figure).exists())

    def test_a_stale_record_ends_the_run_and_writes_no_thesis_figure(self):
        checkpoint, artifact = _record_current_model_inputs(self.model_dir)
        _pretend_source_changed(self)
        message = self._refuse()
        self.assertIn("nb_en.ipynb", message)
        # Naming the records is the whole remedy: which of 31 checkpoints and 16
        # model files has to be rebuilt is not guessable from a count.
        self.assertIn(checkpoint, self.output)
        self.assertIn(artifact, self.output)
        self.assertEqual(self._exported(), [])
        self.assertFalse((self.module.THESIS_FIG_DIR / self.thesis_figure).exists())

    def test_a_model_that_lost_its_provenance_record_ends_the_run(self):
        # Notebook 05 refuses to simulate with a model carrying no sidecar; the
        # export has to refuse the same file, or the two halves of one contract
        # disagree and the scheduling figures reach the thesis anyway.
        _, artifact = _record_current_model_inputs(self.model_dir)
        (self.model_dir / (artifact + ".provenance.json")).unlink()
        message = self._refuse()
        self.assertIn("nb_en.ipynb", message)
        self.assertIn(artifact, self.output)
        self.assertFalse((self.module.THESIS_FIG_DIR / self.thesis_figure).exists())

    def test_no_provenance_record_at_all_ends_the_run(self):
        # Nothing to judge must not read as nothing stale: emptying the records
        # is exactly what an out-of-date tree can do, and it would otherwise
        # reopen the hole in silence.
        message = self._refuse()
        self.assertIn("nb_en.ipynb", message)
        self.assertFalse((self.module.THESIS_FIG_DIR / self.thesis_figure).exists())

    def test_an_unusable_src_tuning_ends_the_run(self):
        # Unverifiable is not the same as current: swallowing the ImportError
        # would turn an environment that cannot answer the question into one
        # that answers "yes".
        _record_current_model_inputs(self.model_dir)
        with mock.patch.dict(sys.modules, {"src.tuning": None}):
            self._refuse()
        self.assertFalse((self.module.THESIS_FIG_DIR / self.thesis_figure).exists())

    #, the two languages ---------------------------------------------------

    def test_a_turkish_run_is_judged_too(self):
        # The Turkish notebooks report the same model results out of the same
        # results/models, and the TR export is an artifact set in its own right;
        # only the copy into the thesis is English-only.
        _record_current_model_inputs(self.model_dir)
        _pretend_source_changed(self)
        message = self._refuse(lang="tr")
        self.assertIn("nb_tr.ipynb", message)
        self.assertEqual(self._exported("tr"), [])


class TestTheJudgedNotebooksAreTheModelDerivedOnes(unittest.TestCase):
    """MODEL_DERIVED_PREFIXES is maintained by hand beside the notebook lists.

    A prefix renamed in NOTEBOOKS_EN/TR and not here leaves the gate matching
    nothing for that notebook, the same hole as deleting it, and just as
    silent, because every other gate still passes.
    """

    def setUp(self):
        self.module = _load_module()

    def test_notebooks_04_and_05_are_judged_and_nothing_else_is(self):
        # 01-03 (data overview, workload analysis, feature engineering) read
        # neither a checkpoint nor a model artifact, so refusing them because a
        # model went stale would be exactly the cry-wolf warning the gate's own
        # docstring cautions against.
        for notebooks in (self.module.NOTEBOOKS_EN, self.module.NOTEBOOKS_TR):
            model_derived = {prefix for nb_file, prefix, *_ in notebooks
                             if nb_file.startswith(("04", "05"))}
            self.assertEqual(model_derived, set(self.module.MODEL_DERIVED_PREFIXES))


if __name__ == "__main__":
    unittest.main()
