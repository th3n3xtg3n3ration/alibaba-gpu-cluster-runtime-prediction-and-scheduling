"""The guards the notebooks have to honour, checked in the notebooks.

Two of them live in src/ and are tested there, ``model_artifact_is_current``
in tests/test_checkpoint_currency.py, the degenerate-predictor refusal in
tests/test_regression_guards.py, and both were still fully bypassable,
because a guard is only worth what its consumer does with it and the consumer
here is notebook JSON that no test read:

* the currency guard was enforced at the WRITER only (notebook 04's save
  cells). Notebook 05 loaded whatever .joblib / .pth happened to be on disk, so
  a model fitted before a feature-engineering fix could still produce every
  scheduling figure and table in the thesis, the exact pairing the guard's own
  docstring cites, "notebook 05 simulated with those stale models". Every
  artifact in results/models is stale as of this writing, which is the normal
  state after a source fix, so this is not hypothetical.

* the refusal is a REPORTABLE RESULT, not a crash and not a silent skip. A
  predictor that emits one value for every job (cnn_lstm_numeric_seq.pth
  predicts 4128.124 for all 16,437 test jobs) ranks nothing, so the policy is
  FIFO wearing a model's name. It must be recorded as excluded and appear as an
  excluded row; reporting it with FIFO's numbers and a 0.00% improvement would
  state that it was evaluated as a predictor and found to gain nothing, which
  never happened. Leaving the exception uncaught is the opposite failure: the
  comparison then aborts partway through and the other 27 policies are lost.

The third is not a src/ guard at all but a contract between two notebooks, and
it is the one that turned the currency gate into a deadlock the moment it
shipped: notebook 05 refuses any artifact without a provenance sidecar, and
notebook 04 wrote ten of the artifacts on that list, the four LSTM scalers,
the two median baselines, the three Alibaba-estimate lookups and the ablation
model, with no sidecar at all. The file exists, so the gate's "absent is a
legitimate skip" branch does not fire; the currency check then rejects it; and
the refusal tells the reader to re-run notebook 04, which writes no sidecar for
those files either. All four notebook 05 variants would have raised before
making a single prediction, with no reachable remedy, and the whole scheduler
chapter with them. Only the reader half had a test. So the gate list is checked
against what notebook 04 actually stamps, not against what its message claims.

The last is notebook 04's own zero-floor convention (``predict_nonneg``): a raw
``.predict()`` in one cell against ``np.maximum(..., 0)`` in another published
two different test MAEs for one model out of a single run (4857.24 s vs
4857.41 s).

The notebooks are read as JSON and parsed, never executed.
"""
import ast
import json
import re
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The four scheduler-evaluation notebooks. EN and TR mirror each other cell
#: for cell, and both cluster sizes replay the same policies, so a guard that
#: holds in one of them says nothing about the other three.
_NOTEBOOK_05 = (
    "notebooks/en/05_scheduler_evaluation_32_gpu.ipynb",
    "notebooks/en/05_scheduler_evaluation_256_gpu.ipynb",
    "notebooks/tr/05_gorev_zamanlayici_degerlendirme_32_gpu.ipynb",
    "notebooks/tr/05_gorev_zamanlayici_degerlendirme_256_gpu.ipynb",
)

#: The two notebook-04 mirrors. Every artifact in results/models comes from one
#: of their cells, written there or by the training code they call, so whatever
#: notebook 05 gates has to be satisfiable from here.
_NOTEBOOK_04 = (
    "notebooks/en/04_runtime_prediction_models.ipynb",
    "notebooks/tr/04_calisma_zamani_tahmin_modelleri.ipynb",
)

_LOADERS = ("joblib.load", "torch.load", "_load")
_WRITERS = ("joblib.dump", "torch.save")


def _code_cells(rel_path):
    """(cell id, source, parsed tree) for every code cell of a notebook."""
    notebook = json.loads((_REPO_ROOT / rel_path).read_text(encoding="utf-8"))
    cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source") or "")
        cells.append((cell.get("id"), source, ast.parse(source)))
    return cells


def _call_name(node):
    """Dotted name of a call target, e.g. 'joblib.load' or '_load'."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _artifact_name(node, local_paths):
    """Resolve an argument to the ``results/models`` file it names, or None.

    Both spellings the notebooks use are accepted: ``MODEL_DIR / "x.joblib"``
    written inline, and the ``_dest = MODEL_DIR / "x.pth"`` variable the
    checkpoint cells bind first and then pass to torch.save and
    record_model_artifact in turn. ``str(...)`` is unwrapped because the
    per-seed template is handed to finalize_dl_model as a string.
    """
    if (isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.left, ast.Name)
            and node.left.id == "MODEL_DIR"
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, str)):
        return node.right.value
    if isinstance(node, ast.Name):
        return local_paths.get(node.id)
    if isinstance(node, ast.Call) and _call_name(node) == "str" and node.args:
        return _artifact_name(node.args[0], local_paths)
    return None


def _path_patterns(rel_path):
    """MODEL_DIR paths built with a placeholder, normalised to ``{}``.

    The per-seed LSTM checkpoints are the only artifacts neither notebook
    spells out in full: notebook 04 passes a ``{seed}`` template down to
    finalize_dl_model, notebook 05 rebuilds the same names with an f-string.
    Normalising both to one shape is what lets the two be compared at all.
    """
    patterns = set()
    for _cid, _src, tree in _code_cells(rel_path):
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.Div)
                    and isinstance(node.left, ast.Name)
                    and node.left.id == "MODEL_DIR"):
                continue
            right = node.right
            if isinstance(right, ast.JoinedStr):
                patterns.add("".join(
                    str(part.value) if isinstance(part, ast.Constant) else "{}"
                    for part in right.values
                ))
            elif (isinstance(right, ast.Constant)
                    and isinstance(right.value, str)
                    and "{" in right.value):
                patterns.add(re.sub(r"\{[^}]*\}", "{}", right.value))
    return patterns


def _artifact_writes(rel_path):
    """(written -> producing cell, stamped names) for one notebook-04 mirror.

    A write is joblib.dump / torch.save into MODEL_DIR; a stamp is
    record_model_artifact on the same destination in the same cell. The pairing
    is per cell rather than per call site because record_model_artifact is
    invoked on the destination variable in the checkpoint cells, so nothing
    outside the cell can tell which file a stamp refers to.
    """
    written, stamped = {}, set()
    for cell_id, _src, tree in _code_cells(rel_path):
        local_paths = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                name = _artifact_name(node.value, local_paths)
                if name:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            local_paths[target.id] = name
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = _call_name(node)
            if called in _WRITERS and len(node.args) >= 2:
                name = _artifact_name(node.args[1], local_paths)
                if name:
                    written.setdefault(name, cell_id)
            elif called == "record_model_artifact" and node.args:
                name = _artifact_name(node.args[0], local_paths)
                if name:
                    stamped.add(name)
    return written, stamped


def _gated_artifact_names(rel_path):
    """The artifact names actually HANDED to the gate, not merely listed.

    Resolving the call rather than every ``*_ARTIFACTS`` assignment is the
    point: the list literal can stay intact while the call passes something
    else, an empty list, say, and a test that reads the assignment reports
    full coverage over a gate that checks nothing.
    """
    lists, gate_calls = {}, []
    for _cid, _src, tree in _code_cells(rel_path):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
                names = {
                    e.value for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                }
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("_ARTIFACTS"):
                        lists[target.id] = names
            elif (isinstance(node, ast.Call)
                    and _call_name(node) == "_reject_stale_artifacts"):
                gate_calls.append(node)

    gated = set()
    for call in gate_calls:
        if not call.args:
            continue
        for node in ast.walk(call.args[0]):
            if isinstance(node, ast.Name) and node.id in lists:
                gated |= lists[node.id]
            else:
                # Paths spelled out in the call itself count just the same.
                inline = _artifact_name(node, {})
                if inline:
                    gated.add(inline)
    return gated, gate_calls


def _outside_function_bodies(tree):
    """Walk a cell's statements, skipping anything inside a def.

    A loader call in a helper's body says nothing about ordering; what matters
    is where the helper is CALLED.
    """
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


class TestNotebook05GatesEveryArtifactLoad(unittest.TestCase):

    def test_the_gate_is_defined_and_actually_refuses(self):
        for rel_path in _NOTEBOOK_05:
            with self.subTest(notebook=rel_path):
                gates = [
                    node
                    for _cid, _src, tree in _code_cells(rel_path)
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "_reject_stale_artifacts"
                ]
                self.assertEqual(len(gates), 1, "exactly one gate helper is expected")
                gate = gates[0]
                # A Call node, not a substring of the dumped tree: the gate's
                # own docstring names the guard, so a dump search is satisfied
                # by prose and stayed green when the call was swapped for
                # `path.exists()`, the presence-only check this whole cell
                # exists to replace.
                self.assertTrue(
                    any(isinstance(n, ast.Call)
                        and _call_name(n) == "model_artifact_is_current"
                        for n in ast.walk(gate)),
                    "the gate must call the guard, not re-implement a weaker test",
                )
                # Refusing rather than warning is the point: everything below
                # is downstream of these files and nothing downstream can see
                # which weights produced a prediction.
                self.assertTrue(
                    [n for stmt in gate.body if isinstance(stmt, ast.If)
                     for n in ast.walk(stmt) if isinstance(n, ast.Raise)],
                    "a warning here is a warning nobody can act on once the "
                    "outputs exist",
                )
                # ...but conditional on something being rejected. A gate that
                # raises whatever it is handed refuses every run, which is the
                # deadlock shape this notebook already shipped once.
                self.assertFalse(
                    [n for n in gate.body if isinstance(n, ast.Raise)],
                    "an unconditional raise refuses artifacts that are current",
                )

    def test_every_named_artifact_is_covered_by_the_gate(self):
        for rel_path in _NOTEBOOK_05:
            with self.subTest(notebook=rel_path):
                named = {}
                for cell_id, _src, tree in _code_cells(rel_path):
                    for node in ast.walk(tree):
                        # MODEL_DIR / "<name>", every artifact path the
                        # notebook spells out.
                        name = _artifact_name(node, {})
                        if name:
                            named.setdefault(name, cell_id)

                # Read off the gate call, because that is what runs. Built from
                # the `_ARTIFACTS = [...]` assignments instead, this reported
                # all 25 required artifacts covered while the call beside them
                # passed an empty list.
                gated, gate_calls = _gated_artifact_names(rel_path)
                self.assertTrue(gate_calls, "nothing calls the gate")
                self.assertTrue(named, "no artifact path found, has the cell moved?")
                ungated = sorted(set(named) - gated)
                self.assertEqual(
                    ungated, [],
                    "these artifacts are loaded but never checked for currency, "
                    "so a model fitted by different source code can produce the "
                    f"scheduling results: {ungated}",
                )

    def test_the_gate_runs_before_the_first_load_in_each_loading_cell(self):
        """Checked in one pass before the first load, in every cell that loads.

        Order is the whole guarantee. A currency check placed after the loads
        would name the stale files only once their predictions already exist,
        and a check in one cell says nothing about a second cell that loads
        more artifacts later (the per-seed robustness cell does exactly that).
        """
        for rel_path in _NOTEBOOK_05:
            for cell_id, source, tree in _code_cells(rel_path):
                if "MODEL_DIR" not in source:
                    continue
                loads = [
                    node.lineno
                    for node in _outside_function_bodies(tree)
                    if isinstance(node, ast.Call) and _call_name(node) in _LOADERS
                ]
                if not loads:
                    continue
                with self.subTest(notebook=rel_path, cell=cell_id):
                    gate_calls = [
                        node.lineno
                        for node in _outside_function_bodies(tree)
                        if isinstance(node, ast.Call)
                        and _call_name(node) == "_reject_stale_artifacts"
                    ]
                    self.assertTrue(
                        gate_calls,
                        f"cell {cell_id} loads model artifacts without asking "
                        "whether they are current",
                    )
                    self.assertLess(
                        min(gate_calls), min(loads),
                        f"cell {cell_id} loads before it checks",
                    )


class TestNotebook04CanSatisfyTheGate(unittest.TestCase):
    """The writer half of the currency contract.

    Notebook 05's gate is only a gate while notebook 04 can clear it. Ten
    artifacts on the required/optional lists were dumped with no sidecar, which
    is not staleness but a deadlock: the file exists, so the gate's "absent is a
    legitimate skip" branch does not fire; the currency check rejects it; and
    the refusal's own instruction, re-run notebook 04, writes no sidecar for
    those files either. Nothing could clear it, and the scheduler chapter had no
    reachable path back. Only the reader half was tested, which is how a gate
    could ship with nothing able to satisfy it.
    """

    def test_every_gated_artifact_is_stamped_by_notebook_04(self):
        gated = set()
        for rel_path in _NOTEBOOK_05:
            names, _calls = _gated_artifact_names(rel_path)
            gated |= names
        self.assertTrue(gated, "the gate list is empty, has the cell moved?")

        for rel_path in _NOTEBOOK_04:
            with self.subTest(notebook=rel_path):
                _written, stamped = _artifact_writes(rel_path)
                unwritable = sorted(gated - stamped)
                self.assertEqual(
                    unwritable, [],
                    "notebook 05 refuses these artifacts unless they carry a "
                    "provenance sidecar, and no cell here records one for them, "
                    "so re-running this notebook cannot clear the refusal: "
                    f"{unwritable}",
                )

    def test_every_artifact_notebook_04_writes_is_stamped(self):
        """The same contract read from the writer's side.

        A new dump added without record_model_artifact beside it does not fail
        anywhere until notebook 05 refuses to start, by which point the file
        looks perfectly ordinary on disk.
        """
        for rel_path in _NOTEBOOK_04:
            with self.subTest(notebook=rel_path):
                written, stamped = _artifact_writes(rel_path)
                self.assertTrue(written, "no model dump found, has the cell moved?")
                unstamped = sorted(
                    f"{name} (cell {cell})"
                    for name, cell in written.items() if name not in stamped
                )
                self.assertEqual(
                    unstamped, [],
                    "these artifacts are written without the sidecar the "
                    f"consumer demands: {unstamped}",
                )

    def test_the_per_seed_checkpoints_are_written_under_the_names_gated(self):
        """The multi-seed robustness check gates files built from an f-string.

        Those are not on the artifact lists and are written by
        finalize_dl_model rather than by a dump cell, so the two halves agree
        only if the filename patterns match. They already disagreed once,
        notebook 05 built the paths from (0, 1, 2) while notebook 04 saved by
        seed VALUE, and the cell reported "not found" on every run without a
        line of it ever executing.
        """
        for consumer in _NOTEBOOK_05:
            gated = _path_patterns(consumer)
            with self.subTest(notebook=consumer):
                self.assertTrue(gated, "no per-seed checkpoint path found")
            for writer in _NOTEBOOK_04:
                with self.subTest(notebook=consumer, writer=writer):
                    unwritten = sorted(gated - _path_patterns(writer))
                    self.assertEqual(
                        unwritten, [],
                        "no save template in notebook 04 produces these names, "
                        f"so nothing can ever satisfy the gate on them: {unwritten}",
                    )


class TestNotebook04FloorsEveryReportedPrediction(unittest.TestCase):
    """A negative runtime is not a prediction, it is an artifact of the model.

    Notebook 04 floors predictions at zero through one helper, ``predict_nonneg``.
    The convention is enforced nowhere else: a raw ``.predict()`` in one cell
    against ``np.maximum(..., 0)`` in another published two different test MAEs
    for the same model out of a single run (4857.24 s vs 4857.41 s).
    """

    #: Cells that call .predict() directly, with the reason each one does.
    #: ecd03 is deliberate, it floors at 1 s for a log axis, which is a
    #: stricter floor than zero and is stated in the cell. The other two are
    #: recorded here because they exist, not because they are agreed: ablcell01
    #: computes a Spearman rho printed to three decimals beside floored
    #: MAE/MdAE/R2 in the same table row, and ecd05 labels a thesis figure with
    #: a residual mean taken from unfloored predictions. Both belong to
    #: notebook 04's owner; listing them keeps the exemption stated rather than
    #: assumed, and keeps a NEW unfloored call from joining them unnoticed.
    _RAW_PREDICT_CELLS = {"ecd03", "ecd05", "ablcell01"}

    def _helper(self, tree):
        return [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "predict_nonneg"
        ]

    def test_the_helper_actually_floors_at_zero(self):
        for rel_path in _NOTEBOOK_04:
            with self.subTest(notebook=rel_path):
                helpers = [
                    node for _cid, _src, tree in _code_cells(rel_path)
                    for node in self._helper(tree)
                ]
                self.assertEqual(len(helpers), 1, "exactly one floor helper is expected")
                floors = [
                    node for node in ast.walk(helpers[0])
                    if isinstance(node, ast.Call) and _call_name(node) == "np.maximum"
                    and any(isinstance(a, ast.Constant) and a.value == 0
                            for a in node.args)
                ]
                self.assertTrue(
                    floors,
                    "predict_nonneg that does not floor is a plain .predict() "
                    "under a name promising otherwise",
                )

    def test_no_new_cell_predicts_without_the_floor(self):
        for rel_path in _NOTEBOOK_04:
            with self.subTest(notebook=rel_path):
                raw = set()
                for cell_id, _src, tree in _code_cells(rel_path):
                    inside = [
                        (node.lineno, node.end_lineno) for node in self._helper(tree)
                    ]
                    for node in ast.walk(tree):
                        if (isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Attribute)
                                and node.func.attr == "predict"
                                and not any(start <= node.lineno <= end
                                            for start, end in inside)):
                            raw.add(cell_id)
                unlisted = sorted(raw - self._RAW_PREDICT_CELLS)
                self.assertEqual(
                    unlisted, [],
                    "these cells predict without the zero floor the reported "
                    "metrics apply, so one run can publish two numbers for one "
                    f"model: {unlisted}",
                )


class TestNotebook05ReportsTheDegeneratePolicyAsExcluded(unittest.TestCase):

    def _policy_loop(self, rel_path):
        """The cell that replays every policy, and its parsed tree."""
        for cell_id, source, tree in _code_cells(rel_path):
            for node in ast.walk(tree):
                if (isinstance(node, ast.For)
                        and isinstance(node.target, ast.Name)
                        and node.target.id == "policy"
                        and isinstance(node.iter, ast.Name)
                        and node.iter.id == "POLICIES"):
                    return cell_id, source, node
        self.fail(f"{rel_path}: the policy loop was not found")

    def test_the_refusal_is_imported_by_name(self):
        for rel_path in _NOTEBOOK_05:
            with self.subTest(notebook=rel_path):
                imported = {
                    alias.name
                    for _cid, _src, tree in _code_cells(rel_path)
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    for alias in node.names
                }
                self.assertIn("DegeneratePredictionError", imported)

    def test_the_loop_catches_only_the_refusal(self):
        for rel_path in _NOTEBOOK_05:
            with self.subTest(notebook=rel_path):
                _cid, _src, loop = self._policy_loop(rel_path)
                handlers = [
                    h for node in ast.walk(loop) if isinstance(node, ast.Try)
                    for h in node.handlers
                ]
                self.assertTrue(handlers, "the loop must survive a refused policy")
                for handler in handlers:
                    self.assertIsInstance(
                        handler.type, ast.Name,
                        "a bare except (or a tuple including one) would swallow "
                        "a missing prediction column and an unplaceable job too, "
                        "and file a broken pipeline as a reportable exclusion",
                    )
                    self.assertEqual(handler.type.id, "DegeneratePredictionError")

    def test_the_refusal_is_recorded_and_the_comparison_continues(self):
        for rel_path in _NOTEBOOK_05:
            with self.subTest(notebook=rel_path):
                _cid, _src, loop = self._policy_loop(rel_path)
                handlers = [
                    h for node in ast.walk(loop) if isinstance(node, ast.Try)
                    for h in node.handlers
                ]
                self.assertTrue(
                    handlers,
                    "with no handler the refusal aborts the loop at policy 25 of "
                    "28 and the other 27 results are never produced",
                )
                handler = handlers[0]
                body = ast.dump(ast.Module(body=handler.body, type_ignores=[]))
                # Recorded: a refusal that is only printed is a skip, and the
                # results table would be silently one row short.
                self.assertIn("REFUSED_POLICIES", body)
                # Continued: 27 other policies are waiting behind this one.
                self.assertTrue(
                    any(isinstance(n, ast.Continue) for n in ast.walk(handler)),
                    "the refused policy must not take the whole comparison down",
                )
                self.assertFalse(
                    any(isinstance(n, ast.Raise) for n in ast.walk(handler)),
                    "re-raising aborts the run at policy 25 of 28",
                )

    def test_the_results_table_carries_an_excluded_row_not_a_zero_row(self):
        """A refused policy produced no simulated rows, so the table has to add
        it back deliberately, with empty metrics and a stated reason. A 0.00%
        improvement row would say it was evaluated as a predictor and tied
        FIFO; it was never evaluated at all.
        """
        for rel_path in _NOTEBOOK_05:
            with self.subTest(notebook=rel_path):
                refused_rows = []
                for _cid, source, tree in _code_cells(rel_path):
                    if "summary_rows" not in source:
                        continue
                    for node in ast.walk(tree):
                        if (isinstance(node, ast.For)
                                and "REFUSED_POLICIES" in ast.dump(node.iter)):
                            refused_rows.append(ast.unparse(node))

                self.assertTrue(
                    refused_rows,
                    "nothing appends the refused policies to the summary table, "
                    "so the exclusion never reaches the results",
                )
                for block in refused_rows:
                    self.assertIn("summary_rows.append", block)
                    self.assertIn("EXCLUDED", block)
                    self.assertIn("np.nan", block,
                                  "an excluded policy has no metrics to report")

    def test_the_refusal_can_name_the_policy_it_refused(self):
        # Every prediction-driven policy shares one scheduler class, so the
        # instance has to be told which of the 28 it is standing in for,
        # otherwise the excluded row cannot say what was excluded.
        for rel_path in _NOTEBOOK_05:
            with self.subTest(notebook=rel_path):
                constructions = [
                    node
                    for _cid, _src, tree in _code_cells(rel_path)
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and _call_name(node) == "SJFPredScheduler"
                ]
                self.assertTrue(constructions)
                for call in constructions:
                    named = ([kw for kw in call.keywords if kw.arg == "policy_name"]
                             or call.args)
                    self.assertTrue(
                        named,
                        "SJFPredScheduler built without a policy name: a refusal "
                        "would report the class name instead of the policy",
                    )


#: The results tables of notebook 04, per mirror. Six cell ids are shared; the
#: Experiment E and F tables were given different ids in the two languages.
_SUMMARY_CELLS = {
    "notebooks/en/04_runtime_prediction_models.ipynb": (
        "cd11", "cd19", "693a11bd", "c0607428", "db973a20", "95d3ac8e",
        "cd31", "ecd07",
    ),
    "notebooks/tr/04_calisma_zamani_tahmin_modelleri.ipynb": (
        "cd11", "cd19", "693a11bd", "c0607428", "8f518fd6", "350874ad",
        "cd31", "ecd07",
    ),
}

#: Two side studies that also rank on MAE and do not carry the verdict: the
#: feature-ablation table (12i) and the rolling-origin table (12k), both of
#: which re-fit one tuned tree model over different columns or time windows.
#: Listed so the discovery check below stays honest about where the wiring
#: currently stops instead of quietly passing over them.
_TABLES_THE_VERDICT_HAS_NOT_REACHED = {"ablcell01", "fc06cell01"}


def _metric_rows(tree):
    """Row dicts built out of a metrics dict, e.g. ``{"Test MAE (s)": m["mae"]}``."""
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and any(isinstance(value, ast.Subscript)
                and isinstance(value.slice, ast.Constant)
                and value.slice.value == "mae"
                for value in node.values)
    ]


def _calls_to(tree, name):
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _call_name(node) == name]


class TestNotebook04ReportsTheCollapseVerdictBesideTheMetrics(unittest.TestCase):
    """The modelling half of the same contract the class above tests.

    ``evaluate_regression`` had recorded ``pred_std`` and ``pred_unique_frac``
    into every checkpoint for a while before anything read them, and that is the
    whole defect: CNN-LSTM (Numeric Sequence) predicted 4128.124 s for all
    16,437 test jobs and was printed in notebook 04's comparison tables on its
    MAE alone, mid-pack among the models that had learned something, while
    notebook 05 refused to simulate with it. One notebook's excluded model was
    the other notebook's ordinary row.

    Nothing read the notebooks, so the reading could be, and was, dropped
    without a single failure. These assert it as source: the predicate is
    imported, the shared label helper really asks it, its three-valued answer is
    not flattened on the way to the table, and every row a table builds out of a
    metrics dict carries a verdict.

    The notebooks are read as JSON and parsed, never executed.
    """

    def _cells(self, rel_path):
        return {cid: tree for cid, _src, tree in _code_cells(rel_path)}

    def test_the_verdict_is_imported_by_name(self):
        for rel_path in _NOTEBOOK_04:
            with self.subTest(notebook=rel_path):
                imported = {
                    alias.name
                    for _cid, _src, tree in _code_cells(rel_path)
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    and node.module == "src.models.evaluation"
                    for alias in node.names
                }
                self.assertIn("is_degenerate_prediction", imported)

    def test_the_shared_label_helper_asks_the_real_predicate(self):
        # Every table routes through degeneracy_label, so a helper that stopped
        # calling the predicate would leave all eight tables printing a verdict
        # column that decides nothing, and every other test here would still
        # pass.
        for rel_path in _NOTEBOOK_04:
            with self.subTest(notebook=rel_path):
                helpers = [
                    node
                    for _cid, _src, tree in _code_cells(rel_path)
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "degeneracy_label"
                ]
                self.assertEqual(len(helpers), 1,
                                 "exactly one shared verdict helper is expected")
                self.assertTrue(
                    _calls_to(helpers[0], "is_degenerate_prediction"),
                    "degeneracy_label that does not ask is_degenerate_prediction "
                    "is a column of labels with nothing behind them",
                )

    def test_the_unknown_verdict_is_not_flattened_into_a_clean_one(self):
        """``None``, no spread evidence, must not print as ``False``.

        Every deep-learning checkpoint on disk predates the spread fields, so
        the unknown state is the one the tables actually render today. Mapping
        it onto the "no" label would print a clean bill of health for exactly
        the model notebook 05 refuses.
        """
        for rel_path in _NOTEBOOK_04:
            with self.subTest(notebook=rel_path):
                maps = [
                    node.value
                    for _cid, _src, tree in _code_cells(rel_path)
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "DEGENERACY_LABELS"
                            for t in node.targets)
                ]
                self.assertEqual(len(maps), 1)
                labels = {key.value: value.value for key, value in
                          zip(maps[0].keys, maps[0].values)}
                self.assertEqual(set(labels), {True, False, None})
                self.assertEqual(
                    len(set(labels.values())), 3,
                    "two verdicts sharing a label make one of them unreadable; "
                    f"got {labels}",
                )

    def test_every_results_table_reports_the_verdict(self):
        for rel_path, cell_ids in _SUMMARY_CELLS.items():
            cells = self._cells(rel_path)
            for cell_id in cell_ids:
                with self.subTest(notebook=rel_path, cell=cell_id):
                    # Asserted rather than skipped: a renamed cell would
                    # otherwise turn this whole check into a loop over nothing.
                    self.assertIn(cell_id, cells,
                                  "the results table this test names is gone")
                    self.assertTrue(
                        _calls_to(cells[cell_id], "degeneracy_label"),
                        "this table ranks models on MAE with no way to tell a "
                        "collapsed model from one that learned something",
                    )

    def test_every_row_built_from_a_metrics_dict_carries_a_verdict(self):
        # A verdict on the first row only is the failure shape to expect here:
        # the tables are hand-written row by row, and Experiment B's table alone
        # has thirteen of them.
        for rel_path, cell_ids in _SUMMARY_CELLS.items():
            cells = self._cells(rel_path)
            for cell_id in cell_ids:
                if cell_id not in cells:
                    continue  # a renamed cell is the test above's to report
                rows = _metric_rows(cells[cell_id])
                if not rows:
                    continue  # the DL comparison builds its rows in a loop
                with self.subTest(notebook=rel_path, cell=cell_id):
                    self.assertEqual(
                        len(_calls_to(cells[cell_id], "degeneracy_label")),
                        len(rows),
                        "a model row without a verdict is reported as an "
                        "ordinary predictor whatever its predictions were",
                    )

    def test_no_new_table_ranks_on_mae_without_the_verdict(self):
        for rel_path, cell_ids in _SUMMARY_CELLS.items():
            with self.subTest(notebook=rel_path):
                unwired = {
                    cid for cid, _src, tree in _code_cells(rel_path)
                    if _metric_rows(tree) and not _calls_to(tree, "degeneracy_label")
                }
                self.assertEqual(
                    sorted(unwired - _TABLES_THE_VERDICT_HAS_NOT_REACHED), [],
                    "these cells build model rows out of a metrics dict and "
                    "print no collapse verdict beside them, which is the state "
                    "that ranked a constant predictor mid-pack: "
                    f"{sorted(unwired - _TABLES_THE_VERDICT_HAS_NOT_REACHED)}",
                )
                # The allowlist is not a place to park a table: an entry that
                # has since been wired up has to leave it, or it goes on
                # excusing the next one that has not been.
                self.assertEqual(
                    sorted(_TABLES_THE_VERDICT_HAS_NOT_REACHED - unwired), [],
                    "these cells now report the verdict and no longer need an "
                    "exemption",
                )


if __name__ == "__main__":
    unittest.main()
