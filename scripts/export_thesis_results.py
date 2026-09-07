#!/usr/bin/env python3
"""
export_thesis_results.py
========================
Automated extraction of all publication-grade figures (PNG) and benchmark tables (HTML)
directly from executed Jupyter Research Notebooks (01 to 05).

Ensures 100% mathematical and visual consistency between the notebooks and thesis artifacts.

In English mode this script is also the single mechanism that populates
``thesis/latex/figures/``: every extracted figure whose (notebook, position) pair
appears in ``THESIS_FIGURE_MAP`` is additionally written there under the exact
filename the LaTeX sources include. No manual copying or renaming is needed:
re-running the notebooks and then this script brings the thesis figures up to
date in one step. Turkish notebooks are never copied into the thesis (the
thesis is written in English), so ``--lang tr`` only refreshes that language's
export directory.

Each language exports into its OWN directory tree (see ``_export_dirs``): the
two languages produce the same positional filenames, so sharing one directory
meant a ``--lang tr`` run overwrote the English artifact set.

A notebook's stored outputs are exported only when they read as the record of
one clean top-to-bottom run (see ``_audit_stored_outputs``). A notebook holding
the output of a cell that raised, or of a cell that never ran, is skipped in
full, and nothing of it reaches the export directory or the thesis.

That audit judges a notebook against itself, so it cannot see a source change
made after the run. The notebooks whose figures and tables report model results
(04 and both 05 variants) are therefore also checked against the provenance
records the training code writes: ``_stale_model_inputs`` refuses them when a
checkpoint or a saved model artifact is no longer certified current by
``src.tuning``, the same predicate the notebooks' own ``if ckpt:`` branches and
notebook 05's ``_reject_stale_artifacts`` use, over the same set of files (see
``_loaded_model_artifacts``). Without it the default no-``--execute`` path
harvested whatever a notebook happened to have stored, produced under any
earlier source tree, and copied the mapped figures straight into
``thesis/latex/figures``.

Every notebook the script refuses, whether for untrusted stored outputs, stale
model inputs, a figure count that no longer matches ``EXPECTED_FIGURE_COUNT``,
a figure position no longer produced by the cell ``THESIS_FIGURE_MAP`` pins it
to, or no stored outputs at all, is collected and re-reported at the end of the
run. The script then exits non-zero so ``run_all_experiments.sh`` stops instead
of reporting success over an export, and a thesis figure directory, that was
not refreshed. The export files are
buffered during extraction and written only after the figure count is verified,
so a notebook refused up to that point leaves nothing behind in the export
directory. It used to leave files whose content had shifted under unchanged
positional names. A figure-position refusal comes later and stops only the
thesis copy: the export filenames ARE positions, so that set stays internally
consistent even when a figure cell has moved.

Usage:
    python scripts/export_thesis_results.py                 # Fast extraction from existing outputs
    python scripts/export_thesis_results.py --execute       # Auto-executes empty notebooks if needed
    python scripts/export_thesis_results.py --force-execute # Re-runs all notebooks from scratch
    python scripts/export_thesis_results.py --lang tr       # Extracts from Turkish notebooks
"""

import argparse
import ast
import base64
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
EXPORT_DIR = BASE_DIR / "results" / "figures" / "thesis_export"

# Rebound per run by ``_export_dirs``; the English paths are the defaults
# because the English export is the one the thesis quotes.
PNG_DIR = EXPORT_DIR / "png"
HTML_DIR = EXPORT_DIR / "html"


def _export_dirs(lang: str) -> tuple[Path, Path]:
    """Return the (PNG, HTML) export directories for one language.

    The two languages produce the SAME positional filenames (``NB01`` …
    ``NB05_256GPU`` are reused by NOTEBOOKS_TR), and ``_clean_stale_exports``
    deletes by prefix with no language filter, so while both wrote into one
    directory a ``--lang tr`` run deleted the English figures and replaced the
    English tables with Turkish ones under the identical names, silently and
    with exit status 0. It was not even a like-for-like swap: the table position
    index differs between the mirrors, so NB04_Table10.html changed from one
    benchmark table to a different one, and nothing in a filename or in a file
    recorded which language had produced it.

    English keeps the documented top-level ``png/`` and ``html/`` directories:
    that set is git-tracked and is the record the thesis numbers were
    transcribed from. Turkish gets its own subtree beside it.
    """
    root = EXPORT_DIR if lang == "en" else EXPORT_DIR / lang
    return root / "png", root / "html"

HTML_STYLE_HEADER = (
    # charset declaration: these tables carry UTF-8, Turkish labels and unit
    # symbols, and browsers guessed the encoding without it.
    "<html><head><meta charset=\"utf-8\"><style>"
    "table{border-collapse:collapse;font-family:Arial,sans-serif;font-size:12px} "
    "td,th{border:1px solid #ccc;padding:6px 10px} th{background:#f0f0f0}"
    "</style></head><body><div>\n"
)
HTML_STYLE_FOOTER = "\n</div></body></html>"

NOTEBOOKS_EN = [
    ("01_data_overview.ipynb", "NB01", True, False),
    ("02_workload_analysis.ipynb", "NB02", True, False),
    ("03_feature_engineering.ipynb", "NB03", True, False),
    ("04_runtime_prediction_models.ipynb", "NB04", True, True),
    ("05_scheduler_evaluation_32_gpu.ipynb", "NB05_32GPU", True, True),
    ("05_scheduler_evaluation_256_gpu.ipynb", "NB05_256GPU", True, True),
]

NOTEBOOKS_TR = [
    ("01_veri_ozeti.ipynb", "NB01", True, False),
    ("02_is_yuku_analizi.ipynb", "NB02", True, False),
    ("03_ozellik_muhendisligi.ipynb", "NB03", True, False),
    ("04_calisma_zamani_tahmin_modelleri.ipynb", "NB04", True, True),
    ("05_gorev_zamanlayici_degerlendirme_32_gpu.ipynb", "NB05_32GPU", True, True),
    ("05_gorev_zamanlayici_degerlendirme_256_gpu.ipynb", "NB05_256GPU", True, True),
]

# Where the LaTeX sources expect their figures.
THESIS_FIG_DIR = BASE_DIR / "thesis" / "latex" / "figures"

# (notebook prefix, Nth figure produced by that notebook)
#     -> (id of the cell that produces it, thesis filename).
# The position is the order in which the figure appears when the notebook runs
# top to bottom, as the extraction loop below counts them. If a figure cell is
# added, removed or reordered, this table must be updated. A notebook's figures
# are copied into the thesis only when every position it expects was produced;
# a partial run is skipped with a warning rather than overwriting thesis figures.
#
# The cell id is pinned because position alone is not enough: reordering two
# figure cells leaves the count intact, so the map would write the wrong chart
# under the right filename and exit 0. The ids are the English notebooks'; the
# Turkish mirrors differ for notebooks 04 and 05 and are never copied into the
# thesis, so this table is only consulted on an --lang en run.
#
# Figures deliberately absent from this table stay in the export directory only.
THESIS_FIGURE_MAP = {
    ("NB01", 1): ("cd11", "nb01-fig01-runtime-dist.png"),
    ("NB01", 2): ("cd13", "nb01-fig02-runtime-cdf.png"),
    ("NB01", 3): ("cd15", "nb01-fig03-arrival-rate.png"),
    ("NB01", 4): ("cd17", "nb01-fig04-gpu-demand.png"),
    ("NB01", 5): ("cd19", "nb01-fig05-interarrival.png"),
    ("NB01", 6): ("cd21", "nb01-fig06-arrival-heatmap.png"),
    ("NB02", 3): ("cd12", "nb02-fig03-gpu-vs-runtime.png"),
    ("NB03", 1): ("cd12", "nb03-fig01-cluster-load.png"),
    ("NB03", 2): ("cd18", "nb03-fig02-correlation.png"),
    ("NB04", 1): ("cd32", "nb04-fig01-model-comparison.png"),
    ("NB04", 2): ("ecd01", "nb04-fig02-feature-importance.png"),
    ("NB04", 3): ("ecd03", "nb04-fig03-pred-vs-actual.png"),
    ("NB04", 4): ("ecd05", "nb04-fig04-residuals.png"),
    ("NB04", 5): ("ecd07", "nb04-fig05-dl-comparison.png"),
    # Figure 1 of each NB05 variant is the rank-correlation analysis. That cell
    # saves no copy of its own, so this map is the only thing that puts it in
    # the thesis.
    ("NB05_32GPU", 1): ("7a843cee", "mae_spearman_vs_jct_gain_32gpu.png"),
    ("NB05_32GPU", 2): ("7cc8bbfc", "nb05-fig06-load-backfill-sensitivity_32gpu.png"),
    ("NB05_32GPU", 3): ("cd21", "nb05-fig01-scheduler-jct_32gpu.png"),
    ("NB05_32GPU", 4): ("e5cd01", "nb05-fig02-wait-cdf_32gpu.png"),
    ("NB05_32GPU", 5): ("e5cd03", "nb05-fig03-slowdown-box_32gpu.png"),
    ("NB05_32GPU", 6): ("e5cd05", "nb05-fig04-improvement-heatmap_32gpu.png"),
    ("NB05_32GPU", 7): ("e5cd09", "nb05-fig05-wait-percentile_32gpu.png"),
    ("NB05_256GPU", 1): ("8d1a4e10", "mae_spearman_vs_jct_gain_256gpu.png"),
    ("NB05_256GPU", 2): ("d7a6e287", "nb05-fig06-load-backfill-sensitivity_256gpu.png"),
    ("NB05_256GPU", 3): ("cd21", "nb05-fig01-scheduler-jct_256gpu.png"),
    ("NB05_256GPU", 4): ("e5cd01", "nb05-fig02-wait-cdf_256gpu.png"),
    ("NB05_256GPU", 5): ("e5cd03", "nb05-fig03-slowdown-box_256gpu.png"),
    ("NB05_256GPU", 6): ("e5cd05", "nb05-fig04-improvement-heatmap_256gpu.png"),
    ("NB05_256GPU", 7): ("e5cd09", "nb05-fig05-wait-percentile_256gpu.png"),
}

# Total number of figures each notebook is expected to produce in a complete
# top-to-bottom run. THESIS_FIGURE_MAP addresses figures by position, so a
# figure inserted or removed anywhere shifts every later position and the map
# would write the wrong images under the right filenames. Comparing the total
# catches that; a count-preserving shuffle is what the pinned cell ids are for.
EXPECTED_FIGURE_COUNT = {
    "NB01": 6,
    "NB02": 3,
    "NB03": 2,
    "NB04": 5,
    "NB05_32GPU": 7,
    "NB05_256GPU": 7,
}

# Notebooks whose exported figures and tables report model results: notebook 04
# trains the models and reads its numbers back out of checkpoints, and both
# notebook 05 variants simulate with the saved artifacts. Only these are judged
# against the provenance records in ``_stale_model_inputs``. Notebooks 01 to 03
# touch neither a checkpoint nor a model artifact.
MODEL_DERIVED_PREFIXES = frozenset({"NB04", "NB05_32GPU", "NB05_256GPU"})

# The notebooks' MODEL_DIR. Checkpoints are not given a constant here: they are
# listed from the directory ``src.tuning`` itself resolves, so the listing and
# the currency predicate cannot look at two different places.
MODEL_DIR = BASE_DIR / "results" / "models"
# Written by ``src.tuning.record_model_artifact`` next to every model it saves.
_PROVENANCE_SUFFIX = ".provenance.json"

# The notebooks reach results/models through a ``MODEL_DIR`` variable of their
# own, so every artifact they touch is spelled ``MODEL_DIR / "<name>"``, or, for
# the per-seed LSTM checkpoints alone, ``MODEL_DIR / f"..._seed{seed}.pth"``.
_MODEL_DIR_VAR = "MODEL_DIR"
_FORMAT_FIELD_RE = re.compile(r"\{[^}]*\}")


def _loaded_model_artifacts(nb_paths: list[Path]) -> tuple[set[str], list[re.Pattern]]:
    """The results/models filenames the model-derived notebooks actually touch.

    Read out of the notebooks' own source rather than restated here, because
    ``_stale_model_inputs`` and notebook 05's ``_reject_stale_artifacts`` are two
    halves of one contract and a list copied into this file would drift out of
    step with the list the notebook enforces, which is how this gate
    came to judge a strictly smaller set than the notebook does.

    Returns the names spelled out in full, and the compiled form of the names
    built with a format field: the per-seed LSTM checkpoints are the only
    artifacts neither notebook writes out literally (notebook 04 hands
    ``finalize_dl_model`` a ``{seed}`` template, notebook 05 rebuilds the same
    names with an f-string), so both spellings are normalised to one shape.

    A notebook that is missing or that no longer parses contributes nothing,
    since the extraction loop reports a missing or unreadable notebook itself, and
    ``_stale_model_inputs`` treats an empty result as unverifiable rather than
    as nothing to check.
    """
    names: set[str] = set()
    patterns: set[str] = set()

    for nb_path in nb_paths:
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            try:
                tree = ast.parse("".join(cell.get("source") or []))
            except SyntaxError:
                # A cell that does not parse cannot have produced an artifact:
                # it raises on execution, which ``_audit_stored_outputs`` sees.
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.BinOp)
                        and isinstance(node.op, ast.Div)
                        and isinstance(node.left, ast.Name)
                        and node.left.id == _MODEL_DIR_VAR):
                    continue
                right = node.right
                if isinstance(right, ast.JoinedStr):
                    patterns.add("".join(
                        str(part.value) if isinstance(part, ast.Constant) else "{}"
                        for part in right.values
                    ))
                elif isinstance(right, ast.Constant) and isinstance(right.value, str):
                    if "{" in right.value:
                        patterns.add(_FORMAT_FIELD_RE.sub("{}", right.value))
                    else:
                        names.add(right.value)

    compiled = [
        re.compile(".+".join(re.escape(part) for part in pattern.split("{}")) + r"\Z")
        for pattern in sorted(patterns)
    ]
    return names, compiled


def _stale_model_inputs(
    loaded_names: set[str], loaded_patterns: list[re.Pattern]
) -> list[str]:
    """Checkpoints and model artifacts the current source tree no longer certifies.

    ``_audit_stored_outputs`` judges a notebook against itself: it sees a
    crashed cell or a mixture of kernel sessions, but nothing about the source
    code the run used. So in the default no-``--execute`` mode this script
    harvested outputs produced under ANY earlier source tree and copied the
    mapped figures into ``thesis/latex/figures``, with the whole
    ``_provenance_is_current`` mechanism bypassed at the last hop to the thesis.

    The records to compare against already exist: ``save_checkpoint`` stores a
    provenance snapshot in every checkpoint and ``record_model_artifact`` writes
    a sidecar beside every saved model. This asks the same predicate the
    notebooks' own ``if ckpt:`` branches use, so the export cannot certify a
    result the notebook that produced it would now recompute.

    Which artifacts are judged is decided by ``_loaded_model_artifacts``, i.e.
    by the names the notebooks themselves load and write. A missing sidecar is a
    reason to REFUSE, exactly as it is in notebook 05's
    ``_reject_stale_artifacts``. Selecting instead the files that already carry
    a sidecar inverted that: a model written without one was never judged at
    all, so the two halves of the contract disagreed about the same 16 files,
    the scalers, the median and Alibaba-estimate baselines and the per-seed LSTM
    checkpoints, and once the checkpoints were refreshed this gate passed
    notebooks that notebook 05 would have refused to simulate with, publishing
    their scheduling figures into ``thesis/latex/figures``.

    A file nothing loads is left out by not being NAMED, not by lacking a
    sidecar: ``results/models`` also accumulates leftovers from earlier
    revisions, and demanding a sidecar from a file no run writes any more would
    be a gate nothing could satisfy. Everything that IS named is stamped by a
    full notebook 04 run, so the refusal always has a remedy.

    Absence is deliberately not a refusal here. Notebook 05 can insist on its
    required list because it is about to load those files; this script judges
    the record of a run that already happened, and the optional baselines and
    the per-seed checkpoints are legitimately absent on some runs, and reading
    "the reader never produced it" as a stale result would be the cry-wolf
    warning ``src.tuning._compute_provenance``'s docstring cautions against.

    Returns the reasons to refuse, or an empty list when every record still
    matches the tree.
    """
    try:
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        from src.tuning import (
            _CHECKPOINT_DIR,
            checkpoint_is_current,
            model_artifact_is_current,
        )
    except ImportError as err:  # unverifiable is not the same as current
        raise RuntimeError(
            "❌ [HATA] src.tuning içe aktarılamadı, bu yüzden notebook 04/05 "
            f"çıktılarının güncelliği doğrulanamıyor: {err}"
        ) from err

    if not loaded_names and not loaded_patterns:
        # Nothing was read out of the notebook sources, so an unstamped file
        # cannot be told apart from one a notebook loads. Reading that as
        # nothing to check is the shape of the hole this gate closes.
        return [
            "notebook 04/05 kaynaklarında hiçbir model dosyası adı bulunamadı; "
            "hangi model dosyalarının güncel olması gerektiği belirlenemiyor"
        ]

    checkpoints = sorted(_CHECKPOINT_DIR.glob("*.json"))
    artifacts = [
        path for path in sorted(MODEL_DIR.glob("*"))
        if path.is_file()
        and not path.name.endswith(_PROVENANCE_SUFFIX)
        and (path.name in loaded_names
             or any(pattern.match(path.name) for pattern in loaded_patterns)
             # A stamped file stays judged even when the scan did not name it,
             # so a notebook that stops spelling a path the way the scan reads
             # can only widen this set, never quietly narrow it.
             or path.with_name(path.name + _PROVENANCE_SUFFIX).exists())
    ]

    if not checkpoints and not artifacts:
        return [
            "results/checkpoints ve results/models altında hiç köken (provenance) "
            "kaydı yok; model sonuçlarının hangi kaynak koddan geldiği doğrulanamıyor"
        ]

    stale = [
        f"{path.stem}: kontrol noktasının kökeni bu kaynak ağacıyla uyuşmuyor"
        for path in checkpoints if not checkpoint_is_current(path.stem)
    ]
    for path in artifacts:
        if model_artifact_is_current(path):
            continue
        # The two causes need different repairs, so each refusal says which it
        # is, the same distinction ``_reject_stale_artifacts`` draws.
        if not path.with_name(path.name + _PROVENANCE_SUFFIX).exists():
            why = "köken kaydı (sidecar) yok, güncellik denetiminden eski"
        else:
            why = "köken kaydı bu kaynak ağacıyla uyuşmuyor"
        stale.append(f"{path.name}: {why}")
    return stale


def _report_stale_model_inputs(nb_file: str, stale: list[str]) -> None:
    """Print why a model-derived notebook was skipped, in the same loud shape as
    the other gates."""
    print(f"\n⚠️  [UYARI] {nb_file}: bu notebook'un sayıları GÜNCEL DEĞİL. Üretildiği")
    print(f"    kaynak kod ağacı artık bu ağaç değil ({len(stale)} kayıt uyuşmuyor):")
    for name in stale[:10]:
        print(f"      • {name}")
    if len(stale) > 10:
        print(f"      • … ve {len(stale) - 10} kayıt daha")
    print("    Saklı çıktılar bir çalıştırmanın kaydıdır, o çalıştırmanın hangi")
    print("    kaynak kodla yapıldığının değil; bu yüzden karar, eğitim kodunun")
    print("    yazdığı köken kayıtlarına bakılarak veriliyor. Notebook 04'ü baştan")
    print("    sona çalıştırın (--force-execute) ki kontrol noktaları ve model")
    print("    dosyaları güncel kökenle yeniden yazılsın.\n")


class UntrustedNotebookOutputs(RuntimeError):
    """A notebook's stored outputs are not the record of one clean run.

    Carries the individual findings so the caller can print them all instead of
    only the first one.
    """

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def _audit_stored_outputs(nb: dict) -> list[str]:
    """Report every stored output that cannot have come from one clean run.

    The extraction loop below reads whatever a cell has stored and writes it out
    as a result of the current run. Two kinds of output pass that reading while
    being nothing of the sort:

    * Outputs of a cell that RAISED. matplotlib flushes the half-drawn figure
      after the traceback and pandas has usually already displayed the table
      computed from the still-incomplete frame, so a crashed cell contributes a
      picture and a table indistinguishable from valid ones. The figure count is
      unchanged by the crash, so EXPECTED_FIGURE_COUNT cannot see it either.
      notebook 05 raised LinAlgError in its np.polyfit cell and the scatter plot
      with no regression line went to thesis/latex/figures under the mapped
      filename, with only ``[PNG]`` success lines printed and exit status 0.
    * Outputs of a cell that never ran (``execution_count: null``). They are left
      over from an earlier revision of the notebook, yet they occupy a figure
      position and count towards EXPECTED_FIGURE_COUNT exactly like a fresh
      figure, so a stale figure helps the count gate pass instead of tripping
      it, and is then copied into the thesis. Cell d7a6e287 of the 256-GPU
      notebook 05 is exactly this: no execution count, one leftover image/png,
      already exported once as NB05_256GPU-Figure01.png.

    Execution counts must also increase from top to bottom across the cells that
    carry outputs: an output whose count is not above the one above it was
    produced in a different kernel session, so the stored outputs are a mixture
    of runs and their order is not the notebook's order. A GAP in the sequence
    is deliberately not reported: it only shows that some cell was skipped or
    cleared, while every output still present did come from the same increasing
    run, and a skipped figure cell shows up as a count mismatch in
    ``_check_figure_count``.

    The high-water mark is raised by every executed cell, including the ones
    that store no output. Skipping those before reading their count left the
    commonest way a notebook goes stale invisible: editing an import, a path or
    a parameter cell (``N_GPU``, a seed, the load factor) and re-running just
    that cell produces no output of its own, so the figures below it kept
    counts far lower than the cell that now defines their inputs and both this
    gate and the figure count passed. ``--force-execute`` is exactly what fills
    that blind spot with numbered output-less cells.
    """
    problems: list[str] = []
    last_count = 0

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs", [])
        cell_id = cell.get("id", "?")

        failed = [out for out in outputs if out.get("output_type") == "error"]
        if failed:
            enames = ", ".join(out.get("ename", "?") for out in failed)
            problems.append(
                f"hücre {cell_id}: çalışırken hata verdi ({enames}); bu hücrenin "
                f"{len(outputs)} çıktısı yarım kalmış bir hesaplamadan geliyor"
            )

        count = cell.get("execution_count")
        if not outputs:
            # An output-less cell cannot itself be stale, but it did run, so it
            # still fixes how late every cell below it must have run. Only a
            # cell that never ran at all is passed over.
            if count is not None:
                last_count = max(last_count, count)
            continue

        if count is None:
            problems.append(
                f"hücre {cell_id}: hiç çalıştırılmamış (execution_count: null) ama "
                f"{len(outputs)} çıktısı saklı; bunlar notebook'un önceki bir "
                "sürümünden kalma"
            )
        elif count <= last_count:
            problems.append(
                f"hücre {cell_id}: execution_count {count}, yukarısındaki hücrenin "
                f"{last_count} değerinden büyük değil; saklı çıktılar tek bir "
                "baştan sona çalıştırmadan gelmiyor"
            )
        else:
            last_count = count

    return problems


_UNSAFE_CELL_ID_RE = re.compile(r"[^A-Za-z0-9_-]")


def _safe_cell_id(cell_id: object) -> str:
    """Filename-safe form of a notebook cell id, or ``""`` when there is none.

    nbformat already restricts ids to ``[A-Za-z0-9-_]``, but this file is read
    as plain JSON, so a hand-edited notebook could carry anything.
    """
    if not isinstance(cell_id, str):
        return ""
    return _UNSAFE_CELL_ID_RE.sub("-", cell_id)[:40]


def extract_from_nb_dict(
    nb: dict,
    prefix: str,
    extract_png: bool,
    extract_html: bool,
    thesis_buffer: dict | None = None,
    export_files: list | None = None,
) -> tuple[int, int]:
    """Extract figures and tables from a notebook JSON dictionary.

    When ``thesis_buffer`` is a dict, every figure whose (prefix, position)
    appears in ``THESIS_FIGURE_MAP`` is stored in it under that key, as
    ``(producing cell id, PNG bytes)``. Nothing is written to the thesis
    directory here. The caller copies a notebook's figures only after
    verifying the complete expected set was produced by the cells the map pins
    it to (see ``_check_figure_count`` and ``_sync_thesis_figures``).

    When ``export_files`` is a list, the export files are appended to it
    instead of being written, for the caller to flush with ``_flush_exports``
    once the figure count is verified. Writing them here meant a count refusal
    left the git-tracked export directory holding files whose content had
    shifted under unchanged positional names, ``NB01-Figure01.png`` holding
    the chart the export convention calls figure 2, after a run that had
    already failed. Left as None the files are written immediately, which is
    what a caller extracting a single notebook wants.

    Raises ``UntrustedNotebookOutputs``, before writing or buffering anything so a
    rejected notebook leaves no half-written export behind, when the
    stored outputs fail ``_audit_stored_outputs``. The check lives here rather
    than only in the caller so that no future call site can extract from a
    crashed or unexecuted notebook by forgetting to ask first.
    """
    problems = _audit_stored_outputs(nb)
    if problems:
        raise UntrustedNotebookOutputs(problems)

    fig_idx = 1
    table_idx = 1
    extracted_pngs = 0
    extracted_htmls = 0
    pending: list[tuple[str, Path, bytes]] = []

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        cell_id = _safe_cell_id(cell.get("id"))

        for out in cell.get("outputs", []):
            data = out.get("data", {})

            # Extract PNG
            if extract_png and "image/png" in data:
                png_b64 = data["image/png"]
                if isinstance(png_b64, list):
                    png_b64 = "".join(png_b64)
                png_bytes = base64.b64decode(png_b64)

                out_name = f"{prefix}-Figure{fig_idx:02d}.png"
                pending.append(("PNG", PNG_DIR / out_name, png_bytes))

                if thesis_buffer is not None and (prefix, fig_idx) in THESIS_FIGURE_MAP:
                    # The raw id, not the filename-safe form: it is compared
                    # against THESIS_FIGURE_MAP's pin, not put in a name.
                    thesis_buffer[(prefix, fig_idx)] = (cell.get("id"), png_bytes)

                fig_idx += 1
                extracted_pngs += 1

            # Extract HTML Table
            if extract_html and "text/html" in data:
                html_content = data["text/html"]
                if isinstance(html_content, list):
                    html_content = "".join(html_content)

                if "<table" in html_content:
                    html_content = _strip_positional_index(html_content)
                    full_html = HTML_STYLE_HEADER + html_content + HTML_STYLE_FOOTER
                    # The producing cell's id is part of the name, not just the
                    # running position. Figures are protected against a shifted
                    # position by THESIS_FIGURE_MAP and EXPECTED_FIGURE_COUNT;
                    # tables had nothing, so "NB04_Table10.html" meant one table
                    # in one export and a different one in the next. Pinning the
                    # cell id makes a shifted table land under a new filename, so
                    # a stale reference fails to resolve instead of resolving to
                    # the wrong table. The position stays in front so the
                    # directory still sorts in notebook order.
                    stem = f"{prefix}_Table{table_idx:02d}"
                    out_name = f"{stem}_{cell_id}.html" if cell_id else f"{stem}.html"
                    pending.append(
                        ("HTML", HTML_DIR / out_name, full_html.encode("utf-8"))
                    )
                    table_idx += 1
                    extracted_htmls += 1

    if export_files is None:
        _flush_exports(pending)
    else:
        export_files.extend(pending)

    return extracted_pngs, extracted_htmls


def _flush_exports(pending: list[tuple[str, Path, bytes]]) -> None:
    """Write the export files buffered during extraction, and report each one.

    The success lines are printed here rather than during extraction so that
    every ``[PNG]``/``[HTML]`` line on screen corresponds to a file that really
    is on disk.
    """
    for label, out_path, payload in pending:
        with open(out_path, "wb") as f_out:
            f_out.write(payload)
        print(f"  [{label}]".ljust(9) + out_path.name)

_THEAD_RE = re.compile(r"<thead>.*?</thead>", re.S)
_TBODY_RE = re.compile(r"<tbody>.*?</tbody>", re.S)
# pandas writes body rows as a bare <tr> but header rows carry attributes
# (<tr style="text-align: right;">), so the opening tag must allow them.
_ROW_RE = re.compile(r"<tr[^>]*>.*?</tr>", re.S)
_LEADING_INT_TH_RE = re.compile(r"\s*<th>\s*(\d+)\s*</th>", re.S)
_LEADING_EMPTY_TH_RE = re.compile(r"\s*<th>\s*</th>", re.S)


def _strip_positional_index(html: str) -> str:
    """Drop the DataFrame's positional index from a scraped table.

    Jupyter renders ``df`` with ``to_html()``, which writes the index as a
    leading ``<th>`` on every body row under an EMPTY header cell. For a
    RangeIndex that column carries no information; in the thesis it reads as
    a nameless "0, 1, 2, ..." first column next to the model names.

    Only a pure positional index is removed. A table with a MEANINGFUL index
    such as model names or policies is left untouched, as is a
    table whose header cell is not empty, because there the index is data.
    """
    thead_m = _THEAD_RE.search(html)
    tbody_m = _TBODY_RE.search(html)
    if thead_m is None or tbody_m is None:
        return html

    rows = _ROW_RE.findall(tbody_m.group(0))
    if not rows:
        return html
    # Every body row must start with a plain-integer <th>, and those integers
    # must be exactly 0..n-1, otherwise the index means something.
    def _split_open_tag(row):
        open_tag, rest = row.split(">", 1)
        return open_tag + ">", rest

    labels = []
    for row in rows:
        _, inner = _split_open_tag(row)
        m = _LEADING_INT_TH_RE.match(inner)
        if m is None:
            return html
        labels.append(int(m.group(1)))
    if labels != list(range(len(labels))):
        return html

    # The matching header cell must be empty; a named index is data.
    header_rows = _ROW_RE.findall(thead_m.group(0))
    if not header_rows:
        return html
    new_header_rows = []
    for hrow in header_rows:
        open_tag, rest = _split_open_tag(hrow)
        m = _LEADING_EMPTY_TH_RE.match(rest)
        if m is None:
            return html
        new_header_rows.append(open_tag + rest[m.end():])

    new_thead = thead_m.group(0)
    for old, new in zip(header_rows, new_header_rows):
        new_thead = new_thead.replace(old, new, 1)

    new_tbody = tbody_m.group(0)
    for row in rows:
        open_tag, inner = _split_open_tag(row)
        new_tbody = new_tbody.replace(
            row, open_tag + inner[_LEADING_INT_TH_RE.match(inner).end():], 1
        )

    return html.replace(thead_m.group(0), new_thead, 1).replace(
        tbody_m.group(0), new_tbody, 1
    )


def _clean_stale_exports(prefix: str) -> None:
    """Remove this notebook's previously-exported PNG/HTML files before a
    fresh extraction.

    Filenames lead with a POSITION (``{prefix}-Figure{N}.png``,
    ``{prefix}_Table{N}_{cell id}.html``): if a notebook used to produce, say, 7
    figures and now produces 6, the old Figure07 file was never removed by the
    code that follows: it just sat in the export directory looking like a
    current file from this run. Deleting every file for this
    prefix up front makes "not present after export" mean "not produced this
    run," not "produced by some earlier run and never cleaned up." Its
    counterpart is that ``run_pipeline`` flushes the new files only after the
    figure count is verified, so "present after export" means "produced by a
    run this script trusts" rather than merely "written before the refusal".

    The table glob is deliberately ``{prefix}_Table*.html`` rather than the exact
    new name shape, so it also clears tables exported before the cell id became
    part of the name, otherwise those would linger forever under names no run
    writes any more.
    """
    for stale_file in PNG_DIR.glob(f"{prefix}-Figure*.png"):
        stale_file.unlink()
    for stale_file in HTML_DIR.glob(f"{prefix}_Table*.html"):
        stale_file.unlink()

def _report_untrusted_outputs(nb_file: str, problems: list[str]) -> None:
    """Print why a notebook was skipped, in the same loud shape as the count gate."""
    print(f"\n⚠️  [UYARI] {nb_file}: saklı çıktılar GÜVENİLİR DEĞİL. Bu notebook'tan")
    print("    hiçbir şekil veya tablo aktarılmadı:")
    for problem in problems:
        print(f"      • {problem}")
    print("    Hata vermiş ya da hiç çalışmamış bir hücrenin çıktısı, geçerli bir")
    print("    sonuçtan dosya olarak ayırt edilemez. Notebook'u baştan sona")
    print("    çalıştırıp (--force-execute) bu betiği tekrar koşun.\n")


def _check_figure_count(prefix: str, produced: int, sync_thesis: bool) -> str | None:
    """Compare a notebook's figure count with EXPECTED_FIGURE_COUNT.

    A figure added or removed anywhere shifts every later position, so
    THESIS_FIGURE_MAP would still find all the positions it expects and would
    write the wrong images under the right thesis filenames, which the
    per-position presence check below cannot see. The count is checked for both
    languages, not only English: the export filenames are positional in the same
    way, and the Turkish export is an artifact set in its own right.

    ``sync_thesis`` only shapes the wording: on a Turkish run nothing is copied
    into the thesis, so the message must not claim thesis figures were held back.

    Returns the reason to refuse this notebook, or None when the count matches.
    """
    expected_total = EXPECTED_FIGURE_COUNT.get(prefix)
    if expected_total is None or produced == expected_total:
        return None

    print(f"\n⚠️  [UYARI] {prefix}: bu notebook {produced} şekil üretti, "
          f"beklenen {expected_total}.")
    print("    Bir şekil eklenmiş, silinmiş ya da hücresi hiç çalıştırılmamış")
    print("    demektir; dosya adları konuma göre verildiği için sonraki tüm")
    print("    şekiller kayar ve aynı ad bir sonraki aktarımda başka bir grafiği")
    print("    gösterir.")
    if sync_thesis:
        print("    THESIS_FIGURE_MAP de konumla eşleştiğinden yanlış görseller doğru")
        print("    dosya adlarıyla teze yazılırdı; bu yüzden bu notebook'un tez")
        print("    şekilleri güncellenmedi.")
    print("    Notebook'u baştan sona çalıştırın; şekil sırası gerçekten değiştiyse")
    print("    THESIS_FIGURE_MAP ve EXPECTED_FIGURE_COUNT güncellenmelidir.\n")

    reason = f"{produced} şekil üretti, beklenen {expected_total}"
    return f"{reason} (tez şekilleri güncellenmedi)" if sync_thesis else reason


def _sync_thesis_figures(prefix: str, thesis_buffer: dict) -> tuple[int, str | None]:
    """Copy one notebook's buffered figures into the thesis, all-or-nothing.

    Returns (files written, refusal reason). Nothing from this notebook is
    copied, and a warning explains why, when a position THESIS_FIGURE_MAP
    expects is absent, after a stale or partial notebook run, or when the figure at
    a position was produced by a cell other than the one the map pins it to. A
    mislabeled thesis figure is far worse than a stale one, and a swap of two
    figure cells is precisely a mislabel the count gate cannot see: both
    positions are still produced, so the wrong chart went to the thesis under
    the right filename with no warning and exit status 0.

    The figure COUNT is checked before this, by ``_check_figure_count``; the
    reason is returned rather than only printed because a refusal has to reach
    ``run_pipeline`` and fail the run. Returning 0 silently was how a run that
    left thesis/latex/figures untouched still ended in "İşlem Tamamlandı" and
    exit 0, i.e. in a green "PIPELINE COMPLETED SUCCESSFULLY" from
    run_all_experiments.sh over a LaTeX build using the previous run's images.
    """
    expected = {i: pin for (p, i), pin in THESIS_FIGURE_MAP.items() if p == prefix}
    if not expected:
        return 0, None

    missing = sorted(i for i in expected if (prefix, i) not in thesis_buffer)
    if missing:
        print(f"\n⚠️  [UYARI] {prefix}: tez şekilleri GÜNCELLENMEDİ. Beklenen "
              f"{len(expected)} şekilden şu konumlar üretilmemiş: {missing}.")
        print("    Notebook baştan sona çalıştırılmamış olabilir ya da şekil sırası")
        print("    değişmiş olabilir (THESIS_FIGURE_MAP ile karşılaştırın). Yanlış")
        print("    içerikli kopyalamayı önlemek için bu notebook'un tümü atlandı.\n")
        return 0, f"THESIS_FIGURE_MAP'in beklediği {missing} konumları üretilmemiş"

    shifted = [
        f"konum {i}: {name} beklenen hücre {pinned_id} yerine "
        f"{thesis_buffer[(prefix, i)][0]} hücresinden geldi"
        for i, (pinned_id, name) in sorted(expected.items())
        if thesis_buffer[(prefix, i)][0] != pinned_id
    ]
    if shifted:
        print(f"\n⚠️  [UYARI] {prefix}: tez şekilleri GÜNCELLENMEDİ. Şekiller "
              "THESIS_FIGURE_MAP'te kayıtlı hücrelerden gelmiyor:")
        for problem in shifted:
            print(f"      • {problem}")
        print("    Şekil hücreleri yer değiştirmiş demektir; şekil sayısı aynı")
        print("    kaldığı için sayı denetimi bunu göremez ve yanlış grafik doğru")
        print("    tez dosya adıyla yazılırdı. Sıra gerçekten değiştiyse")
        print("    THESIS_FIGURE_MAP güncellenmelidir.\n")
        return 0, f"tez şekilleri kayıtlı hücrelerden gelmiyor ({len(shifted)} konum)"

    for i, (_, name) in sorted(expected.items()):
        with open(THESIS_FIG_DIR / name, "wb") as f_thesis:
            f_thesis.write(thesis_buffer[(prefix, i)][1])
        print(f"         └─ thesis/latex/figures/{name}")
    return len(expected), None

def execute_notebook(nb_path: Path):
    """Execute a notebook and save the executed notebook with outputs."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    print(f"🔄 [ÇALIŞTIRILIYOR] '{nb_path.name}' arka planda çalıştırılıyor...")
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_node = nbformat.read(f, as_version=4)

    # No per-cell timeout: notebook 04 trains 18+ models and its search cells run
    # for hours, and the previous 1800 s cap killed it mid-training.
    #
    # kernel_name="thesis-venv", not the generic "python3": this machine has more
    # than one Jupyter kernel registered under "python3", so that name resolves
    # to whichever is found first. "thesis-venv" is registered for venv/bin/python
    # and matches the kernelspec every notebook in this repository declares.
    ep = ExecutePreprocessor(timeout=None, kernel_name="thesis-venv")
    try:
        ep.preprocess(nb_node, {"metadata": {"path": str(nb_path.parent)}})
    except Exception as err:
        raise RuntimeError(f"❌ [HATA] '{nb_path.name}' çalıştırılırken hata oluştu:\n{err}") from err

    # Save the executed notebook back to disk
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb_node, f)
    print(f"💾 [KAYDEDİLDİ] '{nb_path.name}' başarıyla tamamlandı ve çıktıları kaydedildi.")

def run_pipeline(lang: str = "en", auto_execute: bool = False, force_execute: bool = False):
    # Point the extraction at this language's own directories before anything is
    # written: the filenames are identical across languages, so one shared
    # directory meant each run deleted the other language's export.
    global PNG_DIR, HTML_DIR
    PNG_DIR, HTML_DIR = _export_dirs(lang)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    nb_dir = BASE_DIR / "notebooks" / lang
    notebooks_list = NOTEBOOKS_TR if lang == "tr" else NOTEBOOKS_EN

    # Which model files the currency gate has to judge, read out of the notebooks
    # themselves. Both languages' mirrors are scanned whatever --lang this run
    # is, because there is only one results/models: reading only the language at
    # hand would let a drifted mirror judge a smaller set than its counterpart.
    # Scanned once, up front, since executing a notebook rewrites its outputs and
    # never its code cells.
    loaded_names, loaded_patterns = _loaded_model_artifacts([
        BASE_DIR / "notebooks" / nb_lang / nb_file
        for nb_lang, notebooks in (("en", NOTEBOOKS_EN), ("tr", NOTEBOOKS_TR))
        for nb_file, prefix, _png, _html in notebooks
        if prefix in MODEL_DERIVED_PREFIXES
    ])

    # The thesis is written in English, so only English notebook figures may be
    # copied into thesis/latex/figures; a Turkish run must never overwrite them.
    sync_thesis = (lang == "en")
    if sync_thesis:
        THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    total_pngs = 0
    total_htmls = 0
    total_thesis = 0
    # Every reason a notebook was refused. All of them end the run non-zero, and
    # none may be left as a warning the caller reports as success.
    refusals: list[str] = []

    print("=" * 65)
    print(f"Tez Çıktı Aktarımı ({lang.upper()}) | Mod: {'Force-Execute' if force_execute else ('Auto-Execute' if auto_execute else 'Hızlı Çıkarım')}")
    print("=" * 65)

    for nb_file, prefix, extract_png, extract_html in notebooks_list:
        nb_path = nb_dir / nb_file
        if not nb_path.exists():
            raise FileNotFoundError(f"❌ [HATA] Notebook dosyası bulunamadı: {nb_path}")

        # If force-execute is requested, run notebook directly
        if force_execute:
            execute_notebook(nb_path)

        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
        except Exception as e:
            raise RuntimeError(f"❌ [HATA] {nb_file} dosyası JSON olarak okunamadı: {e}") from e

        thesis_buffer: dict | None = {} if sync_thesis else None
        # Filled by extraction, written only once the figure count is verified,
        # so a refused notebook contributes nothing to the export directory. The
        # audit runs before the first file is buffered.
        export_files: list = []

        # Try extracting from existing outputs. Stored outputs that cannot come
        # from one clean run are refused here; re-running the notebook is the
        # only remedy, so --execute takes it and every other mode skips it.
        reexecuted = False
        try:
            pngs, htmls = extract_from_nb_dict(
                nb, prefix, extract_png, extract_html, thesis_buffer, export_files
            )
        except UntrustedNotebookOutputs as untrusted:
            if not auto_execute:
                _report_untrusted_outputs(nb_file, untrusted.problems)
                refusals.append(
                    f"{nb_file}: saklı çıktılar güvenilir değil "
                    f"({len(untrusted.problems)} bulgu)"
                )
                continue

            print(f"\n⚠️  [UYARI] '{nb_file}' saklı çıktıları güvenilir değil "
                  f"({len(untrusted.problems)} bulgu); notebook yeniden çalıştırılıyor.")
            execute_notebook(nb_path)
            with open(nb_path, "r", encoding="utf-8") as f:
                nb_executed = json.load(f)
            pngs, htmls = extract_from_nb_dict(
                nb_executed, prefix, extract_png, extract_html, thesis_buffer,
                export_files,
            )
            # Executed once already; the empty-output branch below must not run
            # it a second time, since notebook 04 trains for hours.
            reexecuted = True

        # If 0 outputs found
        if pngs == 0 and htmls == 0:
            if auto_execute:
                if not reexecuted:
                    # Auto-execute empty notebook
                    execute_notebook(nb_path)
                    with open(nb_path, "r", encoding="utf-8") as f:
                        nb_executed = json.load(f)
                    pngs, htmls = extract_from_nb_dict(
                        nb_executed, prefix, extract_png, extract_html,
                        thesis_buffer, export_files,
                    )

                if pngs == 0 and htmls == 0:
                    raise RuntimeError(
                        f"❌ [HATA] '{nb_file}' çalıştırılmasına rağmen hiçbir görsel veya tablo çıktısı üretmedi!"
                    )
            else:
                # A notebook with no stored outputs contributes nothing to the
                # export, so the run did not refresh it: refused like any other
                # gap rather than warned about and reported as success.
                print(
                    f"⚠️  [UYARI] '{nb_file}' çıktısı boş! Otomatik çalıştırmak için '--execute' bayrağını kullanabilirsiniz:\n"
                    f"     python scripts/export_thesis_results.py --execute\n"
                )
                refusals.append(f"{nb_file}: hiçbir şekil veya tablo çıktısı yok")
                continue

        # Asked here, after every path that may have re-executed the notebook, so
        # a run that retrained notebook 04 is judged on the records that run
        # wrote. The stored outputs cannot answer this themselves: they record a
        # run, not which source tree that run used.
        if prefix in MODEL_DERIVED_PREFIXES:
            stale = _stale_model_inputs(loaded_names, loaded_patterns)
            if stale:
                _report_stale_model_inputs(nb_file, stale)
                refusals.append(
                    f"{nb_file}: sonuçları güncel olmayan {len(stale)} köken "
                    "kaydından geliyor"
                )
                continue

        count_problem = _check_figure_count(prefix, pngs, sync_thesis)
        if count_problem:
            # Nothing is written: the positions no longer line up with the export
            # filenames, so flushing would leave the tracked export directory
            # holding shifted images under unchanged names.
            refusals.append(f"{nb_file}: {count_problem}")
            continue

        # Deleting this notebook's previous exports happens here, beside the write
        # that replaces them, not before extraction. Filenames are positional, so
        # the removal is required, but it cannot happen unless the replacement is
        # ready.
        _clean_stale_exports(prefix)
        _flush_exports(export_files)
        total_pngs += pngs
        total_htmls += htmls

        if thesis_buffer is not None:
            written, sync_problem = _sync_thesis_figures(prefix, thesis_buffer)
            total_thesis += written
            if sync_problem:
                refusals.append(f"{nb_file}: {sync_problem}")

    # Raised before the closing summary: a run that refused a notebook must not
    # print "İşlem Tamamlandı" and exit 0, because run_all_experiments.sh would
    # then print "PIPELINE COMPLETED SUCCESSFULLY" over an export missing a
    # notebook's figures and tables. This covers every refusal, not only the
    # untrusted-outputs one.
    if refusals:
        thesis_line = ""
        if sync_thesis:
            thesis_line = (
                f"\n    Tez şekilleri: {len(THESIS_FIGURE_MAP)} dosyanın {total_thesis} "
                "tanesi güncellendi;\n    geri kalanlar eski hâlleriyle bırakıldı."
            )
        raise RuntimeError(
            "❌ [HATA] Aktarım tamamlanmadı; şu notebook'lar atlandı:\n"
            + "\n".join(f"      • {reason}" for reason in refusals)
            + "\n    Gerekçeler yukarıdaki UYARI bloklarında listelendi; ilgili"
            "\n    notebook'lar baştan sona çalıştırılmadan aktarım tamamlanmış"
            "\n    sayılmaz." + thesis_line
        )

    # The summary below is what run_all_experiments.sh turns into a success
    # banner, so it may only be printed when every mapped thesis figure was
    # rewritten. Reaching here with a shortfall would be a bug in this script.
    if sync_thesis and total_thesis != len(THESIS_FIGURE_MAP):
        raise RuntimeError(
            f"❌ [HATA] Tez şekillerinin yalnızca {total_thesis}/{len(THESIS_FIGURE_MAP)} "
            "tanesi güncellendi ama\n    hiçbir notebook için gerekçe kaydedilmedi; "
            "aktarım başarılı sayılamaz."
        )

    print("=" * 65)
    print(f"İşlem Tamamlandı: Toplam {total_pngs} PNG Grafik ve {total_htmls} HTML Tablo aktarıldı.")
    print(f"Çıktı Dizini: {PNG_DIR.parent}")
    if sync_thesis:
        print(f"Tez Şekilleri: {total_thesis}/{len(THESIS_FIGURE_MAP)} dosya "
              "thesis/latex/figures/ altına güncellendi.")
    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(
        description="Jupyter Notebook'lardan tüm görsel ve tabloları dışa aktarma aracı."
    )
    parser.add_argument(
        "--execute", "-e",
        action="store_true",
        help="Çıktısı boş olan notebook'ları otomatik olarak çalıştır ve kaydet."
    )
    parser.add_argument(
        "--force-execute",
        action="store_true",
        help="Tüm notebook'ları dolu olsa bile sıfırdan baştan çalıştırıp çıktıları yenile."
    )
    parser.add_argument(
        "--lang", "-l",
        choices=["en", "tr"],
        default="en",
        help="Hangi dil klasöründeki notebook'ların işleneceği (varsayılan: en)."
    )

    args = parser.parse_args()
    try:
        run_pipeline(lang=args.lang, auto_execute=args.execute, force_execute=args.force_execute)
    except Exception as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
