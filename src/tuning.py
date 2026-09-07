"""
tuning.py

Hyperparameter Tuning and Model Optimization

This module provides a unified framework for optimizing both classical Machine 
Learning (Random Forest, XGBoost, LightGBM) and Deep Learning (CNN, LSTM, Hybrid) 
models. It handles randomized and grid search, cross-validation, and early stopping.

Key Components:
  - run_ml_tuning: Main entry point for tree-based model optimization.
  - run_dl_randomsearch: Randomized search for PyTorch-based DL models.
  - EarlyStopping: Custom callback for DL training regularization.
  - finalize_and_evaluate_dl: Final refit and evaluation for the best DL architecture.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import os
import platform
import subprocess
import warnings
import weakref
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Set, Tuple, Optional, Union, List

# ── MacOS Threading Stability Patch ──────────────────────────────────────────
# Prevents 'OMP: Error #179: Function pthread_mutex_init failed' and 
# associated Segmentation Faults during joblib parallel execution.
if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    # Also limit other library thread pools inside workers
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
import lightgbm as lgb

# Progress bar support
import contextlib
import joblib
from tqdm.auto import tqdm
from sklearn.model_selection import ParameterGrid

import gc
import numpy as np
import pandas as pd
from src.config_utils import load_paths_config
from src.models.evaluation import evaluate_regression, is_degenerate_prediction

# Deep Learning Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.dl_runtime_predictor import RuntimePredictorCNN, RuntimePredictorLSTM, RuntimePredictorCNNLSTM
import copy
import time
import random
import itertools
import json

DL_SEED = 42


def seed_everything(seed: int = DL_SEED) -> None:
    """
    Seed every RNG that influences deep learning results.

    This must be called *before* a model is constructed, not after. PyTorch
    draws initial weights from the global RNG at construction time, so seeding
    inside the training routine leaves the initialisation governed by whatever
    state the previous trial happened to leave behind. Each trial then starts
    from different weights and the reported metrics cannot be reproduced.
    Seeding here makes a given (seed, architecture, hyperparameter) triple
    yield bit-identical weights on every run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_default_device() -> str:
    """Automatically detect Apple Silicon (MPS), CUDA, or fallback to CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

# =====================================================================
# CHECKPOINT SYSTEM — Saves intermediate results to disk
# =====================================================================
_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "results" / "checkpoints"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files whose content determines a checkpoint's numbers: changing any of them
# invalidates every existing checkpoint's provenance match and forces a refit.
#
# The list used to hold only feature_engineering.py and tuning.py, which
# advertised a coverage it did not have: the DL architectures (hidden sizes,
# dropout defaults), the metric definitions the reported numbers come from, the
# loader that resolves the training CSV, and every search grid / cv / seed in
# configs/models.yaml could all change while model_artifact_is_current() kept
# returning True -- so no model was refit and no checkpoint was recomputed,
# even though the published numbers would have been different.
_PROVENANCE_SRC_FILES = (
    "configs/models.yaml",
    "configs/paths.yaml",
    "src/data_loading.py",
    "src/feature_engineering.py",
    "src/models/dl_runtime_predictor.py",
    "src/models/evaluation.py",
    "src/tuning.py",
)

# The driving notebooks belong in the same fingerprint: most of what decides a
# checkpoint's numbers lives in their cells rather than in src/ -- the
# train/test fraction (test_size=0.20), DL_SEEDS, the search and final-refit
# budgets (n_iter, tuning_epochs, final_epochs, patience), seq_len, the ablation
# feature groups, and the entire definition of the non-learned baselines
# including their join keys. Hashing only src/ let all of those change while
# both guards below still answered "yes, computed by the source code in the tree
# now", so the training cells took their `if ckpt:` branch and reported numbers
# produced under the previous settings. Notebook 04 EN and TR both write the
# same checkpoint files, so both are hashed.
#
# Notebook 05 is deliberately absent: it trains nothing and writes no
# checkpoint, so it cannot change a checkpoint's numbers -- hashing it would
# invalidate every training result whenever a scheduling cell is edited, which
# is the cry-wolf failure _compute_provenance's docstring warns about.
_PROVENANCE_NOTEBOOK_FILES = (
    "notebooks/en/04_runtime_prediction_models.ipynb",
    "notebooks/tr/04_calisma_zamani_tahmin_modelleri.ipynb",
)
_PROVENANCE_PACKAGES = ("scikit-learn", "xgboost", "lightgbm", "torch", "numpy", "pandas")

# Checkpoint files load_checkpoint() refused in this process, by absolute path.
# A refusal is this module telling the caller "recompute this", so the next
# result written under that name cannot have come from disk -- see
# save_checkpoint's ``recomputed`` argument.
_RECOMPUTE_REQUESTED: Set[str] = set()

# The tree under which each model fitted in this process was fitted, so that
# record_model_artifact can stamp the provenance of the COMPUTATION rather than
# of the write -- the rule save_checkpoint's carry-forward already follows.
#
# The record is written ONTO the model object, because module-level state does
# not survive the workflow this mechanism exists for: notebook 04's import cell
# ends in ``importlib.reload(src.tuning)``, and a reload re-executes this module
# body. A plain assignment here therefore handed a fresh, empty registry to
# precisely the run that reloads to pick a source edit up, after which
# record_model_artifact fell back to the tree standing at the WRITE -- the
# laundering the docstrings below say it prevents. An attribute rides with the
# object through the reload (and through joblib / torch pickling, so a model
# read back off disk still knows what it was fitted under), and the two globals
# are seeded with setdefault so a reload keeps what the previous module body
# had already recorded rather than starting blank.
#
# The weak registry covers objects that refuse attributes; the most recent fit
# is kept separately because the notebooks' non-learned baselines (per-user
# median lookup tables) are computed in the notebook and handed to
# record_model_artifact as a destination path alone.
_FIT_PROVENANCE_ATTR = "_thesis_fit_provenance"
_FIT_PROVENANCE: "weakref.WeakKeyDictionary[Any, Dict[str, Any]]" = globals().setdefault(
    "_FIT_PROVENANCE", weakref.WeakKeyDictionary()
)
_PROVENANCE_AT_LAST_FIT: Optional[Dict[str, Any]] = globals().setdefault(
    "_PROVENANCE_AT_LAST_FIT", None
)

# Stamped when a model object is handed over that this process demonstrably did
# not fit. It carries no ``src_sha256``, so _provenance_is_current refuses it
# and the artifact is refit -- rather than being certified on the strength of
# some other model's fit, or of the tree standing at the write.
_FIT_UNKNOWN: Dict[str, Any] = {
    "fit_provenance": "unknown",
    "note": (
        "Recorded from a model object with no fit on record in the process that "
        "wrote it, so the source tree it was fitted under is unknown. Re-run "
        "notebook 04 to refit it and get a verifiable sidecar."
    ),
}


def _note_model_fit(*models: Any) -> None:
    """Record the tree ``models`` were just fitted under.

    Called by every path in this module that produces a model destined for
    disk. Without it, provenance could only be sampled at save time, and a
    save cell re-run after a source edit -- with the pre-edit model still bound
    in the kernel, which is the whole reason the save is a separate cell --
    stamped the CURRENT tree onto pre-edit weights.

    Variadic so that objects fitted together (``prepare_dl_datasets``'s two
    scalers) share one snapshot instead of hashing the tree once each.
    """
    global _PROVENANCE_AT_LAST_FIT
    _PROVENANCE_AT_LAST_FIT = _compute_provenance()
    for model in models:
        try:
            setattr(model, _FIT_PROVENANCE_ATTR, _PROVENANCE_AT_LAST_FIT)
        except (AttributeError, TypeError):
            # Refuses attributes (__slots__, a dict-shaped lookup-table
            # "model"); the registry below still covers it if it can be
            # weakly keyed, and the most recent fit above if it cannot.
            pass
        try:
            _FIT_PROVENANCE[model] = _PROVENANCE_AT_LAST_FIT
        except TypeError:
            # Not weak-referenceable (a dict-shaped lookup-table "model");
            # the most recent fit recorded above still applies to it.
            pass


def _recorded_fit(model: Any) -> Optional[Dict[str, Any]]:
    """The fit this process recorded for ``model``, or None if it recorded none."""
    provenance = getattr(model, _FIT_PROVENANCE_ATTR, None)
    if isinstance(provenance, dict):
        return provenance
    try:
        return _FIT_PROVENANCE.get(model)
    except TypeError:
        return None


def _can_record_fit(model: Any) -> bool:
    """Could :func:`_note_model_fit` have left a record on ``model``?

    Weak-referenceability is the test: anything carrying an ordinary
    ``__dict__`` is weak-referenceable too, so the only objects this answers
    False for are the dict-shaped lookup-table "models" the notebooks build,
    which can hold neither an attribute nor a registry entry. For every
    estimator, scaler and nn.Module, a missing record is therefore evidence
    that this process did not fit the object -- not an absence of evidence.
    """
    try:
        weakref.ref(model)
    except TypeError:
        return False
    return True


def _git_commit_hash(root: Path = _PROJECT_ROOT) -> Optional[str]:
    """Current HEAD commit, or None outside a git repo / if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _file_sha256(path: Path) -> Optional[str]:
    """SHA-256 of a file's bytes, or None if it can't be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _notebook_code_sha256(path: Path) -> Optional[str]:
    """SHA-256 of a notebook's code-cell sources, or None if it can't be read.

    Only ``cell['source']`` of the code cells is hashed. Hashing the .ipynb
    bytes would fold in stored outputs and execution counts, so merely running
    the notebook -- or clearing its outputs, or editing a markdown cell --
    would invalidate every checkpoint it just wrote, none of which changes a
    number.
    """
    try:
        cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
    except (OSError, ValueError):
        return None
    digest = hashlib.sha256()
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source") or ""
        if isinstance(source, list):
            source = "".join(source)
        # NUL separator: without it, moving a line between two adjacent cells
        # would leave the concatenation -- and the hash -- unchanged.
        digest.update(source.encode("utf-8") + b"\0")
    return digest.hexdigest()


def _pkg_version(name: str) -> Optional[str]:
    """Installed version of a package, or None if it isn't importable this way."""
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _training_input_files() -> Dict[str, Path]:
    """The data files a training run actually reads, keyed by repo-relative path.

    Every ``prepare_features_for_model`` call in the modelling notebooks leaves
    ``use_processed=False``, so features are rebuilt from the raw trace that
    ``load_sample`` resolves out of paths.yaml -- the processed utilization CSV
    is never opened by a modelling cell.
    """
    try:
        data_cfg = load_paths_config()["data"]
    except Exception:
        return {}

    raw_dir = str(data_cfg.get("raw_data_dir", "data/alibaba_cluster_trace/")).rstrip("/")
    names = [data_cfg.get("main_sample_file"), data_cfg.get("baseline_estimate_file")]
    return {f"{raw_dir}/{n}": _PROJECT_ROOT / raw_dir / n for n in names if n}


def _compute_provenance() -> Dict[str, Any]:
    """
    Snapshot of the environment a checkpoint's numbers were produced under:
    git commit, sha256 of the source files whose logic determines the result
    (``src_sha256``, which covers the driving notebooks' code cells as well as
    the src/ and configs/ files), hashes of the data files the run reads, key
    library versions, and the training device.

    The snapshot is both reported and enforced: load_checkpoint() prints every
    field that differs from the current environment, and
    :func:`_provenance_is_current` refuses the checkpoint on the subset that can
    move a number (``src_sha256`` and the raw-trace hashes). It was
    print-only once, and every caller read ``if ckpt:`` regardless, which is how
    pre-fix numbers kept reaching the results tables. Without the snapshot, a
    checkpoint produced under different code, data, or library versions is
    indistinguishable from a fresh one -- the ``reproducibility-4`` finding.

    ``data_sha256`` fingerprints the raw trace files, not the processed
    utilization CSV it used to hash. That CSV is written by notebook 00 and
    read by nothing in the training path, so regenerating it fired a data-change
    warning on 26 of 31 checkpoints whose numbers it cannot influence -- while a
    swap of the raw trace those numbers do come from went unreported. A guard
    that cries wolf about an unread file teaches the reader to skip the whole
    warning block, including the src_sha256 line that matters.
    """
    return {
        "git_commit": _git_commit_hash(),
        "src_sha256": {
            **{name: _file_sha256(_PROJECT_ROOT / name) for name in _PROVENANCE_SRC_FILES},
            **{name: _notebook_code_sha256(_PROJECT_ROOT / name)
               for name in _PROVENANCE_NOTEBOOK_FILES},
        },
        "data_sha256": {
            name: _file_sha256(path) for name, path in _training_input_files().items()
        },
        "package_versions": {name: _pkg_version(name) for name in _PROVENANCE_PACKAGES},
        "device": get_default_device(),
    }


def _mapping_mismatches(key: str, stored: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Per-entry differences for a dict-valued provenance field, one line each."""
    stored_map = stored.get(key)
    if not isinstance(stored_map, dict):
        # Checkpoints written before this field became a mapping (data_sha256
        # was a single processed-CSV hash) recorded nothing for these entries.
        stored_map = {}
    lines = []
    for name, current_value in (current.get(key) or {}).items():
        if stored_map.get(name) != current_value:
            lines.append(f"{key}[{name}]: {stored_map.get(name)!r} -> {current_value!r}")
    return lines


def _provenance_mismatches(stored: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Human-readable list of provenance fields that changed since a checkpoint was written."""
    if not stored:
        return ["no provenance recorded (checkpoint predates reproducibility-4 fix)"]
    mismatches = []
    for key in ("git_commit", "device"):
        if stored.get(key) != current.get(key):
            mismatches.append(f"{key}: {stored.get(key)!r} -> {current.get(key)!r}")
    for key in ("src_sha256", "data_sha256", "package_versions"):
        mismatches.extend(_mapping_mismatches(key, stored, current))
    return mismatches


def _provenance_is_current(stored: Dict[str, Any]) -> bool:
    """Was a result carrying ``stored`` produced by the tree as it stands now?

    The single predicate behind both :func:`checkpoint_is_current` and
    :func:`model_artifact_is_current`, so a metric and the model artifact beside
    it can never disagree about whether they are stale.

    ``src_sha256`` -- source files, configs and the driving notebooks' code
    cells -- must match exactly, and so must every raw-trace hash the current
    environment can compute. The data half used to be excluded on the grounds
    that ``data_sha256`` hashed a processed file the training path never reads;
    that stopped being true once the fingerprint was repointed at the raw trace
    ``load_sample`` actually opens, and until then a re-sampled or regenerated
    trace left every checkpoint and artifact certified current behind nothing
    but a printed warning -- the warning-only mechanism load_checkpoint's own
    docstring calls insufficient.

    A trace the current environment cannot hash (``None``) is not judged:
    ``baseline_estimate_file`` is optional, notebook 04 skips the Alibaba rows
    when it is missing, and reading "the reader never downloaded it" as a data
    change would discard all 31 checkpoints over a file that contributed to
    three of them.

    ``git_commit`` and ``package_versions`` stay out by decision rather than by
    omission: a commit that touches none of the hashed files cannot move a
    number, and a patch-level library bump would force a full 31-experiment
    retrain for a change that usually moves nothing. Both remain in
    load_checkpoint's printed mismatch report, where a human can judge them.
    """
    current = _compute_provenance()
    if stored.get("src_sha256") != current.get("src_sha256"):
        return False
    stored_data = stored.get("data_sha256")
    if not isinstance(stored_data, dict):
        # Checkpoints predating the mapping form recorded a single
        # processed-CSV hash here, which says nothing about the raw trace:
        # unverifiable, therefore not current.
        stored_data = {}
    return all(
        stored_data.get(name) == digest
        for name, digest in (current.get("data_sha256") or {}).items()
        if digest is not None
    )


_FEATURE_MODE_BY_EXPERIMENT = {
    "exp_a": "numeric_only",
    "exp_b": "with_categorical_onehot",
    "exp_c": "numeric_sequence",
    "exp_d": "categorical_embedding",
    "exp_e": "numeric_plus_sequential",
    "exp_f": "categorical_plus_sequential",
}

_FEATURE_MODE_BY_CHECKPOINT_NAME = {
    "exp_b_lgbm_nat": "with_categorical_native",
}


def chronological_train_validation_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    validation_size: float = 0.15,
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Split ordered training data into chronological train/validation partitions.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Ordered feature matrix.
    y : pd.Series or np.ndarray
        Ordered target vector aligned with ``X``.
    validation_size : float, default 0.15
        Fraction of trailing samples reserved for validation.

    Returns
    -------
    tuple
        ``(X_train, X_val, y_train, y_val)`` preserving original order.

    Raises
    ------
    ValueError
        If ``validation_size`` is not strictly between 0 and 1, or if the
        split would produce an empty train or validation partition.
    """
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be strictly between 0 and 1.")

    n_samples = len(X)
    split_idx = int(n_samples * (1 - validation_size))
    if split_idx <= 0 or split_idx >= n_samples:
        raise ValueError("Chronological split requires at least one train and one validation sample.")

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[:split_idx].copy()
        X_val = X.iloc[split_idx:].copy()
    else:
        X_train = X[:split_idx]
        X_val = X[split_idx:]

    if isinstance(y, pd.Series):
        y_train = y.iloc[:split_idx].copy()
        y_val = y.iloc[split_idx:].copy()
    else:
        y_train = y[:split_idx]
        y_val = y[split_idx:]

    return X_train, X_val, y_train, y_val


def _parse_checkpoint_name(experiment_name: str) -> Tuple[str, str]:
    """Split a checkpoint name into experiment tag and model name."""
    experiment_parts = experiment_name.split("_")
    if len(experiment_parts) < 3 or experiment_parts[0] != "exp":
        raise ValueError(
            "experiment_name must follow the 'exp_<tag>_<model>' pattern, "
            f"got {experiment_name!r}."
        )
    experiment_key = "_".join(experiment_parts[:2]) if len(experiment_parts) >= 2 else experiment_name
    model_name = "_".join(experiment_parts[2:]) if len(experiment_parts) > 2 else ""
    return experiment_key, model_name


def _build_checkpoint_payload(
    experiment_name: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize checkpoint payloads and backfill required metadata fields."""
    clean = {}
    for k, v in data.items():
        if isinstance(v, dict):
            clean[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                        for kk, vv in v.items()}
        elif isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v)
        else:
            clean[k] = v

    experiment_key, model_name = _parse_checkpoint_name(experiment_name)
    feature_mode = clean.get("feature_mode")
    if feature_mode is None:
        feature_mode = _FEATURE_MODE_BY_CHECKPOINT_NAME.get(
            experiment_name,
            _FEATURE_MODE_BY_EXPERIMENT.get(experiment_key),
        )

    clean.setdefault("experiment", experiment_key)
    clean.setdefault("model", model_name)
    clean.setdefault("feature_mode", feature_mode)
    clean.setdefault("train_size", None)
    clean.setdefault("test_size", None)
    clean.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    clean.setdefault("status", "complete")
    clean.setdefault("provenance", _compute_provenance())

    return clean


def save_checkpoint(
    experiment_name: str, data: Dict[str, Any], recomputed: Optional[bool] = None
) -> Path:
    """
    Save experiment results to disk so they survive kernel crashes.

    Parameters
    ----------
    experiment_name : str
        Identifier like 'exp_a_rf', 'exp_c_cnn', etc.
    data : dict
        Must contain JSON-serializable values (metrics, params).
        Model objects should be saved separately via joblib/torch.save.
    recomputed : bool, optional
        Whether ``data`` was computed by this run rather than loaded from the
        checkpoint being overwritten. Leave as None (the default) to infer it
        from whether :func:`load_checkpoint` refused this name earlier in the
        process -- a refusal means the caller was sent down its retrain path,
        so whatever it writes back is by construction a fresh computation.
        Pass it explicitly when the cell that writes is not the cell that
        loaded, since the inference cannot see that link (notebook 04's nbae02
        writes three Alibaba-estimate checkpoints after nbae01 loaded only the
        first of them).

    Returns
    -------
    Path
        Path to the saved checkpoint file.
    """
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _CHECKPOINT_DIR / f"{experiment_name}.json"
    clean = _build_checkpoint_payload(experiment_name, data)
    if recomputed is None:
        recomputed = str(path) in _RECOMPUTE_REQUESTED

    # If an existing checkpoint already reports the same metrics and the caller
    # did not recompute them, this is a re-save of an already-recorded result
    # (e.g. a notebook cell re-run after loading from disk), not a new training
    # run -- keep its original timestamp instead of bumping it to "now", which
    # would otherwise make a stale checkpoint look freshly produced on every
    # kernel restart (reproducibility-4).
    if path.exists():
        try:
            with open(path, "r") as f:
                previous = json.load(f)
            # Identical metrics are NOT by themselves evidence that nothing was
            # recomputed. Eight non-learned baselines hard-code
            # train_time = 0.0 and three further checkpoints store no
            # train_time at all, so a genuine refit of a deterministic baseline
            # reproduces its metrics block bit for bit. Inferring "loaded, not
            # computed" from that froze the OLD provenance onto a freshly
            # computed result, which checkpoint_is_current can then never
            # certify again and load_all_checkpoints drops for good -- so the
            # Experiment B and summary tables could no longer be rebuilt from
            # disk. Only the caller (or load_checkpoint's own refusal, which
            # ``recomputed`` reads) knows which of the two happened.
            #
            # When the caller did load this result, the provenance must be
            # carried over UNCHANGED. The earlier version of this guard also
            # required the provenance to match, so a re-save after a source
            # change fell through and restamped the checkpoint with the CURRENT
            # source hash -- which silently relabelled a stale result as freshly
            # produced and suppressed load_checkpoint's mismatch warning from
            # the next run onwards. Provenance has to describe the computation,
            # not the write.
            #
            # best_params deliberately takes no part in this test. The
            # non-learned baselines record their configuration as a prose
            # 'estimator' string that the Turkish notebook writes translated
            # ("single global training median, no grouping" ->
            # "gruplamasız tek global eğitim medyanı"), so comparing it made a
            # mere language switch look like a fresh computation: opening the
            # TR notebook restamped exp_b_constant_median, exp_b_constant_zero
            # and exp_b_profile_median with the current commit, source hashes
            # and timestamp although their numbers were never recomputed. A
            # human-readable label is not evidence about what produced a number.
            same_result = previous.get("metrics") == clean.get("metrics")
            if same_result and not recomputed:
                if "timestamp" in previous:
                    clean["timestamp"] = previous["timestamp"]
                if "provenance" in previous:
                    clean["provenance"] = previous["provenance"]
        except (OSError, json.JSONDecodeError):
            pass

    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=str)

    # A refusal is a one-shot licence for the write that answers it: consuming
    # it here means a save cell re-run later in the same kernel -- after a
    # source edit, with the old numbers still in memory -- falls back to the
    # conservative carry-forward above instead of relabelling them as freshly
    # produced.
    _RECOMPUTE_REQUESTED.discard(str(path))

    print(f"  [Checkpoint] Saved → {path.name}")
    return path


def model_artifact_is_current(dest: Union[str, Path]) -> bool:
    """Is the model file at ``dest`` still valid for the current source code?

    The notebooks' save cells used to read ``elif dest.exists(): skip``, which
    means a model file, once written, is never refreshed. That is how the
    trained artifacts came to predate a feature-engineering fix while the
    metrics next to them were recomputed: the ``.joblib`` files stayed at the
    pre-fix fit, notebook 05 simulated with those stale models, and the
    per-bucket table evaluated a pre-fix model against a post-fix test set --
    a pairing that never existed in any single coherent run.

    A model is current only when its sidecar still passes
    :func:`_provenance_is_current` -- the same source hashes and the same raw
    trace as the tree holds right now. The sidecar records the tree the model
    was FITTED under rather than the one standing when it was written
    (:func:`record_model_artifact`), so re-saving a model that predates a
    source edit cannot launder it into a current one. Missing sidecar = not
    current, so artifacts predating this check are refit once.

    Returns
    -------
    bool
        True only if ``dest`` exists AND its sidecar matches current sources.
    """
    dest = Path(dest)
    sidecar = dest.with_suffix(dest.suffix + ".provenance.json")
    if not dest.exists() or not sidecar.exists():
        return False
    try:
        with open(sidecar, "r") as f:
            stored = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return _provenance_is_current(stored)


def record_model_artifact(dest: Union[str, Path], model: Any = None) -> Path:
    """Write the provenance sidecar that :func:`model_artifact_is_current` reads.

    Call immediately after joblib.dump / torch.save on the same path.

    Parameters
    ----------
    dest : str or Path
        The artifact just written.
    model : object, optional
        The object that was written. Pass it whenever it is to hand: it is the
        only way to distinguish "this model was fitted under the current tree"
        from "some model was", which matters once a kernel has fitted more than
        one and the source changed in between.

    Notes
    -----
    The sidecar records the tree the model was FITTED under, not the tree
    standing at the moment of the write. Sampling ``_compute_provenance()``
    here was the one place in this module where provenance described the write:
    ``save_checkpoint`` refuses to relabel a metric it did not recompute, but
    the ``joblib.dump`` two lines below it in every notebook 04 save cell
    relabelled the model unconditionally. Re-running only the save cell -- the
    reason the save is a separate cell at all -- after a source edit, with the
    trained model still bound in the kernel, therefore left the checkpoint
    correctly marked stale and the ``.joblib``/``.pth`` beside it certified
    current with pre-edit weights. That artifact then passed notebook 05's
    stale-artifact gate, which exists to exclude exactly it, and reinstated the
    pre-fix-model / post-fix-test-set pairing this whole mechanism was built
    after. The fit record lives on the model object rather than in module state
    for the same reason: notebook 04's import cell reloads this module, and a
    reload that emptied the record put the write-time stamp straight back.

    A ``model`` that could be carrying a fit record and is not was fitted
    somewhere this process cannot see, so it is stamped ``fit_provenance:
    unknown`` -- refused by :func:`_provenance_is_current` -- instead of
    borrowing the last fit in the kernel or the tree standing at the write.

    Falling back to the current tree when this process fitted nothing keeps the
    notebooks' non-learned baselines (the per-user and per-profile median
    lookup tables) recordable: they are computed in the notebook and passed as
    a path alone, so this module has no fit to point at, and refusing to stamp
    them would make notebook 05's gate unsatisfiable for artifacts that are in
    fact fresh.
    """
    dest = Path(dest)
    sidecar = dest.with_suffix(dest.suffix + ".provenance.json")
    provenance = None
    if model is not None:
        provenance = _recorded_fit(model)
        if provenance is None and _can_record_fit(model):
            provenance = dict(_FIT_UNKNOWN)
    if provenance is None:
        provenance = _PROVENANCE_AT_LAST_FIT
    if provenance is None:
        provenance = _compute_provenance()
    with open(sidecar, "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    return sidecar


def checkpoint_is_current(experiment_name: str) -> bool:
    """Was this checkpoint's result computed by the source code in the tree now?

    Judged by :func:`_provenance_is_current`, the same predicate
    :func:`model_artifact_is_current` uses, so a metric and the model artifact
    beside it can never disagree about whether they are stale. That predicate's
    docstring records what is compared and what is deliberately left out.
    """
    path = _CHECKPOINT_DIR / f"{experiment_name}.json"
    if not path.exists():
        return False
    try:
        with open(path, "r") as f:
            stored = json.load(f).get("provenance") or {}
    except (OSError, json.JSONDecodeError):
        return False
    return _provenance_is_current(stored)


def load_checkpoint(
    experiment_name: str, allow_stale: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Load experiment results from disk.

    Returns None if the checkpoint does not exist, and -- by default -- also if
    it was computed by different source code than the tree now holds.

    That second case used to be a printed warning only, and the callers all
    read `if ckpt: metrics = ckpt['metrics']`, so a stale result was reported
    anyway: after the sweep-line correction (09cf225) every model checkpoint
    still held pre-fix numbers while the .joblib artifacts beside them had been
    refit, and the two disagreed by up to 507 s of MAE -- enough to invert the
    Random Forest / XGBoost ranking the results chapter states. A warning that
    every run prints and every caller ignores is not a guard.

    Returning None instead sends the caller down its own retrain path, which is
    what a changed pipeline requires. Pass ``allow_stale=True`` to read a
    checkpoint for inspection rather than for reporting.

    Every refusal is remembered for the process, because it is the only place
    the module learns that the result the caller writes next was computed
    rather than loaded -- see :func:`save_checkpoint`'s ``recomputed``.
    """
    path = _CHECKPOINT_DIR / f"{experiment_name}.json"
    if not path.exists():
        _RECOMPUTE_REQUESTED.add(str(path))
        return None

    with open(path, "r") as f:
        data = json.load(f)

    mismatches = _provenance_mismatches(data.get("provenance"), _compute_provenance())
    if mismatches:
        print(f"  [Checkpoint] WARNING: {path.name} provenance differs from the current environment:")
        for m in mismatches:
            print(f"    - {m}")

    if not allow_stale and not checkpoint_is_current(experiment_name):
        print(
            f"  [Checkpoint] STALE → {path.name} was computed by different source code; "
            "ignoring it so this experiment is recomputed."
        )
        _RECOMPUTE_REQUESTED.add(str(path))
        return None

    # The caller now holds a result that did come from disk, so any earlier
    # refusal for this name no longer describes what it will write back.
    _RECOMPUTE_REQUESTED.discard(str(path))
    print(f"  [Checkpoint] Loaded ← {path.name}")
    return data


def load_all_checkpoints() -> Dict[str, Dict[str, Any]]:
    """
    Load ALL saved checkpoints. Used by the final summary cells.

    Returns
    -------
    dict
        {experiment_name: data_dict, ...}
    """
    if not _CHECKPOINT_DIR.exists():
        return {}

    results = {}
    skipped = []
    for f in sorted(_CHECKPOINT_DIR.glob("*.json")):
        # The summary tables are built from this dict, so it has to apply the
        # same currency rule as load_checkpoint -- otherwise a stale result the
        # training cell correctly recomputed could still reach a table through
        # the fallback path.
        if not checkpoint_is_current(f.stem):
            skipped.append(f.stem)
            continue
        with open(f, "r") as fh:
            results[f.stem] = json.load(fh)

    print(f"  [Checkpoint] Loaded {len(results)} experiment results from disk.")
    if skipped:
        print(f"  [Checkpoint] Skipped {len(skipped)} stale checkpoint(s): {sorted(skipped)}")
    return results


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def _run_search_with_progress(search, X, y, **fit_params):
    """Helper to run a search object with a tqdm progress bar."""
    # Estimate total fits
    if hasattr(search, "n_iter"):
        n_candidates = search.n_iter
    elif hasattr(search, "param_grid"):
        n_candidates = len(ParameterGrid(search.param_grid))
    else:
        n_candidates = 1
        
    cv = search.cv
    if hasattr(cv, "get_n_splits"):
        n_splits = cv.get_n_splits(X, y)
    else:
        try:
            n_splits = cv
        except (TypeError, AttributeError):
            n_splits = 3  # fallback default
            
    total_fits = n_candidates * n_splits
    desc = search.__class__.__name__

    print(f"Starting {desc} with {total_fits} fits...")
    with tqdm_joblib(tqdm(desc=desc, total=total_fits)) as _:
        # ── MacOS Stability Enforcement ──────────────────────────────────────
        # On Darwin (macOS), we force the 'threading' backend for search.fit.
        # This avoids the 'multiple OMP runtimes' conflict common with 'loky' (forking).
        if platform.system() == "Darwin":
            with joblib.parallel_backend("threading"):
                search.fit(X, y, **fit_params)
        else:
            search.fit(X, y, **fit_params)
        
    return search.best_estimator_, search.best_params_, float(-search.best_score_)



PROJECT_ROOT = Path(__file__).resolve().parent.parent
TUNING_CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"


def _load_tuning_config(config_path: Path = TUNING_CONFIG_PATH) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Tuning config not found at {config_path}")
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("tuning", {})


def _get_common(cfg: Dict[str, Any]) -> Dict[str, Any]:
    common = cfg.get("common", {})
    return {
        "cv": int(common.get("cv", 3)),
        "scoring": str(common.get("scoring", "neg_mean_absolute_error")),
        "n_jobs": int(common.get("n_jobs", -1)),
        "verbose": int(common.get("verbose", 1)),
        "random_state": int(common.get("random_state", 42)),
        "n_iter": int(common.get("n_iter", 30)),
    }


def _make_cv(common: Dict[str, Any]) -> TimeSeriesSplit:
    """
    Build the inner cross-validation splitter used during hyperparameter search.

    Uses an expanding-window temporal split rather than a shuffled ``KFold``.
    The job table is ordered by submission time, so shuffled folds would let a
    configuration be validated on jobs that precede its own training data --
    the same leakage the outer chronological train/test split exists to avoid
    (Chapter 4). ``TimeSeriesSplit`` always trains on a prefix and validates on
    the segment that follows it, so hyperparameter selection is subject to the
    same "arrow of time" constraint as the final evaluation.

    The trade-off is that the earliest fold trains on a smaller prefix than a
    ``KFold`` fold would; ``random_state`` is therefore unused here, since the
    splits are fully determined by row order.
    """
    return TimeSeriesSplit(n_splits=common["cv"])


def get_param_distributions(model_key: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = _load_tuning_config() if cfg is None else cfg
    mapping = {
        "rf": "random_forest",
        "random_forest": "random_forest",
        "xgb": "xgboost",
        "xgboost": "xgboost",
        "lgbm": "lightgbm",
        "lightgbm": "lightgbm",
    }
    key = mapping.get(model_key)
    if key is None:
        raise ValueError("model_key must be one of: rf/xgb/lgbm")
    return cfg.get(key, {}) or {}


# ---------------------------------------------------------------------
# Training loss, per model family
#
# ``objective`` is a fixed constructor argument, not a tuned hyperparameter:
# it appears in no search grid in configs/models.yaml, so it is never in
# ``best_params`` and has to be set at every construction site or the library
# default takes over. Each family therefore names its loss here, once.
#
# LightGBM is L1, matching what everything downstream selects and reports on:
# _get_common fixes ``scoring: neg_mean_absolute_error`` for all three
# families, the results tables rank on MAE, and notebook 05 picks its SJF-Pred
# model on MAE. Its search already used ``regression_l1`` while its final refit
# silently fell back to LightGBM's default L2; naming the objective here is
# what closes that gap.
#
# XGBoost deliberately does NOT follow it onto L1, even though that leaves one
# booster fitted on a loss it is not ranked by. ``reg:absoluteerror`` starts
# from the training median (base_score 596 s here) and on this heavy-tailed
# target the chronological 10% validation split in finalize_ml_model sees its
# MAE rise monotonically from round 3 onward, so early stopping ends the refit
# at 3 trees in Experiment A and 4 in Experiment B. Measured on the real split
# with the stored best_params, switching only this constant: Exp A R2
# 0.019 -> -0.111, MdAE 3032 -> 805, distinct predictions 2689 -> 25 over
# 16,437 test jobs; Exp B (native categorical, the best tree-model R2 in the
# thesis) R2 0.119 -> -0.108. The MAE it buys (-8.8% in Exp A) is the model
# degenerating toward the conditional median -- the collapse
# _warn_if_ensemble_collapsed exists to flag -- and a 3-tree, 25-value
# predictor cannot rank jobs, which is the only thing notebook 05 asks of it.
# Nor would it remove the confound it was meant to remove: it would leave two
# collapsed L1 boosters against one L2 forest.
#
# RandomForest stays on squared error for an unrelated reason: scikit-learn's
# ``criterion="absolute_error"`` evaluates a median at every candidate split
# and is orders of magnitude slower than ``squared_error`` on an 80k-row
# training set, which puts a CV grid search out of reach.
#
# The loss is therefore NOT uniform across the three families and cannot be
# made uniform here. It travels in the metrics dict instead (``train_loss``,
# set by _training_loss below), so every checkpoint and every results table
# states per row which loss produced it rather than the asymmetry being
# visible only in this file.
_XGB_OBJECTIVE = "reg:squarederror"
_LGBM_OBJECTIVE = "regression_l1"
_RF_DEFAULT_CRITERION = "squared_error"


# ============================================
# IMPROVED: XGBoost with Early Stopping in CV
# ============================================
class XGBRegressorCV(xgb.XGBRegressor):
    """
    XGBoost wrapper that uses early stopping during CV.
    
    This ensures n_estimators search is realistic and prevents
    overfitting during hyperparameter tuning.
    """
    def fit(self, X, y, **fit_params):
        # Split X, y into train/validation
        import gc
        
        X_tr, X_val, y_tr, y_val = chronological_train_validation_split(
            X, y, validation_size=0.15
        )
        
        # Add eval set for early stopping
        fit_params['eval_set'] = [(X_val, y_val)]
        fit_params['verbose'] = False # Disable verbose to reduce notebook output bloat
        
        # Call parent fit
        super().fit(X_tr, y_tr, **fit_params)
        
        # Explicit cleanup
        del X_tr, X_val, y_tr, y_val
        gc.collect()
        return self


class LGBMRegressorCV(lgb.LGBMRegressor):
    """
    LightGBM wrapper that uses early stopping during CV.
    """
    def fit(self, X, y, **fit_params):
        import gc
        
        X_tr, X_val, y_tr, y_val = chronological_train_validation_split(
            X, y, validation_size=0.15
        )
        
        # Add eval set
        fit_params['eval_set'] = [(X_val, y_val)]
        fit_params['eval_metric'] = 'mae'
        
        # Early stopping callback
        if 'callbacks' not in fit_params:
            fit_params['callbacks'] = [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0) # Disable logging to reduce bloat
            ]
        
        super().fit(X_tr, y_tr, **fit_params)
        
        # Explicit cleanup
        del X_tr, X_val, y_tr, y_val
        gc.collect()
        return self


# ============================================
# Public API: RandomizedSearch (Improved)
# ============================================
def run_randomsearch_rf(X, y, n_iter: Optional[int] = None, random_state: Optional[int] = None):
    """
    Perform randomized hyperparameter search for Random Forest.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    n_iter : int, optional
        Number of parameter settings that are sampled.
    random_state : int, optional
        Seed used by the random number generator.

    Returns
    -------
    tuple
        (best_estimator, best_params, best_score)
    """

    cfg = _load_tuning_config()
    common = _get_common(cfg)
    param_dist = get_param_distributions("rf", cfg)

    rs = common["random_state"] if random_state is None else int(random_state)
    estimator = RandomForestRegressor(
        random_state=rs,
        n_jobs=1,  # Single-threaded tree building to avoid nested parallelism crash
    )
    
    cv = _make_cv(common)
    local_n_iter = n_iter if n_iter is not None else common["n_iter"]
    
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=local_n_iter,
        scoring=common["scoring"],
        cv=cv,
        random_state=rs,
        n_jobs=common["n_jobs"],  # Optimized core usage from config
        verbose=common["verbose"],
    )
    
    result = _run_search_with_progress(search, X, y)
    gc.collect()  # Free memory after search
    return result


def run_randomsearch_xgb(X, y, n_iter: Optional[int] = None, random_state: Optional[int] = None,
                         enable_categorical: bool = False):
    """
    Perform randomized hyperparameter search for XGBoost.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    n_iter : int, optional
        Number of iterations for random search.
    random_state : int, optional
        Seed for reproducibility.
    enable_categorical : bool, default False
        Let XGBoost split on pandas ``category`` columns natively instead of
        requiring one-hot input. Without this, Experiment B compares LightGBM's
        native categorical handling against one-hot XGBoost, which confounds the
        library with the encoding.

    Returns
    -------
    tuple
        (best_estimator, best_params, best_score)
    """
    cfg = _load_tuning_config()
    common = _get_common(cfg)
    param_dist = get_param_distributions("xgb", cfg)

    rs = common["random_state"] if random_state is None else int(random_state)
    
    # Use CV wrapper with early stopping
    estimator = XGBRegressorCV(
        random_state=rs,
        n_jobs=1,  # Single-threaded to avoid OpenMP conflicts on macOS M1
        objective=_XGB_OBJECTIVE,
        eval_metric="mae",
        tree_method="hist",
        early_stopping_rounds=30,
        enable_categorical=enable_categorical,
    )
    
    cv = _make_cv(common)
    local_n_iter = n_iter if n_iter is not None else common["n_iter"]
    
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=local_n_iter,
        scoring=common["scoring"],
        cv=cv,
        random_state=rs,
        n_jobs=common["n_jobs"], # Parallel search enabled
        verbose=2,
    )
    
    result = _run_search_with_progress(search, X, y)
    gc.collect()
    return result


def run_randomsearch_lgbm(X, y, n_iter: Optional[int] = None, random_state: Optional[int] = None, 
                          categorical_feature=None):
    """
    Perform randomized hyperparameter search for LightGBM.
    """
    cfg = _load_tuning_config()
    common = _get_common(cfg)
    param_dist = get_param_distributions("lgbm", cfg)

    rs = common["random_state"] if random_state is None else int(random_state)
    
    # Use CV wrapper with early stopping
    estimator = LGBMRegressorCV(
        random_state=rs,
        n_jobs=1,
        objective=_LGBM_OBJECTIVE,
        verbose=-1,
    )
    
    cv = _make_cv(common)
    local_n_iter = n_iter if n_iter is not None else common["n_iter"]
    
    # For categorical features
    fit_params = {}
    if categorical_feature is not None:
        fit_params['categorical_feature'] = categorical_feature
    
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=local_n_iter,
        scoring=common["scoring"],
        cv=cv,
        random_state=rs,
        n_jobs=common["n_jobs"],
        verbose=2,
    )
    
    result = _run_search_with_progress(search, X, y, **fit_params)
    gc.collect()
    return result


# ============================================
# Public API: GridSearch
# ============================================
def run_gridsearch_rf(X, y, param_grid: Dict[str, Any], random_state: Optional[int] = None):
    cfg = _load_tuning_config()
    common = _get_common(cfg)

    rs = common["random_state"] if random_state is None else int(random_state)
    estimator = RandomForestRegressor(random_state=rs, n_jobs=1)
    
    cv = _make_cv(common)
    search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring=common["scoring"],
        cv=cv,
        n_jobs=common["n_jobs"], # Parallel core usage
        verbose=2,
    )
    
    result = _run_search_with_progress(search, X, y)
    gc.collect()
    return result


def run_gridsearch_xgb(X, y, param_grid: Dict[str, Any], random_state: Optional[int] = None,
                       enable_categorical: bool = False):
    cfg = _load_tuning_config()
    common = _get_common(cfg)

    rs = common["random_state"] if random_state is None else int(random_state)
    estimator = XGBRegressorCV(
        random_state=rs,
        n_jobs=1,
        objective=_XGB_OBJECTIVE,
        eval_metric="mae",
        tree_method="hist",
        early_stopping_rounds=30,
        enable_categorical=enable_categorical,
    )
    
    cv = _make_cv(common)
    search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring=common["scoring"],
        cv=cv,
        n_jobs=common["n_jobs"],
        verbose=2,
    )
    
    result = _run_search_with_progress(search, X, y)
    gc.collect()
    return result


def run_gridsearch_lgbm(X, y, param_grid: Dict[str, Any], random_state: Optional[int] = None,
                        categorical_feature=None):
    cfg = _load_tuning_config()
    common = _get_common(cfg)

    rs = common["random_state"] if random_state is None else int(random_state)
    estimator = LGBMRegressorCV(
        random_state=rs,
        n_jobs=1,
        objective=_LGBM_OBJECTIVE,
        verbose=-1,
    )
    
    cv = _make_cv(common)
    
    fit_params = {}
    if categorical_feature is not None:
        fit_params['categorical_feature'] = categorical_feature
    
    search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring=common["scoring"],
        cv=cv,
        n_jobs=common["n_jobs"],
        verbose=2,
    )
    
    result = _run_search_with_progress(search, X, y, **fit_params)
    gc.collect()
    return result


def _training_loss(model_name: str, best_params: Dict[str, Any]) -> str:
    """The loss the final refit of ``model_name`` actually minimises."""
    if model_name == "xgb":
        return _XGB_OBJECTIVE
    if model_name == "lgbm":
        return _LGBM_OBJECTIVE
    # RF's criterion IS searchable (configs/models.yaml offers it), so read the
    # winner rather than assume the default.
    return str(best_params.get("criterion", _RF_DEFAULT_CRITERION))


def _searched_n_estimators(best_params: Dict[str, Any]) -> Optional[int]:
    """The tree budget the search was allowed, or None if it never set one."""
    value = best_params.get("n_estimators")
    return None if value is None else int(value)


# A boosted "ensemble" this small is not a tuned model, it is a statement about
# the internal early-stopping split: the loss curve turned upward before the
# search budget was anywhere near spent.
_MIN_ENSEMBLE_TREES = 10


def _warn_if_ensemble_collapsed(
    model_name: str, effective: int, searched: Optional[int]
) -> None:
    """Say so when early stopping left a boosted model with almost no trees.

    Exp A LightGBM reached the thesis's best Experiment A MAE with a refit of
    exactly ONE tree, alongside an R2 of -0.1219 -- the signature of a single
    L1 tree predicting group medians -- while the only tree count that reached
    the hyperparameter table was ``best_params['n_estimators']``, i.e. 1300.
    Nothing in the pipeline noticed: ``is_degenerate_prediction`` (called by
    evaluate_regression) judges prediction-value collapse, and one tree with
    dozens of leaves still emits dozens of distinct values, so it passes.
    """
    floor = _MIN_ENSEMBLE_TREES
    if searched:
        floor = max(floor, searched // 100)
    if effective > floor:
        return
    budget = "unset" if searched is None else str(searched)
    warnings.warn(
        f"{model_name.upper()} final refit early-stopped at {effective} tree(s) "
        f"out of a search budget of {budget}. The returned model is not the "
        "tuned ensemble its best_params describe -- a table printing "
        "n_estimators for this row overstates the model by orders of "
        "magnitude -- and stopping this early is evidence about the internal "
        "10% chronological holdout, not a converged fit. Report "
        "n_estimators_effective, and treat the score as provisional.",
        UserWarning,
        stacklevel=3,
    )


def finalize_ml_model(
    model_name: str,
    best_params: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    random_state: int = 42,
    verbose: bool = True,
    categorical_feature: Optional[List[str]] = None,
    enable_categorical: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """
    Final refit of the ML model with best parameters and evaluation on test set.

    Parameters
    ----------
    model_name : str
        'rf', 'xgb', or 'lgbm'.
    best_params : dict
        Hyperparameters found during tuning.
    X_train, y_train : array-like
        Full training data.
    X_test, y_test : array-like
        Test data for final metric reporting.
    random_state : int, default=42
        Seed for reproducibility.
    verbose : bool, default=True
        Whether to print status.
    categorical_feature : list of str, optional
        Column names LightGBM should treat as categorical.
    enable_categorical : bool, default False
        XGBoost equivalent: split on pandas ``category`` columns natively
        rather than requiring one-hot input.

    Returns
    -------
    tuple
        ``(final_model, metrics)``. Besides the error metrics, ``metrics``
        carries what a results table needs in order to state what was actually
        fitted: ``train_loss`` (the loss this family minimises -- L1 for
        LightGBM, squared error for XGBoost and RandomForest, see
        ``_XGB_OBJECTIVE``) and, for the boosters,
        ``n_estimators_effective`` beside
        ``n_estimators_searched``.
    """
    start_time = time.time()
    
    cfg = _load_tuning_config()
    common = _get_common(cfg)
    safe_n_jobs = common.get("n_jobs", 1)
    
    if model_name == "rf":
        final_model = RandomForestRegressor(**best_params, random_state=random_state, n_jobs=safe_n_jobs)
        final_model.fit(X_train, y_train)
    else:
        # Split for internal validation: this first fit's only purpose is to
        # use early stopping to find how many trees the loss curve actually
        # supports (best_iteration). It is NOT the model that gets returned
        # or evaluated below.
        X_tr_split, X_val_split, y_tr_split, y_val_split = chronological_train_validation_split(
            X_train, y_train, validation_size=0.10
        )

        if model_name == "xgb":
            search_model = xgb.XGBRegressor(
                **best_params,
                random_state=random_state,
                n_jobs=safe_n_jobs,
                tree_method="hist",
                enable_categorical=enable_categorical,
                # Match the loss AND the eval_metric used during hyperparameter
                # search (run_randomsearch_xgb / run_gridsearch_xgb build
                # XGBRegressorCV with the same pair), so the tree count found
                # here is the one those hyperparameters were selected under.
                # eval_metric="mae" steers early stopping and scoring only; the
                # gradient stays squared error, and _XGB_OBJECTIVE above says
                # why that asymmetry is kept rather than closed here.
                objective=_XGB_OBJECTIVE,
                eval_metric="mae",
                early_stopping_rounds=50
            )
            search_model.fit(
                X_tr_split, y_tr_split,
                eval_set=[(X_val_split, y_val_split)],
                verbose=False
            )
            # +1: best_iteration is the 0-indexed round with the best score.
            best_n_estimators = max(1, int(search_model.best_iteration) + 1)

            # Refit on ALL of X_train -- not the 90% split above -- with the
            # tree count early stopping found, and no further early
            # stopping. Without this second fit, the ~10% of chronologically
            # last (closest-to-test-period) training rows held out above
            # never contributed a single gradient update to XGB/LGBM, while
            # RF's bagging sees 100% of X_train: an unequal data budget
            # between model families (code_bugs-6).
            final_model = xgb.XGBRegressor(
                **{**best_params, "n_estimators": best_n_estimators},
                random_state=random_state,
                n_jobs=safe_n_jobs,
                tree_method="hist",
                enable_categorical=enable_categorical,
                objective=_XGB_OBJECTIVE,
                eval_metric="mae",
            )
            final_model.fit(X_train, y_train, verbose=False)
        else: # lgbm
            search_model = lgb.LGBMRegressor(
                **best_params,
                random_state=random_state,
                n_jobs=safe_n_jobs,
                # Match the objective used during hyperparameter search
                # (run_randomsearch_lgbm / run_gridsearch_lgbm build
                # LGBMRegressorCV with the same constant). Without setting it
                # here, the final refit silently falls back to LightGBM's
                # default "regression" (L2 / squared-error) objective -- a
                # different loss function than the one hyperparameters were
                # chosen for.
                objective=_LGBM_OBJECTIVE,
            )
            search_model.fit(
                X_tr_split, y_tr_split,
                eval_set=[(X_val_split, y_val_split)],
                eval_metric="mae",
                callbacks=[lgb.early_stopping(50, verbose=False)],
                categorical_feature=categorical_feature
            )
            best_n_estimators = max(1, int(search_model.best_iteration_))

            # Same rationale as the XGB branch above: refit on 100% of
            # X_train with the discovered tree count, no further early
            # stopping, so LGBM's data budget matches RF's (code_bugs-6).
            final_model = lgb.LGBMRegressor(
                **{**best_params, "n_estimators": best_n_estimators},
                random_state=random_state,
                n_jobs=safe_n_jobs,
                objective=_LGBM_OBJECTIVE,
            )
            final_model.fit(X_train, y_train, categorical_feature=categorical_feature)
    
    train_time = time.time() - start_time
    
    # Test Evaluation
    # Clipped to non-negative for the same reason _finalize_dl_single already
    # clips DL predictions: a negative predicted runtime is never physically
    # meaningful, and an unclipped tree prediction (observed for LightGBM
    # with native categoricals) would let notebook 05's SJF-Pred treat that
    # job as the shortest in the queue (code_bugs-5).
    y_pred = np.maximum(final_model.predict(X_test), 0)
    metrics = evaluate_regression(y_test, y_pred)
    metrics['train_time'] = train_time
    # The loss this family was actually fitted with, carried into the
    # checkpoint so a results table can state it per row. Only LightGBM
    # minimises the metric it is ranked on; XGBoost and RandomForest do not
    # (see _XGB_OBJECTIVE above for why neither is forced onto L1), and a table
    # that prints MAE without this column cannot tell a reader which comparison
    # is controlled for loss.
    metrics['train_loss'] = _training_loss(model_name, best_params)
    if model_name in ("xgb", "lgbm"):
        # The tree count early stopping actually found and that final_model
        # was refit with -- distinct from best_params['n_estimators'], which
        # is only the search-phase upper bound (modeling-3 / modeling-11).
        metrics['n_estimators_effective'] = best_n_estimators
        # Both numbers, so a hyperparameter table can print the size of the
        # model instead of the size of the search. best_params['n_estimators']
        # is the only tree count that ever reached the thesis table, and for
        # Exp A LightGBM it says 1300 for a model that is one tree.
        metrics['n_estimators_searched'] = _searched_n_estimators(best_params)
        _warn_if_ensemble_collapsed(
            model_name, best_n_estimators, metrics['n_estimators_searched']
        )

    # The provenance of the model that just came out of this call, so a save
    # cell re-run later in the same kernel -- after a source edit, with this
    # object still bound -- cannot stamp the current tree onto it.
    _note_model_fit(final_model)

    if verbose:
        print(f"[{model_name.upper()}][FinalRefit] Completed in {train_time:.2f}s")
        print(f"[{model_name.upper()}][TestMetrics] MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.4f}")
        if 'n_estimators_effective' in metrics:
            print(f"[{model_name.upper()}][ModelSize] trees refit: "
                  f"{metrics['n_estimators_effective']} "
                  f"(search budget: {metrics['n_estimators_searched']}), "
                  f"loss: {metrics['train_loss']}")

    return final_model, metrics


# -----------------------------
# Grid convergence (modeling-11)
# -----------------------------
def grid_boundary_params(param_grid: Dict[str, Any], best_params: Dict[str, Any]) -> List[str]:
    """Tuned parameters whose winning value sits on the edge of the grid searched.

    A narrow grid centred on the random-search winner can only return values
    it was offered. When the winner is the smallest or largest value on the
    grid, the search has not converged: the true optimum may lie beyond the
    range that was searched, and the reported "best" is as likely to be a
    boundary artefact as a real optimum (``modeling-11``). Parameters whose
    grid holds a single value, or whose values are not order-comparable
    (``max_features='sqrt'``), cannot be at a boundary and are skipped.
    """
    def _orderable(v) -> bool:
        # bool is a subclass of int but has no meaningful "edge"; strings compare
        # lexicographically, which says nothing about whether 'sqrt' is an extreme
        # choice. Only genuinely numeric values can sit at a boundary.
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    at_edge: List[str] = []
    for name, values in param_grid.items():
        vals = [v for v in values if v is not None]
        if len(vals) < 2 or not all(_orderable(v) for v in vals):
            continue
        chosen = best_params.get(name)
        if chosen is None or not _orderable(chosen):
            continue
        if chosen == min(vals) or chosen == max(vals):
            at_edge.append(name)
    return at_edge


def run_gridsearch_iterative(
    model_name: str,
    search_fn,
    X,
    y,
    seed_params: Dict[str, Any],
    max_rounds: int = 2,
    verbose: bool = True,
    **search_kwargs,
):
    """Grid search that widens once more when the winner lands on the grid's edge.

    ``make_narrow_grid`` builds a grid *around* the parameters it is given, so
    re-centring it on a boundary winner automatically extends the search in
    that direction. Repeating until the winner is interior (or ``max_rounds``
    is reached) is the cheap version of the iterative-narrowing the fix for
    ``modeling-11`` asks for -- one extra GridSearchCV per model at worst.

    Returns
    -------
    tuple
        ``(model, best_params, score, diagnostics)`` where ``diagnostics``
        records how many rounds ran and which parameters (if any) were still
        at a boundary when the search stopped -- so a run that hit
        ``max_rounds`` without converging is visible in the checkpoint rather
        than silently reported as a converged optimum.
    """
    grid = make_narrow_grid(model_name, seed_params)
    model, best, score = search_fn(X, y, param_grid=grid, **search_kwargs)
    rounds = 1

    while rounds < max_rounds:
        edge = grid_boundary_params(grid, best)
        if not edge:
            break
        if verbose:
            print(f"[{model_name}] grid winner sits at the edge for {edge}; "
                  f"re-centring and searching once more (round {rounds + 1}/{max_rounds})")
        grid = make_narrow_grid(model_name, best)
        model, best, score = search_fn(X, y, param_grid=grid, **search_kwargs)
        rounds += 1

    still_at_edge = grid_boundary_params(grid, best)
    if verbose and still_at_edge:
        print(f"[{model_name}] NOTE: after {rounds} round(s) these parameters are still at a "
              f"grid boundary: {still_at_edge} -- treat the reported optimum as unconverged.")
    return model, best, score, {
        "grid_rounds": rounds,
        "params_at_boundary": still_at_edge,
        "seed_params": dict(seed_params),
    }


# -----------------------------
# Narrow grid builder
# -----------------------------
def make_narrow_grid(
    model_name: str,
    best_params: Dict[str, Any],
    max_grid_size: int = 81,
) -> Dict[str, Any]:
    """Programmatic narrow grid around best_params.

    Rules:
    - No hard-coded "full grid": derive from best_params
    - Integer params: small deltas
    - Float params: multiplicative factors (0.5, 0.8, 1.0, 1.2, 1.5)
    - subsample/colsample clipped to the library-legal (0, 1.0]
    - max_features / max_samples (if float) clipped to [0.1, 1.0]
    - min_samples_split>=2, min_samples_leaf>=1
    - XGB/LGBM gamma/min_split_gain>=0, reg_alpha/reg_lambda>=0
    - LGBM max_depth: if -1 -> [-1, 10, 20], else neighborhood
    - list values unique + sorted when possible
    - grid size capped by max_grid_size (shrinks less-important params first)
    """
    if model_name.lower() not in {"rf", "xgb", "lgbm", "cnn", "lstm", "hybrid"}:
        raise ValueError("model_name must be one of: rf/xgb/lgbm/cnn/lstm/hybrid")

    if not best_params:
        return {}

    def _uniq_sorted(values):
        uniq = []
        for v in values:
            if v not in uniq:
                uniq.append(v)
        try:
            return sorted(uniq)
        except TypeError:
            return uniq

    def _clip_fraction(v: float) -> float:
        """Keep a sampling fraction inside the (0, 1] both libraries accept.

        This used to floor at 0.5, which is well inside the range the search
        itself offers: configs/models.yaml lists colsample_bytree [0.3, 0.5]
        for XGBoost and [0.3, 0.5, 0.7] for LightGBM. When random search picked
        0.3, every multiplicative variant (0.15 ... 0.45) collapsed onto 0.5 and
        the narrow grid degenerated to the single value [0.5] -- a point the
        search had never scored against the 0.3 it actually selected, silently
        substituted into best_params and into the final refit.
        """
        return max(1e-6, min(1.0, float(v)))

    def _clip_01_10(v: float) -> float:
        return max(0.1, min(1.0, float(v)))

    def _int_deltas(v: int) -> list[int]:
        v = int(v)
        if v >= 2000:
            steps = [-500, 0, 500]
        elif v >= 800:
            steps = [-200, 0, 200]
        elif v >= 300:
            steps = [-100, 0, 100]
        elif v >= 50:
            steps = [-10, 0, 10]
        elif v >= 10:
            steps = [-5, 0, 5]
        elif v >= 5:
            steps = [-2, 0, 2]
        else:
            steps = [-1, 0, 1]
        return [v + s for s in steps]

    def _narrow_int(name: str, v: Any) -> list[Any]:
        if v is None:
            return [None]
        vals = _int_deltas(int(v))

        if name == "min_samples_split":
            vals = [x for x in vals if x >= 2]
        if name == "min_samples_leaf":
            vals = [x for x in vals if x >= 1]
        if name in {
            "n_estimators", "num_leaves", "max_bin", "min_child_samples",
            "num_filters", "hidden_size", "lstm_hidden_size",
        }:
            vals = [x for x in vals if x >= 2]
        if name in {"num_layers", "lstm_num_layers"}:
            # A single-layer (LSTM/hybrid) recurrent stack is a valid,
            # commonly-best configuration -- filtering at >=2 here silently
            # dropped 1 from the grid even when random search selected it,
            # so the narrow grid and final refit could never reproduce that
            # choice (modeling-5).
            vals = [x for x in vals if x >= 1]

        if name == "subsample_freq":
            vals = [x for x in vals if x >= 0]

        if name == "max_depth":
            # RF: can be None; XGB/LGBM: int >= 1 or -1 (LGBM handled specifically)
            vals = [x for x in vals if x >= 1]

        if name == "min_child_weight":
            vals = [x for x in vals if x >= 0]

        if name == "max_delta_step":
            vals = [x for x in vals if x >= 0]

        if name == "kernel_size":
            vals = [x for x in vals if x >= 3]

        return _uniq_sorted(vals) or [int(v)]

    def _narrow_float(name: str, v: Any) -> list[float]:
        v = float(v)
        factors = [0.5, 0.8, 1.0, 1.2, 1.5]
        vals = [v * f for f in factors]

        if name in {"subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode"}:
            vals = [_clip_fraction(x) for x in vals]

        if name in {"max_features", "max_samples"}:
            vals = [_clip_01_10(x) for x in vals]

        if name in {"reg_alpha", "reg_lambda", "gamma", "min_split_gain"}:
            vals = [max(0.0, float(x)) for x in vals]

        if name == "learning_rate":
            vals = [max(1e-6, float(x)) for x in vals]

        vals = [round(float(x), 6) for x in vals]
        return _uniq_sorted(vals) or [round(v, 6)]

    skip = {
        "random_state",
        "n_jobs",
        "objective",
        "eval_metric",
        "tree_method",
        "verbosity",
        "verbose",
        "device",
        "booster",
        "early_stopping_rounds",
    }

    if model_name == "rf":
        allowed = {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "max_samples",
            "criterion",
            "ccp_alpha",
        }
    elif model_name == "xgb":
        allowed = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "colsample_bylevel",
            "colsample_bynode",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "max_delta_step",
        }
    elif model_name == "lgbm":
        allowed = {
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "subsample",
            "subsample_freq",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "min_child_samples",
            "min_split_gain",
            "max_bin",
        }
    elif model_name.lower() == "cnn":
        allowed = {"num_filters", "kernel_size", "learning_rate", "batch_size", "dropout"}
    elif model_name.lower() == "lstm":
        allowed = {"hidden_size", "num_layers", "learning_rate", "batch_size", "dropout"}
    elif model_name.lower() == "hybrid":
        allowed = {"num_filters", "kernel_size", "lstm_hidden_size", "lstm_num_layers", "learning_rate", "batch_size", "dropout"}
    else:  # fallback
        allowed = set(best_params.keys())

    grid: Dict[str, Any] = {}

    for name, v in best_params.items():
        if name in skip:
            continue
        if name not in allowed:
            continue

        # special case: LGBM max_depth
        if model_name == "lgbm" and name == "max_depth":
            try:
                iv = int(v)
            except (TypeError, ValueError):
                grid[name] = [v]
                continue
            if iv == -1:
                grid[name] = [-1, 10, 20]
            else:
                grid[name] = _narrow_int(name, iv)
            continue

        # RF max_depth may be None
        if model_name == "rf" and name == "max_depth" and v is None:
            grid[name] = [None]
            continue

        # dropout is sampled from a small fixed set in configs/models.yaml
        # (e.g. [0.2, 0.3]), not a value meant to be widened by the generic
        # multiplicative float narrowing below -- that would search dropout
        # rates the random search was never configured to consider. Carry
        # whatever random search selected through unchanged (code_bugs-3):
        # previously "dropout" was not in `allowed` for any DL model, so it
        # never reached the grid or final refit at all, which silently
        # trained the grid/final models at the framework default (0.2)
        # regardless of what random search had actually picked.
        if name == "dropout":
            grid[name] = [v]
            continue

        if isinstance(v, bool) or isinstance(v, str):
            grid[name] = [v]
        elif isinstance(v, (int, np.integer)):
            grid[name] = _narrow_int(name, int(v))
        elif isinstance(v, (float, np.float32, np.float64)):
            grid[name] = _narrow_float(name, float(v))
        else:
            grid[name] = [v]

    # Validations / clipping
        # RF: max_samples cannot be used when bootstrap=False
    if model_name == "rf":
        if best_params.get("bootstrap") is False and "max_samples" in grid:
                grid["max_samples"] = [None]
    if "min_samples_split" in grid:
        grid["min_samples_split"] = (
            [x for x in grid["min_samples_split"] if int(x) >= 2] or [int(best_params["min_samples_split"])]
        )
    if "min_samples_leaf" in grid:
        grid["min_samples_leaf"] = (
            [x for x in grid["min_samples_leaf"] if int(x) >= 1] or [int(best_params["min_samples_leaf"])]
        )
        
    for c in ["subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode"]:
        if c in grid:
            # Union the incoming value back in: clipping must never be able to
            # drop the point the grid is supposed to be centred on, or the
            # refinement stage reports a value the search never evaluated.
            vals = [_clip_fraction(x) for x in grid[c]] + [_clip_fraction(best_params[c])]
            grid[c] = _uniq_sorted([round(float(x), 6) for x in vals])

    for c in ["max_features", "max_samples"]:
        if c in grid and any(isinstance(x, float) for x in grid[c]):
            grid[c] = _uniq_sorted([_clip_01_10(x) for x in grid[c]])

    if "reg_alpha" in grid:
        grid["reg_alpha"] = _uniq_sorted([max(0.0, float(x)) for x in grid["reg_alpha"]])
    if "reg_lambda" in grid:
        grid["reg_lambda"] = _uniq_sorted([max(0.0, float(x)) for x in grid["reg_lambda"]])

    if "gamma" in grid:
        grid["gamma"] = _uniq_sorted([max(0.0, float(x)) for x in grid["gamma"]])
    if "min_split_gain" in grid:
        grid["min_split_gain"] = _uniq_sorted([max(0.0, float(x)) for x in grid["min_split_gain"]])

    if "min_child_weight" in grid:
        grid["min_child_weight"] = (
            [x for x in grid["min_child_weight"] if float(x) >= 0.0] or [float(best_params["min_child_weight"])]
        )

    if "min_child_samples" in grid:
        grid["min_child_samples"] = (
            [x for x in grid["min_child_samples"] if int(x) >= 1] or [int(best_params["min_child_samples"])]
        )
    if "subsample_freq" in grid:
        grid["subsample_freq"] = (
            [x for x in grid["subsample_freq"] if int(x) >= 0] or [int(best_params["subsample_freq"])]
        )
    if "max_bin" in grid:
        grid["max_bin"] = (
            [x for x in grid["max_bin"] if int(x) >= 32] or [int(best_params["max_bin"])]
        )
    if "max_delta_step" in grid:
        grid["max_delta_step"] = (
            [x for x in grid["max_delta_step"] if int(x) >= 0] or [int(best_params["max_delta_step"])]
        )

    def _grid_size(g: Dict[str, Any]) -> int:
        size = 1
        for values in g.values():
            size *= max(1, len(values))
        return size

    # Prevent combinatorial explosions by shrinking secondary params first
    if max_grid_size is not None and int(max_grid_size) > 0:
        max_grid_size_int = int(max_grid_size)

        if model_name == "rf":
            shrink_order = [
                "ccp_alpha",
                "criterion",
                "max_samples",
                "bootstrap",
                "max_features",
                "min_samples_leaf",
                "min_samples_split",
                "max_depth",
                "n_estimators",
            ]
        elif model_name == "xgb":
            shrink_order = [
                "max_delta_step",
                "gamma",
                "min_child_weight",
                "colsample_bynode",
                "colsample_bylevel",
                "colsample_bytree",
                "subsample",
                "reg_lambda",
                "reg_alpha",
                "max_depth",
                "n_estimators",
                "learning_rate",
            ]
        elif model_name.lower() == "cnn":
            shrink_order = ["kernel_size", "num_filters", "learning_rate", "batch_size"]
        elif model_name.lower() == "lstm":
            shrink_order = ["num_layers", "hidden_size", "learning_rate", "batch_size"]
        elif model_name.lower() == "hybrid":
            shrink_order = ["lstm_num_layers", "lstm_hidden_size", "kernel_size", "num_filters", "learning_rate", "batch_size"]
        else:  # lgbm
            shrink_order = [
                "max_bin",
                "subsample_freq",
                "min_split_gain",
                "min_child_samples",
                "colsample_bytree",
                "subsample",
                "reg_lambda",
                "reg_alpha",
                "max_depth",
                "num_leaves",
                "n_estimators",
                "learning_rate",
            ]

        while _grid_size(grid) > max_grid_size_int:
            shrunk = False
            for param in shrink_order:
                if param in grid and len(grid[param]) > 1:
                    grid[param] = [best_params[param]]
                    shrunk = True
                    break
            if not shrunk:
                break

    return grid


# ---------------------------------------------------------------------
# Dedicated DL wrappers to mirror ML experience (birebir aynı yapı)
# ---------------------------------------------------------------------

def run_randomsearch_cnn(*args, **kwargs):
    kwargs['model_name'] = 'CNN'
    if 'search_space' not in kwargs:
        kwargs['search_space'] = load_dl_config('CNN')
    return run_dl_randomsearch(*args, **kwargs)

def run_gridsearch_cnn(*args, **kwargs):
    kwargs['model_name'] = 'CNN'
    return run_dl_gridsearch(*args, **kwargs)

def run_randomsearch_lstm(*args, **kwargs):
    kwargs['model_name'] = 'LSTM'
    if 'search_space' not in kwargs:
        kwargs['search_space'] = load_dl_config('LSTM')
    return run_dl_randomsearch(*args, **kwargs)

def run_gridsearch_lstm(*args, **kwargs):
    kwargs['model_name'] = 'LSTM'
    return run_dl_gridsearch(*args, **kwargs)

def run_randomsearch_hybrid(*args, **kwargs):
    kwargs['model_name'] = 'Hybrid'
    if 'search_space' not in kwargs:
        kwargs['search_space'] = load_dl_config('Hybrid')
    return run_dl_randomsearch(*args, **kwargs)

def run_gridsearch_hybrid(*args, **kwargs):
    kwargs['model_name'] = 'Hybrid'
    return run_dl_gridsearch(*args, **kwargs)

# =====================================================================
# DEEP LEARNING TUNING
# =====================================================================

def load_dl_config(model_name: str = None, config_path: str = "configs/models.yaml"):
    """
    Loads Deep Learning hyperparameter search spaces directly from the central YAML configuration.
    """
    full_path = Path(__file__).resolve().parent.parent / config_path
    
    with open(full_path, 'r') as f:
        config = yaml.safe_load(f)

    if model_name is None:
        return config    
    
    tuning_config = config.get('tuning', {})
    
    if model_name == 'CNN':
        return tuning_config.get('cnn', {})
    elif model_name == 'LSTM':
        return tuning_config.get('lstm', {})
    elif model_name == 'Hybrid':
        return tuning_config.get('cnn_lstm', {})
    else:
        raise ValueError(f"Unknown deep learning model: {model_name}")


class SequenceJobDataset(torch.utils.data.Dataset):
    """
    Sliding window dataset for Deep Learning models.
    Converts tabular rows into 3D sequential tensors (batch, seq_len, features).
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 10):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len
        
    def __len__(self):
        return max(0, len(self.features) - self.seq_len + 1)
        
    def __getitem__(self, idx):
        x_seq = self.features[idx : idx + self.seq_len]
        y_target = self.targets[idx + self.seq_len - 1]
        return x_seq, y_target


def prepare_dl_datasets(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    val_split: float = 0.2,
    random_state: int = 42,
    seq_len: int = 10,
):
    """
    Standardizes DL data preparation: Scaling, Sequence conversion, and Dataset creation.

    Automatically splits the training data into a train and a validation set so
    that Early Stopping can monitor generalisation performance rather than
    in-sample loss.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices.
    y_train, y_test : np.ndarray
        Target vectors.
    val_split : float, default=0.2
        Fraction of trailing training samples reserved for validation.
    random_state : int, default=42
        Unused. The train/validation split is chronological, not random, so it
        is fully determined by row order. Retained for call-site compatibility.
    seq_len : int, default=10
        Length of the sliding window for temporal prediction.

    Returns
    -------
    tuple
        (train_dataset, val_dataset, test_dataset, y_test_raw, scaler_x, scaler_y, input_features)
    """
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit & Transform
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled  = scaler_x.transform(X_test)

    # Target scaling (reshape for scaler)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1))

    # The scalers are artifacts in their own right -- notebook 04 saves all
    # four and notebook 05 lists them among the files it refuses to simulate
    # without -- and this is the only place they are fitted. Recording the fit
    # here is what lets record_model_artifact stamp the tree they were fitted
    # under, and what keeps that gate satisfiable for a scaler that is fresh.
    _note_model_fit(scaler_x, scaler_y)

    # To handle testing for the very first item without losing sequence length,
    # we prepend the last (seq_len - 1) items of the training set to the test set.
    if seq_len > 1:
        prefix_x = X_train_scaled[-(seq_len - 1):]
        X_test_scaled_ext = np.vstack([prefix_x, X_test_scaled])
        
        # Prepend zeros for targets as well to maintain alignment
        prefix_y = np.zeros((seq_len - 1, 1)) 
        y_test_scaled_ext = np.vstack([prefix_y, y_test_scaled])
    else:
        X_test_scaled_ext = X_test_scaled
        y_test_scaled_ext = y_test_scaled

    full_dataset = SequenceJobDataset(X_train_scaled, y_train_scaled.flatten(), seq_len=seq_len)
    test_dataset = SequenceJobDataset(X_test_scaled_ext, y_test_scaled_ext.flatten(), seq_len=seq_len)

    # --- chronological train / val split -----------------------------------
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    # Strict chronological split (Temporal Integrity)
    # Validation data is the last chronological portion of the training data
    indices = list(range(n_total))
    train_dataset = torch.utils.data.Subset(full_dataset, indices[:n_train])
    val_dataset = torch.utils.data.Subset(full_dataset, indices[n_train:])
    # -----------------------------------------------------------------------

    input_features = X_train.shape[1]
    y_test_raw     = y_test.flatten()

    return train_dataset, val_dataset, test_dataset, y_test_raw, scaler_x, scaler_y, input_features

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    ``delta`` is a *relative* improvement threshold (a fraction of the
    current best loss), not an absolute one. The DL runtime models here
    train on a MinMax-scaled target whose total variance is on the order of
    1e-3; a fixed absolute delta of 1e-4 -- roughly 14% of that variance --
    could dwarf genuine, useful improvements once training loss has already
    dropped anywhere near that scale, stopping the model on what looks like
    a plateau but is actually still shrinking. A relative threshold instead
    requires the new score to beat the current best by ``delta`` as a
    fraction of its own magnitude, so the bar naturally tightens in absolute
    terms as the loss itself shrinks, rather than staying fixed regardless
    of scale.
    """
    def __init__(self, patience=5, verbose=False, delta=1e-4):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        else:
            required_gain = self.delta * max(abs(self.best_score), 1e-12)
            if score < self.best_score + required_gain:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        self.val_loss_min = val_loss

def train_dl_model(model, train_loader, val_loader, criterion, optimizer, epochs=50, patience=5, device=None):
    """
    Trains a generic PyTorch model with Early Stopping and Learning Rate Scheduling.
    """
    # NOTE: seeding happens in seed_everything(), called by the caller *before*
    # the model is constructed. Reseeding here would be too late to affect the
    # weight initialisation and would additionally reset the shuffle stream to
    # the same order for every trial.
    if device is None:
        device = get_default_device()
        
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, verbose=False)
    
    # Academic Standard: Learning Rate Scheduler (Reduces LR when validation loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1), y_batch.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                loss = criterion(val_outputs.view(-1), y_val.view(-1))
                val_loss += loss.item() * X_val.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        # Track best weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        
        # Update scheduler
        scheduler.step(val_loss)
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"    [Early Stopping] Triggered at epoch {epoch+1}")
            break
            
    # Load best weights
    model.load_state_dict(best_model_wts)
    return model

def create_model_instance(model_name, input_features, params):
    dropout = params.get('dropout', 0.2)
    if model_name == 'CNN':
        return RuntimePredictorCNN(input_features, params['num_filters'], params.get('kernel_size', 1), dropout=dropout)
    elif model_name == 'LSTM':
        return RuntimePredictorLSTM(input_features, params['hidden_size'], params.get('num_layers', 1), dropout=dropout)
    elif model_name == 'Hybrid':
        return RuntimePredictorCNNLSTM(
            input_features, 
            params['num_filters'], 
            params.get('kernel_size', 1), 
            params['lstm_hidden_size'], 
            params.get('lstm_num_layers', 1),
            dropout=dropout
        )
    else:
        raise ValueError("Unsupported model architecture")

def run_dl_randomsearch(model_name, search_space, train_dataset, val_dataset, input_features, 
                        scaler_y, y_test_raw, test_dataset, num_trials=10, tuning_epochs=10, patience=5, device=None):
    """
    Perform randomized hyperparameter search for Deep Learning models.

    Parameters
    ----------
    model_name : str
        Architecture type ('CNN', 'LSTM', 'Hybrid').
    search_space : dict
        Hyperparameter search space with list values for sampled keys.
    train_dataset, val_dataset : torch.utils.data.Dataset
        Training and validation datasets.
    input_features : int
        Number of input features.
    scaler_y : sklearn.preprocessing.MinMaxScaler
        Scaler for inverse transformation of targets.
    y_test_raw : np.ndarray
        True unscaled target values for evaluation.
    test_dataset : torch.utils.data.Dataset
        Test features in format of Sequence dataset.
    num_trials : int, default=10
        Number of random parameter samples.
    tuning_epochs : int, default=10
        Epochs per trial for tuning.
    patience : int, default=5
        Early stopping patience.
    device : str, default=None
        Compute device ('cpu', 'cuda', or 'mps').

    Returns
    -------
    tuple
        (best_params, best_rmse)
    """
    if device is None:
        device = get_default_device()

    best_rmse = float('inf')
    best_params = None
    
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    # test_dataset / scaler_y / y_test_raw are accepted for signature symmetry with
    # finalize_dl_model but are deliberately never touched here: hyperparameter
    # selection uses the validation split only, so the test set stays unseen.

    # Dedicated RNG for hyperparameter sampling. It must be independent of the
    # global streams, because seed_everything() resets those once per trial and
    # would otherwise make every trial sample the same configuration.
    sampler = random.Random(DL_SEED)

    for i in range(num_trials):
        # Only sample from values that are lists; others are treated as constant
        params = {}
        for k, v in search_space.items():
            if isinstance(v, list):
                params[k] = sampler.choice(v)
            else:
                params[k] = v
                
        print(f"  Random Trial {i+1}/{num_trials} -> trying: {params}")
        
        # Seed immediately before construction so weights, shuffle order and
        # dropout masks are identical for this trial on every run, and so that
        # trials differ only by their hyperparameters, not by initialisation luck.
        seed_everything(DL_SEED)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        model = create_model_instance(model_name, input_features, params)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Train with early stopping internally to tune
        model = train_dl_model(model, train_loader, val_loader, criterion, optimizer, epochs=tuning_epochs, patience=patience, device=device)
        
        # Evaluate on the validation set for unbiased hyperparameter selection
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
        current_rmse = np.sqrt(val_loss / len(val_dataset)) # Scaled RMSE for selection
            
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_params = params
            
    return best_params, best_rmse

def run_dl_gridsearch(model_name, grid_search_space, train_dataset, val_dataset, input_features, 
                      scaler_y, y_test_raw, test_dataset, tuning_epochs=10, patience=5, device=None):
    """
    Evaluates every combinatorial pair inside `grid_search_space` (Narrow Grid)
    Optimized with GPU acceleration.
    """
    if device is None:
        device = get_default_device()
    # Separate tunable params (lists) from constants
    tunable_keys = [k for k, v in grid_search_space.items() if isinstance(v, list)]
    tunable_values = [v for k, v in grid_search_space.items() if isinstance(v, list)]
    constants = {k: v for k, v in grid_search_space.items() if not isinstance(v, list)}
    
    grid_combinations = []
    for v in itertools.product(*tunable_values):
        p = dict(zip(tunable_keys, v))
        p.update(constants)
        grid_combinations.append(p)
    
    best_rmse = float('inf')
    best_params = None
    
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    # test_dataset / scaler_y / y_test_raw are accepted for signature symmetry with
    # finalize_dl_model but are deliberately never touched here: grid selection
    # uses the validation split only, so the test set stays unseen.

    for i, params in enumerate(grid_combinations):

        # Seed before construction so every grid point starts from identical
        # weights and differs only by its hyperparameters.
        seed_everything(DL_SEED)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        model = create_model_instance(model_name, input_features, params)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Train with early stopping
        model = train_dl_model(model, train_loader, val_loader, criterion, optimizer, epochs=tuning_epochs, patience=patience, device=device)
        
        # Evaluate on the validation set for unbiased hyperparameter selection
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
        current_rmse = np.sqrt(val_loss / len(val_dataset)) # Scaled RMSE for selection
            
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_params = params

    # modeling-11: report (but do not act on) a winner sitting on the grid's
    # edge. The ML searches widen and re-search automatically
    # (run_gridsearch_iterative); doing the same here would double an already
    # long DL tuning stage, so the DL side settles for making the
    # non-convergence visible rather than silently reporting a boundary value
    # as if it were an interior optimum.
    if best_params:
        _edge = grid_boundary_params(
            {k: v for k, v in grid_search_space.items() if isinstance(v, list)},
            best_params,
        )
        if _edge:
            print(f"[{model_name}] NOTE: grid winner is at a boundary for {_edge} -- "
                  f"the optimum may lie outside the range searched (modeling-11).")

    return best_params, best_rmse

def _warn_if_any_seed_collapsed(model_name, seeds, runs) -> None:
    """Name the seeds of a multi-seed refit whose predictions collapsed.

    ``evaluate_regression`` judges each run as it is scored, but that warning
    is raised once per call site per process and lands in the middle of the
    training log, while the number that reaches the results table is the mean
    over seeds -- which one collapsed seed out of three cannot move far enough
    to look wrong. Said once, against the seed VALUES the refit ran, so the
    entry can be reported as excluded rather than as a predictor that happened
    to score badly.
    """
    collapsed = [
        seed for seed, metrics in zip(seeds, runs)
        if is_degenerate_prediction(metrics) is True
    ]
    if not collapsed:
        return
    warnings.warn(
        f"{model_name} collapsed to a constant output on "
        f"{len(collapsed)} of {len(runs)} seeds ({', '.join(str(s) for s in collapsed)}). "
        "The aggregate returned here averages those runs' error metrics in with "
        "the healthy ones, so its MAE/RMSE/R2 describe no model that exists. A "
        "constant prediction imposes no ordering, so the scheduling policy built "
        "on it is the no-prediction baseline wearing the model's name: report "
        "the entry as excluded, and judge it on pred_unique_frac_per_seed rather "
        "than on the mean.",
        UserWarning,
        stacklevel=2,
    )


def finalize_dl_model(model_name, best_params, train_dataset, val_dataset, input_features,
                     scaler_y, y_test_raw, test_dataset, final_epochs=50, patience=10, device=None,
                     seeds=None, save_all_seeds_to=None):
    """
    Train the selected configuration and report its test-set performance.

    Parameters
    ----------
    seeds : sequence of int, optional
        Random seeds to repeat the final training with. A single seed gives one
        deterministic model, which is reproducible but says nothing about how
        much of the reported score is initialisation luck. Passing several seeds
        trains the same configuration once per seed and returns the mean of each
        metric, with the standard deviation added under a ``*_std`` key. The
        returned model is the one from the first seed, so the artefact on disk
        stays a single concrete network whose own score is recorded under
        ``*_seed0``. Every seed's degeneracy evidence is kept alongside the mean
        (``pred_unique_frac_per_seed`` / ``n_predictions_per_seed``, aligned
        with ``seeds``), because a mean over three seeds cannot show that one of
        them predicted a constant. Defaults to ``(DL_SEED,)``.
    save_all_seeds_to : str, optional
        Path template containing a ``{seed}`` placeholder (e.g.
        ``"results/models/lstm_categorical_pt_seed{seed}.pth"``). When given,
        EVERY seed's trained model is saved to disk under this pattern, not
        just the first one -- letting a downstream simulation (notebook 05)
        run all seeds instead of only ever replaying seed0 (robustness-4).
        A single-seed call still trains and saves only that one model.

    Returns
    -------
    tuple
        ``(final_model, metrics)``.
    """
    if device is None:
        device = get_default_device()

    seed_list = [DL_SEED] if seeds is None else [int(x) for x in seeds]
    runs = []
    first_model = None

    for seed in seed_list:
        model, metrics = _finalize_dl_single(
            model_name, best_params, train_dataset, val_dataset, input_features,
            scaler_y, y_test_raw, test_dataset, final_epochs, patience, device, seed,
        )
        runs.append(metrics)
        if first_model is None:
            first_model = model
        if save_all_seeds_to is not None:
            _seed_path = save_all_seeds_to.format(seed=seed)
            torch.save(model, _seed_path)
            # The per-seed files are read back by notebook 05's multi-seed
            # robustness check, which refuses any artifact without a
            # provenance sidecar. Writing the model without the sidecar left
            # that check permanently unsatisfiable: the file exists, so the
            # "absent is a legitimate skip" branch does not fire, and the
            # currency check then rejects it with no way to repair it.
            # The model is passed so the sidecar carries THIS seed's fit rather
            # than the loop's most recent one.
            record_model_artifact(_seed_path, model)

    if len(runs) == 1:
        return first_model, runs[0]

    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = [r[k] for r in runs]
        agg[k] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    agg["n_seeds"] = len(seed_list)
    agg["seeds"] = list(seed_list)
    # Score of the network actually written to disk, so the saved artefact can
    # still be verified against a single number rather than against a mean.
    for k in keys:
        agg[f"{k}_seed0"] = float(runs[0][k])
    # Every seed's spread evidence, positionally aligned with ``seeds``, not
    # just the mean and seed 0's. A single collapsed seed survives the
    # averaging above -- two healthy seeds out of three leave a mean fraction
    # two-thirds of a healthy one -- while its error metrics are averaged into
    # the reported headline number regardless. Dropping runs[1:]'s
    # pred_unique_frac left that collapse with no trace any reader could judge:
    # the results table printed "no" in the constant-output column and the
    # exclusion notice named nothing.
    agg["pred_unique_frac_per_seed"] = [
        None if r.get("pred_unique_frac") is None else float(r["pred_unique_frac"])
        for r in runs
    ]
    agg["n_predictions_per_seed"] = [
        None if r.get("n_predictions") is None else int(r["n_predictions"])
        for r in runs
    ]
    _warn_if_any_seed_collapsed(model_name, seed_list, runs)
    return first_model, agg


def _finalize_dl_single(model_name, best_params, train_dataset, val_dataset, input_features,
                        scaler_y, y_test_raw, test_dataset, final_epochs, patience, device, seed):
    """One deterministic final-refit run at a given seed."""
    # Seed before construction: PyTorch draws initial weights at construction
    # time, so seeding afterwards would leave them governed by leftover state.
    seed_everything(seed)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False) # Large batch for valid evaluation speed
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    model = create_model_instance(model_name, input_features, best_params)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    
    start_time = time.time()
    
    # Train robust final model with strictly more epochs and deeper early stopping patience setup
    final_model = train_dl_model(model, train_loader, val_loader, criterion, optimizer, epochs=final_epochs, patience=patience, device=device)
    
    train_time = time.time() - start_time
    
    final_model.eval()
    preds_scaled = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds = final_model(X_batch).cpu().numpy()
            preds_scaled.extend(preds)
            
    preds_scaled = np.array(preds_scaled)
    preds_unscaled = np.maximum(scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten(), 0)
        
    metrics = evaluate_regression(y_test_raw, preds_unscaled)
    metrics['train_time'] = train_time

    # Same reason as finalize_ml_model's call: the .pth save cells run
    # separately from the training cells, so the sidecar has to be able to
    # describe this fit rather than whatever the tree looks like when the
    # network is eventually written out.
    _note_model_fit(final_model)

    return final_model, metrics
