"""
evaluation.py

Centralized Regression Evaluation Metrics

This module provides the standard evaluation function used across all runtime
prediction experiments in the thesis. All metrics are computed in a single pass
for consistency and to avoid ad-hoc per-notebook implementations.

Key Components
--------------
evaluate_regression
    Returns a dictionary of MAE, RMSE, R², MAPE, and MdAE, plus the spread of
    the predictions themselves, and warns when that spread shows a collapse.
is_degenerate_prediction
    Flags a model that collapsed to a constant output -- invisible in the error
    metrics, fatal to any ranking built on it. Three-valued: it also reports
    when a metrics dict carries no evidence to judge on.
is_near_constant_prediction
    Flags the weaker finding: a model that ranks, but out of only a handful of
    distinct values. Reported beside the metrics, never as an exclusion.
prediction_ranks_nothing
    The one rule both of the above and the scheduling simulator's refusal are
    written in terms of, so the two cannot disagree about the same model.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


__all__ = [
    "evaluate_regression",
    "is_degenerate_prediction",
    "is_near_constant_prediction",
    "prediction_ranks_nothing",
    "MIN_RANKING_DISTINCT_VALUES",
]


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute standard regression metrics for model evaluation.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth (correct) target runtime values.
    y_pred : array-like of shape (n_samples,)
        Model-predicted runtime values.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``mae``   : Mean Absolute Error (seconds).
        - ``rmse``  : Root Mean Squared Error (seconds).
        - ``r2``    : Coefficient of Determination R².
        - ``mape``  : Mean Absolute Percentage Error, expressed on a 0-100
          percentage scale (a value of ``50.0`` means 50%, not 0.5) --
          this scale is what this function always returns, but it is not
          a guarantee that any particular copy of a thesis table is on
          the same scale: a LaTeX table is a separate, hand-transcribed
          artefact that can drift out of sync with this code
          (statistics-7). Only computed over samples where ``y_true > 0``
          to avoid division by zero; on this heavy-tailed runtime target,
          MAPE routinely reaches values in the hundreds or thousands of
          percent, driven by jobs with a small true runtime, and is not a
          well-behaved summary statistic here on its own.
        - ``mdae``  : Median Absolute Error (seconds).
        - ``pred_std``, ``pred_unique_frac``, ``n_predictions`` : spread of the
          PREDICTIONS, so a model that collapsed to a constant is visible in
          the metrics rather than only downstream. See
          :func:`is_degenerate_prediction`.

    Warns
    -----
    UserWarning
        When :func:`is_degenerate_prediction` judges these predictions
        collapsed. The metrics are still returned -- the caller decides what
        to do -- but the collapse is stated at the point it happens instead of
        depending on a results table asking for it. A second, milder warning
        covers :func:`is_near_constant_prediction`: predictions coarse enough
        to be worth mentioning, but which do rank and are therefore not an
        exclusion.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mdae = float(median_absolute_error(y_true, y_pred))

    # MAPE: guard against zero-runtime jobs (would produce inf / nan).
    # Scaled by 100 to report a genuine percentage, matching notebook 05's
    # own mean_absolute_percentage_error(...) * 100 convention -- the raw
    # scikit-learn-style ratio understates the true percentage by 100x.
    # Whether any given copy of a thesis table is on this same scale is a
    # separate question this function cannot enforce (statistics-7).
    nonzero_mask = y_true > 0
    if nonzero_mask.any():
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
        ) * 100.0
    else:
        mape = float("nan")

    # Spread of the predictions themselves, not of the error.
    #
    # A model can collapse to a single constant output and still land a
    # perfectly ordinary MAE: CNN-LSTM (Numeric Sequence) predicted 4128.12 s
    # for all 16,437 test jobs, scoring MAE 6717 s and R2 -0.01, which sits
    # mid-pack among the working models. Nothing in mae/rmse/r2/mape/mdae
    # separates that from a model that has learned something, and downstream
    # the difference is total: a constant prediction imposes no ordering, so
    # the SJF policy built on it degenerates to arrival order and matches FIFO
    # exactly (0.00% JCT improvement).
    #
    # pred_std and pred_unique_frac put the evidence in every metrics dict, and
    # therefore in every checkpoint and results table. Visible is not the same
    # as read, so the verdict is taken from them below rather than left for a
    # table to derive.
    pred_std = float(np.std(y_pred))
    pred_unique_frac = (
        float(len(np.unique(np.round(y_pred, 6))) / len(y_pred)) if len(y_pred) else float("nan")
    )

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "mdae": mdae,
        "pred_std": pred_std,
        "pred_unique_frac": pred_unique_frac,
        "n_predictions": int(len(y_pred)),
    }

    # Writing the evidence is not the same as reading it. Recording pred_std
    # and pred_unique_frac left the collapse detectable but undetected: the
    # values travelled into every checkpoint and every results table while
    # nothing looked at them, so a constant predictor was still ranked on MAE
    # alongside the working models. The verdict is therefore taken here, on
    # the evidence this function has just produced, rather than left to a
    # downstream table remembering to ask for it.
    n_distinct = round(pred_unique_frac * len(y_pred)) if len(y_pred) else 0
    if is_degenerate_prediction(metrics):
        warnings.warn(
            f"Predictions collapsed to {n_distinct} distinct value(s) across "
            f"{len(y_pred)} samples (std {pred_std:.6g}). The error metrics "
            "returned alongside this warning are ordinary and say nothing about "
            "it, but the model imposes no ordering: an SJF policy built on it is "
            "the no-prediction baseline wearing the model's name. Report it as "
            "excluded, not as a predictor that happened to score no improvement.",
            UserWarning,
            stacklevel=2,
        )
    # Stated separately, and deliberately not as an exclusion: a handful of
    # distinct values still orders the queue, so condemning it here would put
    # this function at odds with the simulator, which schedules such a column
    # without complaint. What it does say is that almost nothing was fitted.
    elif is_near_constant_prediction(metrics):
        warnings.warn(
            f"Predictions take only {n_distinct} distinct value(s) across "
            f"{len(y_pred)} samples (std {pred_std:.6g}). This still imposes an "
            "ordering -- it is not an exclusion, and a scheduling policy built on "
            "it can genuinely improve on FIFO -- but a model this coarse is "
            "usually one that barely fitted (a boosted refit early-stopped after "
            "a tree or two, a network stuck near the target's median). Report the "
            "coarseness beside the error metrics rather than the MAE alone.",
            UserWarning,
            stacklevel=2,
        )

    return metrics


#: Fewest distinct predicted values that can still put one job ahead of
#: another. Two buckets are an ordering; one is not.
MIN_RANKING_DISTINCT_VALUES = 2


def prediction_ranks_nothing(n_distinct: int) -> bool:
    """Is a prediction column with this many distinct values unable to order?

    The single place the "the predictor collapsed" rule is written down, so
    the metrics side (which counts distinct values back out of
    ``pred_unique_frac``) and :meth:`SJFPredScheduler.validate_workload`
    (which counts them in the column itself) cannot drift into giving opposite
    verdicts on the same model. They did: a share-based threshold here
    condemned Exp A LightGBM (Numeric) -- 15 distinct values over 16,437 test
    jobs, a fraction of 0.0009 -- as excluded, while the simulator happily ran
    it, because 15 buckets really do order a queue (Spearman rho 0.24, 20.8%
    JCT improvement in the 32-GPU run). One chapter said the model ranks
    nothing while the next ranked with it.
    """
    return n_distinct < MIN_RANKING_DISTINCT_VALUES


def _distinct_count(frac, n) -> int | None:
    """Recover the number of distinct predictions, or ``None`` if unknowable."""
    if frac is None or not np.isfinite(frac):
        return None
    # A share is not a count, and only the count says whether jobs can be
    # ordered: one distinct value out of four is a fraction of 0.25 and orders
    # nothing, while 15 out of 16,437 is a fraction of 0.0009 and orders
    # coarsely but genuinely. Without the sample count there is no count to
    # judge, which is the module's third state rather than a guess.
    if not n:
        return None
    return int(round(frac * n))


def _collect_verdicts(metrics: dict, judge) -> bool | None:
    """Apply ``judge`` to every spread record in ``metrics``; any True condemns.

    A multi-seed refit averages every numeric metric, so ``pred_unique_frac``
    in such a dict is a mean over seeds while ``<key>_seed0`` belongs to the
    one network actually written to disk and replayed by the scheduling
    notebook. A single collapsed seed survives that averaging (two healthy
    seeds out of three leave a mean fraction two-thirds of a healthy one), so
    the mean, the saved network and each of the n seeds are judged separately,
    under the same rule.
    """
    n_all = metrics.get("n_predictions")
    verdicts = [
        judge(metrics.get("pred_unique_frac"), n_all),
        judge(
            metrics.get("pred_unique_frac_seed0"),
            metrics.get("n_predictions_seed0", n_all),
        ),
    ]
    # A seed that is neither the mean nor seed 0 is invisible in both verdicts
    # above -- a middle seed collapsing to a constant left the aggregate dict
    # answering False -- so finalize_dl_model keeps every seed's spread evidence
    # aligned with its 'seeds' list and each entry is judged on its own. A
    # checkpoint written before those keys existed carries neither, which adds
    # no verdict and so still answers None rather than False.
    per_seed = metrics.get("pred_unique_frac_per_seed") or []
    counts = metrics.get("n_predictions_per_seed") or []
    for i, frac in enumerate(per_seed):
        n = counts[i] if i < len(counts) and counts[i] is not None else n_all
        verdicts.append(judge(frac, n))
    verdicts = [v for v in verdicts if v is not None]
    if not verdicts:
        return None
    return any(verdicts)


def is_degenerate_prediction(metrics: dict) -> bool | None:
    """Did this model collapse to a constant output?

    A predictor whose test predictions take a single value cannot rank jobs at
    all, so any scheduling result built on it is really the no-prediction
    baseline wearing the model's name. Judged on the number of distinct
    predicted values -- recovered from ``pred_unique_frac`` and
    ``n_predictions`` -- rather than on pred_std, because the target is
    heavy-tailed and a genuinely low-variance model is not the same thing as a
    constant one.

    The rule is :func:`prediction_ranks_nothing`, the same one the scheduling
    simulator refuses on, so a model excluded here is refused there and a model
    simulated there is not excluded here. A predictor that emits few but more
    than one distinct value is a different finding -- it orders the queue, only
    coarsely -- and belongs to :func:`is_near_constant_prediction`, which does
    not read as an exclusion.

    Returns ``True`` for a collapsed predictor, ``False`` for one that ranks,
    and ``None`` when the metrics dict carries no spread evidence to judge on.
    The third state is what makes this safe to render in a results table: a
    checkpoint written before this module recorded ``pred_unique_frac`` -- which
    is every deep-learning checkpoint currently on disk -- has nothing to judge,
    and answering ``False`` there would print a clean bill of health for exactly
    the model the scheduling notebook refuses. A caller must show that state as
    unknown, not as "not degenerate".
    """

    def judge(frac, n) -> bool | None:
        n_distinct = _distinct_count(frac, n)
        if n_distinct is None:
            return None
        return prediction_ranks_nothing(n_distinct)

    return _collect_verdicts(metrics, judge)


def is_near_constant_prediction(metrics: dict, min_unique_frac: float = 1e-3) -> bool | None:
    """Does this model rank, but out of only a handful of distinct values?

    Not an exclusion: a column of 15 distinct values orders 16,437 jobs into 15
    buckets, and the 32-GPU simulation measures a real improvement from it. It
    is still worth stating, because that is the fingerprint of a model that
    barely fitted -- Exp A LightGBM's "best" Experiment A MAE came from a refit
    that early-stopped at ONE tree -- and a reader comparing MAEs cannot see it.
    Report it next to ``n_estimators_effective``, not in the excluded list.

    Same three-valued contract, and the same per-seed judging, as
    :func:`is_degenerate_prediction`. A genuinely constant predictor answers
    ``False`` here: it is degenerate, and the two flags name different findings.
    """

    def judge(frac, n) -> bool | None:
        n_distinct = _distinct_count(frac, n)
        if n_distinct is None:
            return None
        if prediction_ranks_nothing(n_distinct):
            return False
        return bool(frac < min_unique_frac)

    return _collect_verdicts(metrics, judge)
