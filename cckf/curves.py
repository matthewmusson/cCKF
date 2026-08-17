"""Display-resolution ROC/PR curves and before/after metric bundles.

Why a threshold grid instead of ``sklearn.roc_curve``
-----------------------------------------------------
``roc_curve`` sorts every score and returns one point per distinct threshold.
On the 24.1M-row val split that is tens of millions of vertices to draw what is
visually a smooth line, and tens of megabytes to move off the Modal volume per
arm per estimator.

``TP(t)`` and ``FP(t)`` are just *counts above a threshold*, which are reverse
cumulative sums of histograms. So fixing a grid of thresholds and histogramming
the positive and negative score distributions over it gives the exact counts at
those thresholds in one O(N) pass and O(G) memory.

The grid is uniform in log-odds, which puts the displayed points densely in the
high-threshold / low-FPR region where the gate actually operates, instead of
densely where the curve is uninformative.

What is exact and what is not
-----------------------------
The counts are **exact** at the grid thresholds -- the grid subsamples which
thresholds are displayed, it does not approximate anything about them.

The AUCs are **not** computed from the grid. Trapezoidal integration over 4,000
points would put a small bias into a headline number, so :func:`metric_bundle`
takes them from scikit-learn on the full data. Grid for display, sklearn for
the numbers.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import metrics

#: Log-odds range spanned by the display grid. |z| = 20 is a probability within
#: 2e-9 of 0 or 1 -- past any resolution this project cares about. Scores are
#: clipped into this range rather than dropped, so no row silently vanishes
#: from a curve.
GRID_LOGIT_RANGE: tuple[float, float] = (-20.0, 20.0)

#: Number of grid intervals. 4,000 gives ~0.01 in log-odds per step, far finer
#: than any visible feature at figure resolution.
GRID_POINTS: int = 4000


def grid_curves(
    logits: np.ndarray,
    labels: np.ndarray,
    n_points: int = GRID_POINTS,
    logit_range: tuple[float, float] = GRID_LOGIT_RANGE,
) -> dict:
    """Exact ROC/PR quantities at a fixed grid of log-odds thresholds.

    Parameters
    ----------
    logits : numpy.ndarray
        Raw log-odds scores. Pass logits rather than probabilities so the grid
        spacing is uniform in the space the thresholds live in.
    labels : numpy.ndarray
        Binary truth, cast to bool.
    n_points : int
        Number of grid intervals; returned arrays have ``n_points + 1`` entries,
        one per grid edge.
    logit_range : tuple of float
        Inclusive log-odds range of the grid.

    Returns
    -------
    dict
        ``threshold_logit``, ``threshold_prob``, ``tpr``, ``fpr``, ``precision``
        (NaN where nothing is accepted), ``tp``, ``fp``, ``n_pos``, ``n_neg``,
        ``base_rate``. All arrays share length ``n_points + 1``, ordered from
        the lowest threshold (accept everything) to the highest (accept
        nothing).
    """
    lo, hi = logit_range
    z = np.clip(np.asarray(logits, dtype=np.float64), lo, hi)
    labels = np.asarray(labels).astype(bool)

    edges = np.linspace(lo, hi, n_points + 1)
    pos_hist, _ = np.histogram(z[labels], bins=edges)
    neg_hist, _ = np.histogram(z[~labels], bins=edges)

    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())

    # Counts at or above each edge. np.histogram's last bin is closed on the
    # right, so a score exactly at ``hi`` is counted in it and tp/fp reach 0 at
    # the final edge -- the "accept nothing" endpoint.
    tp = n_pos - np.concatenate([[0], np.cumsum(pos_hist)])
    fp = n_neg - np.concatenate([[0], np.cumsum(neg_hist)])

    accepted = tp + fp
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = tp / n_pos if n_pos else np.full(tp.shape, np.nan)
        fpr = fp / n_neg if n_neg else np.full(fp.shape, np.nan)
        # 0/0 is "no data", not "perfect precision".
        precision = np.where(accepted > 0, tp / np.maximum(accepted, 1), np.nan)

    total = n_pos + n_neg
    return {
        "threshold_logit": edges,
        "threshold_prob": 1.0 / (1.0 + np.exp(-edges)),
        "tpr": np.asarray(tpr, dtype=np.float64),
        "fpr": np.asarray(fpr, dtype=np.float64),
        "precision": np.asarray(precision, dtype=np.float64),
        "tp": tp.astype(np.int64),
        "fp": fp.astype(np.int64),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "base_rate": (n_pos / total) if total else float("nan"),
    }


#: Fine equal-width probability bins whose sufficient statistics are shipped in
#: the plot bundle. 1000 divides evenly by 2, 4, 5, 8, 10, 20, 25, 40, 50 and
#: 100, so any of those coarser equal-width binnings is recoverable *exactly*
#: by summing adjacent fine bins -- no interpolation, no re-running inference.
PROB_HIST_BINS: int = 1000


def prob_histogram(
    prob: np.ndarray, labels: np.ndarray, n_bins: int = PROB_HIST_BINS
) -> dict:
    """Sufficient statistics for equal-width reliability binning.

    A reliability point needs three numbers per bin: how many rows fell in it,
    the mean prediction of those rows, and the fraction that were positive. All
    three are sums, so accumulating ``count``, ``sum_prob`` and ``sum_label`` on
    a fine fixed grid lets any coarser *equal-width* binning be rebuilt exactly
    by adding adjacent bins.

    This exists because bin midpoints are not an acceptable substitute for the
    mean prediction on a linear axis. At a 0.57% base rate the gate puts ~99.5%
    of its mass below 0.01, so a bin spanning [0, 0.05] has a mean prediction
    around 0.001 -- plotting it at the midpoint 0.025 would move the point by
    more than an order of magnitude and invent a calibration error that is not
    there.

    Parameters
    ----------
    prob : numpy.ndarray
        Probabilities in [0, 1].
    labels : numpy.ndarray
        Binary truth, cast to bool.
    n_bins : int
        Number of equal-width bins spanning [0, 1].

    Returns
    -------
    dict
        ``edges`` (``n_bins + 1`` values), ``count``, ``sum_prob``,
        ``sum_label`` (each ``n_bins`` long).
    """
    p = np.clip(np.asarray(prob, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(labels).astype(bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # ``np.histogram`` with weights gives the sums; the unweighted call gives
    # the counts. All three share the identical binning, so they stay aligned.
    count, _ = np.histogram(p, bins=edges)
    sum_prob, _ = np.histogram(p, bins=edges, weights=p)
    sum_label, _ = np.histogram(p, bins=edges, weights=y.astype(np.float64))
    return {
        "edges": edges,
        "count": count.astype(np.int64),
        "sum_prob": sum_prob,
        "sum_label": sum_label,
    }


def rebin_equal_width(hist: dict, n_bins: int, min_count: int) -> dict:
    """Collapse :func:`prob_histogram` statistics into ``n_bins`` equal bins.

    Exact, not approximate: ``PROB_HIST_BINS`` must be an integer multiple of
    ``n_bins`` so every coarse bin is a whole number of fine bins.

    Parameters
    ----------
    hist : dict
        Output of :func:`prob_histogram`.
    n_bins : int
        Target number of equal-width bins. Must divide ``len(hist["count"])``.
    min_count : int
        Bins with fewer rows than this are flagged ``sparse``; they are still
        returned, so a thin region reads as uncertain rather than absent.

    Returns
    -------
    dict
        ``bins`` -- a list of dicts with ``bin_lo``, ``bin_hi``,
        ``mean_predicted``, ``observed_fraction``, ``count``, ``n_positive``,
        ``ci_lower``, ``ci_upper``, ``sparse`` -- and ``edges``.

        ``ci_lower``/``ci_upper`` are Wilson score intervals on the observed
        fraction, matching what :func:`cckf.metrics.reliability_bins` reports so
        the linear and log-odds reliability views quote the same uncertainty.
        Wilson rather than the normal approximation because bin occupancy here
        spans five orders of magnitude and the observed fraction is often near 0
        or 1, where the normal interval misbehaves badly (it can even leave
        [0, 1]).

    Raises
    ------
    ValueError
        If ``n_bins`` does not divide the fine binning evenly, which would make
        the rebinning silently approximate.
    """
    fine = len(hist["count"])
    if fine % n_bins:
        raise ValueError(
            f"{n_bins} does not divide the {fine}-bin fine histogram evenly; "
            "an uneven split would make this rebinning approximate rather "
            "than exact"
        )
    group = fine // n_bins
    count = np.asarray(hist["count"]).reshape(n_bins, group).sum(axis=1)
    sum_prob = np.asarray(hist["sum_prob"]).reshape(n_bins, group).sum(axis=1)
    sum_label = np.asarray(hist["sum_label"]).reshape(n_bins, group).sum(axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins = []
    for b in range(n_bins):
        n = int(count[b])
        # sum_label accumulates float weights from a boolean, so it is an exact
        # integer up to float representation; round rather than truncate.
        k = int(round(float(sum_label[b])))
        lo_ci, hi_ci = metrics.wilson_ci(k, n) if n else (0.0, 0.0)
        bins.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                # Midpoint only when the bin is empty and there is nothing to
                # average; never as a stand-in for a populated bin.
                "mean_predicted": (
                    float(sum_prob[b] / n)
                    if n
                    else float(0.5 * (edges[b] + edges[b + 1]))
                ),
                "observed_fraction": (float(sum_label[b] / n) if n else 0.0),
                "count": n,
                "n_positive": k,
                "ci_lower": lo_ci,
                "ci_upper": hi_ci,
                "sparse": n < min_count,
            }
        )
    return {"bins": bins, "edges": [float(e) for e in edges]}


def metric_bundle(prob: np.ndarray, labels: np.ndarray) -> dict:
    """Every scalar metric for one (estimator, split) pair.

    Discrimination (``auc_roc``, ``auc_pr``) depends only on the ranking of
    ``prob``; calibration (``ece``, ``mce``, ``dr_ece``) depends on its values.
    That split is why a monotone recalibration moves the second group and
    provably not the first -- see ``tests/test_curves.py::
    test_two_param_platt_leaves_auc_invariant_but_moves_ece``.

    Parameters
    ----------
    prob : numpy.ndarray
        Probabilities in [0, 1].
    labels : numpy.ndarray
        Binary truth, cast to bool.

    Returns
    -------
    dict
        ``auc_roc``, ``auc_pr``, ``ece``, ``mce``, ``dr_ece``, ``dr_n_rows``,
        ``threshold_region_ece``, ``threshold_region_n_rows``, ``base_rate``.
    """
    prob = np.asarray(prob, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    edges = metrics.logit_bin_edges(30)

    wide = metrics.decision_region_ece(prob, labels, region=metrics.DECISION_REGION)
    narrow = metrics.decision_region_ece(prob, labels, region=metrics.THRESHOLD_REGION)

    return {
        "auc_roc": float(roc_auc_score(labels, prob)),
        "auc_pr": float(average_precision_score(labels, prob)),
        "ece": float(metrics.expected_calibration_error(prob, labels, edges=edges)),
        "mce": float(metrics.max_calibration_error(prob, labels, edges=edges)),
        "dr_ece": float(wide["ece"]),
        "dr_n_rows": int(wide["n_rows"]),
        "threshold_region_ece": float(narrow["ece"]),
        "threshold_region_n_rows": int(narrow["n_rows"]),
        "base_rate": float(labels.mean()),
    }
