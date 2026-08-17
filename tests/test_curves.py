"""Tests for grid-threshold ROC/PR curves and before/after metric bundles."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from cckf import curves


def _synthetic(n: int = 50_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=2.0, size=n)
    labels = rng.random(n) < expit(z - 4.0)  # ~2% positive
    return z, labels


def test_grid_curve_endpoints_are_the_trivial_classifiers():
    """At the lowest threshold everything is accepted; at the highest,
    nothing is."""
    z, labels = _synthetic()
    c = curves.grid_curves(z, labels)

    assert c["tpr"][0] == pytest.approx(1.0)
    assert c["fpr"][0] == pytest.approx(1.0)
    assert c["tp"][-1] == 0
    assert c["fp"][-1] == 0


def test_grid_counts_are_exact_at_grid_thresholds():
    """The grid subsamples which thresholds are displayed; it must not
    approximate the counts at those thresholds."""
    z, labels = _synthetic(n=10_000)
    c = curves.grid_curves(z, labels, n_points=50, logit_range=(-6.0, 6.0))
    z_clipped = np.clip(z, -6.0, 6.0)

    for j in (0, 7, 25, 49):
        t = c["threshold_logit"][j]
        assert c["tp"][j] == int(((z_clipped >= t) & labels).sum())
        assert c["fp"][j] == int(((z_clipped >= t) & ~labels).sum())


def test_rates_are_monotone_in_threshold():
    z, labels = _synthetic()
    c = curves.grid_curves(z, labels)
    assert np.all(np.diff(c["tpr"]) <= 1e-12)
    assert np.all(np.diff(c["fpr"]) <= 1e-12)


def test_precision_is_nan_where_nothing_is_accepted():
    """0/0 must not be reported as a precision value."""
    z, labels = _synthetic()
    c = curves.grid_curves(z, labels)
    empty = (c["tp"] + c["fp"]) == 0
    assert empty.any()
    assert np.all(np.isnan(c["precision"][empty]))


def test_clipping_keeps_every_row_on_the_curve():
    """Scores beyond the grid range are clipped in, never dropped."""
    z = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    labels = np.array([False, False, True, True, True])
    c = curves.grid_curves(z, labels, n_points=10, logit_range=(-5.0, 5.0))
    # Lowest threshold accepts all 5 rows despite two being far out of range.
    assert c["tp"][0] + c["fp"][0] == 5
    assert c["n_pos"] == 3
    assert c["n_neg"] == 2


def test_metric_bundle_auc_matches_sklearn_on_full_data():
    """AUC must come from sklearn on all rows, not from the display grid."""
    z, labels = _synthetic()
    prob = expit(z)
    bundle = curves.metric_bundle(prob, labels)

    assert bundle["auc_roc"] == pytest.approx(roc_auc_score(labels, prob), abs=1e-12)
    assert bundle["auc_pr"] == pytest.approx(
        average_precision_score(labels, prob), abs=1e-12
    )


def test_two_param_platt_leaves_auc_invariant_but_moves_ece():
    """The central fact behind the before/after figure.

    2-param Platt is z' = a*z + b with a > 0, a strictly increasing map, so it
    cannot change the ranking -- ROC and PR curves and both AUCs are identical.
    ECE is a property of the values, so it does move. A figure that plots ROC
    'before vs after' 2-param calibration draws two superimposed lines; this
    test is what keeps that from looking like a bug.
    """
    z, labels = _synthetic()
    before = curves.metric_bundle(expit(z), labels)
    after = curves.metric_bundle(expit(0.5 * z - 2.0), labels)

    assert after["auc_roc"] == pytest.approx(before["auc_roc"], abs=1e-12)
    assert after["auc_pr"] == pytest.approx(before["auc_pr"], abs=1e-12)
    assert abs(after["ece"] - before["ece"]) > 1e-4


def test_metric_bundle_reports_both_regions():
    z, labels = _synthetic()
    bundle = curves.metric_bundle(expit(z), labels)
    assert bundle["dr_n_rows"] >= bundle["threshold_region_n_rows"]
    assert set(bundle) >= {
        "auc_roc",
        "auc_pr",
        "ece",
        "mce",
        "dr_ece",
        "dr_n_rows",
        "threshold_region_ece",
        "threshold_region_n_rows",
        "base_rate",
    }


def test_prob_histogram_sums_match_the_raw_data():
    z, labels = _synthetic()
    prob = expit(z)
    h = curves.prob_histogram(prob, labels)

    assert h["count"].sum() == len(prob)
    assert h["sum_prob"].sum() == pytest.approx(prob.sum(), rel=1e-12)
    assert h["sum_label"].sum() == pytest.approx(labels.sum(), rel=1e-12)


def test_rebin_equal_width_is_exact_against_direct_binning():
    """Rebinning must equal what direct binning of the raw data would give.

    This is the property that lets the bundle ship 1000 fine bins instead of
    24M scores: any coarser equal-width binning is recoverable by summation.
    """
    z, labels = _synthetic()
    prob = expit(z)
    coarse = curves.rebin_equal_width(
        curves.prob_histogram(prob, labels), n_bins=20, min_count=100
    )

    edges = np.linspace(0.0, 1.0, 21)
    for b, spec in enumerate(coarse["bins"]):
        lo, hi = edges[b], edges[b + 1]
        # Match np.histogram's half-open bins, with the last bin closed.
        mask = (prob >= lo) & (prob < hi) if b < 19 else (prob >= lo) & (prob <= hi)
        assert spec["count"] == int(mask.sum())
        if mask.any():
            assert spec["mean_predicted"] == pytest.approx(prob[mask].mean(), rel=1e-9)
            assert spec["observed_fraction"] == pytest.approx(
                labels[mask].mean(), rel=1e-9
            )


def test_rebin_uses_the_mean_prediction_not_the_midpoint():
    """The reason prob_histogram exists.

    A bin spanning [0, 0.05] whose rows all sit near 0.001 must plot at 0.001.
    Using the midpoint 0.025 would move the point by an order of magnitude and
    invent calibration error that is not in the data.
    """
    prob = np.full(10_000, 0.001)
    labels = np.zeros(10_000, dtype=bool)
    coarse = curves.rebin_equal_width(
        curves.prob_histogram(prob, labels), n_bins=20, min_count=100
    )
    first = coarse["bins"][0]
    assert first["count"] == 10_000
    assert first["mean_predicted"] == pytest.approx(0.001, rel=1e-9)
    assert first["mean_predicted"] < 0.5 * (first["bin_lo"] + first["bin_hi"])


def test_rebin_refuses_an_uneven_split():
    z, labels = _synthetic(n=1_000)
    h = curves.prob_histogram(prob=expit(z), labels=labels)
    with pytest.raises(ValueError, match="does not divide"):
        curves.rebin_equal_width(h, n_bins=3, min_count=10)


def test_rebin_flags_sparse_bins_without_dropping_them():
    """The realistic shape: nearly all mass at one end, a thin tail elsewhere.

    This is what equal-width bins do to gate output at a 0.57% base rate --
    the first bin holds almost everything and the upper bins go thin. Sparse
    bins must still be returned, flagged, so the plot can show them as
    uncertain rather than silently omit them.
    """
    prob = np.concatenate([np.full(50_000, 0.002), np.full(30, 0.97)])
    labels = np.zeros(prob.size, dtype=bool)
    labels[-10:] = True

    coarse = curves.rebin_equal_width(
        curves.prob_histogram(prob, labels), n_bins=20, min_count=100
    )

    assert len(coarse["bins"]) == 20  # every bin returned, none dropped
    assert coarse["bins"][0]["count"] == 50_000
    assert not coarse["bins"][0]["sparse"]
    thin = [b for b in coarse["bins"] if 0 < b["count"] < 100]
    assert thin and all(b["sparse"] for b in thin)


def test_rebin_reports_wilson_intervals_bracketing_the_observed_fraction():
    z, labels = _synthetic()
    coarse = curves.rebin_equal_width(
        curves.prob_histogram(expit(z), labels), n_bins=20, min_count=100
    )
    checked = 0
    for b in coarse["bins"]:
        if not b["count"]:
            continue
        assert b["ci_lower"] <= b["observed_fraction"] <= b["ci_upper"]
        assert 0.0 <= b["ci_lower"] <= 1.0
        assert 0.0 <= b["ci_upper"] <= 1.0
        checked += 1
    assert checked > 5


def test_rebin_intervals_match_metrics_wilson_ci():
    """The linear and log-odds reliability views must quote the same
    uncertainty, so both must go through metrics.wilson_ci."""
    from cckf import metrics

    z, labels = _synthetic()
    coarse = curves.rebin_equal_width(
        curves.prob_histogram(expit(z), labels), n_bins=20, min_count=100
    )
    for b in coarse["bins"]:
        if not b["count"]:
            continue
        lo, hi = metrics.wilson_ci(b["n_positive"], b["count"])
        assert b["ci_lower"] == pytest.approx(lo, abs=1e-12)
        assert b["ci_upper"] == pytest.approx(hi, abs=1e-12)


def test_rebin_interval_narrows_as_the_bin_fills():
    """A thin bin must carry a visibly wider interval than a well-filled one --
    the whole point of drawing them on a plot whose bin occupancy spans five
    orders of magnitude."""
    thin = curves.rebin_equal_width(
        curves.prob_histogram(np.full(50, 0.52), np.array([True] * 25 + [False] * 25)),
        n_bins=20,
        min_count=100,
    )["bins"][10]
    fat = curves.rebin_equal_width(
        curves.prob_histogram(
            np.full(50_000, 0.52), np.array([True] * 25_000 + [False] * 25_000)
        ),
        n_bins=20,
        min_count=100,
    )["bins"][10]

    assert thin["count"] == 50 and fat["count"] == 50_000
    assert (thin["ci_upper"] - thin["ci_lower"]) > 10 * (
        fat["ci_upper"] - fat["ci_lower"]
    )
