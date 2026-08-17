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
