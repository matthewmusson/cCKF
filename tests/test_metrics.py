"""Tests for calibration metrics."""

from __future__ import annotations

import numpy as np
import pytest

from cckf import metrics


def test_perfectly_calibrated_predictions_have_near_zero_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(np.uint8)
    assert metrics.expected_calibration_error(p, y, n_bins=15) < 0.01


def test_systematically_overconfident_predictions_have_large_ece():
    rng = np.random.default_rng(0)
    p = np.full(100_000, 0.9)
    y = (rng.uniform(size=p.size) < 0.5).astype(np.uint8)
    assert metrics.expected_calibration_error(p, y, n_bins=15) > 0.3


def test_ece_is_bounded_in_zero_one():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=10_000)
    y = rng.integers(0, 2, size=10_000).astype(np.uint8)
    ece = metrics.expected_calibration_error(p, y)
    assert 0.0 <= ece <= 1.0


def test_reliability_bins_returns_requested_bin_count():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=10_000)
    y = (rng.uniform(size=p.size) < p).astype(np.uint8)
    out = metrics.reliability_bins(p, y, n_bins=15)
    assert len(out["bins"]) == 15
    assert sum(b["count"] for b in out["bins"]) == 10_000


def test_logit_bin_edges_are_uniform_in_log_odds():
    edges = metrics.logit_bin_edges(n_bins=10)
    logits = np.log(edges / (1.0 - edges))
    spacing = np.diff(logits)
    np.testing.assert_allclose(spacing, spacing[0], rtol=1e-9)


def test_logit_bin_edges_are_data_independent_and_shared():
    """Two different distributions must produce identical x-axes, otherwise
    figure G3's side-by-side comparison is meaningless."""
    a = metrics.logit_bin_edges(20)
    b = metrics.logit_bin_edges(20)
    np.testing.assert_array_equal(a, b)


def test_logit_bins_resolve_the_decision_region_on_imbalanced_predictions():
    """The failure that motivated dropping quantile bins.

    With 99% of predictions in [0, 0.01], quantile binning puts every edge
    inside that spike and collapses the whole tau range into one bin. Log-odds
    bins must instead spread several populated bins across [0.01, 0.5].
    """
    rng = np.random.default_rng(0)
    p = np.concatenate(
        [
            rng.uniform(1e-5, 0.01, size=990_000),  # the easy-negative spike
            rng.uniform(0.01, 0.5, size=9_000),  # the decision region
            rng.uniform(0.5, 1.0, size=1_000),
        ]
    )
    y = (rng.uniform(size=p.size) < p).astype(np.uint8)

    bins = metrics.reliability_bins(p, y)["bins"]
    # Bins overlapping the decision region — the honest measure of resolution
    # there, since a bin straddling an endpoint still resolves inside it.
    lo, hi = metrics.DECISION_REGION
    in_region = [
        b
        for b in bins
        if b["bin_hi"] > lo and b["bin_lo"] < hi and b["count"] >= metrics.MIN_BIN_COUNT
    ]
    assert (
        len(in_region) >= 4
    ), f"only {len(in_region)} populated bins over [{lo}, {hi}]"

    # Contrast: 15 equal-count bins put 15 of 16 edges below 0.01 and leave the
    # top bin spanning ~0.0094 to ~0.999 — one point covering the whole τ range.
    q_edges = np.quantile(p, np.linspace(0, 1, 16))
    assert (q_edges[:-1] < 0.01).sum() >= 14
    assert q_edges[-2] < 0.01 and q_edges[-1] > 0.9
    n_quantile_in_region = sum(
        1 for i in range(15) if q_edges[i] >= lo and q_edges[i + 1] <= hi
    )
    assert n_quantile_in_region == 0


def test_reliability_bins_accounts_for_every_row():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=50_000)
    y = (rng.uniform(size=p.size) < p).astype(np.uint8)
    bins = metrics.reliability_bins(p, y)["bins"]
    assert sum(b["count"] for b in bins) == 50_000


def test_reliability_bins_clips_rather_than_drops_out_of_range_predictions():
    """Predictions below P_MIN must land in bin 0, not vanish."""
    p = np.array([0.0, 1e-9, 0.5, 1.0])
    y = np.array([0, 0, 1, 1], dtype=np.uint8)
    bins = metrics.reliability_bins(p, y, n_bins=5)["bins"]
    assert sum(b["count"] for b in bins) == 4


def test_sparse_bins_are_flagged_not_silently_dropped():
    p = np.concatenate([np.full(5_000, 0.5), np.array([0.001])])
    y = np.concatenate([np.ones(5_000, dtype=np.uint8), np.zeros(1, dtype=np.uint8)])
    bins = metrics.reliability_bins(p, y, n_bins=20, min_count=100)["bins"]
    sparse = [b for b in bins if b["count"] > 0 and b["sparse"]]
    assert len(sparse) >= 1
    assert sum(b["count"] for b in bins) == 5_001


def test_max_calibration_error_catches_a_localised_failure_that_ece_hides():
    """A small badly-calibrated region swamped by a large good one: ECE stays
    low, MCE does not."""
    rng = np.random.default_rng(0)
    # 200k well-calibrated rows near p=0.002 ...
    p_easy = rng.uniform(1e-4, 4e-3, size=200_000)
    y_easy = (rng.uniform(size=p_easy.size) < p_easy).astype(np.uint8)
    # ... plus 5k rows claiming p=0.3 that are actually never positive.
    p_bad = np.full(5_000, 0.3)
    y_bad = np.zeros(5_000, dtype=np.uint8)

    p = np.concatenate([p_easy, p_bad])
    y = np.concatenate([y_easy, y_bad])

    ece = metrics.expected_calibration_error(p, y)
    mce = metrics.max_calibration_error(p, y)
    assert ece < 0.02, ece  # looks fine on the count-weighted average
    assert mce > 0.2, mce  # the localised failure is visible


def test_decision_region_ece_restricts_to_the_tau_range():
    rng = np.random.default_rng(0)
    p = np.concatenate(
        [rng.uniform(1e-5, 0.01, 100_000), rng.uniform(0.01, 0.5, 20_000)]
    )
    y = (rng.uniform(size=p.size) < p).astype(np.uint8)
    out = metrics.decision_region_ece(p, y)
    assert out["n_rows"] == 20_000
    assert 0.0 <= out["ece"] <= 1.0
    assert out["region"] == [0.01, 0.5]


def test_decision_region_ece_is_nan_when_the_region_is_empty():
    """No data is not the same as perfect calibration."""
    p = np.full(1_000, 1e-4)
    y = np.zeros(1_000, dtype=np.uint8)
    out = metrics.decision_region_ece(p, y)
    assert np.isnan(out["ece"])
    assert out["n_rows"] == 0


def test_reliability_bins_observed_fraction_tracks_prediction():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(np.uint8)
    # Only non-sparse bins carry a usable estimate; sparse ones are expected to
    # scatter and are excluded from the plotted line for exactly this reason.
    checked = 0
    for b in metrics.reliability_bins(p, y, n_bins=10)["bins"]:
        if b["sparse"]:
            continue
        assert abs(b["observed_fraction"] - b["mean_predicted"]) < 0.05
        checked += 1
    assert checked >= 5


def test_wilson_ci_brackets_the_point_estimate():
    lo, hi = metrics.wilson_ci(50, 100)
    assert lo < 0.5 < hi


def test_wilson_ci_narrows_with_more_data():
    narrow = metrics.wilson_ci(5_000, 10_000)
    wide = metrics.wilson_ci(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_ci_handles_zero_count():
    lo, hi = metrics.wilson_ci(0, 0)
    assert lo == 0.0 and hi == 1.0


def test_soft_reliability_bins_recovers_a_calibrated_soft_target():
    """Calibration for a soft target means E[target | pred] == pred."""
    rng = np.random.default_rng(0)
    pred = rng.uniform(size=200_000)
    # Target scatters around pred with zero conditional bias.
    target = np.clip(pred + rng.normal(0, 0.1, size=pred.size), 0.0, 1.0)
    for b in metrics.soft_reliability_bins(pred, target, n_bins=10)["bins"]:
        if b["sparse"]:
            continue
        assert abs(b["mean_target"] - b["mean_predicted"]) < 0.03


def test_soft_calibration_error_is_near_zero_when_unbiased():
    rng = np.random.default_rng(0)
    pred = rng.uniform(size=200_000)
    target = np.clip(pred + rng.normal(0, 0.1, size=pred.size), 0.0, 1.0)
    assert metrics.soft_calibration_error(pred, target) < 0.02


def test_soft_calibration_error_detects_a_systematic_offset():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0.0, 0.7, size=100_000)
    target = np.clip(pred + 0.2, 0.0, 1.0)  # V_phi systematically pessimistic
    assert metrics.soft_calibration_error(pred, target) > 0.15


def test_soft_reliability_uses_sem_not_wilson():
    """The interval must reflect the spread of a continuous target, so a bin
    whose targets are all identical has (essentially) zero width — whereas a
    Wilson interval on the same bin would report a wide binomial interval."""
    pred = np.full(1_000, 0.5)
    target = np.full(1_000, 0.5)
    b = [x for x in metrics.soft_reliability_bins(pred, target)["bins"] if x["count"]][
        0
    ]
    assert b["ci_upper"] - b["ci_lower"] < 1e-6
    wilson_width = metrics.wilson_ci(500, 1000)
    assert (wilson_width[1] - wilson_width[0]) > 0.05


def test_soft_reliability_widens_with_target_spread():
    rng = np.random.default_rng(0)
    pred = np.full(1_000, 0.5)
    tight = metrics.soft_reliability_bins(
        pred, np.clip(0.5 + rng.normal(0, 0.01, 1_000), 0, 1)
    )
    broad = metrics.soft_reliability_bins(
        pred, np.clip(0.5 + rng.normal(0, 0.30, 1_000), 0, 1)
    )
    tb = [b for b in tight["bins"] if b["count"]][0]
    bb = [b for b in broad["bins"] if b["count"]][0]
    assert (bb["ci_upper"] - bb["ci_lower"]) > (tb["ci_upper"] - tb["ci_lower"])


def test_soft_reliability_accounts_for_every_row():
    rng = np.random.default_rng(0)
    pred = rng.uniform(size=5_000)
    target = rng.uniform(size=5_000)
    bins = metrics.soft_reliability_bins(pred, target, n_bins=20)["bins"]
    assert sum(b["count"] for b in bins) == 5_000


def test_soft_reliability_single_row_bin_does_not_crash():
    """std(ddof=1) is undefined for n=1; must yield a finite zero-width band."""
    bins = metrics.soft_reliability_bins(
        np.array([0.05]), np.array([0.4]), n_bins=20, min_count=1
    )["bins"]
    b = [x for x in bins if x["count"]][0]
    assert np.isfinite(b["ci_lower"]) and np.isfinite(b["ci_upper"])
    assert b["mean_target"] == pytest.approx(0.4)


def test_eta_strata_has_five_bins_spanning_minus3_to_3():
    eta = np.linspace(-3.5, 3.5, 1000)
    strata = metrics.eta_strata(eta)
    assert len(strata) == 5
    covered = np.zeros(1000, dtype=bool)
    for mask in strata.values():
        covered |= mask
    # Everything inside |eta| <= 3 is covered.
    assert covered[np.abs(eta) <= 3.0].all()


def test_quintile_strata_splits_into_five_roughly_equal_groups():
    values = np.arange(1000, dtype=float)
    strata = metrics.quintile_strata(values)
    assert len(strata) == 5
    counts = [int(m.sum()) for m in strata.values()]
    assert all(150 <= c <= 250 for c in counts), counts
