"""Tests for the three §2.5 sampling strategies."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cckf import samplers


@pytest.fixture
def imbalanced_labels():
    """1000 rows at a 1:99 positive:negative ratio."""
    y = np.zeros(1000, dtype=np.uint8)
    y[:10] = 1
    return y


def test_batch_sizes_match_spec():
    assert samplers.SAMPLER_BATCH_SIZES == {"A": 40_960, "B": 4_096, "C": 4_096}


def test_large_batch_covers_every_row_exactly_once_per_epoch():
    rng = np.random.default_rng(0)
    seen = np.concatenate(list(samplers.large_batch_indices(1000, 256, rng)))
    assert np.array_equal(np.sort(seen), np.arange(1000))


def test_large_batch_shuffles_between_epochs():
    a = np.concatenate(list(samplers.large_batch_indices(1000, 256, np.random.default_rng(0))))
    b = np.concatenate(list(samplers.large_batch_indices(1000, 256, np.random.default_rng(1))))
    assert not np.array_equal(a, b)


def test_uniform_subsample_hits_target_ratio(imbalanced_labels):
    idx = samplers.uniform_subsample(imbalanced_labels, neg_per_pos=5, rng=np.random.default_rng(0))
    y = imbalanced_labels[idx]
    assert y.sum() == 10  # all positives kept
    assert (y == 0).sum() == 50


def test_uniform_subsample_keeps_all_positives_when_negatives_are_scarce():
    y = np.array([1, 1, 1, 0], dtype=np.uint8)
    idx = samplers.uniform_subsample(y, neg_per_pos=5, rng=np.random.default_rng(0))
    assert y[idx].sum() == 3
    assert (y[idx] == 0).sum() == 1  # only one negative exists


def test_hard_negative_subsample_prefers_low_chi2_negatives(imbalanced_labels):
    # Negatives 10..1009 get chi2 increasing with index; low chi2 = hard.
    chi2 = np.full(1000, 50.0)
    chi2[10:60] = 0.5  # 50 "hard" negatives
    idx = samplers.hard_negative_subsample(
        imbalanced_labels, chi2, neg_per_pos=5, rng=np.random.default_rng(0)
    )
    chosen_neg = idx[imbalanced_labels[idx] == 0]
    frac_hard = np.mean(chi2[chosen_neg] < 1.0)
    assert frac_hard > 0.5, frac_hard


def test_hard_negative_subsample_still_hits_target_count(imbalanced_labels):
    chi2 = np.linspace(0.1, 100.0, 1000)
    idx = samplers.hard_negative_subsample(
        imbalanced_labels, chi2, neg_per_pos=5, rng=np.random.default_rng(0)
    )
    y = imbalanced_labels[idx]
    assert y.sum() == 10 and (y == 0).sum() == 50


def test_hard_negative_subsample_handles_zero_and_nan_chi2(imbalanced_labels):
    chi2 = np.full(1000, 1.0)
    chi2[20] = 0.0
    chi2[21] = np.nan
    idx = samplers.hard_negative_subsample(
        imbalanced_labels, chi2, neg_per_pos=5, rng=np.random.default_rng(0)
    )
    assert len(idx) == 60
    assert np.all(np.isin(idx, np.arange(1000)))


def test_hard_negative_subsample_handles_positive_infinity_chi2(imbalanced_labels):
    """Production reality (Task 6): the cache writer maps NaN chi2 to +inf, so
    +inf is the value that actually appears in aux[:, 0], not NaN. A +inf
    chi2_inc must be treated as maximally easy (selection weight -> 0) while
    the returned negative count still hits the target exactly.
    """
    chi2 = np.full(1000, 1.0)
    # A block of unambiguously "easy" (+inf) negatives among otherwise
    # uniform-difficulty negatives.
    chi2[10:110] = np.inf  # 100 negatives with +inf chi2 (out of 990 negatives)
    idx = samplers.hard_negative_subsample(
        imbalanced_labels, chi2, neg_per_pos=5, rng=np.random.default_rng(0)
    )
    y = imbalanced_labels[idx]
    # Count still matches the target: 10 positives + 50 negatives.
    assert y.sum() == 10
    assert (y == 0).sum() == 50
    assert np.all(np.isin(idx, np.arange(1000)))

    chosen_neg = idx[imbalanced_labels[idx] == 0]
    # +inf negatives get selection weight exactly 0 relative to the finite
    # chi2=1.0 negatives (880 of them), so with only 50 slots and no
    # replacement, an +inf row should essentially never be drawn.
    assert not np.any(np.isinf(chi2[chosen_neg]))


def test_hard_negative_subsample_all_infinite_chi2_falls_back_to_uniform(imbalanced_labels):
    """If every negative's chi2 is +inf (all weights collapse to 0), the
    function must not raise or divide 0/0 -- it should fall back to a
    uniform draw and still hit the exact target count.
    """
    chi2 = np.full(1000, np.inf)
    idx = samplers.hard_negative_subsample(
        imbalanced_labels, chi2, neg_per_pos=5, rng=np.random.default_rng(0)
    )
    y = imbalanced_labels[idx]
    assert y.sum() == 10
    assert (y == 0).sum() == 50


def test_hard_negative_subsample_fewer_hard_negatives_than_requested_does_not_crash(
    imbalanced_labels,
):
    """Regression test (fix round 1): when the number of finite-chi2 (nonzero
    weight) negatives is smaller than n_take but strictly positive, the naive
    weighted rng.choice(..., p=...) call raises
    ``ValueError: Fewer non-zero entries in p than size`` because it cannot
    draw more items than there are nonzero-probability entries. Production
    hits this: the cache writer maps NaN chi2 to +inf (Task 6), so a window
    with many degenerate fits and few finite negatives is expected at scale.

    990 negatives: 950 at chi2=+inf (easy, zero weight), 40 finite (hard,
    nonzero weight). neg_per_pos=5 with 10 positives requests n_take=50,
    i.e. 10 more than the 40 available hard negatives.

    Required degrade-gracefully behavior: return exactly n_take negatives,
    include *all* 40 hard (finite-chi2) negatives, and fill the remaining 10
    slots uniformly from the 950 easy (+inf) negatives.
    """
    chi2 = np.full(1000, np.inf)
    chi2[10:50] = 1.0  # exactly 40 finite ("hard") negatives among the 990

    idx = samplers.hard_negative_subsample(
        imbalanced_labels, chi2, neg_per_pos=5, rng=np.random.default_rng(0)
    )

    y = imbalanced_labels[idx]
    assert y.sum() == 10
    assert (y == 0).sum() == 50  # exactly n_take, no crash, no shortfall

    chosen_neg = idx[imbalanced_labels[idx] == 0]
    hard_pool = np.arange(10, 50)
    easy_pool = np.setdiff1d(np.flatnonzero(imbalanced_labels == 0), hard_pool)

    # All 40 hard negatives must be present -- they're strictly preferred.
    assert np.array_equal(np.intersect1d(chosen_neg, hard_pool), hard_pool)
    # The remaining 10 slots come from the easy (+inf) pool.
    filled = np.setdiff1d(chosen_neg, hard_pool)
    assert len(filled) == 10
    assert np.all(np.isin(filled, easy_pool))


def test_hard_negative_subsample_exact_boundary_uses_weighted_path(imbalanced_labels):
    """Boundary case: the number of nonzero-weight (finite-chi2) negatives
    exactly equals n_take -- no shortfall and no surplus. This must still
    succeed via the ordinary weighted rng.choice(..., p=...) branch (it must
    NOT trip the new shortfall/fill path, since there is nothing to fill).
    """
    chi2 = np.full(1000, np.inf)
    chi2[10:60] = 1.0  # exactly 50 finite negatives == n_take (neg_per_pos=5 * 10 pos)

    idx = samplers.hard_negative_subsample(
        imbalanced_labels, chi2, neg_per_pos=5, rng=np.random.default_rng(0)
    )

    y = imbalanced_labels[idx]
    assert y.sum() == 10
    assert (y == 0).sum() == 50

    chosen_neg = idx[imbalanced_labels[idx] == 0]
    # With exactly 50 nonzero-weight negatives and 50 slots, every one of
    # them must be chosen -- none of the +inf ("easy") negatives should
    # appear.
    expected = np.arange(10, 60)
    assert np.array_equal(np.sort(chosen_neg), expected)


def test_prior_logit_shift_matches_definition():
    # Original 1:99, resampled to 1:5.
    shift = samplers.prior_logit_shift(10, 990, 10, 50)
    expected = math.log(10 / 50) - math.log(10 / 990)
    assert shift == pytest.approx(expected)
    assert shift > 0  # upsampling positives raises the learned logit


def test_prior_logit_shift_is_zero_when_ratio_unchanged():
    assert samplers.prior_logit_shift(10, 990, 20, 1980) == pytest.approx(0.0)


def test_subsample_is_reproducible_from_seed(imbalanced_labels):
    a = samplers.uniform_subsample(imbalanced_labels, 5, np.random.default_rng(42))
    b = samplers.uniform_subsample(imbalanced_labels, 5, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
