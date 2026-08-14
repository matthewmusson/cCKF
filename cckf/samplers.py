"""The three §2.5 strategies for coping with a 1:193 class imbalance.

All three are index selections over the *same* cache, so switching strategy
never requires rebuilding features. The S1 ablation runs all three and compares
them on ECE, reliability and AUC-PR.

Approach A — large batch
    Batch 40,960 with no subsampling. At a 0.52% positive rate that is ~200
    positives per batch, enough for a stable gradient. The natural prior is
    preserved, so σ(z) estimates P(y=1 | x) directly and Platt has nothing to
    undo. Costs a full pass over ~137M rows per epoch.

Approach B — uniform negative subsampling
    Keep all positives, sample negatives uniformly to a 1:5 ratio, batch 4,096.
    Because negatives are dropped *independently of x*, P(x | y=0) is unchanged
    and the only effect on the Bayes-optimal logit is a constant additive shift::

        Δ = log(P'(y=1)/P'(y=0)) − log(P(y=1)/P(y=0))

    Platt's intercept b absorbs Δ exactly. This is the theoretically clean way
    to rebalance, and it is ~30× cheaper per epoch than A.

Approach C — hard-negative enrichment
    As B, but negatives are sampled with weight ∝ 1/χ², concentrating on
    plausible-looking wrong candidates near the decision boundary. More gradient
    signal per negative. The caveat is real and is why this is an ablation and
    not the default: reweighting by a function of x *reshapes* P(x | y=0), so
    the induced logit shift is x-dependent, not constant. Platt's (a, b) removes
    only the affine part; whether the residual non-affine bias matters is an
    empirical question that the §4.2 stratified reliability diagrams answer.
"""
from __future__ import annotations

import math
from typing import Iterator

import numpy as np

#: Batch size per §2.5 approach.
SAMPLER_BATCH_SIZES: dict[str, int] = {"A": 40_960, "B": 4_096, "C": 4_096}

_CHI2_FLOOR = 1e-3  # keeps 1/chi2 finite for a perfectly-aligned negative


def large_batch_indices(
    n_rows: int, batch_size: int, rng: np.random.Generator
) -> Iterator[np.ndarray]:
    """Yield shuffled index batches covering every row once (Approach A).

    The final batch is short rather than dropped, so no row is silently
    excluded from an epoch.
    """
    order = rng.permutation(n_rows)
    for start in range(0, n_rows, batch_size):
        yield order[start : start + batch_size]


def uniform_subsample(
    y: np.ndarray, neg_per_pos: int, rng: np.random.Generator
) -> np.ndarray:
    """Keep all positives and a uniform sample of negatives (Approach B).

    Parameters
    ----------
    y : numpy.ndarray
        Labels in {0, 1}.
    neg_per_pos : int
        Target negatives per positive (5 gives the spec's 1:5 ratio).
    rng : numpy.random.Generator
        Seeded generator; the returned indices are reproducible from it.

    Returns
    -------
    numpy.ndarray
        Sorted int64 indices into ``y``. If fewer negatives exist than
        requested, all are kept (the ratio is a target, not a guarantee).
    """
    y = np.asarray(y)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_take = min(len(neg), neg_per_pos * len(pos))
    picked = rng.choice(neg, size=n_take, replace=False) if n_take else neg[:0]
    return np.sort(np.concatenate([pos, picked]))


def hard_negative_subsample(
    y: np.ndarray, chi2: np.ndarray, neg_per_pos: int, rng: np.random.Generator
) -> np.ndarray:
    """Keep all positives and a 1/χ²-weighted sample of negatives (Approach C).

    Parameters
    ----------
    y : numpy.ndarray
        Labels in {0, 1}.
    chi2 : numpy.ndarray
        Per-row ``chi2_inc``, aligned with ``y``. Non-finite entries are treated
        as maximally easy (weight → 0) rather than dropped, so the returned
        count still matches the target.
    neg_per_pos : int
        Target negatives per positive.
    rng : numpy.random.Generator
        Seeded generator.

    Returns
    -------
    numpy.ndarray
        Sorted int64 indices into ``y``.
    """
    y = np.asarray(y)
    chi2 = np.asarray(chi2, dtype=np.float64)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    n_take = min(len(neg), neg_per_pos * len(pos))
    if n_take == 0:
        return np.sort(pos)

    c = chi2[neg]
    c = np.where(np.isfinite(c), np.maximum(c, _CHI2_FLOOR), np.inf)
    weights = 1.0 / c
    total = weights.sum()
    n_nonzero = int(np.count_nonzero(weights))

    if not np.isfinite(total) or total <= 0.0:
        # Every negative is non-finite chi2 (weight 0 everywhere): no signal
        # to weight by, so fall back to a uniform draw.
        picked = rng.choice(neg, size=n_take, replace=False)
    elif n_nonzero < n_take:
        # Fewer hard (finite-chi2, nonzero-weight) negatives exist than we
        # need. rng.choice(..., p=weights) cannot draw more items than there
        # are nonzero-probability entries, so take all of the hard negatives
        # outright and fill the remainder uniformly from the easy
        # (non-finite-chi2, zero-weight) pool.
        hard_mask = weights > 0.0
        hard = neg[hard_mask]
        easy = neg[~hard_mask]
        fill = rng.choice(easy, size=n_take - len(hard), replace=False)
        picked = np.concatenate([hard, fill])
    else:
        picked = rng.choice(neg, size=n_take, replace=False, p=weights / total)
    return np.sort(np.concatenate([pos, picked]))


def prior_logit_shift(
    n_pos_orig: int, n_neg_orig: int, n_pos_new: int, n_neg_new: int
) -> float:
    """Additive logit shift induced by changing the class prior.

    Under label-conditional resampling that leaves P(x | y) intact, the
    Bayes-optimal logit moves by the change in the log prior odds::

        Δ = log(n_pos_new / n_neg_new) − log(n_pos_orig / n_neg_orig)

    Reported alongside every subsampled run so the fitted Platt intercept can be
    checked against it: b̂ ≈ −Δ confirms the network learned the resampled
    posterior and nothing worse. A large discrepancy means something other than
    the prior changed.
    """
    if min(n_pos_orig, n_neg_orig, n_pos_new, n_neg_new) <= 0:
        raise ValueError("all four counts must be positive")
    return math.log(n_pos_new / n_neg_new) - math.log(n_pos_orig / n_neg_orig)
