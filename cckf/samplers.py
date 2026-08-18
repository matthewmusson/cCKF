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

    Sampling is **with replacement**, so a negative may appear more than once.
    Uniform sampling does not require this -- with or without replacement the
    marginal inclusion probability is uniform either way -- but B and C must
    use the *same* scheme or the S1 ablation differs in two variables
    (weighting and replacement) rather than one. C genuinely requires
    replacement (see :func:`hard_negative_subsample`), so B follows.

    Returns
    -------
    numpy.ndarray
        Sorted int64 indices into ``y``, with duplicates possible among the
        negatives. Exactly ``neg_per_pos * n_pos`` negatives are drawn, capped
        only by there being at least one negative to draw from.
    """
    y = np.asarray(y)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    # With replacement the target is always reachable as long as the pool is
    # non-empty, so n_take no longer has to be clipped to len(neg).
    n_take = neg_per_pos * len(pos) if len(neg) else 0
    picked = rng.choice(neg, size=n_take, replace=True) if n_take else neg[:0]
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

    Sampling is **with replacement**, and for this sampler that is a
    correctness requirement rather than a preference. Weighted sampling
    *without* replacement does not give inclusion probabilities proportional
    to the weights: for i.i.d. draws with replacement ``P(draw = i)`` is
    exactly ``w_i / sum(w)``, whereas without replacement item ``i``'s marginal
    inclusion probability after ``n`` draws is a function of *all* the weights
    that saturates toward 1 for heavy items. So a without-replacement draw
    never realised the intended ``∝ 1/χ²`` distribution at all.

    It also makes the bias analysis exact. Reweighting negatives by ``w(x)``
    shifts the Bayes-optimal logit by ``log(w(x) / E_p[w])``; that identity
    holds for i.i.d. draws from ``q(x) ∝ w(x) p(x)``, which is what
    with-replacement sampling produces and what without-replacement sampling
    only approximates. The §4.2 stratified reliability diagrams and the fitted
    Platt intercept are both testing that identity, so it needs to be the one
    the sampler actually implements.

    Weights are normalised linearly (``w / sum(w)``), never by softmax:
    ``1/χ²`` is already non-negative so it needs only a sum to become a
    distribution, and ``softmax`` would give ``P ∝ exp(1/χ²)`` -- a different
    distribution with an implicit temperature, which also overflows here since
    ``_CHI2_FLOOR`` lets ``1/χ²`` reach 1000.

    Returns
    -------
    numpy.ndarray
        Sorted int64 indices into ``y``, with duplicates possible among the
        negatives.
    """
    y = np.asarray(y)
    chi2 = np.asarray(chi2, dtype=np.float64)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    # With replacement the target is always reachable from a non-empty pool.
    n_take = neg_per_pos * len(pos) if len(neg) else 0
    if n_take == 0:
        return np.sort(pos)

    c = chi2[neg]
    c = np.where(np.isfinite(c), np.maximum(c, _CHI2_FLOOR), np.inf)
    weights = 1.0 / c
    total = weights.sum()

    if not np.isfinite(total) or total <= 0.0:
        # Every negative has non-finite chi2 (weight 0 everywhere): no signal
        # to weight by, so fall back to a uniform draw.
        picked = rng.choice(neg, size=n_take, replace=True)
    else:
        # No shortfall branch is needed any more: with replacement, drawing
        # n_take items only requires *one* entry with nonzero probability, not
        # n_take of them. The previous without-replacement implementation had
        # to special-case "fewer hard negatives than we need" and backfill
        # uniformly from the zero-weight pool, which silently mixed two
        # different sampling distributions into one training set.
        picked = rng.choice(neg, size=n_take, replace=True, p=weights / total)
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
