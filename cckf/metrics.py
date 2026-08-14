"""Calibration metrics: reliability bins, ECE, Wilson intervals, strata.

Reliability diagram
-------------------
Bin predictions, then plot the observed positive fraction against the mean
prediction per bin. Perfect calibration is the diagonal.

Bins are uniform in **log-odds**, not quantile and not equal-width in
probability. The reasoning matters, because the obvious choices are both wrong
here:

*Quantile bins are actively harmful at this class balance.* A trained gate puts
~99% of its predictions in [0, 0.01], so 15 equal-count bins place all 15 edges
inside that spike and the top bin spans the 93rd percentile out to 1.0 —
swallowing the entire τ sweep range (§2.8 sweeps 0.01 to 0.5) into a single
point. Perfect resolution where nothing is decided, one dot where everything is.
Quantile binning is a remedy for small-sample noise in sparse bins; with 137M
rows a bin holding one millionth of the mass still has 137 samples, so that
disease is absent and only the side effects remain. Quantile edges are also
data-dependent, which breaks figure G3: χ² and the gate have different output
distributions, so their quantile edges differ and the two curves would sit on
non-comparable x-axes.

*Equal-width-in-probability bins* fail the other way: 14 of 15 bins would be
nearly empty and the first would hold ~99% of the rows, so the near-zero region
gets no resolution at all.

*Log-odds-uniform bins* give roughly constant **relative** resolution in
probability across the several orders of magnitude the gate actually spans,
which is the scale on which a likelihood-ratio gate is used — the calibrated
log-odds feed additively into the ILP's branch score, so an error of 0.001 at
p=0.002 matters as much as 0.1 at p=0.2. The edges are fixed constants,
independent of the data, so every curve and every stratum shares one x-axis.

Unequal bin occupancy is then not engineered away but **displayed**: the Wilson
interval widens where n is small, which is the honest representation. Bins below
``MIN_BIN_COUNT`` are still reported in the JSON but flagged ``sparse`` and
omitted from the plotted line.

ECE and its blind spot
----------------------
    ECE = Σ_b (n_b / N) · |observed_fraction(b) − mean_predicted(b)|

Targets from spec §4.2: < 0.02 overall and < 0.05 in every stratum.

Be aware of what count-weighting does at a 1:193 imbalance: ~99% of the weight
sits on near-zero-probability bins where |obs − pred| is tiny by construction, so
**overall ECE is dominated by the trivial region and can look excellent while
the decision region is badly miscalibrated.** This is a property of ECE itself,
not of the binning, so changing bins does not fix it. Three complementary
numbers are therefore reported alongside it:

``decision_region_ece``
    ECE restricted to predictions inside the τ sweep range [0.01, 0.5]. This is
    the number that speaks to whether the gate can be *used*.
``max_calibration_error``
    Worst single non-sparse bin. Unweighted, so a small badly-calibrated region
    cannot hide behind a large well-calibrated one.
Stratified ECE
    Per §4.2, by η and n_window quintile — catches the case where the average is
    fine but a stratum is not.

Strata
------
η in 5 bins from −3 to 3 and ``n_window`` quintiles (§4.2). Stratified ECE is
the metric that actually matters here: a model can be well calibrated on average
while being badly overconfident in the dense forward region, and averaging hides
exactly the failure the learned gate is meant to fix.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

#: η stratum edges (spec §4.2: 5 bins from −3 to 3).
ETA_EDGES: tuple[float, ...] = (-3.0, -1.8, -0.6, 0.6, 1.8, 3.0)

#: Probability range spanned by the log-odds-uniform bins. The lower end is set
#: by the gate's useful dynamic range: below ~1e-5 a candidate is not a
#: contender under any threshold, and Platt's own clipping sits at 1e-6.
P_MIN: float = 1e-5
P_MAX: float = 1.0 - 1e-5

#: Bins with fewer rows than this are reported but flagged sparse and not drawn.
#: At 137M rows this is a formality for the gate; it matters for thin strata
#: (forward η at high occupancy) and for the ~10^5-row value cache.
MIN_BIN_COUNT: int = 100

#: The τ sweep range from spec §2.8 — the region where calibration must hold for
#: the gate to be deployable at any operating point.
DECISION_REGION: tuple[float, float] = (0.01, 0.5)


def logit_bin_edges(
    n_bins: int = 30, p_min: float = P_MIN, p_max: float = P_MAX
) -> np.ndarray:
    """Bin edges uniform in log-odds, spanning ``[p_min, p_max]``.

    Fixed constants independent of the data, so every curve, stratum and model
    shares one x-axis. Returns ``n_bins + 1`` probability-space edges; they are
    unevenly spaced in probability and evenly spaced in ``log(p/(1−p))``.
    """
    lo = np.log(p_min / (1.0 - p_min))
    hi = np.log(p_max / (1.0 - p_max))
    logit_edges = np.linspace(lo, hi, n_bins + 1)
    return 1.0 / (1.0 + np.exp(-logit_edges))


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because reliability bins in the
    tails have proportions near 0 or 1, where the Wald interval can extend
    outside [0, 1] and understates uncertainty.

    Returns
    -------
    tuple of float
        ``(lower, upper)``. ``(0.0, 1.0)`` when ``n == 0``.
    """
    if n <= 0:
        return 0.0, 1.0
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    margin = z * np.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, float(center + margin))


def reliability_bins(
    pred: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 30,
    edges: np.ndarray | None = None,
    min_count: int = MIN_BIN_COUNT,
) -> dict:
    """Reliability diagram data on log-odds-uniform bins.

    Parameters
    ----------
    pred : numpy.ndarray
        Predicted probabilities in [0, 1].
    labels : numpy.ndarray
        Binary outcomes.
    n_bins : int
        Number of log-odds-uniform bins, used when ``edges`` is None.
    edges : numpy.ndarray, optional
        Explicit probability-space edges. Pass the *same* array for every curve
        that will be overlaid or compared.
    min_count : int
        Bins below this are flagged ``sparse`` and excluded from the plotted
        line, but are still returned and still counted in ECE.

    Returns
    -------
    dict
        ``bins`` : list of dict with ``mean_predicted``, ``observed_fraction``,
        ``count``, ``ci_lower``, ``ci_upper``, ``sparse``, ``bin_lo``,
        ``bin_hi``; and ``edges`` : list of float.

    Notes
    -----
    Predictions outside ``[edges[0], edges[-1]]`` are clipped into the end bins
    rather than dropped, so every row is accounted for and ECE remains a
    weighted average over the whole sample.
    """
    pred = np.asarray(pred, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    if edges is None:
        edges = logit_bin_edges(n_bins)
    edges = np.asarray(edges, dtype=np.float64)
    n_bins = len(edges) - 1

    bin_idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)

    bins = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        k = int(labels[mask].sum()) if n else 0
        lo, hi = wilson_ci(k, n)
        # Geometric midpoint is the representative point for an empty log-odds
        # bin; a populated bin uses its actual mean prediction.
        midpoint = float(np.sqrt(max(edges[b], 1e-300) * max(edges[b + 1], 1e-300)))
        bins.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "mean_predicted": float(pred[mask].mean()) if n else midpoint,
            "observed_fraction": (k / n) if n else 0.0,
            "count": n,
            "ci_lower": lo,
            "ci_upper": hi,
            "sparse": n < min_count,
        })

    return {"bins": bins, "edges": [float(e) for e in edges]}


def expected_calibration_error(
    pred: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 30,
    edges: np.ndarray | None = None,
) -> float:
    """Count-weighted mean absolute deviation from the diagonal.

    Note the weighting caveat in the module docstring: at a 1:193 imbalance most
    of the weight lands on near-zero bins, so this can look excellent while the
    decision region is poor. Read it together with
    :func:`decision_region_ece` and :func:`max_calibration_error`.
    """
    data = reliability_bins(pred, labels, n_bins=n_bins, edges=edges)
    total = sum(b["count"] for b in data["bins"])
    if total == 0:
        return 0.0
    return float(
        sum(
            (b["count"] / total) * abs(b["observed_fraction"] - b["mean_predicted"])
            for b in data["bins"]
        )
    )


def max_calibration_error(
    pred: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 30,
    edges: np.ndarray | None = None,
    min_count: int = MIN_BIN_COUNT,
) -> float:
    """Worst absolute deviation over non-sparse bins, unweighted.

    Complements ECE: a small badly-calibrated region cannot hide behind a large
    well-calibrated one. Sparse bins are excluded so this is not just the
    noisiest bin's sampling error.
    """
    data = reliability_bins(pred, labels, n_bins=n_bins, edges=edges, min_count=min_count)
    gaps = [
        abs(b["observed_fraction"] - b["mean_predicted"])
        for b in data["bins"]
        if not b["sparse"]
    ]
    return float(max(gaps)) if gaps else 0.0


def decision_region_ece(
    pred: np.ndarray,
    labels: np.ndarray,
    region: tuple[float, float] = DECISION_REGION,
    n_bins: int = 10,
) -> dict:
    """ECE restricted to the τ sweep range (spec §2.8).

    Calibration outside this range does not affect any reachable operating
    point. This is the number that speaks to whether the gate is deployable.

    Returns
    -------
    dict
        ``ece``, ``n_rows``, ``positive_fraction``, ``region``. ``ece`` is NaN
        when the region holds too few rows to estimate — reported as NaN rather
        than 0.0, since "no data" is not "perfectly calibrated".
    """
    pred = np.asarray(pred, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    lo, hi = region
    mask = (pred >= lo) & (pred <= hi)
    n = int(mask.sum())
    if n < MIN_BIN_COUNT:
        return {"ece": float("nan"), "n_rows": n, "positive_fraction": float("nan"),
                "region": [lo, hi]}

    edges = logit_bin_edges(n_bins, p_min=lo, p_max=hi)
    return {
        "ece": expected_calibration_error(pred[mask], labels[mask], edges=edges),
        "n_rows": n,
        "positive_fraction": float(labels[mask].mean()),
        "region": [lo, hi],
    }


def soft_reliability_bins(
    pred: np.ndarray,
    target: np.ndarray,
    n_bins: int = 20,
    edges: np.ndarray | None = None,
    min_count: int = MIN_BIN_COUNT,
) -> dict:
    """Reliability bins for a **soft** target in [0, 1] (the value function).

    Used for V_φ against V^{π†}. The gate's target is Bernoulli, so its per-bin
    uncertainty is a binomial proportion and the Wilson interval is correct.
    V^{π†} is not Bernoulli — it is a continuous score in [0, 1] — so applying
    Wilson here would be wrong twice over: it assumes the bin's outcomes are
    0/1, and it would report an interval on a proportion that is not what is
    being estimated. The quantity of interest is the **mean target** in the bin,
    so the interval is the standard error of that mean, ``s / sqrt(n)``.

    Calibration for a soft target means E[V^{π†} | V_φ = p] = p — the conditional
    mean of the target equals the prediction. That is exactly what the soft BCE
    of §3.4 is minimised by, so this plot tests the thing the loss optimises.

    Bins are equal-width in probability, not log-odds. V^{π†} is bimodal with
    mass at both ends and genuine density in between, so it does not have the
    gate's five-orders-of-magnitude dynamic range problem, and a linear axis
    reads more naturally against the y = x diagonal.

    Returns
    -------
    dict
        ``bins`` : list of dict with ``bin_lo``, ``bin_hi``, ``mean_predicted``,
        ``mean_target``, ``count``, ``ci_lower``, ``ci_upper``, ``sparse``;
        and ``edges`` : list of float.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if edges is None:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.asarray(edges, dtype=np.float64)
    n_bins = len(edges) - 1

    bin_idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)

    bins = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n:
            t = target[mask]
            mean_t = float(t.mean())
            # ddof=1 needs n >= 2; a single-row bin has no spread estimate.
            sem = float(t.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            half = 1.96 * sem if sem == sem else 0.0
            lo, hi = max(0.0, mean_t - half), min(1.0, mean_t + half)
            mean_p = float(pred[mask].mean())
        else:
            mean_t, mean_p, lo, hi = 0.0, float(0.5 * (edges[b] + edges[b + 1])), 0.0, 1.0
        bins.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "mean_predicted": mean_p,
            "mean_target": mean_t,
            "count": n,
            "ci_lower": lo,
            "ci_upper": hi,
            "sparse": n < min_count,
        })

    return {"bins": bins, "edges": [float(e) for e in edges]}


def soft_calibration_error(
    pred: np.ndarray,
    target: np.ndarray,
    n_bins: int = 20,
    edges: np.ndarray | None = None,
) -> float:
    """Count-weighted mean |mean_target − mean_predicted| for a soft target.

    The ECE analogue for V_φ. Zero iff the conditional mean of V^{π†} equals the
    prediction in every bin.
    """
    data = soft_reliability_bins(pred, target, n_bins=n_bins, edges=edges)
    total = sum(b["count"] for b in data["bins"])
    if total == 0:
        return 0.0
    return float(
        sum(
            (b["count"] / total) * abs(b["mean_target"] - b["mean_predicted"])
            for b in data["bins"]
        )
    )


def eta_strata(eta: np.ndarray) -> dict[str, np.ndarray]:
    """Five η strata spanning −3 to 3 (spec §4.2).

    Returns
    -------
    dict of str to numpy.ndarray
        Stratum label → boolean mask. Rows with ``|η| > 3`` fall in no stratum,
        which is intended: they are outside the tracker acceptance the plots
        describe.
    """
    eta = np.asarray(eta, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for i in range(len(ETA_EDGES) - 1):
        lo, hi = ETA_EDGES[i], ETA_EDGES[i + 1]
        upper_inclusive = i == len(ETA_EDGES) - 2
        mask = (eta >= lo) & ((eta <= hi) if upper_inclusive else (eta < hi))
        out[f"eta[{lo:+.1f},{hi:+.1f}]"] = mask
    return out


def quintile_strata(values: np.ndarray) -> dict[str, np.ndarray]:
    """Five quantile strata of a continuous or integer quantity.

    Used for ``n_window`` occupancy quintiles. Edges are made strictly
    increasing so that a heavily-atomic distribution (n_window is a small
    integer) still yields five non-empty strata rather than collapsing.
    """
    values = np.asarray(values, dtype=np.float64)
    edges = np.quantile(values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)

    out: dict[str, np.ndarray] = {}
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        mask = (values >= lo) & ((values <= hi) if i == 4 else (values < hi))
        out[f"Q{i + 1}[{lo:.3g},{hi:.3g}]"] = mask
    return out
