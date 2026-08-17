"""Post-hoc Platt calibration (spec §4.1, §4.3).

Two-parameter Platt
-------------------
    p_cal = σ(a·z + b)

fitted by minimising the negative log-likelihood on the **calibration split**.
The objective is convex in (a, b) — it is logistic regression on the single
feature z — so L-BFGS finds the global optimum and the fit is reproducible.

What each parameter fixes:

``a`` (slope)
    Corrects logits that are too spread out (a < 1, overconfident) or too
    compressed (a > 1, underconfident).
``b`` (intercept)
    Corrects a systematic prior offset. This is what absorbs the constant logit
    shift Δ from negative subsampling (§2.5 Approach B): if the network was
    trained at a 1:5 ratio instead of the true 1:193, then b̂ ≈ −Δ, and
    ``samplers.prior_logit_shift`` predicts that value so the fit can be checked
    against theory.

Sanity band: a ≈ 1 and b ≈ 0 for an already-calibrated network. If |a| > 2 the
raw network is badly miscalibrated and should be debugged, not patched — Platt
is a correction, not a rescue.

Four-parameter occupancy-conditional Platt (§4.3, ablation S2)
-------------------------------------------------------------
    a(x) = a₀ + a₁·log(n_window)
    b(x) = b₀ + b₁·log(n_window)
    p_cal = σ(a(x)·z + b(x))

Still convex — it is logistic regression on the three features
``{z, z·log n_window, log n_window}`` — but it can undo miscalibration whose
*severity depends on occupancy*, which the 2-parameter form cannot. This is the
form to reach for only if the §4.2 stratified diagrams show occupancy-dependent
bias. Let the diagnostics decide; do not add the parameters pre-emptively.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

_N_WINDOW_FLOOR = 1.0  # log(n_window) must be finite even if n_window == 0


def _nll(design: np.ndarray, params: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood of a logistic model, in the stable form.

    Uses ``log_expit`` so that ``log σ(t)`` and ``log(1 − σ(t)) = log σ(−t)``
    stay finite for |t| of order 10⁴, where computing ``log(expit(t))``
    directly would return −inf.
    """
    t = design @ params
    return float(-(y * log_expit(t) + (1.0 - y) * log_expit(-t)).mean())


def _nll_grad(design: np.ndarray, params: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytic gradient of :func:`_nll` with respect to ``params``."""
    t = design @ params
    return design.T @ (expit(t) - y) / len(y)


def _fit_logistic(
    design: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    trace: list[float] | None = None,
) -> np.ndarray:
    """Fit a logistic model by L-BFGS on the convex NLL.

    Parameters
    ----------
    trace : list of float, optional
        If given, appended in place with the NLL at the initial guess, then
        once per L-BFGS-B iteration, then at the returned optimum. The final
        entry may duplicate the last iterate's value; that is harmless and
        keeps the invariant that ``trace[-1]`` is the NLL of the parameters
        actually returned. Tracing costs one extra NLL evaluation per
        iteration and does not affect the optimisation path.
    """
    y = np.asarray(y, dtype=np.float64)
    if not (np.any(y > 0.5) and np.any(y <= 0.5)):
        raise ValueError(
            "calibration split must contain both classes; cannot fit Platt on "
            "a single-class sample"
        )

    callback = None
    if trace is not None:
        trace.append(_nll(design, x0, y))

        def callback(xk: np.ndarray) -> None:
            trace.append(_nll(design, xk, y))

    result = minimize(
        lambda p: _nll(design, p, y),
        x0,
        jac=lambda p: _nll_grad(design, p, y),
        method="L-BFGS-B",
        callback=callback,
    )
    if trace is not None:
        trace.append(_nll(design, result.x, y))
    return result.x


def fit_platt(
    logits: np.ndarray, labels: np.ndarray, trace: list[float] | None = None
) -> tuple[float, float]:
    """Fit two-parameter Platt scaling on the calibration split.

    Parameters
    ----------
    trace : list of float, optional
        Appended in place with the per-iteration NLL; see ``_fit_logistic``.

    Returns
    -------
    tuple of float
        ``(a, b)``.
    """
    z = np.asarray(logits, dtype=np.float64)
    design = np.column_stack([z, np.ones_like(z)])
    a, b = _fit_logistic(design, labels, np.array([1.0, 0.0]), trace=trace)
    return float(a), float(b)


def apply_platt(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply two-parameter Platt scaling, returning probabilities."""
    return expit(a * np.asarray(logits, dtype=np.float64) + b)


def fit_platt_occupancy(
    logits: np.ndarray,
    labels: np.ndarray,
    n_window: np.ndarray,
    trace: list[float] | None = None,
) -> tuple[float, float, float, float]:
    """Fit four-parameter occupancy-conditional Platt scaling.

    Parameters
    ----------
    trace : list of float, optional
        Appended in place with the per-iteration NLL; see ``_fit_logistic``.

    Returns
    -------
    tuple of float
        ``(a0, a1, b0, b1)`` for ``a(x) = a0 + a1·log n_window`` and
        ``b(x) = b0 + b1·log n_window``.
    """
    z = np.asarray(logits, dtype=np.float64)
    log_nw = np.log(np.maximum(np.asarray(n_window, dtype=np.float64), _N_WINDOW_FLOOR))
    design = np.column_stack([z, z * log_nw, np.ones_like(z), log_nw])
    a0, a1, b0, b1 = _fit_logistic(
        design, labels, np.array([1.0, 0.0, 0.0, 0.0]), trace=trace
    )
    return float(a0), float(a1), float(b0), float(b1)


def apply_platt_occupancy(
    logits: np.ndarray,
    n_window: np.ndarray,
    params: tuple[float, float, float, float],
) -> np.ndarray:
    """Apply four-parameter occupancy-conditional Platt scaling."""
    a0, a1, b0, b1 = params
    z = np.asarray(logits, dtype=np.float64)
    log_nw = np.log(np.maximum(np.asarray(n_window, dtype=np.float64), _N_WINDOW_FLOOR))
    return expit((a0 + a1 * log_nw) * z + (b0 + b1 * log_nw))


def platt_occupancy_slope_violations(
    n_window: np.ndarray, params: tuple[float, float, float, float]
) -> dict:
    """Report rows where the 4-param calibrator's slope is non-positive.

    The occupancy-conditional form multiplies the logit by
    ``a(x) = a0 + a1·log n_window``. When ``a(x) <= 0`` the map is
    *decreasing* in the logit: a more confident model output becomes a lower
    calibrated probability, so the calibrator inverts the model's ranking for
    those rows. Two-parameter Platt cannot do this -- ``a`` is one fitted
    scalar and its sign is checkable once -- but the four-parameter form can,
    because ``a1 < 0`` makes ``a(x)`` cross zero at ``n_window =
    exp(-a0/a1)``.

    This is not hypothetical: the 2026-08-17 arm C fit gave
    ``a0 = 0.7007, a1 = -0.1408``, crossing at ``n_window ~= 145``.

    Parameters
    ----------
    n_window : numpy.ndarray
        Per-row occupancy, on the same rows the calibrator will be applied to.
    params : tuple of float
        ``(a0, a1, b0, b1)`` from :func:`fit_platt_occupancy`.

    Returns
    -------
    dict
        ``n_window_at_slope_zero`` (``inf`` when ``a1 == 0``; a value below 1
        means the slope never inverts for physical occupancies),
        ``n_rows_slope_nonpositive``, ``frac_rows_slope_nonpositive``,
        ``min_slope``, ``max_n_window``.
    """
    a0, a1, _, _ = params
    log_nw = np.log(np.maximum(np.asarray(n_window, dtype=np.float64), _N_WINDOW_FLOOR))
    slope = a0 + a1 * log_nw
    n_bad = int((slope <= 0.0).sum())
    crossing = float(np.exp(-a0 / a1)) if a1 != 0.0 else float("inf")
    return {
        "n_window_at_slope_zero": crossing,
        "n_rows_slope_nonpositive": n_bad,
        "frac_rows_slope_nonpositive": n_bad / max(len(log_nw), 1),
        "min_slope": float(slope.min()) if len(slope) else float("nan"),
        "max_n_window": float(np.exp(log_nw.max())) if len(log_nw) else float("nan"),
    }
