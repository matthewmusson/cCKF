"""Feature construction for the gate g_ψ and value function V_φ.

Feature counts
--------------
The training spec's prose says "~23" gate features but its own group tables sum
to 26 (6 Kalman + 6 raw cluster + 3 normalised cluster + 1 occupancy + 4
context + 3 sensor + 3 branch history). This module uses **26**. ``is_pixel``
is omitted from the primary vector — the spec notes it is largely redundant
with the pitch values — and returns only as an A4a ablation toggle. The value
function's groups likewise sum to **11**, not the stated "~12".

Derived quantities and why
--------------------------
``eta``
    There is no ``eta`` column in the Parquet. Pseudorapidity comes from the
    predicted polar angle: η = −ln(tan(θ/2)), using ``state_theta``.

``chol_S_00 / chol_S_10 / chol_S_11``
    The lower-triangular Cholesky factor of the 2×2 innovation covariance
    S = H C Hᵀ + R, read straight from the stored ``S00/S01/S11`` columns
    (these already include the measurement noise R — do not add it again).
    Passing L rather than S gives the network three numbers on the scale of
    *lengths* rather than squared lengths, and L is the natural parameterisation
    because r̃ = L⁻¹r is the whitened residual whose norm² is χ².

``kappa_u / kappa_v``
    Observed cluster extent divided by the extent predicted from the track's
    incidence angle and the sensor geometry::

        expected_size_u = 1 + thickness · |tan α_u| / pitch_u

    One channel minimum, plus one per pitch-width of lateral travel through the
    sensor's depth. A track at steep incidence *should* leave an elongated
    cluster; κ ≈ 1 means the cluster looks the way this track would make it,
    while κ ≫ 1 flags a merged cluster or a different particle.

``q_tilde``
    Charge divided by the expected MIP deposit, which scales with the path
    length through the sensor::

        expected_q = thickness · sqrt(1 + tan²α_u + tan²α_v)

    The absolute MIP charge-per-unit-length constant is deliberately **not**
    included: q̃ is standardised downstream, and a constant multiplicative
    factor is entirely absorbed by the per-feature σ. Only the *shape* of the
    dependence on thickness and incidence matters. (Q̃ assumes MIP energy loss;
    the gate is expected to cross-reference it against ``state_qop`` for
    non-MIP particles on the rising part of the Bethe–Bloch curve, which is why
    both features are present.)

``chi2_log_odds``
    The χ²-implied log-odds under the Gaussian-linear model with ν=2 degrees of
    freedom, where the survival function is exactly Λ = exp(−χ²/2)::

        logit(Λ) = log(Λ / (1 − Λ))

    Used only as the iteration-0 stand-in for the value function's accumulated
    gate log-odds features (spec §3.2), so those features live on the same
    probability scale before and after the learned gate replaces them.

Standardisation
---------------
Per spec §2.3, integer branch counters (``n_hits``, ``n_holes``,
``n_seq_holes``) are **not** standardised — they are small-support ordinals
whose absolute values carry meaning, and z-scoring them would make the value
"three sequential holes" depend on the dataset composition. Binary features are
likewise exempt (none are in the primary vector). Everything else is
standardised with train-split μ and σ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- Numerical guards -----------------------------------------------------
_THETA_EPS = 1e-6  # keeps eta finite at the poles
_VAR_FLOOR = 1e-30  # keeps sqrt and division finite on degenerate S
_LAMBDA_CLIP = 1e-6  # keeps logit(Lambda) finite at chi2 -> 0 and chi2 -> inf

# --- Feature name lists (order is the network's input order; do not reorder)
GATE_GROUPS: dict[str, tuple[str, ...]] = {
    "kalman": ("residual_l0", "residual_l1", "chol_S_00", "chol_S_10", "chol_S_11", "chi2_inc"),
    "cluster_raw": (
        "clus_s_u", "clus_s_v", "clus_q_tot",
        "clus_sigma_uu", "clus_sigma_uv", "clus_sigma_vv",
    ),
    "cluster_norm": ("kappa_u", "kappa_v", "q_tilde"),
    "occupancy": ("n_window",),
    "context": ("eta", "state_qop", "step_k", "pathInX0_interval"),
    "sensor": ("pitch_u", "pitch_v", "thickness"),
    "history": ("n_hits", "n_holes", "n_seq_holes"),
}

GATE_FEATURES: tuple[str, ...] = tuple(f for g in GATE_GROUPS.values() for f in g)

VALUE_FEATURES: tuple[str, ...] = (
    "eta",
    "state_qop",
    "sigma2_l0",
    "sigma2_l1",
    "n_hits",
    "n_holes",
    "n_seq_holes",
    "sum_gate_logodds",
    "min_gate_logodds",
    "step_k",
    "x0_accumulated",
)

NO_STANDARDIZE: frozenset[str] = frozenset({"n_hits", "n_holes", "n_seq_holes"})

#: Parquet columns needed to build the gate feature matrix.
GATE_SOURCE_COLUMNS: tuple[str, ...] = (
    "residual_l0", "residual_l1", "S00", "S01", "S11", "chi2_inc",
    "clus_s_u", "clus_s_v", "clus_q_tot",
    "clus_sigma_uu", "clus_sigma_uv", "clus_sigma_vv",
    "alpha_u", "alpha_v",
    "n_window",
    "state_theta", "state_qop", "step_k", "pathInX0_interval",
    "pitch_u", "pitch_v", "thickness",
    "n_hits", "n_holes", "n_seq_holes",
)

#: Parquet columns needed to build the value feature matrix.
VALUE_SOURCE_COLUMNS: tuple[str, ...] = (
    "event_id", "seed_id", "branch_id", "step_k",
    "state_theta", "state_qop", "cov_00", "cov_06",
    "n_hits", "n_holes", "n_seq_holes",
    "chi2_inc", "cand_hit_id", "pathInX0_interval",
)


def eta_from_theta(theta: np.ndarray) -> np.ndarray:
    """Pseudorapidity from polar angle: η = −ln(tan(θ/2)).

    θ is clipped away from 0 and π so that forward/backward states yield a
    large finite η rather than ±inf (which would poison standardisation).
    """
    t = np.clip(np.asarray(theta, dtype=np.float64), _THETA_EPS, np.pi - _THETA_EPS)
    return -np.log(np.tan(t / 2.0))


def cholesky_S(
    S00: np.ndarray, S01: np.ndarray, S11: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower-triangular Cholesky factor of the 2×2 innovation covariance.

    For S = [[S00, S01], [S01, S11]] the factor L satisfies L Lᵀ = S with::

        L00 = sqrt(S00)
        L10 = S01 / L00
        L11 = sqrt(S11 − L10²)

    Returns
    -------
    tuple of numpy.ndarray
        ``(L00, L10, L11)``, each float64. Variances are floored and the
        Schur complement is clamped at zero so a numerically degenerate or
        non-positive-definite S yields finite output instead of NaN.
    """
    s00 = np.maximum(np.asarray(S00, dtype=np.float64), _VAR_FLOOR)
    s01 = np.asarray(S01, dtype=np.float64)
    s11 = np.maximum(np.asarray(S11, dtype=np.float64), _VAR_FLOOR)

    l00 = np.sqrt(s00)
    l10 = s01 / l00
    l11 = np.sqrt(np.maximum(s11 - l10 * l10, _VAR_FLOOR))
    return l00, l10, l11


def chi2_log_odds(chi2: np.ndarray) -> np.ndarray:
    """χ²-implied log-odds, log(Λ/(1−Λ)) with Λ = exp(−χ²/2).

    Λ is the ν=2 χ² survival function, so this is the probability the
    Gaussian-linear model assigns to a residual at least this large. Λ is
    clipped into ``[1e-6, 1−1e-6]`` so χ²→0 (Λ→1) and χ²→∞ (Λ→0) stay finite.
    NaN input (hole rows) maps to the most-negative clipped value, i.e. "no
    evidence this is the right hit".
    """
    c = np.asarray(chi2, dtype=np.float64)
    c = np.where(np.isfinite(c), np.maximum(c, 0.0), np.inf)
    lam = np.clip(np.exp(-c / 2.0), _LAMBDA_CLIP, 1.0 - _LAMBDA_CLIP)
    return np.log(lam / (1.0 - lam))


def kappa_u(df: pd.DataFrame) -> np.ndarray:
    """Observed / expected cluster extent along local u."""
    expected = 1.0 + (
        df["thickness"].to_numpy(dtype=np.float64)
        * np.abs(np.tan(df["alpha_u"].to_numpy(dtype=np.float64)))
        / np.maximum(df["pitch_u"].to_numpy(dtype=np.float64), _VAR_FLOOR)
    )
    return df["clus_s_u"].to_numpy(dtype=np.float64) / expected


def kappa_v(df: pd.DataFrame) -> np.ndarray:
    """Observed / expected cluster extent along local v."""
    expected = 1.0 + (
        df["thickness"].to_numpy(dtype=np.float64)
        * np.abs(np.tan(df["alpha_v"].to_numpy(dtype=np.float64)))
        / np.maximum(df["pitch_v"].to_numpy(dtype=np.float64), _VAR_FLOOR)
    )
    return df["clus_s_v"].to_numpy(dtype=np.float64) / expected


def q_tilde(df: pd.DataFrame) -> np.ndarray:
    """Cluster charge / expected MIP deposit for this path length."""
    tan_u = np.tan(df["alpha_u"].to_numpy(dtype=np.float64))
    tan_v = np.tan(df["alpha_v"].to_numpy(dtype=np.float64))
    secant = np.sqrt(1.0 + tan_u * tan_u + tan_v * tan_v)
    expected = np.maximum(
        df["thickness"].to_numpy(dtype=np.float64) * secant, _VAR_FLOOR
    )
    return df["clus_q_tot"].to_numpy(dtype=np.float64) / expected


def build_gate_features(df: pd.DataFrame) -> np.ndarray:
    """Build the (n_rows, 26) float32 gate feature matrix.

    Columns follow :data:`GATE_FEATURES` order exactly. Non-finite values
    (hole rows carry NaN cluster features, degenerate states carry NaN
    covariance) are replaced with 0.0 *after* all derivations, so the network
    never sees NaN. Rows that matter for gate training are filtered upstream by
    ``labels.derive_labels``' mask; this zero-fill only protects the arithmetic.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain :data:`GATE_SOURCE_COLUMNS`.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(df), 26)``, dtype float32, all entries finite.
    """
    l00, l10, l11 = cholesky_S(
        df["S00"].to_numpy(dtype=np.float64),
        df["S01"].to_numpy(dtype=np.float64),
        df["S11"].to_numpy(dtype=np.float64),
    )

    derived: dict[str, np.ndarray] = {
        "chol_S_00": l00,
        "chol_S_10": l10,
        "chol_S_11": l11,
        "kappa_u": kappa_u(df),
        "kappa_v": kappa_v(df),
        "q_tilde": q_tilde(df),
        "eta": eta_from_theta(df["state_theta"].to_numpy(dtype=np.float64)),
    }

    n_rows = len(df)
    out = np.empty((n_rows, len(GATE_FEATURES)), dtype=np.float64)
    for j, name in enumerate(GATE_FEATURES):
        if name in derived:
            out[:, j] = derived[name]
        else:
            out[:, j] = df[name].to_numpy(dtype=np.float64)

    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)
