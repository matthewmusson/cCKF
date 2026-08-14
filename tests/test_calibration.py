"""Tests for Platt scaling."""

from __future__ import annotations

import numpy as np
import pytest

from cckf import calibration, metrics


def _synthetic_logits(n=200_000, a_true=1.0, b_true=0.0, seed=0):
    """Labels generated from sigma(a_true * z + b_true)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, size=n)
    p = 1.0 / (1.0 + np.exp(-(a_true * z + b_true)))
    y = (rng.uniform(size=n) < p).astype(np.uint8)
    return z, y


def test_fit_platt_recovers_identity_on_calibrated_logits():
    z, y = _synthetic_logits(a_true=1.0, b_true=0.0)
    a, b = calibration.fit_platt(z, y)
    assert a == pytest.approx(1.0, abs=0.05)
    assert b == pytest.approx(0.0, abs=0.05)


def test_fit_platt_recovers_a_known_slope_and_intercept():
    z, y = _synthetic_logits(a_true=0.5, b_true=-1.5)
    a, b = calibration.fit_platt(z, y)
    assert a == pytest.approx(0.5, abs=0.05)
    assert b == pytest.approx(-1.5, abs=0.08)


def test_platt_reduces_ece_on_miscalibrated_logits():
    z, y = _synthetic_logits(a_true=0.4, b_true=1.0)
    raw = 1.0 / (1.0 + np.exp(-z))
    a, b = calibration.fit_platt(z, y)
    cal = calibration.apply_platt(z, a, b)
    assert metrics.expected_calibration_error(
        cal, y
    ) < metrics.expected_calibration_error(raw, y)


def test_apply_platt_returns_probabilities():
    out = calibration.apply_platt(np.array([-50.0, 0.0, 50.0]), 1.0, 0.0)
    assert np.all((out >= 0.0) & (out <= 1.0))
    assert out[1] == pytest.approx(0.5)


def test_apply_platt_is_numerically_stable_in_the_tails():
    out = calibration.apply_platt(np.array([-1e4, 1e4]), 2.0, 5.0)
    assert np.all(np.isfinite(out))


def test_fit_platt_rejects_single_class_input():
    with pytest.raises(ValueError, match="both classes"):
        calibration.fit_platt(np.array([0.0, 1.0]), np.array([1, 1]))


def test_fit_platt_occupancy_recovers_occupancy_dependent_slope():
    rng = np.random.default_rng(0)
    n = 400_000
    z = rng.normal(0.0, 2.0, size=n)
    n_window = rng.integers(1, 16, size=n)
    log_nw = np.log(n_window)
    a = 1.0 + 0.2 * log_nw
    b = -0.5 - 0.3 * log_nw
    p = 1.0 / (1.0 + np.exp(-(a * z + b)))
    y = (rng.uniform(size=n) < p).astype(np.uint8)

    a0, a1, b0, b1 = calibration.fit_platt_occupancy(z, y, n_window)
    assert a0 == pytest.approx(1.0, abs=0.1)
    assert a1 == pytest.approx(0.2, abs=0.08)
    assert b0 == pytest.approx(-0.5, abs=0.1)
    assert b1 == pytest.approx(-0.3, abs=0.08)


def test_occupancy_platt_beats_two_param_when_miscalibration_is_occupancy_dependent():
    rng = np.random.default_rng(1)
    n = 400_000
    z = rng.normal(0.0, 2.0, size=n)
    n_window = rng.integers(1, 16, size=n)
    a = 1.0 + 0.4 * np.log(n_window)
    p = 1.0 / (1.0 + np.exp(-(a * z)))
    y = (rng.uniform(size=n) < p).astype(np.uint8)

    a2, b2 = calibration.fit_platt(z, y)
    p2 = calibration.apply_platt(z, a2, b2)
    p4 = calibration.apply_platt_occupancy(
        z, n_window, calibration.fit_platt_occupancy(z, y, n_window)
    )

    worst2 = max(
        metrics.expected_calibration_error(p2[m], y[m])
        for m in metrics.quintile_strata(n_window).values()
        if m.sum() > 1000
    )
    worst4 = max(
        metrics.expected_calibration_error(p4[m], y[m])
        for m in metrics.quintile_strata(n_window).values()
        if m.sum() > 1000
    )
    assert worst4 < worst2


def test_occupancy_platt_handles_n_window_of_zero():
    """log(0) must not appear; n_window is floored at 1."""
    z = np.array([0.0, 1.0, -1.0, 2.0])
    out = calibration.apply_platt_occupancy(
        z, np.array([0, 1, 2, 3]), (1.0, 0.1, 0.0, 0.0)
    )
    assert np.all(np.isfinite(out))
