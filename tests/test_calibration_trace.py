"""Tests for the Platt NLL trace and the 4-param slope-inversion guard."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from cckf import calibration


def _synthetic(n: int = 20_000, seed: int = 0):
    """Miscalibrated logits with a known ground-truth link."""
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=2.0, size=n)
    labels = (rng.random(n) < expit(0.5 * z - 1.0)).astype(np.float64)
    return z, labels


def test_two_param_trace_decreases_and_ends_at_the_optimum():
    z, labels = _synthetic()
    trace: list[float] = []
    calibration.fit_platt(z, labels, trace=trace)

    assert len(trace) >= 2, "trace must record at least the start and the optimum"
    # Convex NLL under L-BFGS-B: no iterate may be worse than the start, and
    # the last entry must be the best seen.
    assert trace[-1] <= trace[0]
    assert trace[-1] == pytest.approx(min(trace), abs=1e-12)


def test_trace_is_optional_and_does_not_perturb_the_fit():
    z, labels = _synthetic()
    a_traced, b_traced = calibration.fit_platt(z, labels, trace=[])
    a_plain, b_plain = calibration.fit_platt(z, labels)
    assert a_traced == pytest.approx(a_plain, abs=1e-12)
    assert b_traced == pytest.approx(b_plain, abs=1e-12)


def test_four_param_trace_records_a_lower_optimum_than_two_param():
    """The 4-param family contains the 2-param family, so its NLL optimum
    cannot be worse. Checks the trace measures the objective it claims to."""
    z, labels = _synthetic()
    n_window = np.exp(np.linspace(0.0, 5.0, len(z)))

    t2: list[float] = []
    calibration.fit_platt(z, labels, trace=t2)
    t4: list[float] = []
    calibration.fit_platt_occupancy(z, labels, n_window, trace=t4)

    assert t4[-1] <= t2[-1] + 1e-9


def test_slope_violation_detects_an_inverting_calibrator():
    """a1 < 0 makes a(x) cross zero; above that occupancy the calibrator
    reverses the model's ranking, which no affine-in-logit map should do."""
    params = (0.7007, -0.1408, -3.6135, -0.7317)  # arm C's actual fit
    n_window = np.array([2.0, 10.0, 100.0, 200.0, 1000.0])

    report = calibration.platt_occupancy_slope_violations(n_window, params)

    assert report["n_window_at_slope_zero"] == pytest.approx(145.1, rel=0.01)
    assert report["n_rows_slope_nonpositive"] == 2  # 200 and 1000
    assert report["frac_rows_slope_nonpositive"] == pytest.approx(0.4)
    assert report["min_slope"] < 0.0


def test_slope_violation_is_empty_when_a1_is_positive():
    params = (0.9648, 0.0032878, -0.0989, -0.0283)  # arm A's actual fit
    n_window = np.array([1.0, 10.0, 1000.0, 100_000.0])

    report = calibration.platt_occupancy_slope_violations(n_window, params)

    assert report["n_rows_slope_nonpositive"] == 0
    assert report["frac_rows_slope_nonpositive"] == 0.0
    assert report["min_slope"] > 0.0
