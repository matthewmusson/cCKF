"""tests/test_winfail_uncensored.py"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winfail_uncensored import (
    ETA_BINS, N_VALUES, OCC_EDGES, PT_EDGES, wilson_interval, assign_strata,
)


def test_eta_binning_is_dense():
    assert len(ETA_BINS) == 141
    assert np.isclose(ETA_BINS[0], -3.5) and np.isclose(ETA_BINS[-1], 3.5)
    assert np.allclose(np.diff(ETA_BINS), 0.05)


def test_pt_edges():
    assert PT_EDGES == (0.0, 0.7, 0.9, 1.0)


def test_wilson_interval_basic():
    lo, hi = wilson_interval(np.array([5]), np.array([10]), z=1.0)
    assert 0.0 < lo[0] < 0.5 < hi[0] < 1.0


def test_wilson_interval_zero_denominator_is_nan():
    lo, hi = wilson_interval(np.array([0]), np.array([0]))
    assert np.isnan(lo[0]) and np.isnan(hi[0])


def test_wilson_interval_extremes_stay_in_unit_interval():
    lo, hi = wilson_interval(np.array([0, 10]), np.array([10, 10]))
    assert lo[0] >= 0.0 and hi[1] <= 1.0


def test_assign_strata_sensor_groups_and_vol20():
    eta = np.array([0.0, 0.0, 0.0, 0.0])
    vol = np.array([17, 24, 29, 20])
    occ = np.array([0, 3, 12, 50])
    ei, si, oi = assign_strata(eta, vol, occ)
    assert list(si) == [0, 1, 2, -1]
    assert list(oi) == [0, 1, 3, 4]
    assert ei[0] == 70
