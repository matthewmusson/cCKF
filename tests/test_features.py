"""Tests for feature derivations."""
from __future__ import annotations

import numpy as np
import pytest

from cckf import features


def test_gate_features_has_26_unique_names():
    assert len(features.GATE_FEATURES) == 26
    assert len(set(features.GATE_FEATURES)) == 26


def test_value_features_has_11_unique_names():
    assert len(features.VALUE_FEATURES) == 11
    assert len(set(features.VALUE_FEATURES)) == 11


def test_gate_groups_partition_the_feature_list():
    covered = [f for g in features.GATE_GROUPS.values() for f in g]
    assert sorted(covered) == sorted(features.GATE_FEATURES)


def test_source_columns_all_exist_in_schema():
    from cckf.splits import SCHEMA_76

    for col in features.GATE_SOURCE_COLUMNS:
        assert col in SCHEMA_76, col


def test_no_standardize_is_exactly_the_three_branch_counters():
    assert features.NO_STANDARDIZE == frozenset({"n_hits", "n_holes", "n_seq_holes"})


def test_eta_from_theta_at_ninety_degrees_is_zero():
    assert features.eta_from_theta(np.array([np.pi / 2]))[0] == pytest.approx(0.0)


def test_eta_from_theta_matches_closed_form():
    theta = np.array([0.5, 1.0, 2.5])
    expected = -np.log(np.tan(theta / 2.0))
    np.testing.assert_allclose(features.eta_from_theta(theta), expected, rtol=1e-12)


def test_eta_from_theta_is_finite_at_the_poles():
    """theta = 0 or pi would give +/-inf; must clip instead of emitting inf."""
    out = features.eta_from_theta(np.array([0.0, np.pi]))
    assert np.all(np.isfinite(out))


def test_cholesky_S_reproduces_S():
    S00, S01, S11 = np.array([4.0]), np.array([2.0]), np.array([5.0])
    l00, l10, l11 = features.cholesky_S(S00, S01, S11)
    assert l00[0] == pytest.approx(2.0)
    assert l10[0] == pytest.approx(1.0)
    assert l11[0] == pytest.approx(2.0)
    # L L^T == S
    assert (l00[0] ** 2) == pytest.approx(S00[0])
    assert (l00[0] * l10[0]) == pytest.approx(S01[0])
    assert (l10[0] ** 2 + l11[0] ** 2) == pytest.approx(S11[0])


def test_cholesky_S_handles_non_positive_definite_input():
    """Numerically degenerate S must yield finite values, not NaN."""
    l00, l10, l11 = features.cholesky_S(
        np.array([0.0]), np.array([5.0]), np.array([1.0])
    )
    assert np.all(np.isfinite([l00[0], l10[0], l11[0]]))


def test_chi2_log_odds_is_monotonically_decreasing():
    chi2 = np.array([0.0, 1.0, 4.0, 25.0, 1e4])
    out = features.chi2_log_odds(chi2)
    assert np.all(np.diff(out) < 0)
    assert np.all(np.isfinite(out))


def test_chi2_log_odds_matches_definition_mid_range():
    chi2 = np.array([2.0])
    lam = np.exp(-chi2 / 2.0)
    expected = np.log(lam / (1.0 - lam))
    np.testing.assert_allclose(features.chi2_log_odds(chi2), expected, rtol=1e-9)


def test_build_gate_features_shape_and_dtype(synthetic_df):
    X = features.build_gate_features(synthetic_df)
    assert X.shape == (len(synthetic_df), 26)
    assert X.dtype == np.float32


def test_build_gate_features_known_values(synthetic_df):
    X = features.build_gate_features(synthetic_df)
    idx = {name: i for i, name in enumerate(features.GATE_FEATURES)}
    # Row 0 of the fixture: theta = pi/2 → eta = 0
    assert X[0, idx["eta"]] == pytest.approx(0.0, abs=1e-6)
    # S = [[4,2],[2,5]] → L00 = 2, L10 = 1, L11 = 2
    assert X[0, idx["chol_S_00"]] == pytest.approx(2.0)
    assert X[0, idx["chol_S_10"]] == pytest.approx(1.0)
    assert X[0, idx["chol_S_11"]] == pytest.approx(2.0)
    # alpha_u = 0 → expected_size_u = 1 → kappa_u = clus_s_u = 2
    assert X[0, idx["kappa_u"]] == pytest.approx(2.0)
    assert X[0, idx["kappa_v"]] == pytest.approx(3.0)
    # q_tilde = q_tot / (thickness * secant) = 0.4 / (0.2 * 1) = 2.0
    assert X[0, idx["q_tilde"]] == pytest.approx(2.0)


def test_build_gate_features_is_finite_on_hole_rows(synthetic_df):
    """Hole rows carry NaN cluster features; the cache must not emit NaN."""
    X = features.build_gate_features(synthetic_df)
    assert np.all(np.isfinite(X)), "non-finite feature emitted"


def test_kappa_grows_with_incidence_angle():
    """A steeper track crosses more channels, so expected size grows and
    kappa (observed / expected) shrinks for a fixed observed cluster."""
    import pandas as pd

    df = pd.DataFrame({
        "clus_s_u": [4.0, 4.0], "alpha_u": [0.0, 1.2],
        "pitch_u": [0.05, 0.05], "thickness": [0.2, 0.2],
    })
    kappa = features.kappa_u(df)
    assert kappa[1] < kappa[0]
