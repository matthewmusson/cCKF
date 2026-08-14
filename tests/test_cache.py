"""Tests for the streaming feature cache."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cckf import cache, features


def test_build_gate_cache_keeps_only_trainable_rows(synthetic_parquet, tmp_path):
    out = tmp_path / "gate_train"
    report = cache.build_gate_cache([synthetic_parquet], out, batch_rows=2)
    # Fixture has 6 rows; 3 are trainable (2 positives + 1 negative).
    assert report["n_rows"] == 3
    assert report["n_positive"] == 2


def test_cache_files_exist_and_have_consistent_length(synthetic_parquet, tmp_path):
    out = tmp_path / "gate_train"
    cache.build_gate_cache([synthetic_parquet], out, batch_rows=2)
    loaded = cache.load_cache(out)
    n = loaded["meta"]["n_rows"]
    assert loaded["X"].shape == (n, len(features.GATE_FEATURES))
    assert loaded["y"].shape == (n,)
    assert loaded["aux"].shape == (n, 3)
    assert loaded["X"].dtype == np.float32
    assert loaded["y"].dtype == np.uint8


def test_cache_batching_does_not_change_contents(synthetic_parquet, tmp_path):
    a = cache.build_gate_cache([synthetic_parquet], tmp_path / "a", batch_rows=1)
    b = cache.build_gate_cache([synthetic_parquet], tmp_path / "b", batch_rows=1000)
    assert a["n_rows"] == b["n_rows"]
    Xa = cache.load_cache(tmp_path / "a")["X"]
    Xb = cache.load_cache(tmp_path / "b")["X"]
    np.testing.assert_array_equal(np.asarray(Xa), np.asarray(Xb))


def test_cached_features_are_all_finite(synthetic_parquet, tmp_path):
    out = tmp_path / "gate_train"
    cache.build_gate_cache([synthetic_parquet], out, batch_rows=2)
    X = np.asarray(cache.load_cache(out)["X"])
    assert np.all(np.isfinite(X))


def test_aux_columns_are_chi2_nwindow_eta(synthetic_parquet, tmp_path):
    out = tmp_path / "gate_train"
    cache.build_gate_cache([synthetic_parquet], out, batch_rows=2)
    loaded = cache.load_cache(out)
    assert loaded["meta"]["aux_columns"] == ["chi2_inc", "n_window", "eta"]
    aux = np.asarray(loaded["aux"])
    # Fixture trainable rows have chi2 = 0.0, 1.0, 9.0 in order.
    np.testing.assert_allclose(aux[:, 0], [0.0, 1.0, 9.0], rtol=1e-6)


def test_ambiguous_flag_is_recorded_for_A8b(synthetic_parquet, tmp_path):
    out = tmp_path / "gate_train"
    cache.build_gate_cache([synthetic_parquet], out, batch_rows=2)
    loaded = cache.load_cache(out)
    # Second trainable row is the merged cluster.
    assert np.asarray(loaded["ambiguous"]).tolist() == [False, True, False]


def test_compute_norm_stats_skips_branch_counters(synthetic_parquet, tmp_path):
    out = tmp_path / "gate_train"
    cache.build_gate_cache([synthetic_parquet], out, batch_rows=2)
    loaded = cache.load_cache(out)
    mu, sigma = cache.compute_norm_stats(
        loaded["X"], skip=features.NO_STANDARDIZE, names=features.GATE_FEATURES
    )
    idx = {n: i for i, n in enumerate(features.GATE_FEATURES)}
    for name in features.NO_STANDARDIZE:
        assert mu[idx[name]] == 0.0
        assert sigma[idx[name]] == 1.0
    # A standardised feature with real spread gets a non-unit sigma.
    assert sigma[idx["chi2_inc"]] != 1.0


def test_compute_norm_stats_never_returns_zero_sigma(synthetic_parquet, tmp_path):
    out = tmp_path / "gate_train"
    cache.build_gate_cache([synthetic_parquet], out, batch_rows=2)
    loaded = cache.load_cache(out)
    mu, sigma = cache.compute_norm_stats(
        loaded["X"], skip=features.NO_STANDARDIZE, names=features.GATE_FEATURES
    )
    # pitch_u is constant in the fixture -> variance 0 -> must be floored to 1.
    assert np.all(sigma > 0)


def test_meta_json_records_feature_names(synthetic_parquet, tmp_path):
    out = tmp_path / "gate_train"
    cache.build_gate_cache([synthetic_parquet], out, batch_rows=2)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["feature_names"] == list(features.GATE_FEATURES)
