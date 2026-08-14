"""Tests for the streaming feature cache."""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cckf import cache, features

#: All array outputs an on-disk cache directory holds, keyed by the name
#: returned in ``load_cache``'s dict.
_ARRAY_KEYS: tuple[str, ...] = ("X", "y", "aux", "ambiguous")


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
    # batch_rows=1 forces a fresh batch (and a fresh derive_labels/build_gate_features
    # call) on every single row of the 6-row fixture, so this exercises every
    # possible intra-file batch boundary against the batch_rows=1000 case where
    # the whole file is one batch.
    a = cache.build_gate_cache([synthetic_parquet], tmp_path / "a", batch_rows=1)
    b = cache.build_gate_cache([synthetic_parquet], tmp_path / "b", batch_rows=1000)
    assert a["n_rows"] == b["n_rows"] == 3
    assert a["n_positive"] == b["n_positive"] == 2
    loaded_a = cache.load_cache(tmp_path / "a")
    loaded_b = cache.load_cache(tmp_path / "b")
    for key in _ARRAY_KEYS:
        np.testing.assert_array_equal(
            np.asarray(loaded_a[key]), np.asarray(loaded_b[key])
        )


def test_cache_batching_does_not_change_contents_multi_file(synthetic_df, tmp_path):
    # Production streams 24 files per split through one CacheWriter; a bug that
    # carries writer state wrongly across a file boundary, or mishandles the
    # transition between two ParquetFile objects, is invisible to a single-file
    # test. Write the fixture out to two separate Parquet files (different
    # event_id, as real per-event files would be) and require batching
    # invariance across both files and across an internal batch boundary within
    # each file (batch_rows=2 against a 6-row-per-file fixture forces 3
    # batches per file, i.e. a boundary mid-file as well as one between files).
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    paths = []
    for event_id in (0, 1):
        df = synthetic_df.copy()
        df["event_id"] = event_id
        path = src_dir / f"expanded_event{event_id:09d}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
        paths.append(path)

    small = cache.build_gate_cache(paths, tmp_path / "small", batch_rows=2)
    large = cache.build_gate_cache(paths, tmp_path / "large", batch_rows=1000)

    # Each file contributes 3 trainable rows (2 positive, 1 negative).
    assert small["n_rows"] == large["n_rows"] == 6
    assert small["n_positive"] == large["n_positive"] == 4

    loaded_small = cache.load_cache(tmp_path / "small")
    loaded_large = cache.load_cache(tmp_path / "large")
    for key in _ARRAY_KEYS:
        np.testing.assert_array_equal(
            np.asarray(loaded_small[key]), np.asarray(loaded_large[key])
        )


def test_meta_n_rows_matches_actual_file_sizes(synthetic_parquet, tmp_path):
    # A recorded-vs-actual n_rows mismatch would make load_cache's np.memmap
    # shape wrong (silently misaligning every row past the discrepancy) rather
    # than raising, so check the byte counts directly against what meta.json
    # claims instead of trusting load_cache's own shape math.
    out = tmp_path / "gate_train"
    cache.build_gate_cache([synthetic_parquet], out, batch_rows=2)
    meta = json.loads((out / "meta.json").read_text())
    n, f = meta["n_rows"], meta["n_features"]

    assert (out / "X.f32").stat().st_size == n * f * np.dtype(np.float32).itemsize
    assert (out / "y.u8").stat().st_size == n * np.dtype(np.uint8).itemsize
    assert (out / "aux.f32").stat().st_size == n * 3 * np.dtype(np.float32).itemsize
    assert (out / "ambiguous.u8").stat().st_size == n * np.dtype(np.uint8).itemsize


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
