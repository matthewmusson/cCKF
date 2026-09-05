"""Tests for the windowed-cache loading path in ``scripts/train_value.py``.

Window-conditioned tier-3 value plan, Task 7. The trainer must accept either
a flat cache directory (today's layout: ``X.f32``/``y.f32``/``aux.f32``/
``meta.json``/``norm_stats.npz`` directly inside it) or a parent directory
holding one ``nsig*/`` subdirectory per rollout window (Task 6's windowed
layout), concatenating the subdirs' ``X``/``y``/``aux``. These tests build
tiny synthetic caches with ``numpy`` rather than real Parquet/simhits data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cckf import features as feat
from scripts import train_value


def _write_tiny_cache(
    d: Path,
    n_rows: int,
    n_features: int,
    feature_names: list[str],
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write a minimal ``X.f32``/``y.f32``/``aux.f32``/``meta.json`` cache."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    y = rng.uniform(size=n_rows).astype(np.float32)
    aux = rng.normal(size=(n_rows, 3)).astype(np.float32)
    d.mkdir(parents=True, exist_ok=True)
    X.tofile(d / "X.f32")
    y.tofile(d / "y.f32")
    aux.tofile(d / "aux.f32")
    meta = {
        "n_rows": n_rows,
        "n_features": n_features,
        "feature_names": list(feature_names),
    }
    (d / "meta.json").write_text(json.dumps(meta))
    return X, y, aux


# --- discover_cache_dirs ----------------------------------------------------


def test_discover_cache_dirs_flat(tmp_path):
    d = tmp_path / "flatcache"
    d.mkdir()
    (d / "meta.json").write_text("{}")
    assert train_value.discover_cache_dirs(str(d)) == [d]


def test_discover_cache_dirs_windowed_parent_sorted(tmp_path):
    parent = tmp_path / "windowed"
    (parent / "nsig10").mkdir(parents=True)
    (parent / "nsig10" / "meta.json").write_text("{}")
    (parent / "nsig3").mkdir(parents=True)
    (parent / "nsig3" / "meta.json").write_text("{}")

    dirs = train_value.discover_cache_dirs(str(parent))

    assert dirs == sorted(dirs)
    assert {d.name for d in dirs} == {"nsig10", "nsig3"}


def test_discover_cache_dirs_raises_when_neither(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        train_value.discover_cache_dirs(str(d))


# --- load_value_cache: concatenation ----------------------------------------


def test_load_value_cache_flat_passthrough(tmp_path):
    d = tmp_path / "flat"
    X, y, aux = _write_tiny_cache(d, 9, 11, list(feat.VALUE_FEATURES))

    loaded = train_value.load_value_cache(str(d))

    assert loaded["cache_dirs"] == [d]
    assert loaded["meta"]["n_rows"] == 9
    assert loaded["meta"]["n_features"] == 11
    np.testing.assert_allclose(np.asarray(loaded["X"]), X)
    np.testing.assert_allclose(np.asarray(loaded["y"]), y)


def test_load_value_cache_concatenates_windowed_subdirs(tmp_path):
    feature_names = list(feat.VALUE_FEATURES_WINDOWED)
    parent = tmp_path / "windowed_parent"
    x3, y3, aux3 = _write_tiny_cache(parent / "nsig3", 5, 12, feature_names, seed=1)
    x5, y5, aux5 = _write_tiny_cache(parent / "nsig5", 7, 12, feature_names, seed=2)

    loaded = train_value.load_value_cache(str(parent))

    assert loaded["meta"]["n_rows"] == 12
    assert loaded["meta"]["n_features"] == 12
    assert loaded["meta"]["feature_names"] == feature_names
    assert [d.name for d in loaded["cache_dirs"]] == ["nsig3", "nsig5"]

    X = np.asarray(loaded["X"])
    np.testing.assert_allclose(X[:5], x3)
    np.testing.assert_allclose(X[5:], x5)
    np.testing.assert_allclose(loaded["y"][:5], y3)
    np.testing.assert_allclose(loaded["y"][5:], y5)
    np.testing.assert_allclose(np.asarray(loaded["aux"])[:5], aux3)
    np.testing.assert_allclose(np.asarray(loaded["aux"])[5:], aux5)


def test_load_value_cache_raises_on_n_features_mismatch(tmp_path):
    parent = tmp_path / "mismatch"
    _write_tiny_cache(parent / "nsig3", 5, 12, list(feat.VALUE_FEATURES_WINDOWED))
    _write_tiny_cache(parent / "nsig5", 5, 11, list(feat.VALUE_FEATURES))

    with pytest.raises(ValueError, match="n_features"):
        train_value.load_value_cache(str(parent))


def test_load_value_cache_raises_on_feature_names_mismatch(tmp_path):
    feature_names_a = list(feat.VALUE_FEATURES_WINDOWED)
    feature_names_b = feature_names_a[:-1] + ["not_window_nsigma"]
    parent = tmp_path / "mismatch_names"
    _write_tiny_cache(parent / "nsig3", 5, 12, feature_names_a)
    _write_tiny_cache(parent / "nsig5", 5, 12, feature_names_b)

    with pytest.raises(ValueError, match="feature_names"):
        train_value.load_value_cache(str(parent))


# --- get_norm_stats: recompute over concatenation ---------------------------


def test_get_norm_stats_single_dir_uses_norm_stats_npz(tmp_path):
    d = tmp_path / "train"
    _write_tiny_cache(d, 10, 11, list(feat.VALUE_FEATURES))
    mu = np.arange(11, dtype=np.float32)
    sigma = np.full(11, 2.0, dtype=np.float32)
    np.savez(d / "norm_stats.npz", mu=mu, sigma=sigma)

    tr = train_value.load_value_cache(str(d))
    got_mu, got_sigma = train_value.get_norm_stats(tr)

    np.testing.assert_allclose(got_mu, mu)
    np.testing.assert_allclose(got_sigma, sigma)


def test_get_norm_stats_recomputes_for_concatenated_cache(tmp_path):
    feature_names = list(feat.VALUE_FEATURES_WINDOWED)
    parent = tmp_path / "windowed"
    _write_tiny_cache(parent / "nsig3", 50, 12, feature_names, seed=3)
    _write_tiny_cache(parent / "nsig5", 50, 12, feature_names, seed=4)

    tr = train_value.load_value_cache(str(parent))
    mu, sigma = train_value.get_norm_stats(tr)

    assert mu.shape == (12,)
    assert sigma.shape == (12,)
    window_idx = feature_names.index("window_nsigma")
    # window_nsigma is exempt from standardisation (cckf.features.NO_STANDARDIZE).
    assert mu[window_idx] == 0.0
    assert sigma[window_idx] == 1.0
    # Sanity: a standardised column's mean matches a direct mean over the
    # concatenated X, i.e. this really is a recompute, not a copy of one
    # subdir's own norm_stats.
    X = np.asarray(tr["X"])
    np.testing.assert_allclose(mu[0], X[:, 0].mean(), atol=1e-4)


# --- meta-driven n_features, end to end -------------------------------------


def test_main_builds_model_with_cache_meta_n_features(tmp_path, monkeypatch):
    feature_names = list(feat.VALUE_FEATURES_WINDOWED)
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _write_tiny_cache(train_dir, 64, 12, feature_names, seed=5)
    np.savez(
        train_dir / "norm_stats.npz",
        mu=np.zeros(12, dtype=np.float32),
        sigma=np.ones(12, dtype=np.float32),
    )
    _write_tiny_cache(val_dir, 16, 12, feature_names, seed=6)

    out_dir = tmp_path / "out"
    argv = [
        "train_value.py",
        "--train-cache",
        str(train_dir),
        "--val-cache",
        str(val_dir),
        "--out-dir",
        str(out_dir),
        "--max-epochs",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_value.main()

    ckpt = torch.load(
        out_dir / "value_model.pt", map_location="cpu", weights_only=False
    )
    assert ckpt["n_features"] == 12
    assert ckpt["feature_names"] == feature_names


def test_main_raises_on_train_val_schema_mismatch(tmp_path, monkeypatch):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _write_tiny_cache(train_dir, 20, 12, list(feat.VALUE_FEATURES_WINDOWED))
    np.savez(
        train_dir / "norm_stats.npz",
        mu=np.zeros(12, dtype=np.float32),
        sigma=np.ones(12, dtype=np.float32),
    )
    _write_tiny_cache(val_dir, 10, 11, list(feat.VALUE_FEATURES))

    out_dir = tmp_path / "out"
    argv = [
        "train_value.py",
        "--train-cache",
        str(train_dir),
        "--val-cache",
        str(val_dir),
        "--out-dir",
        str(out_dir),
        "--max-epochs",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError):
        train_value.main()
