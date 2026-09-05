"""Tests for ``scripts/eval_value_cal.py --window-nsigma``.

Window-conditioned tier-3 value plan, Task 7: the cal-split audit must be
able to restrict to one rollout window's rows (12th feature == N) for
per-window reliability reporting, and must error rather than silently no-op
on an 11-feature (un-windowed) cache where no such column exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cckf import features as feat
from scripts.eval_value_cal import filter_to_window


def _make_split(n_features: int, feature_names: list[str], window_values=None):
    n_rows = 30
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    if window_values is not None:
        idx = feature_names.index("window_nsigma")
        X[:, idx] = window_values
    y = rng.uniform(size=n_rows).astype(np.float32)
    return {
        "X": X,
        "y": y,
        "meta": {"n_features": n_features, "feature_names": feature_names},
    }


def test_filter_to_window_keeps_only_matching_rows():
    feature_names = list(feat.VALUE_FEATURES_WINDOWED)
    window_values = np.array([10.0] * 20 + [3.0] * 10, dtype=np.float32)
    split = _make_split(12, feature_names, window_values)

    filtered = filter_to_window(split, 3.0)

    assert filtered["X"].shape[0] == 10
    assert filtered["y"].shape[0] == 10
    idx = feature_names.index("window_nsigma")
    assert np.all(filtered["X"][:, idx] == 3.0)


def test_filter_to_window_raises_on_unwindowed_cache():
    split = _make_split(11, list(feat.VALUE_FEATURES))
    with pytest.raises(ValueError, match="window_nsigma"):
        filter_to_window(split, 3.0)


def test_filter_to_window_raises_when_value_absent():
    feature_names = list(feat.VALUE_FEATURES_WINDOWED)
    window_values = np.full(30, 10.0, dtype=np.float32)
    split = _make_split(12, feature_names, window_values)

    with pytest.raises(ValueError, match="no rows"):
        filter_to_window(split, 3.0)
