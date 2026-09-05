"""Tests for ``scripts/value_window_monotonicity.py``'s pure scoring logic.

Window-conditioned tier-3 value plan, Task 7 acceptance check: mean predicted
V at fixed state features must be non-increasing as the rollout window
shrinks (spot-check by re-scoring 1k cal states with the 12th feature,
``window_nsigma``, swapped 10 -> 3). These tests exercise
``monotonicity_stats`` directly on a stub model whose output is a known
linear function of the 12th feature, so the expected fraction and mean
difference can be computed by hand instead of trusting a real trained model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cckf import features as feat
from scripts.value_window_monotonicity import monotonicity_stats, score_at_window


class _LinearInWindow(nn.Module):
    """Stub model: logit = weight * x[:, window_idx] + bias, ignoring the rest.

    Increasing in the window column when ``weight > 0`` -- i.e. V should come
    out *higher* for a larger acceptance window (10 sigma) than a smaller one
    (3 sigma), matching the plan's "non-increasing as the window shrinks"
    acceptance direction.
    """

    def __init__(self, window_idx: int, weight: float, bias: float) -> None:
        super().__init__()
        self.window_idx = window_idx
        self.weight = weight
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x[:, self.window_idx] + self.bias


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def test_score_at_window_overrides_only_the_window_column():
    window_idx = list(feat.VALUE_FEATURES_WINDOWED).index("window_nsigma")
    n_features = len(feat.VALUE_FEATURES_WINDOWED)
    model = _LinearInWindow(window_idx, weight=1.0, bias=0.0)
    model.eval()

    X = np.zeros((3, n_features), dtype=np.float32)
    X[:, window_idx] = 999.0  # should be overridden, not read
    mu = np.zeros(n_features, dtype=np.float32)
    sigma = np.ones(n_features, dtype=np.float32)

    pred = score_at_window(model, X, mu, sigma, window_idx, window_value=5.0)

    np.testing.assert_allclose(pred, _sigmoid(5.0), atol=1e-6)


def test_monotonicity_stats_on_linear_stub_fully_monotonic():
    window_idx = list(feat.VALUE_FEATURES_WINDOWED).index("window_nsigma")
    n_features = len(feat.VALUE_FEATURES_WINDOWED)
    model = _LinearInWindow(window_idx, weight=0.5, bias=-1.0)
    model.eval()

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, n_features)).astype(np.float32)
    mu = np.zeros(n_features, dtype=np.float32)
    sigma = np.ones(n_features, dtype=np.float32)

    stats = monotonicity_stats(model, X, mu, sigma, window_idx, big=10.0, small=3.0)

    expected_v_big = _sigmoid(0.5 * 10.0 - 1.0)
    expected_v_small = _sigmoid(0.5 * 3.0 - 1.0)
    assert stats["frac_non_increasing"] == 1.0
    assert abs(stats["mean_diff"] - (expected_v_big - expected_v_small)) < 1e-5
    assert abs(stats["v_big_mean"] - expected_v_big) < 1e-5
    assert abs(stats["v_small_mean"] - expected_v_small) < 1e-5


def test_monotonicity_stats_flags_violations_for_reversed_stub():
    """A model with weight < 0 (V rises as the window shrinks) should be
    reported as fully violating the acceptance direction, not silently
    averaged away -- ``frac_non_increasing`` must read 0.0."""
    window_idx = list(feat.VALUE_FEATURES_WINDOWED).index("window_nsigma")
    n_features = len(feat.VALUE_FEATURES_WINDOWED)
    model = _LinearInWindow(window_idx, weight=-0.5, bias=0.0)
    model.eval()

    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, n_features)).astype(np.float32)
    mu = np.zeros(n_features, dtype=np.float32)
    sigma = np.ones(n_features, dtype=np.float32)

    stats = monotonicity_stats(model, X, mu, sigma, window_idx, big=10.0, small=3.0)

    assert stats["frac_non_increasing"] == 0.0
    assert stats["mean_diff"] < 0.0
