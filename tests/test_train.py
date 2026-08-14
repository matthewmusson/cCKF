"""Tests for the shared training loop."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cckf import models, train


@pytest.fixture
def separable_data():
    """A linearly separable problem the MLP must solve nearly perfectly."""
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.uint8)
    return X[:3000], y[:3000], X[3000:], y[3000:]


def test_standardize_produces_zero_mean_unit_variance():
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]], dtype=np.float32)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    Z = train.standardize(X, mu, sigma)
    np.testing.assert_allclose(Z.mean(axis=0), [0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(Z.std(axis=0), [1.0, 1.0], atol=1e-5)


def test_standardize_passes_through_unit_sigma_features():
    X = np.array([[2.0], [4.0]], dtype=np.float32)
    Z = train.standardize(X, np.array([0.0], np.float32), np.array([1.0], np.float32))
    np.testing.assert_allclose(Z, X)


def test_train_model_reduces_validation_loss(separable_data):
    Xtr, ytr, Xva, yva = separable_data
    cfg = train.TrainConfig(batch_size=256, max_epochs=10, seed=0)
    result = train.train_model(models.GateMLP(n_features=4, depth=2), Xtr, ytr, Xva, yva, cfg)
    assert result["history"]["val_loss"][-1] < result["history"]["val_loss"][0]
    assert result["best_val_loss"] < 0.3


def test_train_model_is_reproducible_from_seed(separable_data):
    Xtr, ytr, Xva, yva = separable_data
    cfg = train.TrainConfig(batch_size=256, max_epochs=3, seed=7)
    a = train.train_model(models.GateMLP(n_features=4, depth=2), Xtr, ytr, Xva, yva, cfg)
    b = train.train_model(models.GateMLP(n_features=4, depth=2), Xtr, ytr, Xva, yva, cfg)
    assert a["history"]["val_loss"] == pytest.approx(b["history"]["val_loss"])


def test_train_model_early_stops_on_patience(separable_data):
    Xtr, ytr, Xva, yva = separable_data
    cfg = train.TrainConfig(batch_size=256, max_epochs=100, patience=2, seed=0)
    result = train.train_model(models.GateMLP(n_features=4, depth=2), Xtr, ytr, Xva, yva, cfg)
    assert result["stopped_epoch"] < 100


def test_train_model_returns_the_best_not_the_last_state(separable_data):
    Xtr, ytr, Xva, yva = separable_data
    cfg = train.TrainConfig(batch_size=256, max_epochs=8, seed=0)
    result = train.train_model(models.GateMLP(n_features=4, depth=2), Xtr, ytr, Xva, yva, cfg)
    assert result["best_val_loss"] == pytest.approx(min(result["history"]["val_loss"]))


def test_lr_schedule_decays_from_lr_to_lr_min(separable_data):
    Xtr, ytr, Xva, yva = separable_data
    cfg = train.TrainConfig(batch_size=512, max_epochs=5, patience=99, lr=1e-3, lr_min=1e-5, seed=0)
    result = train.train_model(models.GateMLP(n_features=4, depth=2), Xtr, ytr, Xva, yva, cfg)
    lrs = result["history"]["lr"]
    assert lrs[0] == pytest.approx(1e-3, rel=1e-3)
    assert lrs[-1] < lrs[0]
    assert lrs[-1] >= 1e-5 * 0.99


def test_train_model_accepts_soft_targets_for_the_value_function():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3)).astype(np.float32)
    v = 1.0 / (1.0 + np.exp(-X[:, 0]))
    cfg = train.TrainConfig(batch_size=256, max_epochs=5, seed=0)
    result = train.train_model(
        models.ValueMLP(n_features=3), X[:1500], v[:1500].astype(np.float32),
        X[1500:], v[1500:].astype(np.float32), cfg
    )
    assert np.isfinite(result["best_val_loss"])


def test_predict_logits_matches_direct_forward(separable_data):
    Xtr, _, _, _ = separable_data
    net = models.GateMLP(n_features=4, depth=2).eval()
    batched = train.predict_logits(net, Xtr, batch_size=128)
    with torch.no_grad():
        direct = net(torch.from_numpy(Xtr)).numpy()
    np.testing.assert_allclose(batched, direct, atol=1e-5)


def test_predict_logits_returns_one_value_per_row(separable_data):
    Xtr, _, _, _ = separable_data
    out = train.predict_logits(models.GateMLP(n_features=4, depth=2), Xtr)
    assert out.shape == (len(Xtr),)


def test_gradient_clipping_is_applied(separable_data):
    """A pathological batch must not blow the weights up to NaN."""
    Xtr, ytr, Xva, yva = separable_data
    Xtr = Xtr.copy()
    Xtr[0] = 1e6  # extreme input
    cfg = train.TrainConfig(batch_size=256, max_epochs=3, grad_clip=1.0, seed=0)
    result = train.train_model(models.GateMLP(n_features=4, depth=2), Xtr, ytr, Xva, yva, cfg)
    assert all(np.isfinite(v) for v in result["history"]["val_loss"])
