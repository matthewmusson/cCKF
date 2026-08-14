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


def test_gradient_clipping_bounds_the_post_clip_gradient_norm(separable_data, monkeypatch):
    """The clip actually bounds the gradient norm, not merely "stays finite".

    AdamW's per-parameter second-moment normalisation already keeps a single
    pathological input finite even with the clip deleted -- confirmed
    separately by removing the ``clip_grad_norm_`` call and rerunning the old
    version of this test, which still passed (val losses ~0.53 -> ~0.41, all
    finite). So "stays finite" alone was vacuous: it passed whether or not
    clipping ran.

    This test instruments ``torch.nn.utils.clip_grad_norm_`` directly. That
    function returns the *pre*-clip norm and clips the ``.grad`` tensors in
    place, so a thin spy can record both the pre-clip norm (from the return
    value) and the true post-clip norm (recomputed from the now-modified
    grads) for every batch in the run. The assertion only means something if
    the pre-clip norm genuinely exceeded ``grad_clip`` at least once -- an
    extreme input row (``1e6``) is engineered in to guarantee that -- so we
    check that too, rather than letting the post-clip assertion pass
    trivially because nothing ever needed clipping.

    A plain ``Xtr[0] = 1e6`` is not sufficient on its own: this fixture's
    label rule is ``X[:, 0] + 0.5 * X[:, 1] > 0``, so the extreme row's true
    label agrees with its extreme sign. Once training nudges the model to
    correctly classify ordinary rows along that same feature, it also
    classifies the extreme row correctly -- with a logit so large that
    ``sigmoid(z)`` rounds to exactly ``1.0`` in float32, making the BCE
    gradient ``sigmoid(z) - y`` exactly zero for that row on every later
    epoch (verified directly: after 10 ordinary update steps the row's own
    pre-clip contribution collapsed from ~10⁴ to ~0.3, small enough that the
    "did it ever exceed the threshold" assertion below would fail). The
    row's label is deliberately flipped so the network's learned rule is
    always *wrong* about it, keeping the mismatch -- and the oversized
    gradient -- large on every epoch.
    """
    Xtr, ytr, Xva, yva = separable_data
    Xtr = Xtr.copy()
    ytr = ytr.copy()
    Xtr[0] = 1e6  # extreme input ...
    ytr[0] = 1 - ytr[0]  # ... deliberately mislabelled, so the model can
    # never "solve" it and the gradient conflict persists every epoch

    pre_clip_norms: list[float] = []
    post_clip_norms: list[float] = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def spy_clip(parameters, max_norm, *args, **kwargs):
        params = list(parameters)
        pre_norm = original_clip(params, max_norm, *args, **kwargs)
        pre_clip_norms.append(float(pre_norm))
        post_norm = torch.sqrt(
            sum(p.grad.detach().pow(2).sum() for p in params if p.grad is not None)
        )
        post_clip_norms.append(float(post_norm))
        return pre_norm

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)

    cfg = train.TrainConfig(batch_size=256, max_epochs=3, grad_clip=1.0, seed=0)
    result = train.train_model(models.GateMLP(n_features=4, depth=2), Xtr, ytr, Xva, yva, cfg)

    assert all(np.isfinite(v) for v in result["history"]["val_loss"])
    # The assertion below would be vacuous if no batch ever exceeded the
    # threshold -- confirm at least one genuinely did.
    assert max(pre_clip_norms) > cfg.grad_clip
    assert all(p <= cfg.grad_clip + 1e-4 for p in post_clip_norms)


def test_reinit_default_ignores_caller_weights_reinit_false_preserves_them(separable_data):
    """``config.reinit`` controls whether ``train_model`` discards or keeps
    the caller's starting weights.

    Two runs with the same seed but *different* constant-filled starting
    weights should converge to byte-identical results under the default
    (``reinit=True``): the constant is discarded and replaced by a fresh,
    seed-determined init before training starts, so it cannot influence the
    outcome. With ``reinit=False`` the constant is never touched, so it
    becomes the optimiser's actual starting point and different constants
    must produce different trained weights.
    """
    Xtr, ytr, Xva, yva = separable_data

    def _run(marker: float, reinit: bool) -> dict:
        net = models.GateMLP(n_features=4, depth=2)
        with torch.no_grad():
            for p in net.parameters():
                p.fill_(marker)
        cfg = train.TrainConfig(batch_size=256, max_epochs=1, seed=0, reinit=reinit)
        return train.train_model(net, Xtr, ytr, Xva, yva, cfg)

    default_a = _run(0.1234, reinit=True)
    default_b = _run(-0.9, reinit=True)
    for key in default_a["best_state"]:
        torch.testing.assert_close(default_a["best_state"][key], default_b["best_state"][key])

    warm_a = _run(0.1234, reinit=False)
    warm_b = _run(-0.9, reinit=False)
    differs = any(
        not torch.allclose(warm_a["best_state"][key], warm_b["best_state"][key])
        for key in warm_a["best_state"]
    )
    assert differs
