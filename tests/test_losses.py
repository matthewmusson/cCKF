"""Tests for the BCE losses."""

from __future__ import annotations

import math

import pytest
import torch

from cckf import losses


def test_bce_matches_closed_form_hard_target():
    logits = torch.tensor([0.0, 2.0])
    targets = torch.tensor([1.0, 0.0])
    expected = (math.log(2.0) + (2.0 + math.log(1 + math.exp(-2.0)))) / 2.0
    assert losses.bce_with_logits(logits, targets).item() == pytest.approx(
        expected, rel=1e-6
    )


def test_bce_is_minimised_when_sigmoid_equals_soft_target():
    """With soft target v, the loss is minimised at sigma(z) = v (spec §3.4)."""
    v = 0.3
    targets = torch.tensor([v])
    z_star = math.log(v / (1 - v))
    best = losses.bce_with_logits(torch.tensor([z_star]), targets).item()
    for delta in (-0.5, -0.1, 0.1, 0.5):
        worse = losses.bce_with_logits(torch.tensor([z_star + delta]), targets).item()
        assert worse > best


def test_bce_accepts_soft_targets_in_the_open_interval():
    out = losses.bce_with_logits(torch.zeros(4), torch.tensor([0.0, 0.25, 0.75, 1.0]))
    assert torch.isfinite(out)


def test_bce_applies_no_positive_reweighting():
    """An asymmetric batch pinned to its closed-form unweighted value.

    Spec §2.4: imbalance is handled by the sampler, never by the loss. A
    pos_weight w would shift the learned logit by log(w), so sigma(z) would no
    longer estimate P(y=1|x) and the downstream Platt fit -- which assumes an
    affine correction to a *proper* logit -- would be correcting the wrong
    quantity.

    Two positives and one negative, all at z = 0 so sigma(z) = 0.5 and every
    sample contributes exactly -log(0.5) = log 2 when unweighted. Under a
    pos_weight w the mean becomes (2w + 1)/3 * log 2, which differs from log 2
    for every w != 1. The batch is deliberately NOT permutation-symmetric: a
    symmetric batch lets w cancel, which is why the previous form of this test
    passed even at w = 193.
    """
    loss = losses.bce_with_logits(torch.zeros(3), torch.tensor([1.0, 1.0, 0.0])).item()
    assert loss == pytest.approx(math.log(2.0), rel=1e-6)


def test_bce_is_symmetric_under_simultaneous_logit_and_label_negation():
    """A balanced batch and its label-flipped mirror must give the same loss.

    This symmetry holds for any pos_weight (w cancels in the algebra), so it
    does NOT guard against reweighting being introduced -- that job belongs to
    test_bce_applies_no_positive_reweighting above, which uses an asymmetric
    batch. This test only pins the (weaker, always-true) symmetry property.
    """
    logits = torch.tensor([1.0, -1.0])
    a = losses.bce_with_logits(logits, torch.tensor([1.0, 0.0]))
    b = losses.bce_with_logits(-logits, torch.tensor([0.0, 1.0]))
    assert a.item() == pytest.approx(b.item(), rel=1e-9)


def test_bce_is_numerically_stable_at_extreme_logits():
    out = losses.bce_with_logits(
        torch.tensor([-100.0, 100.0]), torch.tensor([0.0, 1.0])
    )
    assert torch.isfinite(out) and out.item() < 1e-3


def test_bce_gradient_flows():
    logits = torch.tensor([0.5], requires_grad=True)
    losses.bce_with_logits(logits, torch.tensor([1.0])).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_bce_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        losses.bce_with_logits(torch.zeros(3), torch.zeros(4))


def test_bce_rejects_out_of_range_targets():
    with pytest.raises(ValueError):
        losses.bce_with_logits(torch.zeros(2), torch.tensor([-0.1, 0.5]))
