"""Tests for the gate and value networks."""

from __future__ import annotations

import pytest
import torch

from cckf import models


def test_gate_mlp_output_is_one_logit_per_row():
    net = models.GateMLP(n_features=26)
    out = net(torch.randn(7, 26))
    assert out.shape == (7,)


def test_gate_mlp_emits_raw_logits_not_probabilities():
    """Outputs must be unbounded; a sigmoid in the head would break Platt."""
    torch.manual_seed(0)
    net = models.GateMLP(n_features=26)
    with torch.no_grad():
        net[-1].bias.fill_(50.0)
        out = net(torch.randn(4, 26))
    assert out.max().item() > 1.0


def test_gate_mlp_parameter_count_matches_the_specified_architecture():
    # 26->128: 26*128+128 = 3456
    # 128->128: 128*128+128 = 16512  (x2)
    # 128->1:   128*1+1    = 129
    assert (
        models.count_parameters(models.GateMLP(n_features=26, width=128, depth=3))
        == 36_609
    )


def test_gate_mlp_depth_is_configurable_for_ablation():
    d2 = models.count_parameters(models.GateMLP(n_features=26, depth=2))
    d3 = models.count_parameters(models.GateMLP(n_features=26, depth=3))
    assert d3 > d2


def test_value_mlp_parameter_count_matches_the_specified_architecture():
    # 11->128: 11*128+128 = 1536
    # 128->128: 16512
    # 128->1:   129
    assert models.count_parameters(models.ValueMLP(n_features=11)) == 18_177


def test_value_mlp_output_shape_and_param_count():
    net = models.ValueMLP(n_features=11)
    assert net(torch.randn(5, 11)).shape == (5,)


def test_models_use_silu_activation():
    net = models.GateMLP(n_features=26)
    assert any(isinstance(m, torch.nn.SiLU) for m in net.modules())


def test_gradients_flow_through_gate():
    net = models.GateMLP(n_features=26)
    net(torch.randn(3, 26)).sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()
    )


def test_value_gru_handles_variable_length_branches():
    net = models.ValueGRU(n_features=11, hidden=64)
    x = torch.randn(3, 5, 11)  # batch 3, max 5 steps
    lengths = torch.tensor([5, 3, 1])
    out = net(x, lengths)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_value_gru_ignores_padding():
    """Padding beyond `lengths` must not change the output."""
    torch.manual_seed(0)
    net = models.ValueGRU(n_features=11, hidden=64).eval()
    x = torch.randn(1, 4, 11)
    lengths = torch.tensor([2])
    a = net(x, lengths)
    x2 = x.clone()
    x2[0, 2:] = 999.0
    b = net(x2, lengths)
    assert torch.allclose(a, b, atol=1e-5)
