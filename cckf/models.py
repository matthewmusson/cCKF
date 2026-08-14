"""Network definitions for the gate g_ψ and value function V_φ.

Both are plain MLPs that emit a single **raw logit** — no sigmoid in the head.
This matters: post-hoc Platt calibration (§4.1) fits ``σ(a·z + b)`` and needs
the pre-sigmoid z. A model that returned probabilities would force a logit
round-trip and lose precision in the saturated tails, which is exactly where
the 1:193 imbalance puts most of the mass.

Sizes follow spec §2.1 and §3.1: gate 26→128→128→128→1 (36,609 parameters),
value 11→128→128→1 (18,177). ``depth`` is exposed so the 2-vs-3 hidden-layer
ablation is a constructor argument rather than a code edit.

SiLU is used throughout. Unlike ReLU it is smooth, which keeps second-order
information available to AdamW near the decision boundary — and the boundary is
where nearly all of this problem's signal lives, since the easy negatives are
already far from it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _LogitMLP(nn.Sequential):
    """MLP stack ending in a scalar logit, squeezed to shape ``(n,)``."""

    def __init__(self, n_features: int, width: int, depth: int) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        super().__init__(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(n,)``."""
        return super().forward(x).squeeze(-1)


class GateMLP(_LogitMLP):
    """Gate g_ψ: P(candidate is from the branch's majority particle).

    Parameters
    ----------
    n_features : int
        Input dimension. 26 for the full feature set; smaller for A4a/A4b
        feature-group ablations.
    width : int
        Hidden width.
    depth : int
        Number of hidden layers. 3 is the primary; 2 is the ablation.
    """

    def __init__(self, n_features: int = 26, width: int = 128, depth: int = 3) -> None:
        super().__init__(n_features, width, depth)


class ValueMLP(_LogitMLP):
    """Value function V_φ: P(branch completes to a DM-matched track)."""

    def __init__(self, n_features: int = 11, width: int = 128, depth: int = 2) -> None:
        super().__init__(n_features, width, depth)


class ValueGRU(nn.Module):
    """Recurrent value function, for ablation A4c (MLP vs GRU).

    The MLP sees only hand-rolled summaries of branch history (``n_hits``,
    ``n_holes``, ``n_seq_holes``, and the sum and min of the gate log-odds).
    A GRU consumes the per-step sequence instead, so it can in principle learn
    that *where* the holes fell matters — three holes at the start of a branch
    is a different situation from three at the end. A4c asks whether that extra
    capacity buys any tracking efficiency.
    """

    def __init__(self, n_features: int = 11, hidden: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return one raw logit per sequence.

        Parameters
        ----------
        x : torch.Tensor
            Padded per-step features, shape ``(batch, max_steps, n_features)``.
        lengths : torch.Tensor
            True step count per sequence, shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Raw logits, shape ``(batch,)``. Padding is excluded via a packed
            sequence, so trailing garbage cannot affect the result.
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        return self.head(h_n[-1]).squeeze(-1)
