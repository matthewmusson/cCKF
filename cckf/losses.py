"""Binary cross-entropy for the gate and the value function.

One function serves both models because both are BCE — they differ only in
whether the target is hard or soft.

Gate (spec §2.4)
    Target y ∈ {0, 1} is ``label_same_particle``. The loss is::

        L = −(1/N) Σ [ y log σ(z) + (1 − y) log(1 − σ(z)) ]

    **Unweighted.** No ``pos_weight``, despite the 1:193 imbalance. A positive
    reweighting by w₊ shifts the learned logit by exactly log w₊, so σ(z) no
    longer estimates P(y=1 | x) but a tilted quantity. That breaks the
    downstream Platt fit (which assumes an affine correction to a *proper*
    logit) and misdirects network capacity toward reproducing a constant offset.
    Imbalance is handled by the sampler instead (§2.5), where the induced logit
    shift is a known constant that Platt's intercept b absorbs exactly.

Value function (spec §3.4)
    Target v = V^{π†} ∈ [0, 1] is soft. The same cross-entropy is used::

        L = −(1/N) Σ [ v log σ(z) + (1 − v) log(1 − σ(z)) ]

    This is the cross-entropy between the Bernoulli(v) and Bernoulli(σ(z))
    distributions. Its unique minimum over z is at σ(z) = v, so minimising it
    makes the network output the value directly rather than a thresholded
    version of it — which is why a soft target is used instead of
    ``1[V^{π†} ≥ 0.5]``. A branch at V = 0.45 is nearly matchable and a branch
    at V = 0.05 is hopeless; hard labels would throw that distinction away.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean binary cross-entropy from raw logits, unweighted.

    Parameters
    ----------
    logits : torch.Tensor
        Raw network output z, shape ``(n,)``. No sigmoid applied yet.
    targets : torch.Tensor
        Hard labels in ``{0, 1}`` (gate) or soft values in ``[0, 1]``
        (value function), shape ``(n,)``.

    Returns
    -------
    torch.Tensor
        Scalar mean loss.

    Raises
    ------
    ValueError
        If shapes disagree or a target lies outside ``[0, 1]``.

    Notes
    -----
    Delegates to :func:`torch.nn.functional.binary_cross_entropy_with_logits`,
    which uses the log-sum-exp stable form and so stays finite for |z| ~ 100.
    Computing ``sigmoid`` then ``log`` separately would underflow there.
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != targets shape {tuple(targets.shape)}"
        )
    if torch.any(targets < 0.0) or torch.any(targets > 1.0):
        raise ValueError("targets must lie in [0, 1]")

    return F.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction="mean"
    )
