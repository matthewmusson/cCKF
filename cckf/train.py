"""Shared training loop for the gate and the value function.

Both models use the identical recipe (spec §2.6, §3.7): AdamW, lr 1e-3 cosine
annealed to 1e-5, weight decay 1e-2, gradient-norm clipping at 1.0, early
stopping with patience 5 on validation BCE. Only the data, the sampler and the
target type differ, so one function serves both.

Two details that matter:

*Best state, not last state.* ``train_model`` returns the parameters from the
epoch with the lowest validation loss, not from the final epoch. With patience
5 the last five epochs are by construction worse than the best one.

*Validation loss is always computed on the natural distribution.* Subsampling
(§2.5 B/C) applies to training batches only. If validation were subsampled too,
the val loss would measure performance on a distribution the model will never
see at inference, and early stopping would optimise the wrong thing.

*Reproducibility reinitialises the model, by default.* ``model`` is
constructed by the caller before ``train_model`` ever sees it, so its initial
weights reflect whatever the global torch RNG happened to be at construction
time -- not ``config.seed``. To make "same seed -> same history" hold
regardless of construction order, ``train_model`` reseeds torch (CPU and, if
available, CUDA) and then reruns ``reset_parameters()`` on every submodule
that has one, so the weights actually trained from are a deterministic
function of ``config.seed`` alone. Set ``config.reinit = False`` to warm-start
from the weights the caller already put on ``model`` (e.g. a loaded
checkpoint) instead -- see ``train_model``'s docstring for the reproducibility
trade-off that implies.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .losses import bce_with_logits


@dataclass
class TrainConfig:
    """Optimiser and schedule settings (spec §2.6)."""

    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    max_epochs: int = 50
    patience: int = 5
    batch_size: int = 4096
    sampler: str = "B"
    seed: int = 0
    device: str = "cpu"
    reinit: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def standardize(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Apply stored per-feature standardisation, ``(x - mu) / sigma``.

    ``mu`` and ``sigma`` come from the *training* split and are applied verbatim
    to val/cal/test. Features exempt from standardisation carry ``(0, 1)`` and
    pass through unchanged.
    """
    return ((np.asarray(X, dtype=np.float32) - mu) / sigma).astype(np.float32)


class StandardizedView:
    """A lazy, standardised, column-subset view over a (memmapped) feature matrix.

    Materialising ``standardize(np.asarray(X)[picked], mu, sigma)[:, col_idx]``
    eagerly forces the *entire* selection into RAM at once -- three copies of
    it, transiently -- even though every consumer in this module
    (``train_model``, ``_evaluate``, ``predict_logits``) only ever touches one
    batch at a time via ``X[idx]`` or ``X[start:start+batch_size]``. When
    ``picked`` covers all ~174M rows of the real train cache that is tens of
    gigabytes for no reason: the memmap backing ``X`` already lets the OS page
    in only what's read.

    This class defers both the row gather and the standardisation to
    ``__getitem__``, so indexing it costs only a batch's worth of memory
    rather than the dataset's. It supports exactly the two access patterns
    ``cckf.train`` uses -- integer-array fancy indexing and slicing -- via a
    single code path, because ``numpy`` indexing (``self.row_idx[key]``)
    already dispatches correctly for both.

    Parameters
    ----------
    source : numpy.ndarray or numpy.memmap
        The full, unstandardised feature matrix, shape ``(n_source_rows,
        n_source_features)``. Never copied; only the rows/columns actually
        requested are read from it.
    row_idx : numpy.ndarray
        Indices into ``source`` selecting (and ordering) the rows this view
        exposes as its own rows ``0..len(row_idx)-1``. May be a strict subset
        (e.g. a sampler's subsample) or ``np.arange(len(source))`` to expose
        every row.
    mu, sigma : numpy.ndarray
        Per-*source*-feature standardisation stats, shape
        ``(n_source_features,)`` -- i.e. indexed the same way as ``source``'s
        columns, before ``col_idx`` narrows them. Features exempt from
        standardisation (``cckf.features.NO_STANDARDIZE``) are expected to
        already carry ``(mu, sigma) = (0, 1)`` here, per
        ``cckf.cache.compute_norm_stats``; this class does not special-case
        them.
    col_idx : numpy.ndarray
        Indices into ``source``'s columns to keep, e.g. a feature-group
        ablation subset. Use ``np.arange(source.shape[1])`` to keep all of
        them.

    Notes
    -----
    Standardisation is a per-column elementwise map: ``(x - mu) / sigma`` at
    column ``c`` never depends on any other column. So subsetting columns
    before or after standardising is not just numerically close, it performs
    the exact same float32 subtraction and division for every surviving
    element -- the two orderings are bit-for-bit identical, not merely equal
    within round-off. This class subsets columns first (on ``mu``/``sigma``
    in ``__init__`` and on the gathered block in ``__getitem__``) purely so
    the standardisation arithmetic itself runs on fewer columns; that
    reordering does not change any output value.
    """

    def __init__(
        self,
        source: np.ndarray,
        row_idx: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        col_idx: np.ndarray,
    ) -> None:
        self.source = source
        self.row_idx = np.asarray(row_idx)
        self.col_idx = np.asarray(col_idx)
        # Column-subset mu/sigma once, up front -- not per batch -- since
        # they're tiny (n_features,) and every batch needs the same slice.
        self.mu = np.asarray(mu, dtype=np.float32)[self.col_idx]
        self.sigma = np.asarray(sigma, dtype=np.float32)[self.col_idx]

    def __len__(self) -> int:
        return len(self.row_idx)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.row_idx), len(self.col_idx))

    def __getitem__(self, key: Any) -> np.ndarray:
        # ``key`` is either a slice (from _evaluate/predict_logits batching)
        # or a sorted integer array (from train_model's per-batch fancy
        # index). ``self.row_idx[key]`` maps either kind straight through to
        # the corresponding rows of ``source`` -- only ``len(key)`` rows'
        # worth of indices, never the full dataset.
        rows = self.row_idx[key]
        # This is the only line that touches disk/RAM for the source data:
        # a memmap fancy-index gather of just the requested rows, which
        # numpy must materialise as a real array (it can't stay a view).
        # Peak memory from here on is O(batch_size), not O(n_rows).
        block = np.asarray(self.source[rows], dtype=np.float32)[:, self.col_idx]
        return ((block - self.mu) / self.sigma).astype(np.float32)


def _evaluate(
    model: torch.nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: str
) -> float:
    """Mean BCE over a dataset, in eval mode without gradients."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.asarray(X[start : start + batch_size])).to(device)
            yb = torch.from_numpy(
                np.asarray(y[start : start + batch_size], dtype=np.float32)
            ).to(device)
            loss = bce_with_logits(model(xb), yb)
            total += float(loss) * len(xb)
            n += len(xb)
    return total / max(n, 1)


def train_model(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
    wandb_run: Any | None = None,
) -> dict:
    """Train one model to early stopping.

    Parameters
    ----------
    model : torch.nn.Module
        Emits raw logits of shape ``(n,)``.
    X_train, y_train : numpy.ndarray
        Standardised features and targets. Targets may be hard ``{0,1}``
        (gate) or soft ``[0,1]`` (value function).
    X_val, y_val : numpy.ndarray
        Validation set on the natural class distribution.
    config : TrainConfig
        ``config.reinit`` (default ``True``) reruns ``reset_parameters()`` on
        every submodule of ``model`` that has one, immediately after seeding
        torch with ``config.seed``, before any training happens. This makes
        the weights actually trained from a deterministic function of
        ``config.seed`` alone, regardless of how or when the caller built
        ``model`` -- so two calls with the same config and freshly
        constructed models are guaranteed byte-identical, not just "usually
        similar". Set ``config.reinit = False`` to warm-start from the
        model's existing weights instead (e.g. resuming from a checkpoint in
        an iterative retraining loop). Doing so gives up the
        same-seed-same-history guarantee: results then also depend on
        whatever weights ``model`` carried in, which ``train_model`` cannot
        see from ``config.seed``.
    wandb_run : optional
        If given, ``.log()`` is called once per epoch.

    Returns
    -------
    dict
        ``best_state`` (state dict at the best epoch), ``best_val_loss``,
        ``stopped_epoch``, ``history`` with per-epoch ``train_loss``,
        ``val_loss`` and ``lr``.
    """
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed)
    device = config.device

    if config.reinit:
        # Reinitialise every submodule that knows how (Linear, GRU,
        # LayerNorm, ...) from the just-set seed. ``model`` arrives already
        # constructed, so its weights otherwise reflect whatever the global
        # torch RNG happened to be at construction time -- not
        # ``config.seed``. Two calls with the same config but freshly
        # constructed models would then start from different weights and
        # diverge, breaking the reproducibility guarantee. Skipped entirely
        # when ``config.reinit`` is False, so callers can warm-start.
        def _reset(module: torch.nn.Module) -> None:
            reset_fn = getattr(module, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()

        model.apply(_reset)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    n_train = len(X_train)
    steps_per_epoch = max(1, math.ceil(n_train / config.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs * steps_per_epoch, eta_min=config.lr_min
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    stopped_epoch = config.max_epochs

    for epoch in range(config.max_epochs):
        model.train()
        order = rng.permutation(n_train)
        epoch_loss, seen = 0.0, 0
        # Record the LR actually used for this epoch's updates, i.e. its
        # value *before* any of this epoch's per-batch scheduler steps --
        # not the already-decayed value left over after them.
        current_lr = optimizer.param_groups[0]["lr"]

        for start in range(0, n_train, config.batch_size):
            idx = np.sort(order[start : start + config.batch_size])
            xb = torch.from_numpy(np.asarray(X_train[idx])).to(device)
            yb = torch.from_numpy(np.asarray(y_train[idx], dtype=np.float32)).to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = bce_with_logits(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * len(idx)
            seen += len(idx)

        train_loss = epoch_loss / max(seen, 1)
        val_loss = _evaluate(model, X_val, y_val, config.batch_size, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": current_lr,
                }
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                stopped_epoch = epoch + 1
                break

    model.load_state_dict(best_state)
    return {
        "best_state": best_state,
        "best_val_loss": best_val,
        "stopped_epoch": stopped_epoch,
        "history": history,
    }


def predict_logits(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 65_536,
    device: str = "cpu",
) -> np.ndarray:
    """Batched raw-logit inference over a (possibly memmapped) feature matrix."""
    model = model.to(device).eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.asarray(X[start : start + batch_size])).to(device)
            out[start : start + len(xb)] = model(xb).cpu().numpy()
    return out
