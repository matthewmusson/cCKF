"""Train the value function V_φ on soft V^{π†} targets.

Usage
-----
    python scripts/train_value.py \\
        --train-cache /data/cache/value/train \\
        --val-cache /data/cache/value/val \\
        --out-dir /data/models/value_v0 --wandb-project cckf-value

Cache layouts
-------------
``--train-cache``/``--val-cache`` accept either a flat cache directory
(``X.f32``/``y.f32``/``aux.f32``/``meta.json``, optionally ``norm_stats.npz``
-- today's un-windowed Tier-2 layout) or a parent directory holding one
``nsig*/`` subdirectory per rollout window (window-conditioned Tier-3 value
plan, Task 6's windowed layout, e.g. ``.../train/nsig3/``,
``.../train/nsig5/``, ``.../train/nsig10/``). In the second case the subdirs'
``X``/``y``/``aux`` are concatenated into one training set -- the same state
appears once per window with a different 12th feature (``window_nsigma``) and
a different soft target, which is the intended design (see
:data:`cckf.features.VALUE_FEATURES_WINDOWED`). The model's input width and
feature order come from the cache's own ``meta.json`` (``n_features``,
``feature_names``), not a hardcoded constant, so this script trains an
11-feature (Tier 2) or 12-feature (windowed Tier 3) value function
identically. The training recipe itself (loss, optimiser, epochs, early
stopping, splits) is unchanged and untouched by any of this.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from cckf import features as feat
from cckf import models, train
from cckf.cache import compute_norm_stats

#: spec §3.8 expectations
RED_FLAGS = {"train_bce": 0.20, "val_bce": 0.25, "auc_roc_min": 0.90}


def _load(cache_dir: str) -> dict:
    """Open a value cache, memmapping the two matrices rather than reading them.

    ``X`` and ``aux`` are memmapped, not loaded: the training loop and
    ``predict_logits`` only ever touch one batch at a time, so paging rows in
    on demand costs a batch of RAM instead of the whole split. ``y`` is a
    single float32 column and is read eagerly -- the samplers and the marginal
    oversampler both need to scan all of it to build row indices anyway.
    """
    d = Path(cache_dir)
    meta = json.loads((d / "meta.json").read_text())
    n, f = meta["n_rows"], meta["n_features"]
    return {
        "X": np.memmap(d / "X.f32", dtype=np.float32, mode="r", shape=(n, f)),
        "y": np.fromfile(d / "y.f32", dtype=np.float32),
        "aux": np.memmap(d / "aux.f32", dtype=np.float32, mode="r", shape=(n, 3)),
        "meta": meta,
    }


def discover_cache_dirs(cache_dir: str) -> list[Path]:
    """Resolve one ``--*-cache`` argument to a list of leaf cache directories.

    A leaf cache directory is one with its own ``meta.json`` directly inside
    it. ``cache_dir`` is either such a directory itself (today's flat
    layout), or a parent directory holding one or more ``nsig*/`` leaf
    directories (Task 6's windowed layout). The returned list is sorted for
    deterministic concatenation order.

    Raises
    ------
    FileNotFoundError
        If ``cache_dir`` is neither -- no ``meta.json`` in it, and no
        ``nsig*/meta.json`` subdirectory either.
    """
    d = Path(cache_dir)
    if (d / "meta.json").exists():
        return [d]
    subdirs = sorted(p for p in d.glob("nsig*") if (p / "meta.json").exists())
    if not subdirs:
        raise FileNotFoundError(
            f"{d} is neither a flat value cache (no meta.json found) nor a "
            "windowed parent directory (no nsig*/meta.json subdirs found)"
        )
    return subdirs


def load_value_cache(cache_dir: str) -> dict:
    """Load one ``--*-cache`` argument, concatenating windowed subdirs.

    Delegates to :func:`discover_cache_dirs` to find the leaf directories,
    then :func:`_load`\\ s each. A single leaf directory is returned as-is
    (``X``/``aux`` stay memmapped, as before). Multiple leaf directories
    (a windowed parent) are concatenated into real in-memory arrays -- the
    training set genuinely is their union, one row per ``(state, window)``
    pair, so there is no way to preserve memmapping across the join.

    Returns
    -------
    dict
        ``X``, ``y``, ``aux``, ``meta`` (with ``n_rows``, ``n_features``,
        ``feature_names`` reflecting the concatenation), and ``cache_dirs``
        (the sorted list of leaf directories this was built from -- used by
        :func:`get_norm_stats` to decide whether a single ``norm_stats.npz``
        applies).

    Raises
    ------
    ValueError
        If the leaf directories disagree on ``n_features`` or
        ``feature_names`` -- concatenating columns that do not mean the same
        thing would silently corrupt training.
    """
    dirs = discover_cache_dirs(cache_dir)
    parts = [_load(str(d)) for d in dirs]

    n_features = parts[0]["meta"]["n_features"]
    feature_names = parts[0]["meta"]["feature_names"]
    for d, p in zip(dirs[1:], parts[1:]):
        if p["meta"]["n_features"] != n_features:
            raise ValueError(
                f"cache subdir n_features mismatch: {dirs[0]} has "
                f"{n_features}, {d} has {p['meta']['n_features']}"
            )
        if p["meta"]["feature_names"] != feature_names:
            raise ValueError(
                f"cache subdir feature_names mismatch: {dirs[0]} has "
                f"{feature_names}, {d} has {p['meta']['feature_names']}"
            )

    if len(parts) == 1:
        return {**parts[0], "cache_dirs": dirs}

    X = np.concatenate([np.asarray(p["X"]) for p in parts], axis=0)
    y = np.concatenate([p["y"] for p in parts], axis=0)
    aux = np.concatenate([np.asarray(p["aux"]) for p in parts], axis=0)
    meta = {
        "n_rows": int(len(y)),
        "n_features": n_features,
        "feature_names": feature_names,
        "concatenated_from": [str(d) for d in dirs],
    }
    return {"X": X, "y": y, "aux": aux, "meta": meta, "cache_dirs": dirs}


def get_norm_stats(tr: dict) -> tuple[np.ndarray, np.ndarray]:
    """Standardisation stats for a cache loaded by :func:`load_value_cache`.

    A single leaf cache directory carries its own ``norm_stats.npz``, fit
    once at cache-build time on exactly that data -- used verbatim. A
    concatenated windowed cache (multiple ``nsig*/`` subdirs) has no single
    ``norm_stats.npz`` describing the concatenation: each subdir's own file
    was fit on that subdir's window slice alone. Rather than picking one
    subdir's stats (which would silently privilege whichever window happens
    to sort first) or combining several ``(mu, sigma)`` pairs after the fact,
    this recomputes mu/sigma directly over the concatenated ``X`` -- using
    :func:`cckf.cache.compute_norm_stats` with the same
    :data:`cckf.features.NO_STANDARDIZE` skip-list the cache builder itself
    uses, so the windowed run is standardised against the population it
    actually trains on.
    """
    if len(tr["cache_dirs"]) == 1:
        stats = np.load(Path(tr["cache_dirs"][0]) / "norm_stats.npz", allow_pickle=True)
        return stats["mu"], stats["sigma"]
    return compute_norm_stats(tr["X"], feat.NO_STANDARDIZE, tr["meta"]["feature_names"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--wandb-project", default="")
    parser.add_argument(
        "--oversample-marginal",
        type=float,
        default=0.0,
        help="spec §3.5: extra copies of states with V in [0.2, 0.8]",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, va = load_value_cache(args.train_cache), load_value_cache(args.val_cache)
    n_features = tr["meta"]["n_features"]
    feature_names = tr["meta"]["feature_names"]
    if (
        va["meta"]["n_features"] != n_features
        or va["meta"]["feature_names"] != feature_names
    ):
        raise ValueError(
            "train/val cache feature schema mismatch: train has "
            f"n_features={n_features}, feature_names={feature_names}; val has "
            f"n_features={va['meta']['n_features']}, "
            f"feature_names={va['meta']['feature_names']}"
        )
    mu, sigma = get_norm_stats(tr)

    # Labels are bimodal: most branches clearly succeed or clearly fail. The
    # marginal band is where the decision actually matters, so it can be
    # oversampled to keep it from being drowned out. Oversampling is expressed
    # as repeated *row indices* rather than a gathered copy of the matrix --
    # StandardizedView resolves them per batch, so duplicating a row costs one
    # int64 rather than one feature vector.
    train_rows = np.arange(len(tr["y"]))
    if args.oversample_marginal > 0:
        marginal = np.flatnonzero((tr["y"] >= 0.2) & (tr["y"] <= 0.8))
        extra = np.repeat(marginal, int(args.oversample_marginal))
        train_rows = np.concatenate([train_rows, extra])
        print(
            f"oversampled {len(marginal):,} marginal states → {len(train_rows):,} rows"
        )
    y_train = tr["y"][train_rows]

    all_cols = np.arange(tr["X"].shape[1])
    X_train = train.StandardizedView(tr["X"], train_rows, mu, sigma, all_cols)
    X_val = train.StandardizedView(
        va["X"], np.arange(len(va["y"])), mu, sigma, all_cols
    )
    y_val = va["y"]

    config = train.TrainConfig(
        batch_size=4096, max_epochs=args.max_epochs, seed=args.seed, device=args.device
    )

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"value_seed{args.seed}",
            config={**vars(args), "n_train_rows": len(y_train)},
        )

    model = models.ValueMLP(n_features=n_features, width=args.width, depth=args.depth)
    print(f"parameters: {models.count_parameters(model):,}")
    result = train.train_model(model, X_train, y_train, X_val, y_val, config, wandb_run)

    val_logits = train.predict_logits(model, X_val, device=args.device)
    val_pred = 1.0 / (1.0 + np.exp(-val_logits))
    # AUC needs a binary reference; 0.5 is the DM-match threshold by definition
    # (min(completeness, purity) < 0.5 cannot double-majority match).
    y_binary = (y_val >= 0.5).astype(np.uint8)
    mse = float(np.mean((val_pred - y_val) ** 2))

    metrics_out = {
        "train_bce": result["history"]["train_loss"][-1],
        "val_bce": result["best_val_loss"],
        "val_mse": mse,
        "auc_roc": (
            float(roc_auc_score(y_binary, val_logits))
            if y_binary.min() != y_binary.max()
            else None
        ),
        "mean_target": float(y_val.mean()),
        "marginal_fraction": float(np.mean((y_val >= 0.3) & (y_val <= 0.7))),
        # Tier-gap diagnostic (see value_target.py on why Tier 3 is absent).
        "tier1_minus_tier2_mean": float(np.mean(va["aux"][:, 0] - y_val)),
        "stopped_epoch": result["stopped_epoch"],
        "history": result["history"],
    }
    warnings = []
    if metrics_out["val_bce"] > RED_FLAGS["val_bce"]:
        warnings.append(
            f"val BCE {metrics_out['val_bce']:.4f} > {RED_FLAGS['val_bce']}"
        )
    if (
        metrics_out["auc_roc"] is not None
        and metrics_out["auc_roc"] < RED_FLAGS["auc_roc_min"]
    ):
        warnings.append(
            f"AUC-ROC {metrics_out['auc_roc']:.4f} < {RED_FLAGS['auc_roc_min']}"
        )
    metrics_out["red_flags"] = warnings

    torch.save(
        {
            "state_dict": result["best_state"],
            "n_features": n_features,
            "width": args.width,
            "depth": args.depth,
            "feature_names": list(feature_names),
            "mu": mu,
            "sigma": sigma,
        },
        out_dir / "value_model.pt",
    )
    (out_dir / "value_metrics.json").write_text(json.dumps(metrics_out, indent=2))
    np.savez(
        out_dir / "value_val_predictions.npz",
        pred=val_pred,
        target=y_val,
        aux=np.asarray(va["aux"]),
    )

    print(
        f"val BCE {metrics_out['val_bce']:.4f}  MSE {mse:.4f}  "
        f"tier gap {metrics_out['tier1_minus_tier2_mean']:.4f}"
    )
    for w in warnings:
        print(f"RED FLAG: {w}")
    if wandb_run is not None:
        wandb_run.summary.update(
            {k: v for k, v in metrics_out.items() if k != "history"}
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
