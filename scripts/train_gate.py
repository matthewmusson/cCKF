"""Train the gate g_ψ.

Usage
-----
    python scripts/train_gate.py \\
        --train-cache /data/cache/gate/train \\
        --val-cache /data/cache/gate/val \\
        --out-dir /data/models/gate_A \\
        --sampler A --wandb-project cckf-gate

Sampler choices (spec §2.5): A = large batch, no subsampling; B = uniform
negative subsampling to 1:5; C = 1/χ²-weighted hard-negative subsampling to 1:5.
Batch size defaults to the spec value for the chosen sampler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from cckf import cache, features, models, samplers, splits, train

#: spec §2.7 red-flag thresholds
RED_FLAGS = {
    "train_bce": 0.15,
    "val_bce": 0.15,
    "auc_roc_min": 0.95,
    "auc_pr_min": 0.80,
}


def _refuse_partial_cache(
    cache_dir: str, full_events: Sequence[int], allow_partial: bool
) -> None:
    """Refuse to silently train on a staged/partial-split cache.

    ``modal_train.py::build_gate_cache_staged`` builds a gate cache from a
    small event subset (e.g. 2 of 24 train events) for smoke-testing the
    training wiring, and ``scripts/build_gate_cache.py`` marks the result
    ``partial_split: true`` with ``events_used`` in ``meta.json``. Nothing
    else reads that flag: a staged cache looks byte-for-byte like a real one
    to every other consumer. Left unchecked, a staged run that happens to
    follow a full build would silently degrade training data from 24 events
    to 2 while ``train_gate.py`` runs to completion without complaint -- a
    silent-quality failure, not a crash. This check makes that impossible by
    default; ``--allow-partial-cache`` is the deliberate opt-in for the
    smoke-test case the staged cache exists for.

    Reads ``meta.json`` directly (not via :func:`cckf.cache.load_cache`) so
    the check can run, and be tested, without the cache's ``X.f32``/``y.u8``
    arrays existing at all.

    Parameters
    ----------
    cache_dir : str
        Cache directory to check.
    full_events : Sequence[int]
        The full event tuple for the split this cache is expected to cover
        (e.g. ``splits.TRAIN_EVENTS``), used only to report "N of M events"
        in the error message.
    allow_partial : bool
        If True, skip the check entirely (``--allow-partial-cache``).

    Raises
    ------
    SystemExit
        If ``meta.json`` has ``partial_split: true`` and ``allow_partial``
        is False.
    """
    if allow_partial:
        return
    meta = json.loads((Path(cache_dir) / "meta.json").read_text())
    if not meta.get("partial_split"):
        return
    events_used = meta.get("events_used", [])
    raise SystemExit(
        f"refusing to train on {cache_dir}: it is a PARTIAL cache "
        f"(meta.json has partial_split=true), built from "
        f"{len(events_used)}/{len(full_events)} events {events_used}. "
        "This is almost certainly a staged smoke-test cache "
        "(modal_train.py::build_gate_cache_staged), not a real split. "
        "Rebuild the cache without --only-events for the full split, or "
        "pass --allow-partial-cache to use it anyway (e.g. to smoke-test "
        "the training wiring)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sampler", choices=["A", "B", "C"], default="B")
    parser.add_argument("--neg-per-pos", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--drop-ambiguous",
        action="store_true",
        help="A8b ablation: exclude merged-cluster rows",
    )
    parser.add_argument(
        "--feature-groups",
        default="",
        help="A4a/A4b ablation: comma-separated subset of "
        "kalman,cluster_raw,cluster_norm,occupancy,context,sensor,history",
    )
    parser.add_argument(
        "--allow-partial-cache",
        action="store_true",
        help="Allow training on a partial/staged cache (meta.json "
        "partial_split=true), e.g. one built by "
        "modal_train.py::build_gate_cache_staged. Off by default so a "
        "smoke-test cache can never be silently used for a real run.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _refuse_partial_cache(
        args.train_cache, splits.TRAIN_EVENTS, args.allow_partial_cache
    )
    _refuse_partial_cache(args.val_cache, splits.VAL_EVENTS, args.allow_partial_cache)

    tr = cache.load_cache(args.train_cache)
    va = cache.load_cache(args.val_cache)

    stats = np.load(Path(args.train_cache) / "norm_stats.npz", allow_pickle=True)
    mu, sigma = stats["mu"], stats["sigma"]

    # Feature-group subsetting for A4a/A4b. Column order within the kept groups
    # follows GATE_FEATURES so the saved model's inputs stay unambiguous.
    if args.feature_groups:
        keep_groups = [g.strip() for g in args.feature_groups.split(",")]
        unknown = set(keep_groups) - set(features.GATE_GROUPS)
        if unknown:
            raise SystemExit(f"unknown feature groups: {sorted(unknown)}")
        # Canonicalise to GATE_FEATURES order. Iterating groups in
        # command-line order would make the column layout depend on how the
        # user typed the flag, so the same ablation arm could train two
        # different, non-comparable models from the same seed.
        selected = {f for g in keep_groups for f in features.GATE_GROUPS[g]}
        col_idx = np.array(
            [i for i, n in enumerate(features.GATE_FEATURES) if n in selected]
        )
        keep_names = [features.GATE_FEATURES[i] for i in col_idx]
    else:
        keep_names = list(features.GATE_FEATURES)
        col_idx = np.arange(len(features.GATE_FEATURES))

    y_train_full = np.asarray(tr["y"])
    row_idx = np.arange(len(y_train_full))
    if args.drop_ambiguous:
        row_idx = row_idx[np.asarray(tr["ambiguous"])[row_idx] == 0]

    rng = np.random.default_rng(args.seed)
    if args.sampler == "A":
        picked = row_idx
    elif args.sampler == "B":
        sub = samplers.uniform_subsample(y_train_full[row_idx], args.neg_per_pos, rng)
        picked = row_idx[sub]
    else:
        chi2 = np.asarray(tr["aux"])[row_idx, 0]
        sub = samplers.hard_negative_subsample(
            y_train_full[row_idx], chi2, args.neg_per_pos, rng
        )
        picked = row_idx[sub]

    # Lazy views over the memmapped caches: row-gather, standardisation and
    # column-subsetting all happen per batch inside train.train_model, not
    # here. The eager equivalent -- materialising standardize(X[picked],
    # mu, sigma)[:, col_idx] up front -- costs three transient full-size
    # copies; for sampler A, where `picked` covers all ~174M train rows at
    # 26 float32 features (~18 GB memmapped), that is 36-54 GB peak RAM to
    # produce a matrix train_model only ever reads one batch of at a time.
    # See cckf.train.StandardizedView.
    X_train = train.StandardizedView(tr["X"], picked, mu, sigma, col_idx)
    y_train = y_train_full[picked].astype(np.float32)
    X_val = train.StandardizedView(va["X"], np.arange(len(va["y"])), mu, sigma, col_idx)
    y_val = np.asarray(va["y"]).astype(np.float32)

    n_pos_orig = int(y_train_full[row_idx].sum())
    n_neg_orig = int(len(row_idx) - n_pos_orig)
    n_pos_new = int(y_train.sum())
    n_neg_new = int(len(y_train) - n_pos_new)
    shift = (
        0.0
        if args.sampler == "A"
        else samplers.prior_logit_shift(n_pos_orig, n_neg_orig, n_pos_new, n_neg_new)
    )
    print(
        f"sampler={args.sampler} train_rows={len(y_train):,} "
        f"pos={n_pos_new:,} neg={n_neg_new:,} prior_logit_shift={shift:+.4f}"
    )

    config = train.TrainConfig(
        batch_size=samplers.SAMPLER_BATCH_SIZES[args.sampler],
        max_epochs=args.max_epochs,
        sampler=args.sampler,
        seed=args.seed,
        device=args.device,
    )

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"gate_{args.sampler}_seed{args.seed}",
            config={
                **vars(args),
                "n_features": len(keep_names),
                "prior_logit_shift": shift,
                "n_train_rows": len(y_train),
            },
        )

    model = models.GateMLP(
        n_features=len(keep_names), width=args.width, depth=args.depth
    )
    print(f"parameters: {models.count_parameters(model):,}")
    result = train.train_model(model, X_train, y_train, X_val, y_val, config, wandb_run)

    val_logits = train.predict_logits(model, X_val, device=args.device)
    metrics_out = {
        "sampler": args.sampler,
        "n_features": len(keep_names),
        "feature_names": keep_names,
        "prior_logit_shift": shift,
        "train_bce": result["history"]["train_loss"][-1],
        "val_bce": result["best_val_loss"],
        "auc_roc": float(roc_auc_score(y_val, val_logits)),
        "auc_pr": float(average_precision_score(y_val, val_logits)),
        "stopped_epoch": result["stopped_epoch"],
        "n_train_rows": int(len(y_train)),
        "n_val_rows": int(len(y_val)),
        "history": result["history"],
    }

    warnings = []
    if metrics_out["train_bce"] > RED_FLAGS["train_bce"]:
        warnings.append(
            f"train BCE {metrics_out['train_bce']:.4f} > {RED_FLAGS['train_bce']}"
        )
    if metrics_out["val_bce"] > RED_FLAGS["val_bce"]:
        warnings.append(
            f"val BCE {metrics_out['val_bce']:.4f} > {RED_FLAGS['val_bce']}"
        )
    if metrics_out["auc_roc"] < RED_FLAGS["auc_roc_min"]:
        warnings.append(
            f"AUC-ROC {metrics_out['auc_roc']:.4f} < {RED_FLAGS['auc_roc_min']}"
        )
    if metrics_out["auc_pr"] < RED_FLAGS["auc_pr_min"]:
        warnings.append(
            f"AUC-PR {metrics_out['auc_pr']:.4f} < {RED_FLAGS['auc_pr_min']}"
        )
    metrics_out["red_flags"] = warnings

    torch.save(
        {
            "state_dict": result["best_state"],
            "n_features": len(keep_names),
            "width": args.width,
            "depth": args.depth,
            "feature_names": keep_names,
            "mu": mu[col_idx],
            "sigma": sigma[col_idx],
            "prior_logit_shift": shift,
        },
        out_dir / "gate_model.pt",
    )
    (out_dir / "gate_metrics.json").write_text(json.dumps(metrics_out, indent=2))

    print(
        f"val BCE {metrics_out['val_bce']:.4f}  "
        f"AUC-ROC {metrics_out['auc_roc']:.4f}  AUC-PR {metrics_out['auc_pr']:.4f}"
    )
    for w in warnings:
        print(f"RED FLAG: {w}")
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                k: v
                for k, v in metrics_out.items()
                if k not in ("history", "feature_names")
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
