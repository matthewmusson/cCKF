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

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from cckf import cache, features, models, samplers, train

#: spec §2.7 red-flag thresholds
RED_FLAGS = {
    "train_bce": 0.15,
    "val_bce": 0.15,
    "auc_roc_min": 0.95,
    "auc_pr_min": 0.80,
}


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
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Materialise the selected rows once; standardise, then subset columns.
    X_train = train.standardize(np.asarray(tr["X"])[picked], mu, sigma)[:, col_idx]
    y_train = y_train_full[picked].astype(np.float32)
    X_val = train.standardize(np.asarray(va["X"]), mu, sigma)[:, col_idx]
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
