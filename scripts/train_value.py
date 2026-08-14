"""Train the value function V_φ on soft V^{π†} targets.

Usage
-----
    python scripts/train_value.py \\
        --train-cache /data/cache/value/train \\
        --val-cache /data/cache/value/val \\
        --out-dir /data/models/value_v0 --wandb-project cckf-value
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

#: spec §3.8 expectations
RED_FLAGS = {"train_bce": 0.20, "val_bce": 0.25, "auc_roc_min": 0.90}


def _load(cache_dir: str) -> dict:
    d = Path(cache_dir)
    meta = json.loads((d / "meta.json").read_text())
    n, f = meta["n_rows"], meta["n_features"]
    return {
        "X": np.fromfile(d / "X.f32", dtype=np.float32).reshape(n, f),
        "y": np.fromfile(d / "y.f32", dtype=np.float32),
        "aux": np.fromfile(d / "aux.f32", dtype=np.float32).reshape(n, 3),
        "meta": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--oversample-marginal", type=float, default=0.0,
                        help="spec §3.5: extra copies of states with V in [0.2, 0.8]")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, va = _load(args.train_cache), _load(args.val_cache)
    stats = np.load(Path(args.train_cache) / "norm_stats.npz", allow_pickle=True)
    mu, sigma = stats["mu"], stats["sigma"]

    X_train = train.standardize(tr["X"], mu, sigma)
    y_train = tr["y"]

    # Labels are bimodal: most branches clearly succeed or clearly fail. The
    # marginal band is where the decision actually matters, so it can be
    # oversampled to keep it from being drowned out.
    if args.oversample_marginal > 0:
        marginal = np.flatnonzero((y_train >= 0.2) & (y_train <= 0.8))
        extra = np.repeat(marginal, int(args.oversample_marginal))
        keep = np.concatenate([np.arange(len(y_train)), extra])
        X_train, y_train = X_train[keep], y_train[keep]
        print(f"oversampled {len(marginal):,} marginal states → {len(y_train):,} rows")

    X_val = train.standardize(va["X"], mu, sigma)
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

    model = models.ValueMLP(
        n_features=len(feat.VALUE_FEATURES), width=args.width, depth=args.depth
    )
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
        "auc_roc": float(roc_auc_score(y_binary, val_logits))
        if y_binary.min() != y_binary.max() else None,
        "mean_target": float(y_val.mean()),
        "marginal_fraction": float(np.mean((y_val >= 0.3) & (y_val <= 0.7))),
        # Tier-gap diagnostic (see value_target.py on why Tier 3 is absent).
        "tier1_minus_tier2_mean": float(np.mean(va["aux"][:, 0] - y_val)),
        "stopped_epoch": result["stopped_epoch"],
        "history": result["history"],
    }
    warnings = []
    if metrics_out["val_bce"] > RED_FLAGS["val_bce"]:
        warnings.append(f"val BCE {metrics_out['val_bce']:.4f} > {RED_FLAGS['val_bce']}")
    if metrics_out["auc_roc"] is not None and metrics_out["auc_roc"] < RED_FLAGS["auc_roc_min"]:
        warnings.append(f"AUC-ROC {metrics_out['auc_roc']:.4f} < {RED_FLAGS['auc_roc_min']}")
    metrics_out["red_flags"] = warnings

    torch.save(
        {
            "state_dict": result["best_state"],
            "n_features": len(feat.VALUE_FEATURES),
            "width": args.width,
            "depth": args.depth,
            "feature_names": list(feat.VALUE_FEATURES),
            "mu": mu,
            "sigma": sigma,
        },
        out_dir / "value_model.pt",
    )
    (out_dir / "value_metrics.json").write_text(json.dumps(metrics_out, indent=2))
    np.savez(out_dir / "value_val_predictions.npz", pred=val_pred, target=y_val, aux=va["aux"])

    print(f"val BCE {metrics_out['val_bce']:.4f}  MSE {mse:.4f}  "
          f"tier gap {metrics_out['tier1_minus_tier2_mean']:.4f}")
    for w in warnings:
        print(f"RED FLAG: {w}")
    if wandb_run is not None:
        wandb_run.summary.update(
            {k: v for k, v in metrics_out.items() if k != "history"}
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
