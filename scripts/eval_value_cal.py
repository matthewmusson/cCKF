"""Calibration audit for the value function V_phi on the CAL split.

Loads a value cache directory (X.f32 / y.f32 soft targets), runs the trained
ValueMLP with the checkpoint's own mu/sigma, and reports ECE plus a
reliability diagram. Calibration for a soft target means E[y | pred] == pred,
so the reliability bins compare mean predicted probability against the mean
soft target per bin -- the same estimator as for hard labels, no binarization.

Also prints the empirical entropy floor mean(H(y)): the BCE a perfectly
calibrated model would still pay because the targets themselves are soft.

Usage:
    python scripts/eval_value_cal.py \
        --model .../value_model.pt --cal-cache .../vcache_v3/cal \
        --out-dir .../models_v3/value_t2_maj/calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from cckf import metrics, models, train  # noqa: E402

EDGES = metrics.logit_bin_edges(n_bins=30)


def load_value_split(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    n, f = meta["n_rows"], meta["n_features"]
    return {
        "X": np.memmap(d / "X.f32", dtype=np.float32, mode="r", shape=(n, f)),
        "y": np.fromfile(d / "y.f32", dtype=np.float32),
        "meta": meta,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cal-cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = models.ValueMLP(
        n_features=ckpt["n_features"], width=ckpt["width"], depth=ckpt["depth"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    split = load_value_split(Path(args.cal_cache))
    mu = np.asarray(ckpt["mu"], dtype=np.float32)
    sigma = np.asarray(ckpt["sigma"], dtype=np.float32)
    sigma = np.where(sigma > 1e-30, sigma, 1.0).astype(np.float32)
    X = ((np.asarray(split["X"]) - mu) / sigma).astype(np.float32)
    y = split["y"].astype(np.float64)

    logits = train.predict_logits(model, X, device=args.device)
    pred = 1.0 / (1.0 + np.exp(-logits))

    eps = 1e-7
    yc = np.clip(y, eps, 1 - eps)
    entropy_floor = float(
        np.mean(-(yc * np.log(yc) + (1 - yc) * np.log(1 - yc)))
    )
    bce = float(
        np.mean(-(y * np.log(np.clip(pred, eps, 1)) + (1 - y) * np.log(np.clip(1 - pred, eps, 1))))
    )

    # cckf.metrics.reliability_bins binarizes labels (astype(bool)), which is
    # wrong for soft targets: 0.24 would count as a positive. Bin inline
    # instead: mean soft target per prediction bin.
    n_bins = len(EDGES) - 1
    bin_idx = np.clip(np.digitize(pred, EDGES[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        bins.append({
            "mean_predicted": float(pred[mask].mean()) if n else None,
            "observed_mean": float(y[mask].mean()) if n else None,
            "count": n,
        })
    total = sum(b["count"] for b in bins)
    ece = sum(
        (b["count"] / total) * abs(b["observed_mean"] - b["mean_predicted"])
        for b in bins if b["count"] > 0
    )
    mce = max(
        (abs(b["observed_mean"] - b["mean_predicted"])
         for b in bins if b["count"] >= metrics.MIN_BIN_COUNT),
        default=0.0,
    )
    audit = {
        "n_cal_rows": int(len(y)),
        "target_mean": float(y.mean()),
        "cal_bce": bce,
        "entropy_floor": entropy_floor,
        "excess_bce": bce - entropy_floor,
        "ece": float(ece),
        "mce": float(mce),
    }
    (out_dir / "value_cal_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    mp = [b["mean_predicted"] for b in bins if b["count"] > 0]
    ob = [b["observed_mean"] for b in bins if b["count"] > 0]
    ct = [b["count"] for b in bins if b["count"] > 0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    sc = ax.scatter(mp, ob, s=np.clip(np.sqrt(ct), 4, 60), c="tab:blue")
    ax.set_xlabel("mean predicted V")
    ax.set_ylabel("mean observed target")
    ax.set_title(f"V_phi reliability (cal split)  ECE={audit['ece']:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "value_reliability_cal.png", dpi=150)


if __name__ == "__main__":
    main()
