"""AUC-ROC / AUC-PR for a trained gate or value checkpoint on a cache split.

Discrimination metrics belong to the VAL split; calibration audits (ECE,
reliability) belong to the CAL split via calibrate_and_audit.py (spec 6.1:
the calibration split is never used for model selection, and val is never
used for calibration fitting).

Value targets are soft labels in [0, 1]; they are binarized at 0.5 for
ranking metrics, mirroring train_value.py's convention.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cckf import cache as cache_mod  # noqa: E402
from cckf import models  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True, help="val-split cache dir")
    ap.add_argument("--model-type", choices=["gate", "value"], required=True)
    ap.add_argument("--norm-stats", required=True,
                    help="norm_stats.npz fit on the TRAIN split")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--batch", type=int, default=2_000_000)
    args = ap.parse_args()

    loaded = cache_mod.load_cache(args.cache)
    X, y = loaded.X, loaded.y
    stats = np.load(args.norm_stats)
    mu, sigma = stats["mu"].astype(np.float32), stats["sigma"].astype(np.float32)
    sigma = np.where(sigma > 1e-30, sigma, 1.0).astype(np.float32)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cls = models.GateMLP if args.model_type == "gate" else models.ValueMLP
    model = cls(n_features=X.shape[1])
    model.load_state_dict(state)
    model.eval()

    logits = np.empty(len(y), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(y), args.batch):
            xb = (X[i:i + args.batch].astype(np.float32) - mu) / sigma
            logits[i:i + args.batch] = model(
                torch.from_numpy(xb)).squeeze(-1).numpy()

    y_bin = (np.asarray(y, dtype=np.float64) >= 0.5).astype(np.int8)
    out = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "cache": args.cache,
        "n_rows": int(len(y_bin)),
        "positive_fraction": float(y_bin.mean()),
        "auc_roc": float(roc_auc_score(y_bin, logits)),
        "auc_pr": float(average_precision_score(y_bin, logits)),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
