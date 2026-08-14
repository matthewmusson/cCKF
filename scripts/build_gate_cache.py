"""Build the gate feature cache for one split.

Usage
-----
    python scripts/build_gate_cache.py --split train \\
        --parquet-dir /data/results/train32/selected \\
        --out-dir /data/cache/gate/train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cckf import cache, features, splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train", "val", "cal"])
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-rows", type=int, default=1_000_000)
    args = parser.parse_args()

    events = splits.events_for(args.split)
    splits.assert_not_test(events)

    paths = []
    for event_id in events:
        path = Path(args.parquet_dir) / f"expanded_event{event_id:09d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        paths.append(path)

    print(f"split={args.split} events={list(events)} files={len(paths)}")
    meta = cache.build_gate_cache(paths, args.out_dir, batch_rows=args.batch_rows)
    print(json.dumps(meta, indent=2))

    # Normalisation statistics are computed on the TRAIN split only and reused
    # verbatim for val/cal/test (spec §2.3). Fitting them per split would leak
    # split-specific distribution information into the inputs.
    if args.split == "train":
        loaded = cache.load_cache(args.out_dir)
        mu, sigma = cache.compute_norm_stats(
            loaded["X"], skip=features.NO_STANDARDIZE, names=features.GATE_FEATURES
        )
        np.savez(
            Path(args.out_dir) / "norm_stats.npz",
            mu=mu,
            sigma=sigma,
            feature_names=np.array(features.GATE_FEATURES),
        )
        print(f"wrote norm_stats.npz (mu[0]={mu[0]:.4g}, sigma[0]={sigma[0]:.4g})")


if __name__ == "__main__":
    main()
