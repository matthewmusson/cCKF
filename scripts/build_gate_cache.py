"""Build the gate feature cache for one split.

Usage
-----
    python scripts/build_gate_cache.py --split train \\
        --parquet-dir /data/results/train32/selected \\
        --out-dir /data/cache/gate/train

    # Staged run: build from a 2-event subset of the train split only.
    python scripts/build_gate_cache.py --split train \\
        --parquet-dir /data/results/train32/expanded \\
        --out-dir /data/cache/gate/train_staged \\
        --only-events 0,1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cckf import cache, features, splits
from cckf.event_selection import resolve_requested_events


def resolve_split_events(split: str, only_events: str) -> tuple[int, ...]:
    """Resolve ``--only-events`` against one split's own events.

    The assigned set passed to :func:`resolve_requested_events` is
    deliberately ``splits.events_for(split)`` -- *not* the full train+val+cal
    union that ``resolve_requested_events`` would default to. Validating
    against the union would let e.g. ``--split train --only-events 4``
    succeed (4 is a validation event), silently mixing splits, which is the
    worst failure mode in this project: results become incomparable and the
    frozen-split guarantee is broken without any error. Scoping the assigned
    set to the requested split closes that hole.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"val"``, ``"cal"``.
    only_events : str
        Comma-separated event id subset, or ``""`` for every event in
        ``split``.

    Returns
    -------
    tuple[int, ...]
        Sorted, de-duplicated event ids to build the cache from -- always a
        subset of ``splits.events_for(split)``.
    """
    split_events = splits.events_for(split)
    return resolve_requested_events(only_events, split_events)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train", "val", "cal"])
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-rows", type=int, default=1_000_000)
    parser.add_argument(
        "--only-events",
        default="",
        help=(
            "Comma-separated event id subset (e.g. '0,1') to build the cache "
            "from, instead of every event assigned to --split. Validated "
            "against --split's own events, so an id from a different split "
            "or the sealed test range raises. Omit for the full split "
            "(default, unchanged behaviour)."
        ),
    )
    args = parser.parse_args()

    events = resolve_split_events(args.split, args.only_events)
    splits.assert_not_test(events)  # guard applies to the resolved subset too
    is_staged = bool(args.only_events.strip())

    if is_staged and args.split == "train":
        # A norm_stats.npz fit on a subset is NOT the statistics a full run
        # would produce (different mean/std per feature), and downstream
        # training/calibration code has no way to detect this from the file
        # alone -- hence also the meta.json flag below.
        full_train = splits.events_for("train")
        print(
            "WARNING: --only-events was given with --split train. "
            f"norm_stats.npz will be fit on {len(events)}/{len(full_train)} "
            f"train events ({list(events)}), NOT the full train split. "
            "This is a STAGED cache -- do not use its norm_stats.npz (or "
            "this cache) as if it were a real training run's output; "
            "rebuild without --only-events for that.",
        )

    paths = []
    for event_id in events:
        path = Path(args.parquet_dir) / f"expanded_event{event_id:09d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        paths.append(path)

    print(f"split={args.split} events={list(events)} files={len(paths)}")
    meta = cache.build_gate_cache(paths, args.out_dir, batch_rows=args.batch_rows)

    if is_staged:
        # Mark this cache as partial so a later consumer (a training script,
        # a human staring at a directory listing) can't mistake it for a
        # real, full-split cache. Rewritten here rather than threaded through
        # cache.build_gate_cache/CacheWriter.close, since those are shared by
        # every caller and "partial split" is a concern specific to this
        # staged-run CLI path.
        meta["partial_split"] = True
        meta["events_used"] = list(events)
        (Path(args.out_dir) / "meta.json").write_text(json.dumps(meta, indent=2))

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
