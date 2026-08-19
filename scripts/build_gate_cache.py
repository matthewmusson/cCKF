"""Build the gate feature cache for one split.

Usage
-----
    python scripts/build_gate_cache.py --split train \\
        --parquet-dir /data/results/train32/selected \\
        --out-dir /data/cache/gate/train

    # Staged run: build from a 2-event subset of the train split only.
    # --out-dir must be a path outside /data/cache/gate/ -- see
    # modal_train.py's gate_staged/ convention -- so a staged cache can
    # never collide with a full split's cache.
    python scripts/build_gate_cache.py --split train \\
        --parquet-dir /data/results/train32/expanded \\
        --out-dir /data/cache/gate_staged/train \\
        --only-events 0,1

    # Pure-seed run: filter to seeds where all 3 seed hits are from the
    # branch's majority particle. --parquet-dir MUST be SELECTED_DIR (the
    # output of scripts/patch_is_selected.py) -- pure-seed classification
    # needs the is_ckf_selected column, which the raw expanded Parquets
    # don't have.
    python scripts/build_gate_cache.py --split train \\
        --parquet-dir /data/results/train32/selected \\
        --out-dir /data/cache/gate_pure/train \\
        --pure-seeds-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from cckf import cache, features, splits
from cckf.event_selection import resolve_requested_events
from cckf.seed_purity import compute_pure_seed_set


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
    parser.add_argument(
        "--pure-seeds-only",
        action="store_true",
        help="Filter to pure seeds (3/3 seed hits from majority particle). "
             "Requires --parquet-dir to point at selected Parquets "
             "(with is_ckf_selected column).",
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

    pure_seed_sets = None
    if args.pure_seeds_only:
        # Seed purity (cckf.seed_purity) needs is_ckf_selected to find each
        # branch's first 3 CKF-selected measurement hits -- only present in
        # the *selected* Parquets (scripts/patch_is_selected.py), not the
        # raw expanded ones. Fail fast with a clear pointer rather than
        # letting compute_pure_seed_set raise a bare KeyError per file.
        schema_columns = set(pq.ParquetFile(paths[0]).schema_arrow.names)
        if "is_ckf_selected" not in schema_columns:
            raise ValueError(
                f"--pure-seeds-only requires --parquet-dir to point at "
                f"selected Parquets (with an is_ckf_selected column), but "
                f"{paths[0]} has no such column. Point --parquet-dir at "
                f"the SELECTED_DIR output of scripts/patch_is_selected.py "
                f"(e.g. .../train32/selected), not the raw expanded dir."
            )
        print(
            "NOTE: --pure-seeds-only is set. Computing pure seed sets from "
            f"{len(paths)} file(s) -- --parquet-dir must be SELECTED_DIR "
            "(is_ckf_selected present), which it is here."
        )
        pure_seed_sets = {path: compute_pure_seed_set(str(path)) for path in paths}
        n_pure = sum(len(s) for s in pure_seed_sets.values())
        print(f"pure_seeds_only: {n_pure} pure (seed_id, branch_id) pairs total")

    print(f"split={args.split} events={list(events)} files={len(paths)}")
    meta = cache.build_gate_cache(
        paths,
        args.out_dir,
        batch_rows=args.batch_rows,
        pure_seed_sets=pure_seed_sets,
    )

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
