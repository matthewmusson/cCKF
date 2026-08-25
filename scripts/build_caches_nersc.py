#!/usr/bin/env python3
"""Build gate train/val/cal caches from the re-expanded Parquets on NERSC.

Splits come from cckf.splits, which is frozen: 24/4/4 over events [0,32) with
[32,64) sealed. Events 0-3 remain in TRAIN -- re-expansion repairs the S11
corruption that made them unusable, so there is no longer a reason to drop
them.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cckf.cache import build_gate_cache
from cckf.splits import TRAIN_EVENTS, VAL_EVENTS, CAL_EVENTS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", default="train,val,cal")
    ap.add_argument("--batch-rows", type=int, default=1_000_000)
    args = ap.parse_args()

    pdir, odir = Path(args.parquet_dir), Path(args.out_dir)
    groups = {"train": TRAIN_EVENTS, "val": VAL_EVENTS, "cal": CAL_EVENTS}

    for split in args.splits.split(","):
        events = groups[split]
        # Memory-bound events (5, 7, 14) are expanded in track_nr chunks and
        # land as ..._p0.parquet ... _p3.parquet. build_gate_cache takes a list
        # of files per split, so parts need no merging -- just collect them.
        paths, missing = [], []
        for e in events:
            whole = pdir / f"expanded_event{e:09d}.parquet"
            parts = sorted(pdir.glob(f"expanded_event{e:09d}_p*.parquet"))
            if whole.exists():
                paths.append(whole)
            elif parts:
                paths.extend(parts)
            else:
                missing.append(f"event{e:09d}")
        if missing:
            raise SystemExit(f"{split}: missing {len(missing)} event(s): {missing[:5]}")
        print(f"[{split}] {len(events)} events -> {len(paths)} parquet file(s)")
        t0 = time.time()
        meta = build_gate_cache(paths, odir / split, batch_rows=args.batch_rows)
        print(f"[{split}] events={len(events)} secs={time.time()-t0:.1f} meta={json.dumps(meta)[:300]}")


if __name__ == "__main__":
    main()
