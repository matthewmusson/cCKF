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
        paths = [pdir / f"expanded_event{e:09d}.parquet" for e in events]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"{split}: missing {len(missing)} parquet(s): {missing[:5]}")
        t0 = time.time()
        meta = build_gate_cache(paths, odir / split, batch_rows=args.batch_rows)
        print(f"[{split}] events={len(events)} secs={time.time()-t0:.1f} meta={json.dumps(meta)[:300]}")


if __name__ == "__main__":
    main()
