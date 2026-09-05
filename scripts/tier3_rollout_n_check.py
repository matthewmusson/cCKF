#!/usr/bin/env python3
"""Check that a tier-3 multi-window rollout generation run is complete.

Verifies all 32 per-event hit files exist under
``$SCRATCH/cckf/tier3_nsig{NSIG}/hits/`` and prints the per-n total
n_findable (sum over all rollouts of rows with ``meas_id >= 0``), so
monotonicity across the chi2 acceptance window NSIG can be eyeballed by
running this once per n and comparing totals (a wider window can only
ever accept a hit that a narrower one rejected, so the total must be
non-decreasing in NSIG).

Counting logic is the same as ``read_findable`` in
``scripts/t3_window_smoke_check.py`` (Task 2's single-event smoke check),
generalized here to sum across all 32 events instead of diffing two.

Tier 3 (infrastructure).

Usage:
    python scripts/tier3_rollout_n_check.py NSIG
    python scripts/tier3_rollout_n_check.py NSIG --hits-dir /path/to/hits

Exit code is 0 iff all 32 files exist and are non-empty; 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

N_EVENTS = 32


def count_findable(path: Path) -> int:
    """Return the number of rows in a rollout-hits CSV with meas_id >= 0.

    ``meas_id == -1`` marks a hole written by TruthRolloutAlgorithm; any
    other value (including 0) is a findable (accepted) hit.
    """
    total = 0
    with path.open() as f:
        header = f.readline().rstrip("\n").split(",")
        i_mid = header.index("meas_id")
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(",")
            if int(parts[i_mid]) >= 0:
                total += 1
    return total


def hits_dir_for(nsig: str, scratch: str | None = None) -> Path:
    """Return $SCRATCH/cckf/tier3_nsig{nsig}/hits."""
    scratch = scratch or os.environ.get("SCRATCH")
    if not scratch:
        raise SystemExit("SCRATCH env var not set; pass --hits-dir explicitly")
    return Path(scratch) / "cckf" / f"tier3_nsig{nsig}" / "hits"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("nsig", help="chi2 window value used for this run, e.g. 3, 5, 10")
    ap.add_argument(
        "--hits-dir",
        default=None,
        help="override $SCRATCH/cckf/tier3_nsig{NSIG}/hits",
    )
    args = ap.parse_args()

    hits_dir = Path(args.hits_dir) if args.hits_dir else hits_dir_for(args.nsig)

    missing = []
    empty = []
    total_findable = 0
    per_event = {}
    for e in range(N_EVENTS):
        p = hits_dir / f"event{e:09d}-rollout-hits.csv"
        if not p.exists():
            missing.append(p)
            continue
        if p.stat().st_size == 0:
            empty.append(p)
            continue
        n = count_findable(p)
        per_event[e] = n
        total_findable += n

    print(f"nsig={args.nsig}  hits_dir={hits_dir}")
    print(f"  files found   : {len(per_event)}/{N_EVENTS}")
    if missing:
        print(f"  MISSING ({len(missing)}): {[str(p) for p in missing[:5]]}"
              + (" ..." if len(missing) > 5 else ""))
    if empty:
        print(f"  EMPTY ({len(empty)}): {[str(p) for p in empty[:5]]}"
              + (" ..." if len(empty) > 5 else ""))
    print(f"  total n_findable (sum over 32 events): {total_findable}")

    ok = not missing and not empty and len(per_event) == N_EVENTS
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
