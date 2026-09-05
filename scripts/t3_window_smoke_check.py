#!/usr/bin/env python3
"""Behavioral check for the tier-3 rollout chi2 acceptance window.

Compares two rollout-hits CSVs produced from the SAME worklist, one with
``rollout_window_nsigma: 0`` (unbounded pi-dagger) and one with a finite
window. The window can only ever turn an accepted true hit into a hole, so:

  * the rollout_id sets must be identical (the window changes acceptance,
    never which rollouts run),
  * per rollout, n_findable(windowed) <= n_findable(unbounded),
  * at least one rollout must strictly decrease (otherwise the window is
    not actually reaching the C++ -- the silent-pybind-drop failure mode).

n_findable is the count of rows with ``meas_id >= 0``; holes are written
with ``meas_id = -1`` by TruthRolloutAlgorithm.

Tier 3 (infrastructure).

Usage:
    python scripts/t3_window_smoke_check.py W0_HITS.csv W3_HITS.csv
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path


def read_findable(path: Path) -> dict[int, int]:
    """Return {rollout_id: n_findable} from a rollout-hits CSV.

    Every rollout_id present in the file gets an entry, including rollouts
    whose states are all holes (n_findable == 0) -- otherwise a window that
    empties a rollout would look like a missing key rather than a decrease.
    """
    counts: dict[int, int] = defaultdict(int)
    with path.open() as f:
        header = f.readline().rstrip("\n").split(",")
        i_rid = header.index("rollout_id")
        i_mid = header.index("meas_id")
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(",")
            rid = int(parts[i_rid])
            counts[rid] += 1 if int(parts[i_mid]) >= 0 else 0
    return dict(counts)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    w0_path, w3_path = Path(sys.argv[1]), Path(sys.argv[2])
    w0, w3 = read_findable(w0_path), read_findable(w3_path)

    print(f"w0 file: {w0_path}  rollouts={len(w0)}  "
          f"sum_n_findable={sum(w0.values())}")
    print(f"w3 file: {w3_path}  rollouts={len(w3)}  "
          f"sum_n_findable={sum(w3.values())}")

    ok = True
    if set(w0) != set(w3):
        only0, only3 = set(w0) - set(w3), set(w3) - set(w0)
        print(f"FAIL: rollout_id sets differ "
              f"(only_w0={len(only0)} only_w3={len(only3)})")
        ok = False

    shared = sorted(set(w0) & set(w3))
    violations = [r for r in shared if w3[r] > w0[r]]
    decreased = [r for r in shared if w3[r] < w0[r]]
    equal = len(shared) - len(violations) - len(decreased)

    print(f"shared rollouts       : {len(shared)}")
    print(f"  n_findable decreased: {len(decreased)}")
    print(f"  n_findable unchanged: {equal}")
    print(f"  n_findable INCREASED: {len(violations)}  (must be 0)")
    if decreased:
        worst = max(decreased, key=lambda r: w0[r] - w3[r])
        print(f"  largest drop        : rollout {worst} "
              f"{w0[worst]} -> {w3[worst]}")

    if violations:
        print("FAIL: window increased n_findable for "
              f"{len(violations)} rollouts, e.g. "
              f"{[(r, w0[r], w3[r]) for r in violations[:5]]}")
        ok = False
    if not decreased:
        print("FAIL: no rollout strictly decreased -- the window is not "
              "reaching the C++ selector (check the pybind member list).")
        ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
