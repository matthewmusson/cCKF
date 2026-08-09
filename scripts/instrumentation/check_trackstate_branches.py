"""Verify cCKF instrumentation branches in a trackstates ROOT file.

Checks, for every branch added by the ACTS instrumentation patches:
  1. the branch exists in the tree;
  2. it is *state-aligned* -- per entry, its length equals ``nStates``;
  3. it is finite wherever the spec says it must be.

Exit code 0 means every check passed.

Usage
-----
    python scripts/instrumentation/check_trackstate_branches.py trackstates.root
    python scripts/instrumentation/check_trackstate_branches.py out.root --patches A B
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import awkward as ak
import numpy as np
import uproot


@dataclass(frozen=True)
class BranchSpec:
    """Expectation for one instrumentation branch.

    Parameters
    ----------
    name
        ROOT branch name.
    patch
        Which patch introduced it: ``"A"``, ``"B"`` or ``"C"``.
    finite_where
        Which entries must be finite. ``"measurement"`` means entries whose
        ``dim_hit``-equivalent state is a measurement; ``"all"`` means every
        entry; ``"any"`` means at least one finite entry in the file.
    """

    name: str
    patch: str
    finite_where: str = "any"


# Extended by Tasks 3, 4 and 5.
EXPECTED: list[BranchSpec] = [
    BranchSpec("S00_prt", "A", finite_where="any"),
    BranchSpec("S01_prt", "A", finite_where="any"),
    BranchSpec("S11_prt", "A", finite_where="any"),
    # Patch B: the CKF stamps every state, so this must be finite everywhere.
    BranchSpec("pathInX0_interval", "B", finite_where="all"),
]

# Branch known to be state-aligned in stock ACTS; used as the reference length.
REFERENCE_BRANCH = "volume_id"


def check_file(
    path: str,
    tree: str = "trackstates",
    patches: tuple[str, ...] = ("A", "B", "C"),
) -> list[str]:
    """Check one trackstates file and return a list of failure messages.

    Parameters
    ----------
    path
        Path to the ROOT file.
    tree
        Name of the TTree.
    patches
        Which patches' branches to require. Lets Task 2 run before Patches
        B and C exist.

    Returns
    -------
    list of str
        Human-readable failures. Empty means everything passed.
    """
    failures: list[str] = []
    specs = [s for s in EXPECTED if s.patch in patches]

    with uproot.open(f"{path}:{tree}") as t:
        available = set(t.keys())

        missing = [s.name for s in specs if s.name not in available]
        if missing:
            failures.append(f"missing branches: {sorted(missing)}")
        specs = [s for s in specs if s.name in available]
        if not specs:
            return failures

        if REFERENCE_BRANCH not in available:
            failures.append(f"reference branch {REFERENCE_BRANCH!r} not in tree")
            return failures

        arrays = t.arrays([REFERENCE_BRANCH] + [s.name for s in specs])
        n_states = ak.num(arrays[REFERENCE_BRANCH], axis=1)

        for spec in specs:
            col = arrays[spec.name]

            lengths = ak.num(col, axis=1)
            bad = ak.sum(lengths != n_states)
            if bad:
                failures.append(
                    f"{spec.name}: not state-aligned -- {bad} of "
                    f"{len(lengths)} tracks have length != nStates"
                )

            flat = np.asarray(ak.flatten(col))
            if flat.size == 0:
                failures.append(f"{spec.name}: no entries at all")
                continue

            finite = np.isfinite(flat)
            if spec.finite_where == "all" and not finite.all():
                failures.append(
                    f"{spec.name}: {int((~finite).sum())} of {flat.size} "
                    f"entries are non-finite, expected all finite"
                )
            elif spec.finite_where == "any" and not finite.any():
                failures.append(
                    f"{spec.name}: all {flat.size} entries are non-finite"
                )

    return failures


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="path to trackstates ROOT file")
    parser.add_argument("--tree", default="trackstates", help="TTree name")
    parser.add_argument(
        "--patches",
        nargs="+",
        default=["A", "B", "C"],
        choices=["A", "B", "C"],
        help="which patches' branches to require",
    )
    args = parser.parse_args()

    failures = check_file(args.path, args.tree, tuple(args.patches))
    if failures:
        print(f"FAIL ({len(failures)} problem(s)) in {args.path}")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"PASS  all checks for patches {args.patches} in {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
