"""Tier-3 walker: classify branch states for truth-greedy rollout.

Per docs/superpowers/specs/2026-08-25-tier3-rollout-design.md. Walks each
branch tip -> seed and marks every state:

  collapse   : branch's next action == pi-dagger's choice -> V target reuses
               the child's rollout, zero propagation
  divergence : actions differ -> fresh C++ rollout needed from this state
  tip        : last logged state -> fresh rollout (pi-dagger continues to
               detector exit)

Only divergence + tip states enter the C++ worklist. Reuse is licensed by
state equality, certified by hit-IDENTITY equality (not is-truth equality):
same parent state + same hit -> bit-identical filtered child.
"""

from __future__ import annotations

import argparse
import sys

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_COLS = [
    "seed_id", "step_k", "cand_hit_id", "is_ckf_selected", "chi2_inc",
    "contrib_pids", "branch_majority_pid", "majority_undefined",
    "action_taken",
]


def pi_dagger_pick(cands: pd.DataFrame) -> int:
    """pi-dagger's choice among the truth candidates of ONE state.

    TODO(human): this tie-break is part of the pi-dagger definition
    (spec section 11.1) and is Matthew's to fix. Provisional rule, for the
    divergence-statistics run only: lowest chi2_inc, ties to lowest
    cand_hit_id. If the rule changes, re-run the walker before any rollout.

    Parameters
    ----------
    cands : rows of one (seed_id, step_k) with is_truth == True.
    """
    idx = np.lexsort((cands["cand_hit_id"].to_numpy(),
                      cands["chi2_inc"].to_numpy()))
    return int(cands["cand_hit_id"].to_numpy()[idx[0]])


def classify_event(parquet_path: str) -> pd.DataFrame:
    """Classify every (seed_id, step_k) state of one event.

    Returns a per-state frame: seed_id, step_k, state_class in
    {collapse, divergence, tip}, plus sel_hit / truth_pick for audit.
    """
    tbl = pq.read_table(parquet_path, columns=_COLS)
    df = tbl.to_pandas()

    # is_truth per candidate row: branch majority pid among contributors.
    # awkward broadcast, no python loop over 46M lists.
    contribs = ak.Array(df["contrib_pids"].to_numpy())
    majority = df["branch_majority_pid"].to_numpy()
    is_truth = ak.to_numpy(ak.any(contribs == majority[:, None], axis=1))
    is_truth &= ~df["majority_undefined"].to_numpy(dtype=bool)
    is_truth &= df["cand_hit_id"].to_numpy() >= 0
    df["is_truth"] = is_truth

    # Per-state reductions.
    g = df.groupby(["seed_id", "step_k"], sort=False)
    st = g.agg(
        n_truth=("is_truth", "sum"),
        sel_hit=("cand_hit_id",
                 lambda s: -1),  # placeholder, filled vectorized below
    ).reset_index()

    sel = df.loc[df["is_ckf_selected"],
                 ["seed_id", "step_k", "cand_hit_id"]].rename(
                     columns={"cand_hit_id": "sel_hit_real"})
    st = st.merge(sel, on=["seed_id", "step_k"], how="left")
    st["sel_hit"] = st["sel_hit_real"].fillna(-1).astype(np.int64)
    st = st.drop(columns=["sel_hit_real"])

    # pi-dagger pick per state: single-truth states vectorized; multi-truth
    # states (module overlaps / shared clusters) through pi_dagger_pick.
    tdf = df.loc[df["is_truth"],
                 ["seed_id", "step_k", "cand_hit_id", "chi2_inc"]]
    tdf = tdf.sort_values(["seed_id", "step_k", "chi2_inc", "cand_hit_id"])
    picks = tdf.groupby(["seed_id", "step_k"], sort=False).first().reset_index()
    picks = picks.rename(columns={"cand_hit_id": "truth_pick"})
    st = st.merge(picks[["seed_id", "step_k", "truth_pick"]],
                  on=["seed_id", "step_k"], how="left")
    st["truth_pick"] = st["truth_pick"].fillna(-1).astype(np.int64)

    # Branch's action at the NEXT logged state vs pi-dagger's choice there.
    st = st.sort_values(["seed_id", "step_k"]).reset_index(drop=True)
    same_branch_next = st["seed_id"].shift(-1) == st["seed_id"]
    next_sel = st["sel_hit"].shift(-1).fillna(-1).astype(np.int64)
    next_pick = st["truth_pick"].shift(-1).fillna(-1).astype(np.int64)

    # Agreement: same hit taken, or both hole (no selected hit AND no truth
    # hit available). Any other combination is a divergence.
    agree = (next_sel == next_pick) & same_branch_next
    st["state_class"] = np.where(
        ~same_branch_next, "tip", np.where(agree, "collapse", "divergence"))
    return st


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", required=True)
    args = p.parse_args()

    st = classify_event(args.parquet)
    n = len(st)
    counts = st["state_class"].value_counts()
    n_rollouts = int(counts.get("divergence", 0) + counts.get("tip", 0))
    multi_truth = int((st["n_truth"] > 1).sum())
    print(f"states={n:,}")
    for k in ("collapse", "divergence", "tip"):
        c = int(counts.get(k, 0))
        print(f"  {k:10s} {c:>12,}  ({c / n:.2%})")
    print(f"rollouts needed: {n_rollouts:,}  "
          f"(vs {n:,} naive; saving {1 - n_rollouts / n:.2%})")
    print(f"multi-truth states (tie-break exercised): {multi_truth:,} "
          f"({multi_truth / n:.4%})")


if __name__ == "__main__":
    main()
