"""Tier-3 stitcher: compose V^{pi-dagger} targets from rollout hit sequences.

Inputs
------
- classified states (tier3_walker.classify_event)
- rollout hit sequences (TruthRolloutAlgorithm CSV: rollout_id, step,
  geometry_id, meas_id; meas_id == -1 is a hole)
- worklist (maps rollout_id -> seed_id, step_k)
- the expanded Parquet (past counts per state)

Backward pass: counts flow tip -> seed. A collapse state's future = its
child's future plus the child's own action; divergence/tip states take their
future from their rollout. Every (branch, layer) receives a target.

Under pi-dagger the rollout future contains NO wrong hits by construction:
every accepted hit carries the majority particle. So the future contributes
n_findable (hits) and holes only.

Conventions (locked 2026-09-03 with Matthew)
--------------------------------------------
- A state's OWN accepted hit belongs to its past: ``past`` counts run up to
  and including step_k. The tip's rollout therefore contributes only hits
  beyond the tip, and V(tip) equals the finished candidate's actual
  min(completeness, purity).
- ``state_class(k)`` describes the transition INTO k+1 (walker semantics):
  a divergence at k means the branch's action at k+1 differs from
  pi-dagger's, and k's rollout replaces the future from k+1 onward.
- N_total_true missing or <= 0 can only be a failed PID join, not physics
  (a majority-defined branch has >= 2 majority hits). Those branches are
  DROPPED and counted loudly -- labeling them V = 0 would train on
  confidently wrong targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ANCHOR_CLASSES = ("divergence", "tip")


def rollout_futures(hits: pd.DataFrame, worklist: pd.DataFrame) -> pd.DataFrame:
    """Per rollout: n_findable (future truth hits) and n_future_holes.

    Parameters
    ----------
    hits : rollout_id, step, geometry_id, meas_id (-1 = hole)
    worklist : rollout_id, seed_id, step_k
    """
    g = hits.groupby("rollout_id")
    fut = g.agg(
        n_findable=("meas_id", lambda s: int((s >= 0).sum())),
        n_future_holes=("meas_id", lambda s: int((s < 0).sum())),
    ).reset_index()
    return fut.merge(
        worklist[["rollout_id", "seed_id", "step_k"]], on="rollout_id", how="left"
    )


def compose_targets(
    states: pd.DataFrame,
    futures: pd.DataFrame,
    past: pd.DataFrame,
    n_total_true: pd.DataFrame,
) -> pd.DataFrame:
    """Backward pass producing V^{pi-dagger} per (seed_id, step_k).

    The recurrence is a segmented reversed cumulative sum: rollout-bearing
    states (divergence/tip) anchor their segment with the rollout's
    n_findable; each earlier collapse state adds its child's own action
    (+1 when the child accepted a hit -- agreement guarantees it is the
    true hit; +0 when child and pi-dagger both holed).

    Then per state::

        found        = n_correct + n_findable
        completeness = found / N_total_true
        purity       = found / (n_correct + n_wrong + n_findable)
        V^{pi-dagger} = min(completeness, purity)

    Parameters
    ----------
    states : walker classification (seed_id, step_k, state_class, sel_hit)
    futures : rollout_futures output for divergence/tip states
    past : per (seed_id, step_k): n_correct, n_wrong up to and including
        this state (from the expanded Parquet, is_ckf_selected rows judged
        against branch_majority_pid -- the same membership test as
        selected_correctness)
    n_total_true : per seed_id: N_total_true, the majority particle's total
        hit count (from particle_simhit_counts)

    Returns
    -------
    seed_id, step_k, vstar_tier3 in [0, 1]. Branches with a missing rollout
    or a failed PID join are dropped and counted (printed), never labeled.
    """
    df = states.merge(
        futures[["seed_id", "step_k", "n_findable"]],
        on=["seed_id", "step_k"],
        how="left",
    )
    # Tip-first order: within a seed, row i-1 is the CHILD (step k+1) of
    # row i. Everything below relies on this order.
    df = df.sort_values(["seed_id", "step_k"], ascending=[True, False]).reset_index(
        drop=True
    )

    # --- guards: walker/rollout contract -------------------------------
    is_anchor = df["state_class"].isin(_ANCHOR_CLASSES)
    first_per_seed = ~df["seed_id"].duplicated()
    bad_tip = df.loc[first_per_seed & (df["state_class"] != "tip"), "seed_id"]
    if len(bad_tip):
        raise ValueError(
            f"walker contract broken: {len(bad_tip)} branches whose last "
            f"logged state is not class 'tip' (e.g. seed {bad_tip.iloc[0]})"
        )
    missing_rollout = is_anchor & df["n_findable"].isna()
    seeds_missing = df.loc[missing_rollout, "seed_id"].unique()
    if len(seeds_missing):
        # A crashed or unjoined rollout: the branch cannot receive targets.
        print(
            f"compose_targets: DROPPING {len(seeds_missing):,} branches "
            f"with anchor states missing rollouts "
            f"({int(missing_rollout.sum()):,} states)"
        )
        df = df[~df["seed_id"].isin(seeds_missing)].reset_index(drop=True)
        is_anchor = df["state_class"].isin(_ANCHOR_CLASSES)

    # --- segmented backward recurrence ---------------------------------
    # Segment id: cumulative anchor count in tip-first order, so each
    # anchor is the FIRST row of its own segment.
    df["_seg"] = is_anchor.groupby(df["seed_id"]).cumsum()
    grp = [df["seed_id"], df["_seg"]]

    anchor_findable = df.groupby(["seed_id", "_seg"], sort=False)[
        "n_findable"
    ].transform("first")
    # The child's own action: shift(1) hands row k the sel_hit of step k+1.
    child_sel = df.groupby(["seed_id", "_seg"], sort=False)["sel_hit"].shift(1)
    child_is_hit = (child_sel.fillna(-1) >= 0).astype(np.int64)
    inherited = child_is_hit.groupby(grp).cumsum()
    df["_n_findable_full"] = anchor_findable + inherited

    # --- past counts and truth totals ----------------------------------
    df = df.merge(
        past[["seed_id", "step_k", "n_correct", "n_wrong"]],
        on=["seed_id", "step_k"],
        how="left",
    )
    no_past = df["n_correct"].isna()
    if no_past.any():
        seeds_no_past = df.loc[no_past, "seed_id"].unique()
        print(
            f"compose_targets: DROPPING {len(seeds_no_past):,} branches "
            f"with states missing past counts "
            f"({int(no_past.sum()):,} states)"
        )
        df = df[~df["seed_id"].isin(seeds_no_past)].reset_index(drop=True)

    df = df.merge(n_total_true[["seed_id", "N_total_true"]], on="seed_id", how="left")
    bad_pid = df["N_total_true"].isna() | (df["N_total_true"] <= 0)
    if bad_pid.any():
        seeds_bad = df.loc[bad_pid, "seed_id"].unique()
        print(
            f"compose_targets: DROPPING {len(seeds_bad):,} branches with "
            f"failed PID join (N_total_true missing or <= 0)"
        )
        df = df[~df["seed_id"].isin(seeds_bad)].reset_index(drop=True)

    # --- V^{pi-dagger} ---------------------------------------------------
    found = df["n_correct"].to_numpy(dtype=np.float64) + df[
        "_n_findable_full"
    ].to_numpy(dtype=np.float64)
    completeness = found / df["N_total_true"].to_numpy(dtype=np.float64)
    purity_den = (df["n_correct"] + df["n_wrong"]).to_numpy(dtype=np.float64) + df[
        "_n_findable_full"
    ].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = np.where(purity_den > 0, found / purity_den, 0.0)
    vstar = np.minimum(completeness, purity)

    n_over = int((vstar > 1.0).sum())
    if n_over:
        # Overlap layers can let a rollout re-find a hit variant already
        # counted in the past; clip and report rather than hide.
        print(f"compose_targets: clipping {n_over:,} states with V > 1")
    df["vstar_tier3"] = np.clip(vstar, 0.0, 1.0)

    return (
        df[["seed_id", "step_k", "vstar_tier3"]]
        .sort_values(["seed_id", "step_k"])
        .reset_index(drop=True)
    )


def truth_suffix_check(
    states: pd.DataFrame,
    vstar_tier3: pd.DataFrame,
    tier2_targets: pd.DataFrame,
    tol: float = 0.01,
    value_col: str = "vstar_t2",
) -> dict:
    """The spec's diagonal-covariance bias measurement.

    For states whose branch follows truth to the tip (no divergence
    anywhere), the logged future IS the pi-dagger rollout, so tier-2
    already gives the exact target. Any deficit of the diagonal-seeded
    rollout on those states measures the covariance-seeding error
    directly. Acceptance: < 1% disagreement.

    Parameters
    ----------
    states : walker classification (seed_id, step_k, state_class)
    vstar_tier3 : compose_targets output (seed_id, step_k, vstar_tier3)
    tier2_targets : per (seed_id, step_k) with column ``value_col``
    tol : disagreement threshold on |tier3 - tier2|

    Returns
    -------
    dict with n_suffix_branches, n_states_compared, disagree_rate,
    mean_abs_diff, max_abs_diff, p50/p90/p99 of |diff|.
    """
    has_div = states.groupby("seed_id")["state_class"].apply(
        lambda s: (s == "divergence").any()
    )
    suffix_seeds = has_div.index[~has_div]

    t3 = vstar_tier3[vstar_tier3["seed_id"].isin(suffix_seeds)]
    j = t3.merge(
        tier2_targets[["seed_id", "step_k", value_col]],
        on=["seed_id", "step_k"],
        how="inner",
    )
    if not len(j):
        return {"n_suffix_branches": int(len(suffix_seeds)), "n_states_compared": 0}
    diff = (j["vstar_tier3"] - j[value_col]).abs().to_numpy()
    p50, p90, p99 = np.percentile(diff, [50, 90, 99])
    return {
        "n_suffix_branches": int(len(suffix_seeds)),
        "n_states_compared": int(len(j)),
        "disagree_rate": float((diff > tol).mean()),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "p50_abs_diff": float(p50),
        "p90_abs_diff": float(p90),
        "p99_abs_diff": float(p99),
    }
