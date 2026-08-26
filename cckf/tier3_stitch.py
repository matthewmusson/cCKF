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
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rollout_futures(hits: pd.DataFrame,
                    worklist: pd.DataFrame) -> pd.DataFrame:
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
    return fut.merge(worklist[["rollout_id", "seed_id", "step_k"]],
                     on="rollout_id", how="left")


def compose_targets(states: pd.DataFrame, futures: pd.DataFrame,
                    past: pd.DataFrame, n_total_true: pd.DataFrame
                    ) -> pd.DataFrame:
    """Backward pass producing V^{pi-dagger} per (seed_id, step_k).

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
    seed_id, step_k, vstar_tier3 in [0, 1]
    """
    # TODO(human): the V^{pi-dagger} composition (spec section 11.1) is
    # Tier 1 -- yours. The pieces are assembled below; the definition to
    # implement is:
    #
    #   completeness = (n_correct + n_findable) / N_total_true
    #   purity       = (n_correct + n_findable)
    #                  / (n_correct + n_wrong + n_findable)
    #   V^{pi-dagger} = min(completeness, purity)
    #
    # Decisions that are yours to make here:
    #   - clipping / degenerate denominators (N_total_true == 0)
    #   - whether collapse states inherit the child's future counts plus the
    #     child's own action (the backward recurrence), which this skeleton
    #     sets up but does not finalize
    raise NotImplementedError("TODO(human): V composition, spec 11.1")


def truth_suffix_check(states: pd.DataFrame, futures: pd.DataFrame,
                       tier2_targets: pd.DataFrame) -> dict:
    """The spec's diagonal-covariance bias measurement.

    For states whose branch follows truth to the tip, the logged future IS
    the pi-dagger rollout, so tier-2 already gives the exact target. Any
    deficit of the diagonal-seeded rollout on those states measures the
    covariance-seeding error directly. Acceptance: < 1% disagreement.

    TODO(human): wire once compose_targets is implemented; compare
    vstar_tier3 against tier2_targets on the truth-suffix subset and report
    the disagreement rate and its distribution.
    """
    raise NotImplementedError("TODO(human): truth-suffix comparison")
