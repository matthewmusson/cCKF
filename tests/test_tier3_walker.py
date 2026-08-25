"""Guard: the pi-dagger tie-break's two implementations must agree.

The rule (RATIFIED 2026-08-25: lowest chi2_inc, ties to lowest cand_hit_id)
lives in pi_dagger_pick (the named definition) and, vectorized, in
classify_event's sort+groupby-first hot path. This test feeds a synthetic
multi-truth state through both and fails if an edit to one silently
diverges from the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cckf.tier3_walker import classify_event, pi_dagger_pick


def _multi_truth_state() -> pd.DataFrame:
    # Three truth candidates: hit 7 has the lowest chi2; hits 3 and 9 tie on
    # a higher chi2, so the id tiebreak matters only between them.
    return pd.DataFrame(
        {
            "seed_id": [0, 0, 0],
            "step_k": [4, 4, 4],
            "cand_hit_id": [9, 7, 3],
            "chi2_inc": [2.0, 1.0, 2.0],
        }
    )


def test_pick_rule_lowest_chi2_then_lowest_id():
    assert pi_dagger_pick(_multi_truth_state()) == 7
    tied = _multi_truth_state()
    tied["chi2_inc"] = 2.0
    assert pi_dagger_pick(tied) == 3


def test_pick_rule_sites_agree(tmp_path):
    """Run classify_event on a tiny parquet whose one multi-truth state makes
    the vectorized pick observable, and compare with pi_dagger_pick."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cands = _multi_truth_state()
    n = len(cands) + 1
    tbl = pa.table(
        {
            "seed_id": pa.array([0, 0, 0, 0], pa.int64()),
            # state 3 is the parent whose "next" is the multi-truth state 4
            "step_k": pa.array([3, 4, 4, 4], pa.int64()),
            "cand_hit_id": pa.array([1, 9, 7, 3], pa.int64()),
            "is_ckf_selected": pa.array([True, False, True, False]),
            "chi2_inc": pa.array([0.5, 2.0, 1.0, 2.0], pa.float64()),
            "contrib_pids": pa.array([[11], [11], [11], [11]],
                                     pa.list_(pa.int64())),
            "branch_majority_pid": pa.array([11] * 4, pa.int64()),
            "majority_undefined": pa.array([False] * 4),
            "action_taken": pa.array([0] * 4, pa.int64()),
            "volume_id": pa.array([17] * 4, pa.int64()),
            "layer_id": pa.array([2] * 4, pa.int64()),
        }
    )
    path = tmp_path / "t.parquet"
    pq.write_table(tbl, path)

    st = classify_event(str(path)).set_index("step_k")
    vectorized_pick = int(st.loc[4, "truth_pick"])
    assert vectorized_pick == pi_dagger_pick(cands) == 7
    # And the branch selected hit 7 at step 4, so state 3 collapses.
    assert st.loc[3, "state_class"] == "collapse"
