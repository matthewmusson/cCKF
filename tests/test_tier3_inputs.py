"""Tests for the tier-3 stitch input builders (cckf/tier3_inputs.py)."""

import numpy as np
import pandas as pd

from cckf.tier3_inputs import (
    n_total_true_from_frames,
    past_counts_from_rows,
)


def _rows(records):
    cols = [
        "seed_id",
        "step_k",
        "cand_hit_id",
        "is_ckf_selected",
        "contrib_pids",
        "branch_majority_pid",
        "majority_undefined",
    ]
    return pd.DataFrame(records, columns=cols)


def test_past_counts_cumulative_including_own():
    # branch: correct at 0, wrong at 1, hole at 2, correct at 3
    rows = _rows(
        [
            (0, 0, 10, True, [7], 7, False),  # correct hit
            (0, 1, 11, True, [9], 7, False),  # wrong hit
            (0, 2, -1, True, None, 7, False),  # hole
            (0, 3, 12, True, [7], 7, False),  # correct hit
        ]
    )
    out = past_counts_from_rows(rows).set_index("step_k")
    assert out["n_correct"].tolist() == [1, 1, 1, 2]
    assert out["n_wrong"].tolist() == [0, 1, 1, 1]
    assert out["n_correct"].dtype == np.int64
    assert out["n_wrong"].dtype == np.int64


def test_past_counts_excludes_majority_undefined():
    rows = _rows(
        [
            (0, 0, 10, True, [7], 7, False),
            (1, 0, 20, True, [3], 3, True),  # majority_undefined branch
        ]
    )
    out = past_counts_from_rows(rows)
    assert set(out["seed_id"]) == {0}


def test_n_total_true_counts_majority_measurements():
    # two particles; majority pid has 4 measurements -> N_total_true == 4
    majority_by_seed = pd.DataFrame({"seed_id": [0, 1], "branch_majority_pid": [7, 8]})
    simhits = pd.DataFrame({"particle_id": [7, 7, 7, 7, 8, 8]})
    out = n_total_true_from_frames(majority_by_seed, simhits).set_index("seed_id")
    assert out.loc[0, "N_total_true"] == 4
    assert out.loc[1, "N_total_true"] == 2
    assert out["N_total_true"].dtype == np.int64


def test_n_total_true_missing_pid_is_zero_not_dropped():
    majority_by_seed = pd.DataFrame({"seed_id": [0], "branch_majority_pid": [99]})
    simhits = pd.DataFrame({"particle_id": [7, 7]})
    out = n_total_true_from_frames(majority_by_seed, simhits)
    assert len(out) == 1
    assert out.iloc[0]["N_total_true"] == 0
