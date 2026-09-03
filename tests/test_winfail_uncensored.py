"""tests/test_winfail_uncensored.py"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winfail_uncensored import (
    ETA_BINS, N_VALUES, OCC_EDGES, PT_EDGES, wilson_interval, assign_strata,
    select_ckf_branch, branch_purity, flag_ambi_survivors,
)


def test_eta_binning_is_dense():
    assert len(ETA_BINS) == 141
    assert np.isclose(ETA_BINS[0], -3.5) and np.isclose(ETA_BINS[-1], 3.5)
    assert np.allclose(np.diff(ETA_BINS), 0.05)


def test_pt_edges():
    assert PT_EDGES == (0.0, 0.7, 0.9, 1.0)


def test_wilson_interval_basic():
    lo, hi = wilson_interval(np.array([5]), np.array([10]), z=1.0)
    assert 0.0 < lo[0] < 0.5 < hi[0] < 1.0


def test_wilson_interval_zero_denominator_is_nan():
    lo, hi = wilson_interval(np.array([0]), np.array([0]))
    assert np.isnan(lo[0]) and np.isnan(hi[0])


def test_wilson_interval_extremes_stay_in_unit_interval():
    lo, hi = wilson_interval(np.array([0, 10]), np.array([10, 10]))
    assert lo[0] >= 0.0 and hi[1] <= 1.0


def test_assign_strata_sensor_groups_and_vol20():
    eta = np.array([0.0, 0.0, 0.0, 0.0])
    vol = np.array([17, 24, 29, 20])
    occ = np.array([0, 3, 12, 50])
    ei, si, oi = assign_strata(eta, vol, occ)
    assert list(si) == [0, 1, 2, -1]
    assert list(oi) == [0, 1, 3, 4]
    assert ei[0] == 70


# Tests for Task 3: select_ckf_branch, branch_purity, flag_ambi_survivors

def _rows(records):
    cols = ["seed_id", "branch_id", "step_k", "cand_hit_id",
            "is_ckf_selected", "contrib_pids", "branch_majority_pid"]
    return pd.DataFrame(records, columns=cols)


def test_select_ckf_branch_picks_most_selected_rows():
    rows = _rows([
        (0, 0, 0, 11, True,  [7], 7),
        (0, 0, 1, 12, True,  [7], 7),
        (0, 1, 0, 11, True,  [7], 7),
        (1, 5, 0, 21, True,  [9], 9),
    ])
    sel = select_ckf_branch(rows)
    assert dict(zip(sel.seed_id, sel.branch_id)) == {0: 0, 1: 5}


def test_select_ckf_branch_tie_breaks_to_lowest_branch():
    rows = _rows([
        (3, 2, 0, 1, True, [1], 1),
        (3, 1, 0, 1, True, [1], 1),
    ])
    sel = select_ckf_branch(rows)
    assert dict(zip(sel.seed_id, sel.branch_id)) == {3: 1}


def test_branch_purity_three_of_three_is_pure():
    rows = _rows([
        (0, 0, 0, 11, True, [7],    7),
        (0, 0, 1, 12, True, [7, 8], 7),
        (0, 0, 2, 13, True, [7],    7),
        (0, 0, 3, 14, True, [8],    7),   # step 4 wrong: irrelevant to purity
        (1, 0, 0, 21, True, [9],    9),
        (1, 0, 1, 22, True, [4],    9),   # 2/3 -> majority
        (1, 0, 2, 23, True, [9],    9),
    ])
    sel = select_ckf_branch(rows)
    pur = branch_purity(rows, sel)
    assert dict(zip(pur.seed_id, pur.is_pure)) == {0: True, 1: False}


def _arows(records):
    cols = ["seed_id", "step_k", "cand_hit_id", "is_ckf_selected", "chi2_inc"]
    return pd.DataFrame(records, columns=cols)


def test_flag_ambi_survivors_short_branch_dropped():
    rows = _arows([(0, k, 100 + k, True, 1.0) for k in range(7)]
                  + [(1, k, 200 + k, True, 1.0) for k in range(4)])
    out = flag_ambi_survivors(rows)
    assert dict(zip(out.seed_id, out.survived_ambi)) == {0: True, 1: False}


def test_flag_ambi_survivors_evicts_worse_of_overlapping_pair():
    # Branches 0 and 1 share hits 100-106 (7 shared >= max_shared=3).
    # Branch 1 has the higher shared fraction (7/7 vs 7/9) and is evicted;
    # afterwards branch 0 shares nothing and survives.
    rows = _arows(
        [(0, k, 100 + k, True, 1.0) for k in range(7)]
        + [(0, 7, 300, True, 1.0), (0, 8, 301, True, 1.0)]
        + [(1, k, 100 + k, True, 1.0) for k in range(7)]
    )
    out = flag_ambi_survivors(rows)
    assert dict(zip(out.seed_id, out.survived_ambi)) == {0: True, 1: False}


def test_flag_ambi_survivors_disjoint_branches_both_survive():
    rows = _arows([(0, k, 100 + k, True, 1.0) for k in range(8)]
                  + [(1, k, 500 + k, True, 1.0) for k in range(8)])
    out = flag_ambi_survivors(rows)
    assert dict(zip(out.seed_id, out.survived_ambi)) == {0: True, 1: True}
