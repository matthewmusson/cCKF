"""tests/test_winfail_uncensored.py"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winfail_uncensored import (
    ETA_BINS, N_VALUES, OCC_EDGES, PT_EDGES, wilson_interval, assign_strata,
    select_ckf_branch, branch_purity, flag_ambi_survivors, build_state_table,
    accumulate_event, _as_pid_list, _pick_earliest_by_tt,
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


# Tests for Task 4: build_state_table, accumulate_event

def _prows(records):
    cols = ["seed_id", "branch_id", "step_k", "volume_id", "state_theta",
            "n_window", "cand_hit_id", "residual_l0", "residual_l1",
            "S00", "S11", "is_1d", "is_ckf_selected", "contrib_pids",
            "branch_majority_pid", "majority_undefined",
            "majority_true_hit_on_surface", "chi2_inc"]
    return pd.DataFrame(records, columns=cols)


HALF_PI = float(np.pi / 2)  # eta = 0


def test_build_state_table_taxonomy():
    rows = _prows([
        # state A: true-hit row at d = 4.0  (r0=0.8, S00=0.04; r1 small)
        (0, 0, 0, 17, HALF_PI, 3, 11, 0.8, 0.02, 0.04, 0.04, False, True, [7], 7, False, True, 1.0),
        # ...and a wrong-hit row on the same state (must not affect d_true)
        (0, 0, 0, 17, HALF_PI, 3, 12, 0.1, 0.01, 0.04, 0.04, False, False, [8], 7, False, True, 1.0),
        # state B: on surface, NO true-hit row -> escaped (d_true NaN)
        (1, 0, 0, 17, HALF_PI, 1, 13, 0.1, 0.01, 0.04, 0.04, False, True, [5], 9, False, True, 1.0),
        # state C: hole row, majority not on surface -> module failure
        (2, 0, 0, 17, HALF_PI, 0, -1, np.nan, np.nan, np.nan, np.nan, False, False, None, 4, False, False, 1.0),
    ])
    st = build_state_table(rows)
    assert len(st) == 3
    a = st[st.seed_id == 0].iloc[0]
    assert np.isclose(a.d_true, 4.0)
    b = st[st.seed_id == 1].iloc[0]
    assert b.on_surface and np.isnan(b.d_true)
    c = st[st.seed_id == 2].iloc[0]
    assert not c.on_surface


def test_build_state_table_min_over_two_true_rows_and_1d():
    rows = _prows([
        # two true-hit rows on one state (module overlap): d = 4.0 and d = 1.0
        (0, 0, 0, 29, HALF_PI, 2, 11, 0.8, np.nan, 0.04, np.nan, True, True, [7], 7, False, True, 1.0),
        (0, 0, 0, 29, HALF_PI, 2, 12, 0.2, np.nan, 0.04, np.nan, True, False, [7], 7, False, True, 1.0),
    ])
    st = build_state_table(rows)
    assert len(st) == 1
    assert np.isclose(st.iloc[0].d_true, 1.0)   # min wins; 1D uses l0 leg only
    assert st.iloc[0].n_true_rows == 2


def test_accumulate_event_window_and_module_counts():
    st = pd.DataFrame({
        "seed_id": [0, 1, 2], "branch_id": [0, 0, 0], "step_k": [0, 0, 0],
        "volume_id": [17, 17, 17], "state_theta": [HALF_PI] * 3,
        "n_window": [3, 1, 0], "branch_majority_pid": [7, 9, 4],
        "on_surface": [True, True, False],
        "d_true": [4.0, np.nan, np.nan], "n_true_rows": [1, 0, 0],
        "is_pure": [True, True, True],
        "survived_ambi": [True, False, True],
    })
    pt_lut = pd.DataFrame({"particle_id": [7, 9, 4], "pt_gev": [2.0, 2.0, 2.0]})
    out = accumulate_event(st, pt_lut)
    hi = 3  # pt bin [1.0, inf)
    assert out["mod_total"][:, :, :, :, hi, :].sum() == 3
    assert out["mod_fail"][:, :, :, :, hi, :].sum() == 1
    assert out["win_total"][:, :, :, :, hi, :].sum() == 2   # on-surface states
    # ambi axis: state A is on a surviving branch, state B is not
    assert out["win_total"][:, :, :, :, hi, 1].sum() == 1
    assert out["win_total"][:, :, :, :, hi, 0].sum() == 1
    # state A (d=4.0): fails n=3 only. state B (escaped): fails ALL n incl 10.
    n_idx = {n: i for i, n in enumerate((3.0, 5.0, 7.0, 10.0))}
    assert out["win_fail"][n_idx[3.0]].sum() == 2
    assert out["win_fail"][n_idx[5.0]].sum() == 1
    assert out["win_fail"][n_idx[10.0]].sum() == 1
    assert out["counters"]["n_escaped"] == 1


def test_accumulate_event_pt_binning_and_unmatched():
    st = pd.DataFrame({
        "seed_id": [0, 1], "branch_id": [0, 0], "step_k": [0, 0],
        "volume_id": [17, 17], "state_theta": [HALF_PI] * 2,
        "n_window": [1, 1], "branch_majority_pid": [7, 8],
        "on_surface": [True, True], "d_true": [1.0, 1.0],
        "n_true_rows": [1, 1], "is_pure": [False, False],
        "survived_ambi": [True, True],
    })
    pt_lut = pd.DataFrame({"particle_id": [7], "pt_gev": [0.75]})  # 8 unmatched
    out = accumulate_event(st, pt_lut)
    assert out["mod_total"][:, :, :, :, 1, :].sum() == 1   # [0.7, 0.9)
    assert out["mod_total"][:, :, :, :, 0, :].sum() == 1   # unmatched -> bin 0
    assert out["counters"]["n_pt_unmatched"] == 1


# Tests for Task 5: contrib_pids null normalization, earliest-simhit helper

def test_as_pid_list_none_and_nan_are_empty():
    assert _as_pid_list(None) == []
    assert _as_pid_list(float("nan")) == []
    assert _as_pid_list([7, 8]) == [7, 8]


def test_build_state_table_float_nan_contrib_is_not_a_crash():
    # Parquet null lists can surface to pandas as a bare float NaN, not
    # None. Must be treated as "no contributors" like None is.
    rows = _prows([
        (0, 0, 0, 17, HALF_PI, 3, -1, np.nan, np.nan, np.nan, np.nan, False,
         False, float("nan"), 7, False, False, 1.0),
    ])
    st = build_state_table(rows)
    assert len(st) == 1
    assert not st.iloc[0].on_surface
    assert np.isnan(st.iloc[0].d_true)


def test_branch_purity_float_nan_contrib_counts_as_not_matching():
    rows = _rows([
        (0, 0, 0, 11, True, float("nan"), 7),
        (0, 0, 1, 12, True, [7], 7),
        (0, 0, 2, 13, True, [7], 7),
    ])
    sel = select_ckf_branch(rows)
    pur = branch_purity(rows, sel)
    assert dict(zip(pur.seed_id, pur.is_pure)) == {0: False}


def test_pick_earliest_by_tt_prefers_minimum_tt():
    df = pd.DataFrame({
        "particle_id": [1, 1, 2],
        "pt_gev": [5.0, 1.0, 9.0],
        "tt": [3.0, 1.0, 2.0],
    })
    out = _pick_earliest_by_tt(df)
    got = dict(zip(out.particle_id, out.pt_gev))
    assert got[1] == 1.0  # tt=1.0 beats tt=3.0
    assert got[2] == 9.0


def test_pick_earliest_by_tt_falls_back_to_row_order_when_tt_missing():
    df = pd.DataFrame({
        "particle_id": [3, 3, 3],
        "pt_gev": [7.0, 8.0, 9.0],
        "tt": [np.nan, np.nan, np.nan],
    })
    out = _pick_earliest_by_tt(df)
    assert dict(zip(out.particle_id, out.pt_gev)) == {3: 7.0}


def test_pick_earliest_by_tt_partial_nan_prefers_finite_tt():
    df = pd.DataFrame({
        "particle_id": [4, 4],
        "pt_gev": [1.0, 2.0],
        "tt": [np.nan, 5.0],
    })
    out = _pick_earliest_by_tt(df)
    assert dict(zip(out.particle_id, out.pt_gev)) == {4: 2.0}
