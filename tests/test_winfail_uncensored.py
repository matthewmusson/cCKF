"""tests/test_winfail_uncensored.py"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winfail_uncensored import (
    ETA_BINS,
    N_VALUES,
    OCC_EDGES,
    PT_EDGES,
    CONFIG_EMULATIONS,
    wilson_interval,
    assign_strata,
    select_ckf_branch,
    branch_purity,
    flag_ambi_survivors,
    emulate_config,
    build_state_table,
    accumulate_event,
    _as_pid_list,
    _pick_earliest_by_tt,
    particle_pt_lookup,
)
import expansion


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
    cols = [
        "seed_id",
        "branch_id",
        "step_k",
        "cand_hit_id",
        "is_ckf_selected",
        "contrib_pids",
        "branch_majority_pid",
    ]
    return pd.DataFrame(records, columns=cols)


def test_select_ckf_branch_picks_most_selected_rows():
    rows = _rows(
        [
            (0, 0, 0, 11, True, [7], 7),
            (0, 0, 1, 12, True, [7], 7),
            (0, 1, 0, 11, True, [7], 7),
            (1, 5, 0, 21, True, [9], 9),
        ]
    )
    sel = select_ckf_branch(rows)
    assert dict(zip(sel.seed_id, sel.branch_id)) == {0: 0, 1: 5}


def test_select_ckf_branch_tie_breaks_to_lowest_branch():
    rows = _rows(
        [
            (3, 2, 0, 1, True, [1], 1),
            (3, 1, 0, 1, True, [1], 1),
        ]
    )
    sel = select_ckf_branch(rows)
    assert dict(zip(sel.seed_id, sel.branch_id)) == {3: 1}


def test_branch_purity_three_of_three_is_pure():
    rows = _rows(
        [
            (0, 0, 0, 11, True, [7], 7),
            (0, 0, 1, 12, True, [7, 8], 7),
            (0, 0, 2, 13, True, [7], 7),
            (0, 0, 3, 14, True, [8], 7),  # step 4 wrong: irrelevant to purity
            (1, 0, 0, 21, True, [9], 9),
            (1, 0, 1, 22, True, [4], 9),  # 2/3 -> majority
            (1, 0, 2, 23, True, [9], 9),
        ]
    )
    sel = select_ckf_branch(rows)
    pur = branch_purity(rows, sel)
    assert dict(zip(pur.seed_id, pur.is_pure)) == {0: True, 1: False}


def _arows(records):
    cols = ["seed_id", "step_k", "cand_hit_id", "is_ckf_selected", "chi2_inc"]
    return pd.DataFrame(records, columns=cols)


def test_flag_ambi_survivors_short_branch_dropped():
    rows = _arows(
        [(0, k, 100 + k, True, 1.0) for k in range(7)]
        + [(1, k, 200 + k, True, 1.0) for k in range(4)]
    )
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
    rows = _arows(
        [(0, k, 100 + k, True, 1.0) for k in range(8)]
        + [(1, k, 500 + k, True, 1.0) for k in range(8)]
    )
    out = flag_ambi_survivors(rows)
    assert dict(zip(out.seed_id, out.survived_ambi)) == {0: True, 1: True}


# Tests for Task 4: build_state_table, accumulate_event


def _prows(records):
    cols = [
        "seed_id",
        "branch_id",
        "step_k",
        "volume_id",
        "state_theta",
        "n_window",
        "cand_hit_id",
        "residual_l0",
        "residual_l1",
        "S00",
        "S11",
        "is_1d",
        "is_ckf_selected",
        "contrib_pids",
        "branch_majority_pid",
        "majority_undefined",
        "majority_true_hit_on_surface",
        "chi2_inc",
    ]
    return pd.DataFrame(records, columns=cols)


HALF_PI = float(np.pi / 2)  # eta = 0


def test_build_state_table_taxonomy():
    rows = _prows(
        [
            # state A: true-hit row at d = 4.0  (r0=0.8, S00=0.04; r1 small)
            (
                0,
                0,
                0,
                17,
                HALF_PI,
                3,
                11,
                0.8,
                0.02,
                0.04,
                0.04,
                False,
                True,
                [7],
                7,
                False,
                True,
                1.0,
            ),
            # ...and a wrong-hit row on the same state (must not affect d_true)
            (
                0,
                0,
                0,
                17,
                HALF_PI,
                3,
                12,
                0.1,
                0.01,
                0.04,
                0.04,
                False,
                False,
                [8],
                7,
                False,
                True,
                1.0,
            ),
            # state B: on surface, NO true-hit row -> escaped (d_true NaN)
            (
                1,
                0,
                0,
                17,
                HALF_PI,
                1,
                13,
                0.1,
                0.01,
                0.04,
                0.04,
                False,
                True,
                [5],
                9,
                False,
                True,
                1.0,
            ),
            # state C: hole row, majority not on surface -> module failure
            (
                2,
                0,
                0,
                17,
                HALF_PI,
                0,
                -1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                False,
                False,
                None,
                4,
                False,
                False,
                1.0,
            ),
        ]
    )
    st = build_state_table(rows)
    assert len(st) == 3
    a = st[st.seed_id == 0].iloc[0]
    assert np.isclose(a.d_true, 4.0)
    b = st[st.seed_id == 1].iloc[0]
    assert b.on_surface and np.isnan(b.d_true)
    c = st[st.seed_id == 2].iloc[0]
    assert not c.on_surface


def test_build_state_table_min_over_two_true_rows_and_1d():
    rows = _prows(
        [
            # two true-hit rows on one state (module overlap): d = 4.0 and d = 1.0
            (
                0,
                0,
                0,
                29,
                HALF_PI,
                2,
                11,
                0.8,
                np.nan,
                0.04,
                np.nan,
                True,
                True,
                [7],
                7,
                False,
                True,
                1.0,
            ),
            (
                0,
                0,
                0,
                29,
                HALF_PI,
                2,
                12,
                0.2,
                np.nan,
                0.04,
                np.nan,
                True,
                False,
                [7],
                7,
                False,
                True,
                1.0,
            ),
        ]
    )
    st = build_state_table(rows)
    assert len(st) == 1
    assert np.isclose(st.iloc[0].d_true, 1.0)  # min wins; 1D uses l0 leg only
    assert st.iloc[0].n_true_rows == 2


def test_accumulate_event_window_and_module_counts():
    st = pd.DataFrame(
        {
            "seed_id": [0, 1, 2],
            "branch_id": [0, 0, 0],
            "step_k": [0, 0, 0],
            "volume_id": [17, 17, 17],
            "state_theta": [HALF_PI] * 3,
            "n_window": [3, 1, 0],
            "branch_majority_pid": [7, 9, 4],
            "on_surface": [True, True, False],
            "d_true": [4.0, np.nan, np.nan],
            "n_true_rows": [1, 0, 0],
            "is_pure": [True, True, True],
            "survived_ambi": [True, False, True],
        }
    )
    pt_lut = pd.DataFrame({"particle_id": [7, 9, 4], "pt_gev": [2.0, 2.0, 2.0]})
    out = accumulate_event(st, pt_lut)
    hi = 3  # pt bin [1.0, inf)
    assert out["mod_total"][:, :, :, :, hi, :].sum() == 3
    assert out["mod_fail"][:, :, :, :, hi, :].sum() == 1
    assert out["win_total"][:, :, :, :, hi, :].sum() == 2  # on-surface states
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
    st = pd.DataFrame(
        {
            "seed_id": [0, 1],
            "branch_id": [0, 0],
            "step_k": [0, 0],
            "volume_id": [17, 17],
            "state_theta": [HALF_PI] * 2,
            "n_window": [1, 1],
            "branch_majority_pid": [7, 8],
            "on_surface": [True, True],
            "d_true": [1.0, 1.0],
            "n_true_rows": [1, 1],
            "is_pure": [False, False],
            "survived_ambi": [True, True],
        }
    )
    pt_lut = pd.DataFrame({"particle_id": [7], "pt_gev": [0.75]})  # 8 unmatched
    out = accumulate_event(st, pt_lut)
    assert out["mod_total"][:, :, :, :, 1, :].sum() == 1  # [0.7, 0.9)
    assert out["mod_total"][:, :, :, :, 0, :].sum() == 1  # unmatched -> bin 0
    assert out["counters"]["n_pt_unmatched"] == 1


# Tests for Task 5: contrib_pids null normalization, earliest-simhit helper


def test_as_pid_list_none_and_nan_are_empty():
    assert _as_pid_list(None) == []
    assert _as_pid_list(float("nan")) == []
    assert _as_pid_list([7, 8]) == [7, 8]


def test_build_state_table_float_nan_contrib_is_not_a_crash():
    # Parquet null lists can surface to pandas as a bare float NaN, not
    # None. Must be treated as "no contributors" like None is.
    rows = _prows(
        [
            (
                0,
                0,
                0,
                17,
                HALF_PI,
                3,
                -1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                False,
                False,
                float("nan"),
                7,
                False,
                False,
                1.0,
            ),
        ]
    )
    st = build_state_table(rows)
    assert len(st) == 1
    assert not st.iloc[0].on_surface
    assert np.isnan(st.iloc[0].d_true)


def test_branch_purity_float_nan_contrib_counts_as_not_matching():
    rows = _rows(
        [
            (0, 0, 0, 11, True, float("nan"), 7),
            (0, 0, 1, 12, True, [7], 7),
            (0, 0, 2, 13, True, [7], 7),
        ]
    )
    sel = select_ckf_branch(rows)
    pur = branch_purity(rows, sel)
    assert dict(zip(pur.seed_id, pur.is_pure)) == {0: False}


def test_pick_earliest_by_tt_prefers_minimum_tt():
    df = pd.DataFrame(
        {
            "particle_id": [1, 1, 2],
            "pt_gev": [5.0, 1.0, 9.0],
            "tt": [3.0, 1.0, 2.0],
        }
    )
    out = _pick_earliest_by_tt(df)
    got = dict(zip(out.particle_id, out.pt_gev))
    assert got[1] == 1.0  # tt=1.0 beats tt=3.0
    assert got[2] == 9.0


def test_pick_earliest_by_tt_falls_back_to_row_order_when_tt_missing():
    df = pd.DataFrame(
        {
            "particle_id": [3, 3, 3],
            "pt_gev": [7.0, 8.0, 9.0],
            "tt": [np.nan, np.nan, np.nan],
        }
    )
    out = _pick_earliest_by_tt(df)
    assert dict(zip(out.particle_id, out.pt_gev)) == {3: 7.0}


def test_pick_earliest_by_tt_partial_nan_prefers_finite_tt():
    df = pd.DataFrame(
        {
            "particle_id": [4, 4],
            "pt_gev": [1.0, 2.0],
            "tt": [np.nan, 5.0],
        }
    )
    out = _pick_earliest_by_tt(df)
    assert dict(zip(out.particle_id, out.pt_gev)) == {4: 2.0}


def test_particle_pt_lookup_end_to_end_with_leading_comment_line(tmp_path):
    # Regression test: the raw CSV read must pass comment="#" like
    # expansion.load_simhits does for the same file, or a leading comment
    # line (present on real ACTS/edm4hep exports) desyncs row position
    # (hit_id) between the two reads.
    csv_dir = tmp_path
    event_id = 7
    path = csv_dir / f"event{event_id:09d}-simhits.csv"
    path.write_text(
        "# metadata header line, must be stripped like load_simhits does\n"
        "particle_id_pv,particle_id_sv,particle_id_part,particle_id_gen,"
        "particle_id_subpart,geometry_id,tx,ty,tz,tpx,tpy,tpz,tt\n"
        "1,0,1,1,0,111,0.0,0.0,0.0,3.0,4.0,0.0,5.0\n"  # particle 1, later hit
        "1,0,1,1,0,111,0.0,0.0,0.0,1.0,1.0,0.0,1.0\n"  # particle 1, earliest
        "2,0,1,1,0,111,0.0,0.0,0.0,0.0,5.0,0.0,2.0\n"  # particle 2
    )
    out = particle_pt_lookup(str(csv_dir), event_id)
    assert list(out.columns) == ["particle_id", "pt_gev"]

    p1 = expansion.encode_particle_id(
        np.array([1]), np.array([0]), np.array([1]), np.array([1]), np.array([0])
    )[0]
    p2 = expansion.encode_particle_id(
        np.array([2]), np.array([0]), np.array([1]), np.array([1]), np.array([0])
    )[0]
    got = dict(zip(out.particle_id, out.pt_gev))
    assert np.isclose(got[p1], np.hypot(1.0, 1.0))  # earliest (tt=1) hit wins
    assert np.isclose(got[p2], 5.0)


def test_render_all_produces_expected_files(tmp_path):
    import plot_winfail_uncensored as P

    shape = (140, 3, 2, 5, 4, 2)
    rng = np.random.default_rng(0)
    total = rng.integers(50, 100, size=shape)
    np.savez(
        tmp_path / "winfail_unc_event000.npz",
        eta_bins=np.linspace(-3.5, 3.5, 141),
        n_values=np.array([3.0, 5.0, 7.0, 10.0]),
        occ_edges=np.array([0.0, 2.0, 5.0, 10.0, 20.0]),
        pt_edges=np.array([0.0, 0.7, 0.9, 1.0]),
        mod_total=total,
        mod_fail=total // 10,
        win_total=total,
        win_fail=np.stack([total // (i + 2) for i in range(4)]),
        counter_n_states=np.array(1),
        counter_n_vol20=np.array(0),
        counter_n_escaped=np.array(0),
        counter_n_multi_true=np.array(0),
        counter_n_pt_unmatched=np.array(0),
    )
    made = P.render_all(
        str(tmp_path), str(tmp_path / "out"), threshold_gev=1.0, branch_class="ambi"
    )
    names = {p.split("/")[-1] for p in made}
    assert names == {
        "winfail_vs_eta_pure.png",
        "winfail_vs_eta_majority.png",
        "winfail_vs_eta_occupancy_n3.png",
        "winfail_vs_eta_occupancy_n5.png",
        "winfail_vs_eta_occupancy_n7.png",
        "winfail_vs_eta_occupancy_n10.png",
        "modfail_vs_eta.png",
        "modfail_vs_occupancy.png",
    }


def test_pt_slice_rejects_non_edge_threshold(tmp_path):
    import plot_winfail_uncensored as P
    import pytest

    with pytest.raises(ValueError):
        P.pt_slice(np.zeros((2, 4)), 0.8)


# Tests for emulate_config (tight/fast config-emulation branch filters).
# emulate_config truncates at the running EMULATED hole cap (branchStopper
# semantics: stop the branch, keep the accepted prefix) rather than
# discarding the whole branch -- see its docstring. An emulated hole is a
# `states` row with no matching `is_ckf_selected` `cand` row AND a sensor
# volume_id (one of SENSOR_VOLUMES): the parquet's own n_holes column also
# counts passive-material crossings (e.g. volume 20) and is deliberately
# not read for this. cutoff_step is np.inf when the branch never crosses
# max_holes, so its prefix is the whole branch.

PIXEL_VOL = 17  # a SENSOR_VOLUMES key (pixel)
PASSIVE_VOL = 20  # not in SENSOR_VOLUMES -- never a hole, never a hit


def _erows(records):
    cols = ["seed_id", "step_k", "cand_hit_id", "is_ckf_selected", "chi2_inc"]
    return pd.DataFrame(records, columns=cols)


def _estates(records):
    cols = ["seed_id", "step_k", "state_theta", "volume_id", "state_qop"]
    return pd.DataFrame(records, columns=cols)


def test_config_emulations_has_tight_and_fast_exact_values():
    assert set(CONFIG_EMULATIONS) == {"tight", "fast"}
    assert CONFIG_EMULATIONS["tight"]["nmeas_min"] == 9
    assert CONFIG_EMULATIONS["fast"]["nmeas_min"] == 8


def test_emulate_config_passing_branch():
    # 9 selected rows (meets tight nmeas_min=9), all low chi2, no holes
    # (every state has a matching selected cand row, so cutoff stays inf),
    # pT = |1/1.0| * sin(pi/2) = 1.0 GeV > 0.4598 GeV.
    cand = _erows([(0, k, 100 + k, True, 1.0) for k in range(9)])
    states = _estates([(0, k, HALF_PI, PIXEL_VOL, 1.0) for k in range(9)])
    out = emulate_config(cand, states, "tight")
    assert dict(zip(out.seed_id, out.passes_emulation)) == {0: True}
    assert dict(zip(out.seed_id, out.cutoff_step)) == {0: np.inf}


def test_emulate_config_fails_chi2_ceiling():
    # 9 selected rows (nmeas_min ok), but one row's chi2_inc = 16.5 exceeds
    # tight's 16.255929655134203 ceiling. No holes -> prefix = whole branch.
    cand = _erows(
        [(0, k, 100 + k, True, 1.0) for k in range(8)] + [(0, 8, 108, True, 16.5)]
    )
    states = _estates([(0, k, HALF_PI, PIXEL_VOL, 1.0) for k in range(9)])
    out = emulate_config(cand, states, "tight")
    assert dict(zip(out.seed_id, out.passes_emulation)) == {0: False}


def test_emulate_config_fails_nmeas_min():
    cand = _erows([(0, k, 100 + k, True, 1.0) for k in range(5)])
    states = _estates([(0, k, HALF_PI, PIXEL_VOL, 1.0) for k in range(5)])
    out = emulate_config(cand, states, "tight")
    assert dict(zip(out.seed_id, out.passes_emulation)) == {0: False}


def test_emulate_config_trailing_holes_truncate_and_pass():
    # 10 selected hits (steps 0-9), then 2 trailing sensor-volume states
    # with no selected cand row (steps 10-11): the emulated-hole cumsum
    # crosses tight's max_holes=1 at step 11 -> cutoff_step=11, and the
    # prefix (step_k < 11) keeps all 10 hits, well over nmeas_min=9.
    # Matches ACTS branchStopper: the branch is stopped and its accepted
    # prefix survives, it isn't discarded outright.
    cand = _erows([(0, k, 100 + k, True, 1.0) for k in range(10)])
    states = _estates(
        [(0, k, HALF_PI, PIXEL_VOL, 1.0) for k in range(10)]
        + [(0, 10, HALF_PI, PIXEL_VOL, 1.0)]  # hole #1: no selected cand row
        + [(0, 11, HALF_PI, PIXEL_VOL, 1.0)]  # hole #2: cumsum=2 -> cutoff here
    )
    out = emulate_config(cand, states, "tight")
    assert dict(zip(out.seed_id, out.passes_emulation)) == {0: True}
    assert dict(zip(out.seed_id, out.cutoff_step)) == {0: 11.0}


def test_emulate_config_early_holes_truncate_and_fail_nmeas():
    # Two sensor-volume states with no selected cand row at steps 2-3: the
    # emulated-hole cumsum crosses max_holes=1 at step 3 -> cutoff_step=3.
    # Only the 2 selected hits at steps 0-1 precede the cutoff; more hits
    # follow step 3 but are truncated away, so the prefix's accepted-hit
    # count (2) is well under nmeas_min=9 and the branch fails.
    cand = _erows(
        [(0, k, 100 + k, True, 1.0) for k in range(2)]  # steps 0-1, kept
        + [(0, k, 100 + k, True, 1.0) for k in range(4, 13)]  # steps 4-12, truncated
    )
    states = _estates(
        [
            (0, 0, HALF_PI, PIXEL_VOL, 1.0),
            (0, 1, HALF_PI, PIXEL_VOL, 1.0),
            (0, 2, HALF_PI, PIXEL_VOL, 1.0),  # hole #1: no selected cand row
            (0, 3, HALF_PI, PIXEL_VOL, 1.0),  # hole #2: cumsum=2 -> cutoff here
        ]
    )
    out = emulate_config(cand, states, "tight")
    assert dict(zip(out.seed_id, out.passes_emulation)) == {0: False}
    assert dict(zip(out.seed_id, out.cutoff_step)) == {0: 3.0}


def test_emulate_config_passive_volume_state_is_not_a_hole():
    # A volume-20 (passive material) state with no selected cand row must
    # NOT count as an emulated hole -- only tracker sensor volumes do. 9
    # selected hits (nmeas_min met) with a passive crossing between steps
    # 4 and 6; asserting cutoff_step stays inf (not just "under the cap")
    # pins down that it isn't a hole at all.
    hit_steps = [0, 1, 2, 3, 4, 6, 7, 8, 9]  # 9 hits, skipping step 5
    cand = _erows([(0, k, 100 + k, True, 1.0) for k in hit_steps])
    states = _estates(
        [(0, k, HALF_PI, PIXEL_VOL, 1.0) for k in hit_steps]
        + [(0, 5, HALF_PI, PASSIVE_VOL, 1.0)]  # passive crossing between hits
    )
    out = emulate_config(cand, states, "tight")
    assert dict(zip(out.seed_id, out.cutoff_step)) == {0: np.inf}
    assert dict(zip(out.seed_id, out.passes_emulation)) == {0: True}


def test_emulate_config_fails_pt():
    # qop = 5.0 -> p = 0.2 GeV -> pT = 0.2 GeV, below tight's 0.4598 GeV.
    cand = _erows([(0, k, 100 + k, True, 1.0) for k in range(9)])
    states = _estates([(0, 0, HALF_PI, PIXEL_VOL, 5.0)])
    out = emulate_config(cand, states, "tight")
    assert dict(zip(out.seed_id, out.passes_emulation)) == {0: False}


def test_emulate_config_zero_selected_rows_fails():
    cand = _erows([(0, k, 100 + k, False, 1.0) for k in range(9)])
    states = _estates([(0, 0, HALF_PI, PIXEL_VOL, 1.0)])
    out = emulate_config(cand, states, "tight")
    assert dict(zip(out.seed_id, out.passes_emulation)) == {0: False}


def test_emulate_config_fast_ceiling_stricter_than_tight():
    # max chi2 = 16.0: under tight's ceiling (16.2559...) but over fast's
    # (15.4010...). nmeas_min=9 satisfies both tight (9) and fast (8).
    cand = _erows(
        [(0, k, 100 + k, True, 1.0) for k in range(8)] + [(0, 8, 108, True, 16.0)]
    )
    states = _estates([(0, k, HALF_PI, PIXEL_VOL, 1.0) for k in range(9)])
    tight = emulate_config(cand, states, "tight")
    fast = emulate_config(cand, states, "fast")
    assert dict(zip(tight.seed_id, tight.passes_emulation)) == {0: True}
    assert dict(zip(fast.seed_id, fast.passes_emulation)) == {0: False}


def test_emulate_config_multi_branch_alignment():
    """Four branches, each failing a different cut (or none), built with a
    non-sorted seed_id order and rows interleaved across seeds -- pins the
    per-cut boolean series (chi2/nmeas/pT) and cutoff_step to seed_id-label
    alignment (pandas reindex/map) rather than positional/sorted order,
    which a single-seed test cannot catch.

    seed 0: passes. seed 1: fails chi2. seed 2: early holes (cutoff at
    step 3) truncate its prefix to too few hits, failing nmeas_min. seed
    3: fails pT.
    """

    def sel_rows(seed, steps, bad_chi2_at=None, bad_chi2=1.0):
        rows = []
        for k in steps:
            chi2 = bad_chi2 if k == bad_chi2_at else 1.0
            rows.append((seed, k, seed * 100 + k, True, chi2))
        return rows

    r1 = sel_rows(1, range(9), bad_chi2_at=8, bad_chi2=16.5)  # fails chi2 only
    r3 = sel_rows(3, range(9))  # fails pT only (via states below)
    r0 = sel_rows(0, range(9))  # passes everything
    r2 = sel_rows(2, [0, 1])  # 2 early hits, kept in the prefix
    r2_late = sel_rows(2, range(4, 13))  # more hits after the cutoff, truncated away

    # Interleave in a non-ascending, non-seed-grouped order.
    interleaved = []
    for a, b, c in zip(r1, r3, r0):
        interleaved += [a, b, c]
    interleaved += r2 + r2_late
    cand = _erows(interleaved)

    states = _estates(
        [
            (2, 0, HALF_PI, PIXEL_VOL, 1.0),  # seed 2 first state (for pT)
            (2, 2, HALF_PI, PIXEL_VOL, 1.0),  # seed 2 hole #1: no selected row
            (2, 3, HALF_PI, PIXEL_VOL, 1.0),  # seed 2 hole #2: cumsum=2 -> cutoff
            (0, 0, HALF_PI, PIXEL_VOL, 1.0),  # seed 0: passes
            (3, 0, HALF_PI, PIXEL_VOL, 5.0),  # seed 3: qop=5 -> pT=0.2 GeV, fails
            (1, 0, HALF_PI, PIXEL_VOL, 1.0),  # seed 1: fine holes/pT; chi2 via cand
        ]
    )
    out = emulate_config(cand, states, "tight")
    assert dict(zip(out.seed_id, out.passes_emulation)) == {
        0: True,
        1: False,
        2: False,
        3: False,
    }
    assert dict(zip(out.seed_id, out.cutoff_step)) == {
        0: np.inf,
        1: np.inf,
        2: 3.0,
        3: np.inf,
    }


def test_emulate_config_qop_zero_fails():
    # Explicit spec clause: qop == 0 must fail (not just non-finite pT).
    cand = _erows([(0, k, 100 + k, True, 1.0) for k in range(9)])
    states = _estates([(0, 0, HALF_PI, PIXEL_VOL, 0.0)])
    out = emulate_config(cand, states, "tight")
    assert dict(zip(out.seed_id, out.passes_emulation)) == {0: False}


def test_emulate_config_prefix_filtering_drops_post_cutoff_rows():
    """Directly exercises the filter main() applies: states/cand restricted
    to (passing seed) AND (step_k < that seed's cutoff_step). Confirms the
    post-cutoff hole row is dropped while the whole pre-cutoff prefix
    (through the last hit before the cutoff) is kept.
    """
    cand = _erows([(0, k, 100 + k, True, 1.0) for k in range(10)])
    states = _estates(
        [
            (0, 0, HALF_PI, PIXEL_VOL, 1.0),
            (0, 5, HALF_PI, PIXEL_VOL, 1.0),
            (0, 9, HALF_PI, PIXEL_VOL, 1.0),
            (0, 10, HALF_PI, PIXEL_VOL, 1.0),  # hole #1: no selected cand row
            (0, 11, HALF_PI, PIXEL_VOL, 1.0),  # hole #2: cumsum=2 -> cutoff here
        ]
    )
    emu = emulate_config(cand, states, "tight")
    assert dict(zip(emu.seed_id, emu.cutoff_step)) == {0: 11.0}
    assert dict(zip(emu.seed_id, emu.passes_emulation)) == {0: True}

    # Mirror main()'s wiring exactly.
    passing = set(emu.loc[emu["passes_emulation"], "seed_id"])
    cutoff_map = emu.set_index("seed_id")["cutoff_step"]
    kept_states = states[
        states["seed_id"].isin(passing)
        & (states["step_k"] < states["seed_id"].map(cutoff_map))
    ]
    kept_cand = cand[
        cand["seed_id"].isin(passing)
        & (cand["step_k"] < cand["seed_id"].map(cutoff_map))
    ]
    assert sorted(kept_states["step_k"]) == [0, 5, 9, 10]  # step 11 dropped
    assert len(kept_cand) == 10  # all 10 selected hits (steps 0-9) kept


def test_render_all_extra_footer_no_exception(tmp_path):
    import plot_winfail_uncensored as P

    shape = (140, 3, 2, 5, 4, 2)
    rng = np.random.default_rng(1)
    total = rng.integers(50, 100, size=shape)
    np.savez(
        tmp_path / "winfail_unc_event000.npz",
        eta_bins=np.linspace(-3.5, 3.5, 141),
        n_values=np.array([3.0, 5.0, 7.0, 10.0]),
        occ_edges=np.array([0.0, 2.0, 5.0, 10.0, 20.0]),
        pt_edges=np.array([0.0, 0.7, 0.9, 1.0]),
        mod_total=total,
        mod_fail=total // 10,
        win_total=total,
        win_fail=np.stack([total // (i + 2) for i in range(4)]),
        counter_n_states=np.array(1),
        counter_n_vol20=np.array(0),
        counter_n_escaped=np.array(0),
        counter_n_multi_true=np.array(0),
        counter_n_pt_unmatched=np.array(0),
    )
    made = P.render_all(
        str(tmp_path),
        str(tmp_path / "out"),
        threshold_gev=1.0,
        branch_class="ambi",
        extra_footer="tight emulation (chi2<16.26, nmeas>=9, holes<=1, pT>0.46 GeV)",
    )
    names = {p.split("/")[-1] for p in made}
    assert names == {
        "winfail_vs_eta_pure.png",
        "winfail_vs_eta_majority.png",
        "winfail_vs_eta_occupancy_n3.png",
        "winfail_vs_eta_occupancy_n5.png",
        "winfail_vs_eta_occupancy_n7.png",
        "winfail_vs_eta_occupancy_n10.png",
        "modfail_vs_eta.png",
        "modfail_vs_occupancy.png",
    }
