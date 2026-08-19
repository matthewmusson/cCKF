"""Tests for recovering the CKF-selected candidate per state."""

from __future__ import annotations

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from scripts.patch_is_selected import match_selected


def _cands() -> pd.DataFrame:
    # Two states. State (0,1001) has 3 candidates; state (0,1002) has 2.
    return pd.DataFrame(
        {
            "seed_id": [0, 0, 0, 0, 0],
            "geometry_id": [1001, 1001, 1001, 1002, 1002],
            "cand_hit_id": [10, 11, 12, 20, 21],
            "residual_l0": [0.10, -0.30, 0.90, 0.05, 2.00],
            "residual_l1": [0.20, 0.40, -0.10, 0.15, 1.00],
        }
    )


def _root() -> pd.DataFrame:
    # ROOT says state (0,1001) selected the candidate with residual
    # (-0.30, 0.40) and state (0,1002) selected residual (0.05, 0.15).
    return pd.DataFrame(
        {
            "seed_id": [0, 0],
            "geometry_id": [1001, 1002],
            "res_l0": [-0.30, 0.05],
            "res_l1": [0.40, 0.15],
        }
    )


def test_match_selected_picks_exactly_one_per_state():
    sel = match_selected(_cands(), _root())
    assert sel.tolist() == [False, True, False, True, False]


def test_match_selected_one_flag_per_state_group():
    cands = _cands()
    cands["is_sel"] = match_selected(cands, _root())
    per_state = cands.groupby(["seed_id", "geometry_id"])["is_sel"].sum()
    assert (per_state == 1).all()


def test_match_selected_marks_none_when_root_has_no_state():
    """A state with no ROOT entry (e.g. a hole) gets no selected candidate."""
    root = _root().iloc[:1]  # drop state (0,1002)
    sel = match_selected(_cands(), root)
    assert sel.tolist() == [False, True, False, False, False]


def test_match_selected_respects_tolerance():
    root = _root().copy()
    root.loc[0, "res_l0"] = -0.30 + 1e-2  # outside a 1e-4 tolerance
    sel = match_selected(_cands(), root, tol=1e-4)
    # State (0,1001) now matches nothing; state (0,1002) still matches.
    assert sel.tolist() == [False, False, False, True, False]


def test_match_selected_breaks_ties_deterministically():
    """Two identical residuals in one state: flag the lower cand_hit_id only."""
    cands = pd.DataFrame(
        {
            "seed_id": [0, 0],
            "geometry_id": [1001, 1001],
            "cand_hit_id": [11, 10],
            "residual_l0": [0.5, 0.5],
            "residual_l1": [0.5, 0.5],
        }
    )
    root = pd.DataFrame(
        {
            "seed_id": [0],
            "geometry_id": [1001],
            "res_l0": [0.5],
            "res_l1": [0.5],
        }
    )
    sel = match_selected(cands, root)
    assert sel.sum() == 1
    assert cands.loc[sel, "cand_hit_id"].iloc[0] == 10


# --- Primary route: exact contributor join --------------------------------


def test_expansion_barcode_packing_is_the_documented_layout():
    """Guards the shared encoding. `patch_is_selected` imports this rather than
    reimplementing it, so if expansion ever changes the packing this fails
    loudly instead of silently breaking the majority-membership test."""
    from expansion import encode_particle_id

    # (pv << 48) | (sv << 32) | (part << 16) | (gen << 8) | sub
    out = encode_particle_id(
        np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5])
    )
    assert out[0] == (1 << 48) | (2 << 32) | (3 << 16) | (4 << 8) | 5


def test_patch_is_selected_reuses_expansion_encoding():
    """No local reimplementation of the barcode packing may exist."""
    import expansion
    import scripts.patch_is_selected as mod

    assert mod.encode_particle_id is expansion.encode_particle_id
    assert not hasattr(mod, "encode_barcode")


def test_expansion_barcode_roundtrips_a_realistic_primary_particle():
    from expansion import encode_particle_id

    code = encode_particle_id(
        np.array([0]), np.array([0]), np.array([213]), np.array([0]), np.array([0])
    )[0]
    assert (code >> 16) & 0xFFFF == 213


def test_selected_correctness_flags_membership_not_primary_contributor():
    """A merged cluster where the majority particle is the MINORITY contributor
    must still count as correct: the spec's test is membership in contrib_pids.
    Using ROOT's state_primary_pid (the mode) would wrongly call this wrong."""
    from scripts.patch_is_selected import selected_correctness

    sel = pd.DataFrame(
        {
            "seed_id": [0],
            "geometry_id": [1001],
            "sel_contrib_pids": [[999, 7]],  # 999 dominates, 7 is the majority pid
            "sel_has_hit": [True],
        }
    )
    majority = pd.DataFrame({"seed_id": [0], "branch_majority_pid": [7]})
    out = selected_correctness(sel, majority)
    assert out["sel_correct"].iloc[0] == 1
    assert out["sel_wrong"].iloc[0] == 0


def test_selected_correctness_flags_wrong_when_majority_absent():
    from scripts.patch_is_selected import selected_correctness

    sel = pd.DataFrame(
        {
            "seed_id": [0],
            "geometry_id": [1001],
            "sel_contrib_pids": [[999, 888]],
            "sel_has_hit": [True],
        }
    )
    majority = pd.DataFrame({"seed_id": [0], "branch_majority_pid": [7]})
    out = selected_correctness(sel, majority)
    assert out["sel_correct"].iloc[0] == 0
    assert out["sel_wrong"].iloc[0] == 1


def test_selected_correctness_counts_a_hole_as_neither():
    """A hole neither helps completeness nor pollutes purity."""
    from scripts.patch_is_selected import selected_correctness

    sel = pd.DataFrame(
        {
            "seed_id": [0],
            "geometry_id": [1001],
            "sel_contrib_pids": [[]],
            "sel_has_hit": [False],
        }
    )
    majority = pd.DataFrame({"seed_id": [0], "branch_majority_pid": [7]})
    out = selected_correctness(sel, majority)
    assert out["sel_correct"].iloc[0] == 0
    assert out["sel_wrong"].iloc[0] == 0


def test_selected_correctness_is_exclusive():
    """No state may be both correct and wrong."""
    from scripts.patch_is_selected import selected_correctness

    sel = pd.DataFrame(
        {
            "seed_id": [0, 0, 0],
            "geometry_id": [1001, 1002, 1003],
            "sel_contrib_pids": [[7], [8], []],
            "sel_has_hit": [True, True, False],
        }
    )
    majority = pd.DataFrame({"seed_id": [0], "branch_majority_pid": [7]})
    out = selected_correctness(sel, majority)
    assert ((out["sel_correct"] + out["sel_wrong"]) <= 1).all()
    assert out["sel_correct"].tolist() == [1, 0, 0]
    assert out["sel_wrong"].tolist() == [0, 1, 0]


# --- Primary route: doubly-jagged ROOT parsing -----------------------------
#
# expansion.py:418-425 establishes that the trackstates tree is one entry
# per TRACK, with per-state values as jagged sublists -- not one entry per
# state, as an earlier draft of this module wrongly assumed. These tests
# build synthetic awkward arrays with that real shape (doubly-jagged for
# particle_ids_*, singly-jagged for the res_eLOC*_prt residuals) and drive
# the pure-parsing helpers directly, so the parsing logic is covered without
# needing a ROOT fixture.


def _synthetic_contributor_arrays() -> ak.Array:
    """Two tracks in event 0, one track in event 1.

    Track 0 (seed_id 0) has 5 all-states: measurement, hole, material, measurement,
    material.  The particle_ids_* are doubly-jagged (track, state, contributor) --
    one entry per all-state (not measurement-only).
    Track 1 (seed_id 1) has 2 all-states. Track 2 is event 1 with 2 all-states.

    volume_id/layer_id/module_id are singly-jagged (track, state), matching
    the same all-state shape as particle_ids_*.  This mirrors the real ROOT
    tree where both groups have the same length (all-state).
    """
    return ak.Array(
        {
            "event_nr": [0, 0, 1],
            # doubly-jagged: (track, state, contributor)
            # Track 0: hit, hole, material, hit+merged, material
            "particle_ids_particle": [
                [[101], [], [], [101, 202], []],
                [[303], []],
                [[], [404]],
            ],
            "particle_ids_vertex_primary": [
                [[0], [], [], [0, 0], []],
                [[0], []],
                [[], [0]],
            ],
            "particle_ids_vertex_secondary": [
                [[0], [], [], [0, 0], []],
                [[0], []],
                [[], [0]],
            ],
            "particle_ids_generation": [
                [[0], [], [], [0, 0], []],
                [[0], []],
                [[], [0]],
            ],
            "particle_ids_sub_particle": [
                [[0], [], [], [0, 0], []],
                [[0], []],
                [[], [0]],
            ],
            # singly-jagged: same all-state shape as particle_ids
            "volume_id": [[16, 16, 16, 17, 17], [16, 16], [18, 18]],
            "layer_id": [[2, 2, 2, 4, 4], [2, 2], [6, 6]],
            "module_id": [[100, 101, 101, 200, 200], [102, 103], [300, 301]],
        }
    )


def test_select_contributors_handles_hole_and_merged_cluster():
    """The exact shape that crashed: a 0-contributor hole and a 2-contributor
    merged cluster in the same track."""
    from expansion import encode_particle_id, encode_geometry_id
    from scripts.patch_is_selected import _select_contributors_from_arrays

    df = _select_contributors_from_arrays(_synthetic_contributor_arrays(), event_id=0)

    # State 1 (vol=16/lay=2/mod=101) is a hole -- empty contributor list.
    hole_gid = encode_geometry_id(np.array([16]), np.array([2]), np.array([101]))[0]
    rows_hole = df[(df["seed_id"] == 0) & (df["geometry_id"] == hole_gid)]
    assert rows_hole.iloc[0]["sel_contrib_pids"] == []

    # State 3 (vol=17/lay=4/mod=200) is a merged cluster -- two contributors.
    merged_gid = encode_geometry_id(np.array([17]), np.array([4]), np.array([200]))[0]
    row_merged = df[(df["seed_id"] == 0) & (df["geometry_id"] == merged_gid)].iloc[0]
    expected = encode_particle_id(
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([101, 202]),
        np.array([0, 0]),
        np.array([0, 0]),
    ).tolist()
    assert row_merged["sel_contrib_pids"] == expected


def test_select_contributors_returns_geometry_id():
    from expansion import encode_geometry_id
    from scripts.patch_is_selected import _select_contributors_from_arrays

    df = _select_contributors_from_arrays(_synthetic_contributor_arrays(), event_id=0)

    track0 = df[df["seed_id"] == 0].sort_values("geometry_id")
    track0_gids = encode_geometry_id(
        np.array([16, 16, 16, 17, 17]),
        np.array([2, 2, 2, 4, 4]),
        np.array([100, 101, 101, 200, 200]),
    )
    np.testing.assert_array_equal(track0["geometry_id"].to_numpy(), track0_gids)

    track1 = df[df["seed_id"] == 1]
    track1_gids = encode_geometry_id(
        np.array([16, 16]),
        np.array([2, 2]),
        np.array([102, 103]),
    )
    np.testing.assert_array_equal(track1["geometry_id"].to_numpy(), track1_gids)


def test_select_contributors_sel_has_hit_false_only_for_empty_contrib_list():
    from scripts.patch_is_selected import _select_contributors_from_arrays

    df = _select_contributors_from_arrays(_synthetic_contributor_arrays(), event_id=0)
    df = df.sort_values(["seed_id", "geometry_id"]).reset_index(drop=True)
    # Track 0: 5 states — hit, hole, material(no), merged(yes), material(no)
    # Track 1: 2 states — hit, material(no)
    assert df["sel_has_hit"].tolist() == [True, False, False, True, False, True, False]


def test_select_contributors_filters_to_the_requested_event():
    """Track 2 (event_nr=1) must not appear when event_id=0 is requested."""
    from scripts.patch_is_selected import _select_contributors_from_arrays

    df = _select_contributors_from_arrays(_synthetic_contributor_arrays(), event_id=0)
    assert set(df["seed_id"].unique()) == {0, 1}
    assert len(df) == 7  # 5 states for track 0 + 2 states for track 1

    df1 = _select_contributors_from_arrays(_synthetic_contributor_arrays(), event_id=1)
    assert len(df1) == 2


# --- Primary route: doubly-jagged residual parsing -------------------------


def _synthetic_residual_arrays() -> ak.Array:
    """Mimic the real ROOT tree's two jagged-length groups.

    The ROOT tree has **all-state** branches (one entry per CKF state --
    measurements, holes, material interactions) and **measurement-only**
    branches (one entry per state that has a hit).

    Track 0: 5 all-states, 2 measurements at positions [0, 3] (the others
             are holes / material interactions with no hit).
    Track 1: 2 all-states, 1 measurement at position [0].
    Track 2: event 1, 2 all-states, 1 measurement at position [1].

    All-state: volume_id, layer_id, module_id, l_x_hit (length 5, 2, 2).
    Measurement-only: res_eLOC0_prt, res_eLOC1_prt (length 2, 1, 1).
    l_x_hit is all-state but finite only at measurement positions.
    """
    return ak.Array(
        {
            "event_nr": [0, 0, 1],
            # Measurement-only (2, 1, 1 per track)
            "res_eLOC0_prt": [[0.10, -0.30], [0.05], [9.9]],
            "res_eLOC1_prt": [[0.20, 0.40], [0.15], [9.9]],
            # All-state (5, 2, 2 per track)
            "volume_id": [[16, 16, 16, 17, 17], [16, 16], [18, 18]],
            "layer_id": [[2, 2, 2, 4, 4], [2, 2], [6, 6]],
            "module_id": [[100, 101, 101, 200, 200], [102, 103], [300, 301]],
            # All-state; finite only at measurement positions
            "l_x_hit": [[1.0, np.nan, np.nan, -2.0, np.nan],
                        [3.0, np.nan],
                        [np.nan, 5.0]],
        }
    )


def test_root_residuals_drops_hole_and_returns_geometry_id():
    from expansion import encode_geometry_id
    from scripts.patch_is_selected import _root_residuals_from_arrays

    df = _root_residuals_from_arrays(_synthetic_residual_arrays(), event_id=0)
    df = df.sort_values(["seed_id", "geometry_id"]).reset_index(drop=True)

    # Track 0 has 2 measurement states at all-state positions [0, 3],
    # corresponding to vol/lay/mod = (16,2,100) and (17,4,200).
    # Track 1 has 1 measurement at position [0] = (16,2,102).
    expected_gids = encode_geometry_id(
        np.array([16, 17, 16]),
        np.array([2, 4, 2]),
        np.array([100, 200, 102]),
    )
    assert df["seed_id"].tolist() == [0, 0, 1]
    np.testing.assert_array_equal(df["geometry_id"].to_numpy(), expected_gids)
    np.testing.assert_allclose(df["res_l0"].tolist(), [0.10, -0.30, 0.05])
    np.testing.assert_allclose(df["res_l1"].tolist(), [0.20, 0.40, 0.15])


def test_root_residuals_filters_to_the_requested_event():
    """seed_id is synthesized from position among the *filtered* tracks
    (expansion.py:423's convention), so the sole event-1 track is reindexed
    to seed_id 0, not its original position (2) in the unfiltered array."""
    from expansion import encode_geometry_id
    from scripts.patch_is_selected import _root_residuals_from_arrays

    df = _root_residuals_from_arrays(_synthetic_residual_arrays(), event_id=1)
    assert df["seed_id"].tolist() == [0]
    # Track 2's measurement is at all-state position [1] = (18, 6, 301).
    expected_gid = encode_geometry_id(np.array([18]), np.array([6]), np.array([301]))
    np.testing.assert_array_equal(df["geometry_id"].to_numpy(), expected_gid)


def test_root_and_contributor_parsing_agree_on_geometry_id_layout():
    """Both parsers must derive geometry_id the same way from the same track
    layout, since match_selected and selected_correctness join on
    (seed_id, geometry_id) across their outputs."""
    from expansion import encode_geometry_id
    from scripts.patch_is_selected import (
        _root_residuals_from_arrays,
        _select_contributors_from_arrays,
    )

    contrib = _select_contributors_from_arrays(
        _synthetic_contributor_arrays(), event_id=0
    )
    resid = _root_residuals_from_arrays(_synthetic_residual_arrays(), event_id=0)

    contrib_states = set(zip(contrib["seed_id"], contrib["geometry_id"]))
    # Residuals contain only measurement states; contributors contain all
    # states (measurements + holes + material).  The hole at
    # (0, vol=16/lay=2/mod=101) appears in contributors but not residuals.
    resid_states = set(zip(resid["seed_id"], resid["geometry_id"]))
    assert resid_states <= contrib_states
    hole_gid = int(
        encode_geometry_id(np.array([16]), np.array([2]), np.array([101]))[0]
    )
    assert (0, hole_gid) in contrib_states and (0, hole_gid) not in resid_states
