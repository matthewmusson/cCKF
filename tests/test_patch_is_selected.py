"""Tests for recovering the CKF-selected candidate per state."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.patch_is_selected import match_selected


def _cands() -> pd.DataFrame:
    # Two states. State (0,0) has 3 candidates; state (0,1) has 2.
    return pd.DataFrame({
        "seed_id": [0, 0, 0, 0, 0],
        "step_k": [0, 0, 0, 1, 1],
        "cand_hit_id": [10, 11, 12, 20, 21],
        "residual_l0": [0.10, -0.30, 0.90, 0.05, 2.00],
        "residual_l1": [0.20, 0.40, -0.10, 0.15, 1.00],
    })


def _root() -> pd.DataFrame:
    # ROOT says state (0,0) selected the candidate with residual (-0.30, 0.40)
    # and state (0,1) selected residual (0.05, 0.15).
    return pd.DataFrame({
        "seed_id": [0, 0],
        "step_k": [0, 1],
        "res_l0": [-0.30, 0.05],
        "res_l1": [0.40, 0.15],
    })


def test_match_selected_picks_exactly_one_per_state():
    sel = match_selected(_cands(), _root())
    assert sel.tolist() == [False, True, False, True, False]


def test_match_selected_one_flag_per_state_group():
    cands = _cands()
    cands["is_sel"] = match_selected(cands, _root())
    per_state = cands.groupby(["seed_id", "step_k"])["is_sel"].sum()
    assert (per_state == 1).all()


def test_match_selected_marks_none_when_root_has_no_state():
    """A state with no ROOT entry (e.g. a hole) gets no selected candidate."""
    root = _root().iloc[:1]  # drop state (0,1)
    sel = match_selected(_cands(), root)
    assert sel.tolist() == [False, True, False, False, False]


def test_match_selected_respects_tolerance():
    root = _root().copy()
    root.loc[0, "res_l0"] = -0.30 + 1e-2  # outside a 1e-4 tolerance
    sel = match_selected(_cands(), root, tol=1e-4)
    # State (0,0) now matches nothing; state (0,1) still matches.
    assert sel.tolist() == [False, False, False, True, False]


def test_match_selected_breaks_ties_deterministically():
    """Two identical residuals in one state: flag the lower cand_hit_id only."""
    cands = pd.DataFrame({
        "seed_id": [0, 0],
        "step_k": [0, 0],
        "cand_hit_id": [11, 10],
        "residual_l0": [0.5, 0.5],
        "residual_l1": [0.5, 0.5],
    })
    root = pd.DataFrame({
        "seed_id": [0], "step_k": [0], "res_l0": [0.5], "res_l1": [0.5],
    })
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

    sel = pd.DataFrame({
        "seed_id": [0], "step_k": [0],
        "sel_contrib_pids": [[999, 7]],  # 999 dominates, 7 is the majority pid
        "sel_has_hit": [True],
    })
    majority = pd.DataFrame({"seed_id": [0], "branch_majority_pid": [7]})
    out = selected_correctness(sel, majority)
    assert out["sel_correct"].iloc[0] == 1
    assert out["sel_wrong"].iloc[0] == 0


def test_selected_correctness_flags_wrong_when_majority_absent():
    from scripts.patch_is_selected import selected_correctness

    sel = pd.DataFrame({
        "seed_id": [0], "step_k": [0],
        "sel_contrib_pids": [[999, 888]],
        "sel_has_hit": [True],
    })
    majority = pd.DataFrame({"seed_id": [0], "branch_majority_pid": [7]})
    out = selected_correctness(sel, majority)
    assert out["sel_correct"].iloc[0] == 0
    assert out["sel_wrong"].iloc[0] == 1


def test_selected_correctness_counts_a_hole_as_neither():
    """A hole neither helps completeness nor pollutes purity."""
    from scripts.patch_is_selected import selected_correctness

    sel = pd.DataFrame({
        "seed_id": [0], "step_k": [0],
        "sel_contrib_pids": [[]],
        "sel_has_hit": [False],
    })
    majority = pd.DataFrame({"seed_id": [0], "branch_majority_pid": [7]})
    out = selected_correctness(sel, majority)
    assert out["sel_correct"].iloc[0] == 0
    assert out["sel_wrong"].iloc[0] == 0


def test_selected_correctness_is_exclusive():
    """No state may be both correct and wrong."""
    from scripts.patch_is_selected import selected_correctness

    sel = pd.DataFrame({
        "seed_id": [0, 0, 0], "step_k": [0, 1, 2],
        "sel_contrib_pids": [[7], [8], []],
        "sel_has_hit": [True, True, False],
    })
    majority = pd.DataFrame({"seed_id": [0], "branch_majority_pid": [7]})
    out = selected_correctness(sel, majority)
    assert ((out["sel_correct"] + out["sel_wrong"]) <= 1).all()
    assert out["sel_correct"].tolist() == [1, 0, 0]
    assert out["sel_wrong"].tolist() == [0, 1, 0]
