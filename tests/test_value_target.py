"""Tests for the V^{π†} value target."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cckf import value_target as vt


def _step_table() -> pd.DataFrame:
    """One branch, 4 steps.

    step 0: selected hit, correct        -> n_correct 1
    step 1: selected hit, wrong particle -> n_wrong 1
    step 2: hole, majority DID leave a simhit here (a findable miss)
    step 3: selected hit, correct        -> n_correct 2

    majority_true_hit_on_surface is True at steps 0, 2, 3 and False at step 1.
    """
    return pd.DataFrame({
        "seed_id": [0, 0, 0, 0],
        "branch_id": [0, 0, 0, 0],
        "step_k": [0, 1, 2, 3],
        "sel_correct": [1, 0, 0, 1],
        "sel_wrong": [0, 1, 0, 0],
        "maj_hit_on_surface": [True, False, True, True],
        "branch_majority_pid": [7, 7, 7, 7],
    })


def test_particle_simhit_counts_tallies_per_particle():
    simhits = pd.DataFrame({"particle_id": [7, 7, 8, 7, 8]})
    counts = vt.particle_simhit_counts(simhits)
    assert counts == {7: 3, 8: 2}


def test_backward_counts_are_cumulative_and_inclusive():
    out = vt.compute_value_targets(_step_table(), {7: 4})
    assert out["n_correct"].tolist() == [1, 1, 1, 2]
    assert out["n_wrong"].tolist() == [0, 1, 1, 1]


def test_findable_counts_only_steps_strictly_after_k():
    # maj_hit_on_surface is True at steps 0, 2, 3 -> 3 total.
    # After k=0: steps 1,2,3 -> 2. After k=1: 2. After k=2: 1. After k=3: 0.
    out = vt.compute_value_targets(_step_table(), {7: 4})
    assert out["n_findable_t2"].tolist() == [2, 2, 1, 0]


def test_vstar_matches_hand_computed_value_at_step_zero():
    # k=0: n_correct=1, n_wrong=0, n_findable=2
    # n_shared = 3, n_track = 3, N_total = 4
    # completeness = 3/4 = 0.75, purity = 3/3 = 1.0, V = 0.75
    out = vt.compute_value_targets(_step_table(), {7: 4})
    assert out["vstar_t2"].iloc[0] == pytest.approx(0.75)


def test_vstar_matches_hand_computed_value_at_final_step():
    # k=3: n_correct=2, n_wrong=1, n_findable=0
    # n_shared = 2, n_track = 3, N_total = 4
    # completeness = 0.5, purity = 2/3, V = min = 0.5
    out = vt.compute_value_targets(_step_table(), {7: 4})
    assert out["vstar_t2"].iloc[3] == pytest.approx(0.5)


def test_vstar_is_the_min_of_completeness_and_purity():
    out = vt.compute_value_targets(_step_table(), {7: 4})
    np.testing.assert_allclose(
        out["vstar_t2"], np.minimum(out["completeness"], out["purity"])
    )


def test_vstar_is_bounded_in_zero_one():
    out = vt.compute_value_targets(_step_table(), {7: 4})
    assert ((out["vstar_t2"] >= 0.0) & (out["vstar_t2"] <= 1.0)).all()


def test_tier1_is_never_below_tier2():
    """Tier 1 counts every remaining simhit, so it is the optimistic bound."""
    out = vt.compute_value_targets(_step_table(), {7: 10})
    assert (out["n_findable_t1"] >= out["n_findable_t2"]).all()
    assert (out["vstar_t1"] >= out["vstar_t2"] - 1e-12).all()


def test_no_tier3_columns_are_produced():
    """Tier 3 is deliberately absent: computing it honestly requires re-running
    propagation with truth-greedy selection to regenerate the n-sigma windows,
    because the logged windows belong to the behaviour policy's trajectory.
    Guard against a well-meaning reintroduction of the invalid shortcut."""
    out = vt.compute_value_targets(_step_table(), {7: 4})
    assert not [c for c in out.columns if c.endswith("_t3")]
    assert "maj_hit_in_window" not in out.columns


def test_all_wrong_branch_has_zero_value():
    step = pd.DataFrame({
        "seed_id": [0, 0], "branch_id": [0, 0], "step_k": [0, 1],
        "sel_correct": [0, 0], "sel_wrong": [1, 1],
        "maj_hit_on_surface": [False, False], "branch_majority_pid": [7, 7],
    })
    out = vt.compute_value_targets(step, {7: 5})
    assert (out["vstar_t2"] == 0.0).all()


def test_missing_particle_count_yields_nan_not_a_wrong_number():
    out = vt.compute_value_targets(_step_table(), {})  # pid 7 absent
    assert out["vstar_t2"].isna().all()


def test_branches_are_independent():
    step = pd.concat([
        _step_table(),
        _step_table().assign(seed_id=1),
    ], ignore_index=True)
    out = vt.compute_value_targets(step, {7: 4})
    a = out[out["seed_id"] == 0]["vstar_t2"].to_numpy()
    b = out[out["seed_id"] == 1]["vstar_t2"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_build_step_table_collapses_candidates_to_one_row_per_step():
    df = pd.DataFrame({
        "seed_id": [0, 0, 0],
        "branch_id": [0, 0, 0],
        "step_k": [0, 0, 1],
        "cand_hit_id": [10, 11, -1],
        "is_ckf_selected": [False, True, False],
        "label_same_particle": [0, 1, 0],
        "majority_true_hit_on_surface": [True, True, False],
        "branch_majority_pid": [7, 7, 7],
    })
    step = vt.build_step_table(df)
    assert len(step) == 2
    # The selected candidate at step 0 was correct.
    assert step.loc[step["step_k"] == 0, "sel_correct"].iloc[0] == 1
    assert step.loc[step["step_k"] == 0, "sel_wrong"].iloc[0] == 0
    # Step 1 is a hole: neither correct nor wrong.
    assert step.loc[step["step_k"] == 1, "sel_correct"].iloc[0] == 0
    assert step.loc[step["step_k"] == 1, "sel_wrong"].iloc[0] == 0


def test_build_step_table_marks_wrong_when_selected_candidate_is_negative():
    df = pd.DataFrame({
        "seed_id": [0, 0],
        "branch_id": [0, 0],
        "step_k": [0, 0],
        "cand_hit_id": [10, 11],
        "is_ckf_selected": [True, False],
        "label_same_particle": [0, 1],
        "majority_true_hit_on_surface": [True, True],
        "branch_majority_pid": [7, 7],
    })
    step = vt.build_step_table(df)
    assert step["sel_correct"].iloc[0] == 0
    assert step["sel_wrong"].iloc[0] == 1
