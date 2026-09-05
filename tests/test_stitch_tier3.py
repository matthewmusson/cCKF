"""Tests for the tier-3 stitch driver (scripts/stitch_tier3.py).

The driver is thin -- classify/futures/inputs/compose are Tier-1 or already
tested elsewhere. These tests cover only the driver's own logic: the
``window_nsigma`` column attachment, the truth-suffix gate decision, and its
path construction.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stitch_tier3 import (
    attach_window,
    filter_majority_defined,
    filter_valid_tier2,
    gate,
    paths,
    recompute_tier2_from_frames,
)


def test_attach_window_adds_constant_column():
    targets = pd.DataFrame(
        {"seed_id": [0, 0, 1], "step_k": [0, 1, 0], "vstar_tier3": [0.5, 0.8, 1.0]}
    )
    out = attach_window(targets, 10.0)
    assert (out["window_nsigma"] == 10.0).all()
    assert out["window_nsigma"].dtype == np.float64
    # original columns preserved
    assert list(out["vstar_tier3"]) == [0.5, 0.8, 1.0]


def test_attach_window_does_not_mutate_input():
    targets = pd.DataFrame({"seed_id": [0], "step_k": [0], "vstar_tier3": [0.5]})
    attach_window(targets, 3.0)
    assert "window_nsigma" not in targets.columns


def test_attach_window_casts_int_nsig_to_float():
    targets = pd.DataFrame({"seed_id": [0], "step_k": [0], "vstar_tier3": [0.5]})
    out = attach_window(targets, 5)
    assert out["window_nsigma"].iloc[0] == 5.0
    assert isinstance(out["window_nsigma"].iloc[0], float)


def test_gate_passes_below_tolerance():
    report = {"n_states_compared": 100, "disagree_rate": 0.005}
    assert gate(report, 0.01) is True


def test_gate_fails_at_or_above_tolerance():
    assert gate({"n_states_compared": 100, "disagree_rate": 0.01}, 0.01) is False
    assert gate({"n_states_compared": 100, "disagree_rate": 0.05}, 0.01) is False


def test_gate_passes_when_nothing_to_compare():
    # truth_suffix_check's early-return shape: no disagree_rate key at all.
    assert gate({"n_suffix_branches": 0, "n_states_compared": 0}, 0.01) is True


def test_paths_default_hits_dir_uses_nsig():
    p = paths(4, 10.0, "/scratch/cckf", None)
    assert p["parquet"] == "/scratch/cckf/reexpanded/expanded_event000000004.parquet"
    assert (
        p["worklist"]
        == "/scratch/cckf/tier3/worklists/event000000004-rollout-worklist.csv"
    )
    assert p["hits_dir"] == "/scratch/cckf/tier3_nsig10/hits"
    assert (
        p["hits"] == "/scratch/cckf/tier3_nsig10/hits/event000000004-rollout-hits.csv"
    )
    assert p["out"] == "/scratch/cckf/tier3_targets/vstar_nsig10_event000000004.parquet"


def test_paths_hits_dir_override():
    p = paths(4, 10.0, "/scratch/cckf", "/scratch/cckf/tier3/hits")
    assert p["hits_dir"] == "/scratch/cckf/tier3/hits"
    assert p["hits"] == "/scratch/cckf/tier3/hits/event000000004-rollout-hits.csv"
    # output path is unaffected by the hits-dir override
    assert p["out"] == "/scratch/cckf/tier3_targets/vstar_nsig10_event000000004.parquet"


def test_paths_worklist_independent_of_nsig():
    # the worklist is generated once, window-independent; must be identical
    # across nsig values for the same event.
    p3 = paths(1, 3.0, "/S", None)
    p10 = paths(1, 10.0, "/S", None)
    assert p3["worklist"] == p10["worklist"]
    assert p3["parquet"] == p10["parquet"]
    assert p3["hits_dir"] != p10["hits_dir"]


def test_paths_fractional_nsig_falls_back_to_str():
    p = paths(0, 3.5, "/S", None)
    assert p["hits_dir"] == "/S/tier3_nsig3.5/hits"


# --- filter_valid_tier2 ------------------------------------------------


def test_filter_valid_tier2_drops_nan_target_row():
    targets = pd.DataFrame(
        {
            "seed_id": [0, 1],
            "step_k": [0, 0],
            "vstar_t2": [0.5, np.nan],
            "tier_invariant_violated": [False, False],
        }
    )
    out, n_dropped = filter_valid_tier2(targets)
    assert n_dropped == 1
    assert out["seed_id"].tolist() == [0]


def test_filter_valid_tier2_drops_violated_row():
    targets = pd.DataFrame(
        {
            "seed_id": [0, 1],
            "step_k": [0, 0],
            "vstar_t2": [0.5, 0.9],
            "tier_invariant_violated": [False, True],
        }
    )
    out, n_dropped = filter_valid_tier2(targets)
    assert n_dropped == 1
    assert out["seed_id"].tolist() == [0]


def test_filter_valid_tier2_keeps_valid_rows_and_counts_zero():
    targets = pd.DataFrame(
        {
            "seed_id": [0, 1],
            "step_k": [0, 0],
            "vstar_t2": [0.5, 0.9],
            "tier_invariant_violated": [False, False],
        }
    )
    out, n_dropped = filter_valid_tier2(targets)
    assert n_dropped == 0
    assert len(out) == 2


# --- recompute_tier2_from_frames ----------------------------------------

_TIER2_COLS = [
    "seed_id",
    "branch_id",
    "step_k",
    "cand_hit_id",
    "is_ckf_selected",
    "majority_true_hit_on_surface",
    "branch_majority_pid",
    "majority_undefined",
    "label_same_particle",
]


def _tier2_row(**kw) -> dict:
    row = {
        "seed_id": 0,
        "branch_id": 0,
        "step_k": 0,
        "cand_hit_id": -1,
        "is_ckf_selected": False,
        "majority_true_hit_on_surface": False,
        "branch_majority_pid": 0,
        "majority_undefined": False,
        "label_same_particle": 0,
    }
    row.update(kw)
    return row


def test_recompute_tier2_drops_nan_target_and_violated_keeps_valid():
    # seed 0 (pid 1, N_total=1): single accepted correct hit -> vstar_t2 = 1.0
    # seed 1 (pid 99, no simhits at all): NaN target -> dropped
    # seed 2 (pid 2, N_total=1): two holes both flagged
    # majority_true_hit_on_surface=True (simulated surface revisit) ->
    # step 0 is tier-invariant-violated (v1=0 < v2=1) and dropped; step 1
    # is not violated (v1=v2=0) and survives.
    rows = [
        _tier2_row(
            seed_id=0,
            branch_id=0,
            cand_hit_id=10,
            is_ckf_selected=True,
            majority_true_hit_on_surface=True,
            branch_majority_pid=1,
            label_same_particle=1,
        ),
        _tier2_row(
            seed_id=1,
            branch_id=1,
            cand_hit_id=20,
            is_ckf_selected=True,
            majority_true_hit_on_surface=True,
            branch_majority_pid=99,
            label_same_particle=1,
        ),
        _tier2_row(
            seed_id=2,
            branch_id=2,
            step_k=0,
            majority_true_hit_on_surface=True,
            branch_majority_pid=2,
        ),
        _tier2_row(
            seed_id=2,
            branch_id=2,
            step_k=1,
            majority_true_hit_on_surface=True,
            branch_majority_pid=2,
        ),
    ]
    labeled = pd.DataFrame(rows)[_TIER2_COLS]
    simhits = pd.DataFrame({"particle_id": [1, 2]})  # pid 99 absent -> N_total NaN

    out = recompute_tier2_from_frames(labeled, simhits)

    assert set(zip(out["seed_id"], out["step_k"])) == {(0, 0), (2, 1)}
    assert out.loc[out["seed_id"] == 0, "vstar_t2"].iloc[0] == pytest.approx(1.0)
    assert out.loc[out["seed_id"] == 2, "vstar_t2"].iloc[0] == pytest.approx(0.0)
    # neither the NaN-target seed (1) nor the violated (seed 2, step 0) row
    # survive.
    assert 1 not in set(out["seed_id"])
    assert (2, 0) not in set(zip(out["seed_id"], out["step_k"]))


def test_recompute_tier2_excludes_majority_undefined_branches():
    rows = [
        _tier2_row(
            seed_id=0,
            branch_id=0,
            cand_hit_id=10,
            is_ckf_selected=True,
            majority_true_hit_on_surface=True,
            branch_majority_pid=1,
            label_same_particle=1,
        ),
        _tier2_row(
            seed_id=1,
            branch_id=1,
            branch_majority_pid=-1,
            majority_undefined=True,
        ),
    ]
    labeled = pd.DataFrame(rows)[_TIER2_COLS]
    simhits = pd.DataFrame({"particle_id": [1]})

    out = recompute_tier2_from_frames(labeled, simhits)
    assert set(out["seed_id"]) == {0}


# --- filter_majority_defined ---------------------------------------------


def test_filter_majority_defined_drops_undefined_branches():
    states = pd.DataFrame(
        {
            "seed_id": [0, 0, 1, 2],
            "step_k": [0, 1, 0, 0],
            "state_class": ["tip", "tip", "tip", "tip"],
        }
    )
    # only seeds 0 and 2 have a defined majority (e.g. past_counts' seed set)
    defined_seeds = [0, 2]
    out, n_excluded = filter_majority_defined(states, defined_seeds)
    assert n_excluded == 1  # seed 1
    assert set(out["seed_id"]) == {0, 2}
    assert len(out) == 3  # both seed-0 rows plus the seed-2 row


def test_filter_majority_defined_no_op_when_all_defined():
    states = pd.DataFrame({"seed_id": [0, 1], "step_k": [0, 0]})
    out, n_excluded = filter_majority_defined(states, [0, 1])
    assert n_excluded == 0
    assert len(out) == 2
