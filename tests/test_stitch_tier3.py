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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stitch_tier3 import attach_window, gate, paths


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
