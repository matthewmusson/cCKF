"""State ordering: ROOT stores track states outermost-first, the pipeline wants
propagation order (0 at the seed surface).

Background (experiments/LOG.md 2026-09-08): ACTS' RootTrackStatesWriter
iterates ``track.trackStatesReversed()`` and appends, so ROOT index 0 is the
LAST state the CKF created. Every "past/future along the branch" computation
in this repo (value target, branch history, majority id, tier-3 walker)
assumes ``step_k`` is propagation order. These tests pin the conversion and
the pathLength invariant that guards it.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pandas as pd
import pytest

import expansion


def test_propagation_order_index_reverses_each_track():
    jagged = ak.Array([[10, 11, 12], [20], [30, 31]])
    idx = expansion.propagation_order_index(jagged)
    # ROOT order [a, b, c] -> propagation order index [2, 1, 0]
    assert idx.tolist() == [2, 1, 0, 0, 1, 0]
    assert idx.dtype == np.int64


def test_propagation_order_index_with_measurement_mask():
    jagged = ak.Array([[10, 11, 12, 13], [20, 21]])
    mask = ak.Array([[True, False, True, False], [False, True]])
    idx = expansion.propagation_order_index(jagged, mask)
    # Track 0 keeps ROOT positions 0 and 2 -> propagation indices 3 and 1.
    # Track 1 keeps ROOT position 1 -> propagation index 0.
    assert idx.tolist() == [3, 1, 0]


def test_propagation_order_index_handles_empty_tracks():
    jagged = ak.Array([[], [5, 6], []])
    assert expansion.propagation_order_index(jagged).tolist() == [1, 0]


def test_check_propagation_order_accepts_increasing_path_length():
    track_nr = np.array([0, 0, 0, 1, 1])
    state_idx = np.array([2, 1, 0, 1, 0])          # ROOT order reversed
    path_len = np.array([250.0, 100.0, 0.0, 80.0, 0.0])
    expansion.check_propagation_order(track_nr, state_idx, path_len)


def test_check_propagation_order_tolerates_mid_branch_resets():
    # The CKF restarts pathLength when it resumes a forked branch (82% of
    # event-4 tracks carry a drop). Orientation, not monotonicity, is the
    # invariant: seed at 0, outermost above it, first step up.
    track_nr = np.array([0, 0, 0, 0, 0, 0])
    state_idx = np.array([5, 4, 3, 2, 1, 0])
    path_len = np.array([900.0, 170.0, 4.4, 300.0, 100.0, 0.0])
    expansion.check_propagation_order(track_nr, state_idx, path_len)


def test_check_propagation_order_allows_nan():
    track_nr = np.array([0, 0, 0, 0])
    state_idx = np.array([3, 2, 1, 0])
    path_len = np.array([np.nan, 100.0, 50.0, 0.0])
    expansion.check_propagation_order(track_nr, state_idx, path_len)


def test_check_propagation_order_rejects_root_order():
    track_nr = np.array([0, 0, 0])
    state_idx = np.array([0, 1, 2])                 # NOT reversed
    path_len = np.array([250.0, 100.0, 0.0])
    with pytest.raises(ValueError, match="propagation order"):
        expansion.check_propagation_order(track_nr, state_idx, path_len)


def test_check_propagation_order_rejects_nonzero_seed_path_length():
    track_nr = np.array([0, 0])
    state_idx = np.array([1, 0])
    path_len = np.array([250.0, 5.0])
    with pytest.raises(ValueError, match="pathLength 0"):
        expansion.check_propagation_order(track_nr, state_idx, path_len)


def test_check_propagation_order_fraction_threshold():
    # 1 of 2 tracks mis-oriented -> 0.5 < 0.99 -> raise; 0.5 floor -> pass
    track_nr = np.array([0, 0, 1, 1])
    state_idx = np.array([1, 0, 1, 0])
    path_len = np.array([100.0, 0.0, 0.0, 100.0])
    with pytest.raises(ValueError):
        expansion.check_propagation_order(track_nr, state_idx, path_len)
    expansion.check_propagation_order(track_nr, state_idx, path_len, min_frac=0.5)


def _write_trackstates(path, tracks: list[dict]) -> None:
    """Write a minimal trackstates TTree in ROOT's native (reversed) order."""
    uproot = pytest.importorskip("uproot")
    cols = {k: ak.Array([t[k] for t in tracks]) for k in tracks[0]}
    with uproot.recreate(path) as fh:
        fh["trackstates"] = cols


def test_load_trackstates_returns_propagation_order(tmp_path):
    # Two tracks, written outermost-first exactly as RootTrackStatesWriter does.
    tracks = [
        {
            "volume_id": [23, 16, 17], "layer_id": [2, 8, 2], "module_id": [5, 6, 7],
            "pathLength": [900.0, 300.0, 0.0],
            "eLOC0_prt": [1.0, 2.0, 3.0], "eLOC1_prt": [0.1, 0.2, 0.3],
        },
        {
            "volume_id": [17, 17], "layer_id": [4, 2], "module_id": [9, 8],
            "pathLength": [40.0, 0.0],
            "eLOC0_prt": [4.0, 5.0], "eLOC1_prt": [0.4, 0.5],
        },
    ]
    root = tmp_path / "trackstates_ckf.root"
    _write_trackstates(root, tracks)

    df = expansion.load_trackstates(str(root), event_id=0)

    t0 = df[df.track_nr == 0].sort_values("state_idx")
    assert t0["state_idx"].tolist() == [0, 1, 2]
    # state 0 is the seed-end pixel-barrel state, state 2 the outermost strip
    assert t0["volume_id"].tolist() == [17, 16, 23]
    assert t0["pred_l0"].tolist() == [3.0, 2.0, 1.0]
    t1 = df[df.track_nr == 1].sort_values("state_idx")
    assert t1["layer_id"].tolist() == [2, 4]


def test_load_trackstates_refuses_a_file_in_propagation_order(tmp_path):
    # If some future writer stores states seed-first, pathLength would be
    # DEcreasing along the reversed index. The loader must fail loudly rather
    # than silently invert everything a second time.
    tracks = [{
        "volume_id": [17, 16, 23], "layer_id": [2, 8, 2], "module_id": [7, 6, 5],
        "pathLength": [0.0, 300.0, 900.0],
        "eLOC0_prt": [3.0, 2.0, 1.0], "eLOC1_prt": [0.3, 0.2, 0.1],
    }]
    root = tmp_path / "trackstates_ckf.root"
    _write_trackstates(root, tracks)
    with pytest.raises(ValueError, match="propagation order"):
        expansion.load_trackstates(str(root), event_id=0)


def test_load_predicted_cov_reverses_with_root_state_count(tmp_path):
    # PredictedCovWriter counts step_k over trackStatesReversed (every state,
    # predicted or not), so its step_k is the ROOT index. Reversal needs the
    # per-track state count from ROOT, not the CSV's own row count, because
    # states without a prediction leave no row.
    csv_dir = tmp_path
    (csv_dir / "event000000000-predicted-cov.csv").write_text(
        "track_nr,step_k,eLOC0_prt,eLOC1_prt,P00,P01,P11\n"
        "0,0,1.0,0.1,1.0,0.0,1.0\n"
        "0,2,3.0,0.3,3.0,0.0,3.0\n"   # ROOT index 1 had no prediction: no row
        "1,1,5.0,0.5,5.0,0.0,5.0\n"
    )
    n_states = np.array([3, 2])
    df = expansion.load_predicted_cov(str(csv_dir), 0, n_states_per_track=n_states)
    got = df.sort_values(["track_nr", "state_idx"])[["track_nr", "state_idx", "P00"]]
    assert got.values.tolist() == [[0, 0, 3.0], [0, 2, 1.0], [1, 0, 5.0]]
