"""Tests for the window-conditioned value cache (Task 6).

``scripts.build_value_cache.apply_window_targets`` is the pure function that
joins a per-event tier-3 targets frame (``scripts/stitch_tier3.py``'s output
contract: ``seed_id``, ``step_k``, ``vstar_tier3``, ``window_nsigma``) onto a
per-state feature frame on ``(seed_id, step_k)``, drops rows with no matching
target, and appends the constant ``window_nsigma`` feature. These tests use
small synthetic frames (no Parquet/simhits I/O) so the join and drop-count
logic is verified independently of ``process_event``'s Parquet plumbing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cckf.features import NO_STANDARDIZE, VALUE_FEATURES, VALUE_FEATURES_WINDOWED
from scripts.build_value_cache import apply_window_targets


def _synthetic_state_frame() -> pd.DataFrame:
    """Four states, one of which (seed 2) will have no tier-3 target."""
    return pd.DataFrame(
        {
            "seed_id": [0, 0, 1, 2],
            "branch_id": [0, 0, 1, 2],
            "step_k": [0, 1, 0, 0],
            "eta": [0.1, 0.2, 0.3, 0.4],
            "state_qop": [-0.5, -0.5, 0.5, 0.5],
            "sigma2_l0": [0.25, 0.25, 0.25, 0.25],
            "sigma2_l1": [0.36, 0.36, 0.36, 0.36],
            "n_hits": [1, 2, 1, 1],
            "n_holes": [0, 0, 0, 0],
            "n_seq_holes": [0, 0, 0, 0],
            "sum_gate_logodds": [1.0, 2.0, 1.0, 1.0],
            "min_gate_logodds": [1.0, 1.0, 1.0, 1.0],
            "x0_accumulated": [0.01, 0.02, 0.01, 0.01],
        }
    )


def _synthetic_targets() -> pd.DataFrame:
    """Tier-3 targets covering seed 0's two states and seed 1's one state.

    Seed 2 (step 0) has no row here, so it must be dropped by the join.
    """
    return pd.DataFrame(
        {
            "seed_id": [0, 0, 1],
            "step_k": [0, 1, 0],
            "vstar_tier3": [0.7, 0.9, 0.4],
            "window_nsigma": [3.0, 3.0, 3.0],
        }
    )


def test_apply_window_targets_appends_constant_column_and_joins_y():
    frame = _synthetic_state_frame()
    targets = _synthetic_targets()

    out, n_dropped = apply_window_targets(frame, targets, nsig=3.0)

    assert n_dropped == 1
    assert len(out) == 3
    assert set(zip(out["seed_id"], out["step_k"])) == {(0, 0), (0, 1), (1, 0)}

    # constant column, correct value, right dtype
    assert (out["window_nsigma"].to_numpy() == np.float32(3.0)).all()
    assert out["window_nsigma"].dtype == np.float32

    # y joined correctly per row
    joined = out.set_index(["seed_id", "step_k"])["vstar_tier3"]
    assert joined.loc[(0, 0)] == 0.7
    assert joined.loc[(0, 1)] == 0.9
    assert joined.loc[(1, 0)] == 0.4


def test_apply_window_targets_drops_and_counts_unmatched_rows():
    frame = _synthetic_state_frame()
    # No targets at all -> every row dropped.
    empty_targets = pd.DataFrame(
        {"seed_id": [], "step_k": [], "vstar_tier3": [], "window_nsigma": []}
    )
    out, n_dropped = apply_window_targets(frame, empty_targets, nsig=5.0)
    assert n_dropped == 4
    assert len(out) == 0


def test_apply_window_targets_produces_12_wide_x():
    """Feeding the joined frame's VALUE_FEATURES_WINDOWED columns into
    column_stack must yield a 12-wide matrix whose last column is the
    constant window_nsigma."""
    frame = _synthetic_state_frame()
    targets = _synthetic_targets()
    out, _ = apply_window_targets(frame, targets, nsig=10.0)

    X = np.column_stack(
        [out[name].to_numpy(dtype=np.float64) for name in VALUE_FEATURES_WINDOWED]
    ).astype(np.float32)

    assert X.shape == (3, 12)
    assert X.shape[1] == len(VALUE_FEATURES) + 1
    np.testing.assert_allclose(X[:, -1], np.full(3, 10.0, dtype=np.float32))


def test_window_nsigma_is_not_standardized():
    assert "window_nsigma" in NO_STANDARDIZE
    assert VALUE_FEATURES_WINDOWED == tuple(VALUE_FEATURES) + ("window_nsigma",)
    assert len(VALUE_FEATURES_WINDOWED) == 12


def test_default_path_feature_list_unchanged():
    """VALUE_FEATURES itself (the un-windowed 11-dim vector) must be
    untouched by adding the windowed variant."""
    assert len(VALUE_FEATURES) == 11
    assert "window_nsigma" not in VALUE_FEATURES
