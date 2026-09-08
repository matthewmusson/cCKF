"""Re-expansion audit: each check must catch the defect it exists for."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import audit_expansion as au


def _states(rows):
    df = pd.DataFrame(rows)
    df["is_ckf_selected"] = df["is_ckf_selected"].astype(bool)
    df["majority_undefined"] = df.get("majority_undefined", False)
    return df


def _row(seed, k, vol, lay, cand, sel, pids, maj=1001, n_hits=0, n_holes=0, v=0.5):
    return dict(seed_id=seed, step_k=k, volume_id=vol, layer_id=lay, cand_hit_id=cand,
                is_ckf_selected=sel, contrib_pids=pids, branch_majority_pid=maj,
                majority_undefined=False, n_hits=n_hits, n_holes=n_holes, vstar_soft=v)


def _good_branch(seed=0, maj=1001):
    """Three pixel hits (seed), a hole, a strip hit; history counts states before k."""
    return [
        _row(seed, 0, 17, 2, 10, True, [maj], maj, 0, 0),
        _row(seed, 0, 17, 2, 11, False, [2002], maj, 0, 0),
        _row(seed, 1, 17, 4, 12, True, [maj], maj, 1, 0),
        _row(seed, 2, 17, 6, 13, True, [maj], maj, 2, 0),
        _row(seed, 3, 20, 2, -1, False, [], maj, 3, 0),
        _row(seed, 4, 24, 2, 14, True, [maj], maj, 3, 1),
    ]


def _root_for(states):
    st = states.drop_duplicates(["seed_id", "step_k"]).sort_values(["seed_id", "step_k"])
    return pd.DataFrame({
        "track_nr": st.seed_id.to_numpy(), "step_k": st.step_k.to_numpy(),
        "volume_id": st.volume_id.to_numpy(), "layer_id": st.layer_id.to_numpy(),
        "pathLength": st.step_k.to_numpy() * 100.0,
    })


def test_root_order_passes_on_increasing_path_length():
    root = _root_for(_states(_good_branch()))
    assert au.check_root_order(root).passed


def test_root_order_fails_on_root_native_order():
    root = _root_for(_states(_good_branch()))
    root["pathLength"] = root["pathLength"].to_numpy()[::-1]
    c = au.check_root_order(root)
    assert not c.passed and "propagation order" in c.detail


def test_parquet_vs_root_detects_a_reversed_parquet():
    states = _states(_good_branch())
    root = _root_for(states)
    assert au.check_parquet_vs_root(states, root).passed
    flipped = states.copy()
    flipped["step_k"] = flipped["step_k"].max() - flipped["step_k"]
    c = au.check_parquet_vs_root(flipped, root)
    assert not c.passed and c.value > 0


def test_volumes_flags_missing_sensitive_and_unexpected():
    rows = _good_branch()
    states = _states(rows)
    c = au.check_volumes(states)
    assert not c.passed  # only 17, 20, 24 present
    full = rows + [_row(1, k, v, 2, 100 + k, True, [1001]) for k, v in
                   enumerate((16, 18, 23, 25, 28, 29, 30))]
    assert au.check_volumes(_states(full)).passed
    odd = _states(full + [_row(2, 0, 99, 1, 500, True, [1001])])
    assert "unexpected [99]" in au.check_volumes(odd).detail


def test_hole_fraction_against_reference():
    states = _states(_good_branch())          # 1 hole of 5 states = 0.2
    assert au.hole_fraction(states) == pytest.approx(0.2)
    assert au.check_hole_fraction(states, states, tol=0.05).passed
    ref = _states(_good_branch() + [_row(1, k, 20, 2, -1, False, []) for k in range(5)])
    assert not au.check_hole_fraction(states, ref, tol=0.05).passed


def test_candidate_shares_requires_all_three_sensor_classes_without_reference():
    c = au.check_candidate_shares(_states(_good_branch()), None, tol=0.05)
    assert not c.passed  # no long-strip candidates
    full = _good_branch() + [_row(1, 0, 29, 2, 77, True, [1001])]
    assert au.check_candidate_shares(_states(full), None, tol=0.05).passed


def test_history_catches_outside_in_counting():
    states = _states(_good_branch())
    assert au.check_history(states).passed
    bad = states.copy()
    bad.loc[bad.step_k == 0, "n_hits"] = 3   # counted from the other end
    c = au.check_history(bad)
    assert not c.passed


def test_majority_label_recomputed_from_innermost_hits():
    states = _states(_good_branch(maj=1001))
    assert au.check_majority_label(states, 0.98).passed
    # Outer hits belong to 2002; a label taken from the OUTERMOST three
    # would say 2002 and must be rejected.
    rows = _good_branch(maj=1001)             # innermost three hits: 1001
    rows[5]["contrib_pids"] = [2002]
    rows.append(_row(0, 5, 24, 4, 15, True, [2002], 1001, 4, 1))
    rows.append(_row(0, 6, 24, 6, 16, True, [2002], 1001, 5, 1))
    for r in rows:
        r["branch_majority_pid"] = 2002       # stored label from the OUTER end
    c = au.check_majority_label(_states(rows), 0.98)
    assert not c.passed and c.value == 0.0


def test_selected_flag_rejects_two_selected_rows_on_one_state():
    rows = _good_branch()
    rows[1]["is_ckf_selected"] = True
    c = au.check_selected_flag(_states(rows), 0.95)
    assert not c.passed and "1 states with >1" in c.detail


def test_vstar_range():
    states = _states(_good_branch())
    assert au.check_vstar_range(states).passed
    states.loc[0, "vstar_soft"] = 1.5
    assert not au.check_vstar_range(states).passed


def test_run_all_passes_on_consistent_synthetic_event():
    rows = _good_branch() + [_row(1, k, v, 2, 100 + k, True, [1001], 1001, k, 0)
                             for k, v in enumerate((16, 18, 23, 25, 28, 29, 30))]
    states = _states(rows)
    root = _root_for(states)
    checks = au.run_all(states, root, reference=states, tol=0.05)
    assert all(c.passed for c in checks), [c.line() for c in checks if not c.passed]
