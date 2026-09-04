"""Tests for the tier-3 V^{pi-dagger} composition (cckf/tier3_stitch.py)."""

import numpy as np
import pandas as pd

from cckf.tier3_stitch import compose_targets, truth_suffix_check


def _states(records):
    cols = ["seed_id", "step_k", "state_class", "sel_hit"]
    return pd.DataFrame(records, columns=cols)


def _futures(records):
    cols = ["seed_id", "step_k", "n_findable"]
    return pd.DataFrame(records, columns=cols)


def _past(records):
    cols = ["seed_id", "step_k", "n_correct", "n_wrong"]
    return pd.DataFrame(records, columns=cols)


def _ntot(records):
    cols = ["seed_id", "N_total_true"]
    return pd.DataFrame(records, columns=cols)


def test_truth_follower_is_one_everywhere():
    # 4 correct hits, particle has exactly 4; tip rollout finds nothing.
    st = _states([(0, k, "collapse", 100 + k) for k in range(3)] + [(0, 3, "tip", 103)])
    fut = _futures([(0, 3, 0)])
    past = _past([(0, k, k + 1, 0) for k in range(4)])
    out = compose_targets(st, fut, past, _ntot([(0, 4)]))
    assert len(out) == 4
    assert np.allclose(out["vstar_tier3"], 1.0)


def test_wrong_tip_worked_example():
    # The worked example from the design discussion: 10-hit particle,
    # branch takes 9 correct hits then a wrong hit at the tip.
    # V(tip) = min(9/10, 9/10) = 0.9; V(pre-tip) = 1.0 (rollout finds the
    # displaced 10th hit); all earlier states also 1.0.
    st = _states(
        [(0, k, "collapse", 100 + k) for k in range(8)]
        + [(0, 8, "divergence", 108), (0, 9, "tip", 999)]
    )
    fut = _futures([(0, 8, 1), (0, 9, 0)])
    past = _past([(0, k, k + 1, 0) for k in range(9)] + [(0, 9, 9, 1)])
    out = compose_targets(st, fut, past, _ntot([(0, 10)]))
    v = out.set_index("step_k")["vstar_tier3"]
    assert np.isclose(v[9], 0.9)
    assert np.isclose(v[8], 1.0)
    assert np.allclose(v.loc[list(range(8))], 1.0)


def test_hole_child_contributes_zero():
    # Step 1 is a both-hole agreement (sel_hit -1): step 0's future must
    # inherit the tip's findable + step1's zero, not +1.
    st = _states([(0, 0, "collapse", 100), (0, 1, "collapse", -1), (0, 2, "tip", 102)])
    fut = _futures([(0, 2, 0)])
    # particle has 3 hits total; branch got 2 (steps 0 and 2), 1 unfound.
    past = _past([(0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 2, 0)])
    out = compose_targets(st, fut, past, _ntot([(0, 3)]))
    v = out.set_index("step_k")["vstar_tier3"]
    # step 1: found = 1 + (0 + 1 hit at step2) = 2 -> min(2/3, 2/2) = 2/3
    assert np.isclose(v[1], 2 / 3)
    # step 0: child step1 is a hole (+0): found = 1 + (1) = 2 -> 2/3
    assert np.isclose(v[0], 2 / 3)


def test_failed_pid_join_dropped_not_labeled(capsys):
    st = _states([(0, 0, "tip", 100), (1, 0, "tip", 200)])
    fut = _futures([(0, 0, 0), (1, 0, 0)])
    past = _past([(0, 0, 1, 0), (1, 0, 1, 0)])
    out = compose_targets(st, fut, past, _ntot([(0, 1)]))  # seed 1 missing
    assert set(out["seed_id"]) == {0}
    assert "failed PID join" in capsys.readouterr().out


def test_missing_anchor_rollout_drops_whole_branch(capsys):
    st = _states([(0, 0, "collapse", 100), (0, 1, "tip", 101), (1, 0, "tip", 200)])
    fut = _futures([(1, 0, 0)])  # seed 0's tip rollout missing
    past = _past([(0, 0, 1, 0), (0, 1, 2, 0), (1, 0, 1, 0)])
    out = compose_targets(st, fut, past, _ntot([(0, 2), (1, 1)]))
    assert set(out["seed_id"]) == {1}
    assert "missing rollouts" in capsys.readouterr().out


def test_purity_damage_from_midbranch_wrong_hit():
    # Wrong hit mid-branch: state 1 takes a wrong hit (divergence at 0),
    # states after continue on truth. Past wrong count persists, so V of
    # later states carries the purity damage.
    st = _states(
        [(0, 0, "divergence", 100), (0, 1, "collapse", 555), (0, 2, "tip", 102)]
    )
    # rollout from state 0 replaces the wrong step-1 choice: finds 2 hits.
    fut = _futures([(0, 0, 2), (0, 2, 0)])
    past = _past([(0, 0, 1, 0), (0, 1, 1, 1), (0, 2, 2, 1)])
    out = compose_targets(st, fut, past, _ntot([(0, 3)]))
    v = out.set_index("step_k")["vstar_tier3"]
    # state 0: found = 1 + 2 = 3 -> min(3/3, 3/3) = 1 (mistake avoidable)
    assert np.isclose(v[0], 1.0)
    # state 2 (tip): found = 2 -> min(2/3, 2/3) = 2/3 (mistake sunk)
    assert np.isclose(v[2], 2 / 3)
    # state 1: found = n_correct(1) + inherited future = 1 + 1 = 2;
    # purity denominator = n_correct + n_wrong + findable = 1 + 1 + 1 = 3
    assert np.isclose(v[1], 2 / 3)


def test_truth_suffix_check_reports_agreement():
    st = _states(
        [
            (0, 0, "collapse", 100),
            (0, 1, "tip", 101),
            (1, 0, "divergence", 200),
            (1, 1, "tip", 201),
        ]
    )
    t3 = pd.DataFrame(
        {
            "seed_id": [0, 0, 1, 1],
            "step_k": [0, 1, 0, 1],
            "vstar_tier3": [1.0, 1.0, 0.5, 0.4],
        }
    )
    t2 = pd.DataFrame(
        {
            "seed_id": [0, 0, 1, 1],
            "step_k": [0, 1, 0, 1],
            "vstar_t2": [1.0, 0.995, 0.9, 0.9],
        }
    )
    rep = truth_suffix_check(st, t3, t2)
    # only seed 0 is a truth suffix; its diffs are 0 and 0.005 (< tol)
    assert rep["n_suffix_branches"] == 1
    assert rep["n_states_compared"] == 2
    assert rep["disagree_rate"] == 0.0
    assert np.isclose(rep["max_abs_diff"], 0.005)
