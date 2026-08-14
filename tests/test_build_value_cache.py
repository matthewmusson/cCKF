"""Tests for ``scripts.build_value_cache._state_features``.

These are hand-built synthetic fixtures (no real Parquet/simhits data needed)
that exercise the full per-state reduction, in particular the
``min_gate_logodds`` accumulator. A prior version computed it as
``cummin().fillna(0.0)``, which corrupts every hole step: pandas' ``cummin``
leaves a NaN row's own output NaN (the running minimum still passes through to
later rows correctly), and the trailing ``fillna(0.0)`` then overwrites
exactly those hole-step outputs with a literal ``0.0`` log-odds (``p=0.5``)
instead of carrying the true running-worst score forward. The fix is
``cummin()`` -> grouped ``ffill()`` -> ``fillna(0.0)`` (the last step covering
only leading holes, before any hit has been accepted).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cckf.features import chi2_log_odds
from scripts.build_value_cache import _state_features


def _row(
    seed_id: int,
    branch_id: int,
    step_k: int,
    path_x0: float,
    *,
    chi2: float | None = None,
    n_hits: int = 0,
    n_holes: int = 0,
    n_seq_holes: int = 0,
) -> dict:
    """One candidate row. ``chi2=None`` means a hole (no accepted candidate)."""
    is_hit = chi2 is not None
    return {
        "seed_id": seed_id,
        "branch_id": branch_id,
        "step_k": step_k,
        "cand_hit_id": 100 + step_k if is_hit else -1,
        "is_ckf_selected": is_hit,
        "chi2_inc": chi2 if is_hit else np.nan,
        "state_theta": np.pi / 2,  # eta = 0, irrelevant to the assertions here
        "state_qop": -0.5,
        "cov_00": 0.25,
        "cov_06": 0.36,
        "n_hits": n_hits,
        "n_holes": n_holes,
        "n_seq_holes": n_seq_holes,
        "pathInX0_interval": path_x0,
    }


def _synthetic_df() -> pd.DataFrame:
    """Two branches, back to back.

    Branch (seed 0, branch 0): 5 steps, accepted hits at 0/2/4 (chi2
    1.0/10.0/0.5), holes at 1/3. chi2=10.0 gives the branch's worst (most
    negative) log-odds, and it lands in the *middle* of the branch, so a
    correct implementation must carry that minimum across the hole at step 3
    and past the better (less negative) hit at step 4.

    Branch (seed 0, branch 1): 3 steps, LEADING hole at step 0 (no accepted
    hit yet — min_gate_logodds must be exactly 0.0, not leaked from branch 0's
    accumulated minimum), accepted hit at step 1 (chi2 2.0), hole at step 2.
    """
    rows = [
        _row(0, 0, 0, 0.01, chi2=1.0, n_hits=1, n_holes=0, n_seq_holes=0),
        _row(0, 0, 1, 0.02, n_hits=1, n_holes=1, n_seq_holes=1),
        _row(0, 0, 2, 0.01, chi2=10.0, n_hits=2, n_holes=1, n_seq_holes=0),
        _row(0, 0, 3, 0.02, n_hits=2, n_holes=2, n_seq_holes=1),
        _row(0, 0, 4, 0.01, chi2=0.5, n_hits=3, n_holes=2, n_seq_holes=0),
        _row(0, 1, 0, 0.05, n_hits=0, n_holes=1, n_seq_holes=1),
        _row(0, 1, 1, 0.05, chi2=2.0, n_hits=1, n_holes=1, n_seq_holes=0),
        _row(0, 1, 2, 0.05, n_hits=1, n_holes=2, n_seq_holes=1),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def per_state() -> pd.DataFrame:
    return _state_features(_synthetic_df())


@pytest.fixture
def logodds() -> dict[str, float]:
    """Expected chi2_log_odds for each chi2 value used, computed via the real
    library function so the test does not hardcode a hand-derived constant."""
    return {
        "a": float(chi2_log_odds(np.array([1.0]))[0]),
        "b": float(chi2_log_odds(np.array([10.0]))[0]),
        "c": float(chi2_log_odds(np.array([0.5]))[0]),
        "d": float(chi2_log_odds(np.array([2.0]))[0]),
    }


def _branch(per_state: pd.DataFrame, seed_id: int, branch_id: int) -> pd.DataFrame:
    return (
        per_state.loc[
            (per_state["seed_id"] == seed_id) & (per_state["branch_id"] == branch_id)
        ]
        .sort_values("step_k")
        .reset_index(drop=True)
    )


def test_min_gate_logodds_carries_running_worst_across_holes(per_state, logodds):
    """The core bug fix: a hole step must report the running worst accepted
    score, not reset to 0.0 (p=0.5)."""
    branch_a = _branch(per_state, 0, 0)
    a, b, c = logodds["a"], logodds["b"], logodds["c"]
    # chi2 10.0 (b) is more negative (worse) than both chi2 1.0 (a) and 0.5 (c).
    assert b < a < c
    expected = [a, a, min(a, b), min(a, b), min(min(a, b), c)]
    np.testing.assert_allclose(branch_a["min_gate_logodds"].to_numpy(), expected)
    # Concretely: step 1 (a hole right after the chi2=1.0 hit) must equal `a`,
    # not 0.0.
    assert branch_a.loc[branch_a["step_k"] == 1, "min_gate_logodds"].item() == pytest.approx(a)
    # step 3 (a hole after the chi2=10.0 hit, the branch's worst) must carry
    # that worst value forward, not reset to 0.0.
    assert branch_a.loc[branch_a["step_k"] == 3, "min_gate_logodds"].item() == pytest.approx(b)


def test_min_and_sum_gate_logodds_do_not_leak_across_branches(per_state, logodds):
    """Branch 1 immediately follows branch 0 in sorted order (both are
    seed_id=0). Branch 1's leading hole must read 0.0 — genuinely undefined,
    no accepted hit yet — not branch 0's accumulated minimum (which is very
    negative). An ungrouped ffill would leak it."""
    branch_b = _branch(per_state, 0, 1)
    d = logodds["d"]

    assert branch_b.loc[branch_b["step_k"] == 0, "min_gate_logodds"].item() == pytest.approx(0.0)
    assert branch_b.loc[branch_b["step_k"] == 0, "sum_gate_logodds"].item() == pytest.approx(0.0)

    expected_min = [0.0, d, d]
    np.testing.assert_allclose(branch_b["min_gate_logodds"].to_numpy(), expected_min)


def test_sum_gate_logodds_unaffected_by_the_min_fix(per_state, logodds):
    """sum_gate_logodds already filled NaN with 0.0 *before* cumsum (the
    correct additive identity for 'a hole contributes nothing'); the
    min_gate_logodds fix must not change this behaviour."""
    branch_a = _branch(per_state, 0, 0)
    a, b, c = logodds["a"], logodds["b"], logodds["c"]
    expected = [a, a, a + b, a + b, a + b + c]
    np.testing.assert_allclose(branch_a["sum_gate_logodds"].to_numpy(), expected)

    branch_b = _branch(per_state, 0, 1)
    d = logodds["d"]
    np.testing.assert_allclose(branch_b["sum_gate_logodds"].to_numpy(), [0.0, d, d])


def test_x0_accumulated_is_cumulative_per_branch_not_global(per_state):
    """x0_accumulated must reset per branch, not keep accumulating across the
    seed_id/branch_id boundary."""
    branch_a = _branch(per_state, 0, 0)
    branch_b = _branch(per_state, 0, 1)

    np.testing.assert_allclose(
        branch_a["x0_accumulated"].to_numpy(), [0.01, 0.03, 0.04, 0.06, 0.07]
    )
    # Branch B starts fresh at 0.05, not continuing from branch A's final 0.07.
    np.testing.assert_allclose(branch_b["x0_accumulated"].to_numpy(), [0.05, 0.10, 0.15])
