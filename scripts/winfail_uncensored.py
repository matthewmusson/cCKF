"""Uncensored window-failure and module-failure accumulation (parquet-only).

## Definitions

- **State**: one step of a surviving CKF output branch (`track_nr`; `seed_id`/`branch_id` are its aliases), branch majority defined. Every surviving branch contributes - the data is pre-ambiguity and the envelope disabled all terminal cuts, so the only branch-level conditioning is survival of in-flight pruning (χ² gates, branch cap 5). Branch inflation across a physical seed is present and noted in the footer.
- **On-surface state**: `majority_true_hit_on_surface = True`.
- **True-hit row**: candidate row whose `contrib_pids` contains `branch_majority_pid`. Per-row distance `d = max(|residual_l0|/√S00, |residual_l1|/√S11)`; on 1D rows (`is_1d`) the l0 leg only. Per-state `d_true = min` over its true-hit rows (a particle can leave two measurements on one module via overlaps; the CKF needs any one in the window). NaN if the state has no true-hit row.
- **Module failure** (per state): `NOT on_surface`.
- **Window failure at n** (per on-surface state): `d_true > n`, or `d_true` is NaN (true hit escaped the 10-box or was never digitized). Uncensored: the n=10 line equals the escaped fraction and is nonzero.
- **Seed purity**: pure = the majority particle contributes to the CKF-selected candidate at each of the branch's first 3 measurement steps (3/3); otherwise majority.
- **Occupancy** (`window_occupancy`): the state's `n_window` (candidate count in the n=10 box). Strata edges: [0,2), [2,5), [5,10), [10,20), [20,∞).
- **pT bins**: majority-particle pT binned at `PT_EDGES = (0.0, 0.7, 0.9, 1.0)`, last bin open; renders sum bins at/above a threshold that must be an edge. Primary renders: pT > 1.0 and pT > 0.9 GeV.
- **Ambi survivor**: the branch is kept by an offline replica of ACTS `GreedyAmbiguityResolution` with `maximumSharedHits = 3` and `nMeasurementsMin = 7` (the stage-1 steering's fallback when the CKF cut is disabled). Branch hits = its `is_ckf_selected` rows' `cand_hit_id` (measurement ids, unique per event); branch chi2 = sum of those rows' `chi2_inc`. Replica semantics: drop branches with < 7 accepted hits, then iteratively evict the branch with the highest shared-hit fraction (ties: fewer hits, then higher chi2) until every branch shares < 3 hits. Stage 1 wrote no post-ambi reference, so the replica is the only route; the chi2 tie-break uses the shared-S approximation - a documented fidelity caveat affecting only exact ties.
- η: from `state_theta` (branch state direction), 140 bins of width 0.05 over [−3.5, 3.5].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ETA_BINS: np.ndarray = np.linspace(-3.5, 3.5, 141)  # 140 bins, width 0.05
N_VALUES: tuple[float, ...] = (3.0, 5.0, 7.0, 10.0)
OCC_EDGES: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0)  # last bin open
PT_EDGES: tuple[float, ...] = (0.0, 0.7, 0.9, 1.0)  # last bin open; render
# thresholds must be edges of this tuple
SENSOR_VOLUMES: dict[int, int] = {
    16: 0, 17: 0, 18: 0,   # pixel
    23: 1, 24: 1, 25: 1,   # short strip (2D)
    28: 2, 29: 2, 30: 2,   # long strip (1D)
}

# Single source of truth for external column names (corrected by Task 1).
SCHEMA: dict[str, str] = {
    "simhit_hit_id": "hit_id",
    "simhit_particle_id": "particle_id",
    "raw_simhit_tpx": "tpx",
    "raw_simhit_tpy": "tpy",
    "raw_simhit_tt": "tt",
}


def wilson_interval(
    k: np.ndarray, n: np.ndarray, z: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """1-sigma (z=1) Wilson score interval; NaN where n == 0."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1.0 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        lo = np.clip(center - half, 0.0, 1.0)
        hi = np.clip(center + half, 0.0, 1.0)
    empty = n <= 0
    return np.where(empty, np.nan, lo), np.where(empty, np.nan, hi)


def assign_strata(
    eta: np.ndarray, volume_id: np.ndarray, n_window: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(eta, volume, occupancy) -> (eta_idx, sensor_idx, occ_idx).

    sensor_idx is -1 outside the nine tracker volumes (e.g. passive volume
    20); callers count and exclude those rows, never silently drop them.
    """
    eta_idx = np.clip(np.digitize(eta, ETA_BINS) - 1, 0, len(ETA_BINS) - 2)
    sensor_idx = np.array(
        [SENSOR_VOLUMES.get(int(v), -1) for v in volume_id], dtype=np.int64
    )
    occ = np.nan_to_num(np.asarray(n_window, dtype=float), nan=0.0)
    occ_idx = np.clip(np.digitize(occ, OCC_EDGES) - 1, 0, len(OCC_EDGES) - 1)
    return eta_idx, sensor_idx, occ_idx


def select_ckf_branch(rows: pd.DataFrame) -> pd.DataFrame:
    """One (seed_id, branch_id) per seed: the branch the CKF followed.

    The branch with the most is_ckf_selected rows; ties break to the lowest
    branch_id. Seeds with zero selected rows are excluded.

    On stage-1 envelope data seed_id == branch_id == track_nr, so this is an
    identity over branches with at least one selected row - kept as a guard
    and for datasets that do carry real seed grouping.
    """
    picked = rows[rows["is_ckf_selected"]]
    counts = (
        picked.groupby(["seed_id", "branch_id"])
        .size()
        .rename("n_sel")
        .reset_index()
        .sort_values(
            ["seed_id", "n_sel", "branch_id"], ascending=[True, False, True]
        )
    )
    return counts.drop_duplicates("seed_id")[["seed_id", "branch_id"]].reset_index(
        drop=True
    )


def branch_purity(rows: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Pure (3/3) vs majority seed, from the first 3 selected candidates."""
    sel_rows = rows.merge(selected, on=["seed_id", "branch_id"])
    picks = (
        sel_rows[sel_rows["is_ckf_selected"]]
        .sort_values("step_k")
        .groupby("seed_id")
        .head(3)
    )

    def _all_match(g: pd.DataFrame) -> bool:
        maj = g["branch_majority_pid"].iloc[0]
        return all(
            maj in (c if c is not None else []) for c in g["contrib_pids"]
        )

    return (
        picks.groupby("seed_id")
        .apply(_all_match, include_groups=False)
        .rename("is_pure")
        .reset_index()
    )


def flag_ambi_survivors(
    rows: pd.DataFrame, max_shared: int = 3, nmeas_min: int = 7
) -> pd.DataFrame:
    """Offline replica of ACTS GreedyAmbiguityResolution over parquet branches.

    Branch hits = is_ckf_selected rows' cand_hit_id; branch chi2 = summed
    chi2_inc of those rows. Mirrors
    Core/src/AmbiguityResolution/GreedyAmbiguityResolution.cpp: drop branches
    with fewer than nmeas_min hits, then iteratively evict the branch with
    the highest shared-hit fraction (ties: fewer hits, then higher chi2)
    until every branch shares fewer than max_shared hits.

    Returns one row per seed_id in `rows` with a survived_ambi bool.
    """
    picked = rows[rows["is_ckf_selected"]]
    hits: dict[int, list[int]] = (
        picked.groupby("seed_id")["cand_hit_id"].apply(list).to_dict()
    )
    chi2: dict[int, float] = (
        picked.groupby("seed_id")["chi2_inc"].sum().to_dict()
    )
    all_seeds = rows["seed_id"].unique()

    sel = {s for s, h in hits.items() if len(h) >= nmeas_min}
    tracks_per_hit: dict[int, set[int]] = {}
    for s_ in sel:
        for h in hits[s_]:
            tracks_per_hit.setdefault(h, set()).add(s_)
    shared = {
        s_: sum(1 for h in hits[s_] if len(tracks_per_hit[h]) > 1)
        for s_ in sel
    }

    def evict_key(s_: int) -> tuple[float, int, float]:
        rel = shared[s_] / len(hits[s_]) if hits[s_] else 0.0
        return (rel, -len(hits[s_]), chi2[s_])

    while sel:
        worst = max(sel, key=evict_key)
        if shared[worst] < max_shared:
            break
        for h in hits[worst]:
            tracks_per_hit[h].discard(worst)
            if len(tracks_per_hit[h]) == 1:
                (j,) = tracks_per_hit[h]
                if j in sel:
                    shared[j] -= 1
        sel.discard(worst)

    return pd.DataFrame(
        {"seed_id": all_seeds, "survived_ambi": [s_ in sel for s_ in all_seeds]}
    )
