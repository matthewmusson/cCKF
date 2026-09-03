# Uncensored Window-Failure and Module-Failure Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Window-failure and module-failure rate plots with an *uncensored* denominator, dense η binning (140 bins), three-way sensor stratification, seed-purity and occupancy stratification, 1σ Wilson intervals, and the truth-pT threshold marked on every figure, including the existing Pareto overlay.

**Architecture:** Parquet-only. The escape mass is identifiable by subtraction: a state with `majority_true_hit_on_surface = True` but no true-hit candidate row is a state whose true hit escaped the n=10 expansion box (or was never digitized - conflated, see Known limits). So for n ≤ 10: fail(n) = (true-hit rows with d > n) + (on-surface states with no true-hit row); denominator = all on-surface states of the CKF-selected branch. A per-event NERSC job reduces each re-expanded parquet to small count tensors (one light simhit-momentum join supplies majority-particle pT, which the parquet lacks); a local script renders four plot families. No measurements/predicted-cov joins - those are only needed for n > 10 lines and the digitization split (optional Task 8, not scheduled).

**Tech Stack:** Python 3.10+, pandas, numpy, pyarrow, matplotlib. NERSC Perlmutter (sbatch, `module load python`) for accumulation; local machine for tests and rendering.

## Global Constraints

- Events `[32, 64)` are sealed. Only events `[0, 32)` are ever read.
- Code sync local ↔ NERSC CFS is `git push` local → `git pull` on CFS. Never scp source files.
- Every figure carries the truth-pT threshold and the denominator conditions, briefly (memory rule `mark-pt-threshold`).
- No silent scope cuts: excluded populations (volume 20 states; multi-true-row states are counted, not excluded) are counted and printed.
- Style: type hints, NumPy docstrings, Black at 88 chars.
- SSH: `ssh -i ~/.ssh/nersc -o IdentitiesOnly=yes mussonm@perlmutter.nersc.gov` (anything else → "Too many authentication failures").
- Harness truth selection (recorded from `digi_and_reco.py:620`): pT > 0.999 GeV, |η| < 3, ≥6 measurements, ≥3 pixel hits, ρ < 24 mm, |z| < 1 m, charged. Quote as "pT > 1 GeV".

## Recorded facts (verified 2026-09-02, this session)

- Re-expanded parquets: `$SCRATCH/cckf/reexpanded/expanded_event{E:09d}.parquet`, 82 columns, all 32 events. Columns used here: `seed_id, branch_id, step_k, volume_id, state_theta, cand_hit_id, residual_l0, residual_l1, S00, S11, is_1d, n_window, is_ckf_selected, contrib_pids, branch_majority_pid, majority_undefined, majority_true_hit_on_surface`.
- The expansion stored only candidates inside the n=10 axis-aligned box (audit: zero box violations in 259.9M rows), per-dimension `|r_i| ≤ 10·√S_ii`, l0-leg only on 1D long strips. Per-candidate S is stored (`S00/S11` columns; NaN `S11` on 1D rows).
- `majority_true_hit_on_surface` is simhit-level: the majority particle deposited on the module, whether or not digitization produced a measurement.
- `truth_residual_l0/l1` are NaN placeholders - no escaped-hit distances exist in the parquet (that is what Task 8 would add).
- Sensor groups (LOG 2026-08-25): pixel = 16/17/18, short strip = 23/24/25, long strip = 28/29/30. Volume 20 is passive (event 4: 1,658,237 rows, all "no measurements on surface" holes, zero candidates) - excluded from panels, counted.
- Majority-particle pT is NOT in the parquet (states carry the branch's reconstructed q/p, not truth pT) → one light join: simhit momentum by hit id, per Task 1 recon.
- **Stage-1 collection config (envelope.yaml, spec §6.2)** - report on every figure: seeding 46 seeds/spm, seed minPt 0.587 GeV, impactMax 2.86 mm, σ_scat 2.95; CKF χ²_meas < 16.26, χ²_outlier < 35.75, branch cap 5; **terminal cuts disabled** (no nMeasurementsMin, no maxHoles, no ptMin, no loc0 cut). The expansion reads `trackstates_ckf.root` = the CKF output **before ambiguity resolution**.
- **`seed_id` == `branch_id` == ROOT `track_nr`** (schema alias map, `expansion.py:1588`). The parquet's unit is one *surviving output branch*; there is no seed grouping. Multiple surviving branches of the same physical seed appear as distinct seed_ids, so per-state rates are branch-inflated where near-duplicate branches survive. `select_ckf_branch` is therefore an identity on this dataset (kept as a guard; documented).
- `prune_reason` and `parent_branch_id` are unpopulated placeholders (default −1): branches pruned mid-flight left no states and are NOT recoverable from the parquet. All rates are conditioned on branch survival to CKF output.
- Job 57877462 (`modfail_v1`, censored variant) is an independent cross-check; no dependency.

## Definitions (copy into the module docstring)

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

## File Structure

- Create `scripts/winfail_uncensored.py` - constants, Wilson interval, strata, branch selection, purity, per-state reduction, tensor accumulation, per-event CLI.
- Create `scripts/plot_winfail_uncensored.py` - npz aggregation + four plot families.
- Create `scripts/winfail_unc.sbatch` - batch over 32 events.
- Create `tests/test_winfail_uncensored.py` - unit tests, synthetic frames.
- Create `scripts/plot_pareto_overlay.py` - Pareto overlay with threshold footer (Task 7).

Tensor contract (produced by Task 4, consumed by Tasks 5-6):

```
win_total : int64 [140 eta, 3 sensor, 2 purity, 5 occ, 4 ptbin, 2 ambi]  # on-surface states
win_fail  : int64 [4 n, 140, 3, 2, 5, 4, 2]                              # n in (3, 5, 7, 10)
mod_total : int64 [140, 3, 2, 5, 4, 2]                                   # all states
mod_fail  : int64 [140, 3, 2, 5, 4, 2]                                   # not on surface
counters  : {n_states, n_vol20, n_escaped, n_multi_true, n_pt_unmatched,
             n_branches, n_ambi_survivors}
```

Branch-class (ambi) axis: index 1 = the branch survives an offline replica of
the ACTS greedy ambiguity resolution; index 0 = it does not. "All CKF-output
branches" = sum over the axis; "ambi survivors" = slice 1. Truly pruned
branches (killed in flight) left no states and can never appear in either
class.

---

### Task 1: Recon - simhit momentum source for majority-particle pT

**Files:**
- Modify: this file (fill "Recon results" below)

**Interfaces:**
- Produces: verified columns of `expansion.load_simhits` output and of the raw simhits CSV; the join key linking momentum to the custom-encoded particle id (must be hit id, never barcode re-encoding).

- [ ] **Step 1: Run the recon on NERSC**

```bash
ssh -i ~/.ssh/nersc -o IdentitiesOnly=yes mussonm@perlmutter.nersc.gov '
module load python 2>/dev/null
cd /global/cfs/cdirs/atlas/mussonm/cCKF
CCKF_STAGE1_BASE=$SCRATCH/cckf/modal_backup/results python3 - <<PYEOF
from cckf.stage1_map import csv_dir_for
import expansion as E
import pandas as pd, glob
d = str(csv_dir_for(4))
sim = E.load_simhits(d, 4)
print("load_simhits ->", list(sim.columns))
print(sim.head(2).to_string())
raw = sorted(glob.glob(d + "/*simhits*"))[0]
print("raw:", raw)
print(pd.read_csv(raw, nrows=2).to_string())
PYEOF'
```

- [ ] **Step 2: Record results here**

```markdown
#### Recon results (Task 1)
- load_simhits columns: hit_id, geometry_id, particle_id, tx, ty, tz
- momentum columns (tpx/tpy or equivalent) and where they live: tpx, tpy, tpz in raw CSV; pT = hypot(tpx, tpy)
- hit-id join key between load_simhits and the raw CSV: load_simhits.hit_id is the row index in the raw CSV (0-based, 0 to N-1). Verified on full file (event 4): hit_id == [0..273885), tx values align exactly across all 273,885 rows (verified via full-file NERSC check).
- particle_id encoding status: load_simhits.particle_id is custom-encoded as 19-digit integers (example: 19704042944135862); raw CSV stores particle_id as 5 separate components (particle_id_pv, particle_id_sv, particle_id_part, particle_id_gen, particle_id_subpart); load_simhits' encoding is pre-computed, not derivable from raw components by re-encoding barcodes. Earliest simhit per particle: raw CSV has `tt` column for time ordering; use minimum `tt` (or minimum hit_id as fallback if tt is missing in other events).
```

- [ ] **Step 3: Fix the `SCHEMA` dict of Task 2 if any name differs; column names appear nowhere else.**

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-09-02-uncensored-window-failure-plots.md
git commit -m "docs: record simhit momentum recon for uncensored winfail plan"
```

---

### Task 2: Module skeleton - constants, Wilson interval, strata assignment

**Files:**
- Create: `scripts/winfail_uncensored.py`
- Test: `tests/test_winfail_uncensored.py`

**Interfaces:**
- Produces: `wilson_interval(k, n, z=1.0) -> tuple[np.ndarray, np.ndarray]`; `assign_strata(eta, volume_id, n_window) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (eta_idx, sensor_idx with −1 for unknown volumes, occ_idx); constants `ETA_BINS, N_VALUES, OCC_EDGES, SENSOR_VOLUMES, PT_EDGES, SCHEMA`.

- [ ] **Step 1: Write the failing tests**

```python
"""tests/test_winfail_uncensored.py"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winfail_uncensored import (
    ETA_BINS, N_VALUES, OCC_EDGES, PT_EDGES, wilson_interval, assign_strata,
)


def test_eta_binning_is_dense():
    assert len(ETA_BINS) == 141
    assert np.isclose(ETA_BINS[0], -3.5) and np.isclose(ETA_BINS[-1], 3.5)
    assert np.allclose(np.diff(ETA_BINS), 0.05)


def test_pt_edges():
    assert PT_EDGES == (0.0, 0.7, 0.9, 1.0)


def test_wilson_interval_basic():
    lo, hi = wilson_interval(np.array([5]), np.array([10]), z=1.0)
    assert 0.0 < lo[0] < 0.5 < hi[0] < 1.0


def test_wilson_interval_zero_denominator_is_nan():
    lo, hi = wilson_interval(np.array([0]), np.array([0]))
    assert np.isnan(lo[0]) and np.isnan(hi[0])


def test_wilson_interval_extremes_stay_in_unit_interval():
    lo, hi = wilson_interval(np.array([0, 10]), np.array([10, 10]))
    assert lo[0] >= 0.0 and hi[1] <= 1.0


def test_assign_strata_sensor_groups_and_vol20():
    eta = np.array([0.0, 0.0, 0.0, 0.0])
    vol = np.array([17, 24, 29, 20])
    occ = np.array([0, 3, 12, 50])
    ei, si, oi = assign_strata(eta, vol, occ)
    assert list(si) == [0, 1, 2, -1]
    assert list(oi) == [0, 1, 3, 4]
    assert ei[0] == 70
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/matthewm/SURP/cCKF && python3 -m pytest tests/test_winfail_uncensored.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'winfail_uncensored'`

- [ ] **Step 3: Write the implementation**

```python
"""scripts/winfail_uncensored.py

Uncensored window-failure and module-failure accumulation (parquet-only).

[Paste the plan's "Definitions" section here verbatim.]
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_winfail_uncensored.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/winfail_uncensored.py tests/test_winfail_uncensored.py
git commit -m "feat: winfail_uncensored skeleton - strata, Wilson intervals"
```

---

### Task 3: Selected-branch derivation, seed purity, ambi-survivor flag

**Files:**
- Modify: `scripts/winfail_uncensored.py`
- Test: `tests/test_winfail_uncensored.py`

**Interfaces:**
- Consumes: parquet-shaped DataFrame with `seed_id, branch_id, step_k, cand_hit_id, is_ckf_selected, contrib_pids, branch_majority_pid, chi2_inc`.
- Produces: `select_ckf_branch(rows) -> pd.DataFrame[(seed_id, branch_id)]`, one row per seed; `branch_purity(rows, selected) -> pd.DataFrame[(seed_id, is_pure)]`; `flag_ambi_survivors(rows, max_shared=3, nmeas_min=7) -> pd.DataFrame[(seed_id, survived_ambi)]` covering every seed_id present in `rows`.

- [ ] **Step 1: Write the failing tests**

```python
from winfail_uncensored import select_ckf_branch, branch_purity


def _rows(records):
    cols = ["seed_id", "branch_id", "step_k", "cand_hit_id",
            "is_ckf_selected", "contrib_pids", "branch_majority_pid"]
    return pd.DataFrame(records, columns=cols)


def test_select_ckf_branch_picks_most_selected_rows():
    rows = _rows([
        (0, 0, 0, 11, True,  [7], 7),
        (0, 0, 1, 12, True,  [7], 7),
        (0, 1, 0, 11, True,  [7], 7),
        (1, 5, 0, 21, True,  [9], 9),
    ])
    sel = select_ckf_branch(rows)
    assert dict(zip(sel.seed_id, sel.branch_id)) == {0: 0, 1: 5}


def test_select_ckf_branch_tie_breaks_to_lowest_branch():
    rows = _rows([
        (3, 2, 0, 1, True, [1], 1),
        (3, 1, 0, 1, True, [1], 1),
    ])
    sel = select_ckf_branch(rows)
    assert dict(zip(sel.seed_id, sel.branch_id)) == {3: 1}


def test_branch_purity_three_of_three_is_pure():
    rows = _rows([
        (0, 0, 0, 11, True, [7],    7),
        (0, 0, 1, 12, True, [7, 8], 7),
        (0, 0, 2, 13, True, [7],    7),
        (0, 0, 3, 14, True, [8],    7),   # step 4 wrong: irrelevant to purity
        (1, 0, 0, 21, True, [9],    9),
        (1, 0, 1, 22, True, [4],    9),   # 2/3 -> majority
        (1, 0, 2, 23, True, [9],    9),
    ])
    sel = select_ckf_branch(rows)
    pur = branch_purity(rows, sel)
    assert dict(zip(pur.seed_id, pur.is_pure)) == {0: True, 1: False}


def _arows(records):
    cols = ["seed_id", "step_k", "cand_hit_id", "is_ckf_selected", "chi2_inc"]
    return pd.DataFrame(records, columns=cols)


def test_flag_ambi_survivors_short_branch_dropped():
    rows = _arows([(0, k, 100 + k, True, 1.0) for k in range(7)]
                  + [(1, k, 200 + k, True, 1.0) for k in range(4)])
    out = flag_ambi_survivors(rows)
    assert dict(zip(out.seed_id, out.survived_ambi)) == {0: True, 1: False}


def test_flag_ambi_survivors_evicts_worse_of_overlapping_pair():
    # Branches 0 and 1 share hits 100-106 (7 shared >= max_shared=3).
    # Branch 1 has the higher shared fraction (7/7 vs 7/9) and is evicted;
    # afterwards branch 0 shares nothing and survives.
    rows = _arows(
        [(0, k, 100 + k, True, 1.0) for k in range(7)]
        + [(0, 7, 300, True, 1.0), (0, 8, 301, True, 1.0)]
        + [(1, k, 100 + k, True, 1.0) for k in range(7)]
    )
    out = flag_ambi_survivors(rows)
    assert dict(zip(out.seed_id, out.survived_ambi)) == {0: True, 1: False}


def test_flag_ambi_survivors_disjoint_branches_both_survive():
    rows = _arows([(0, k, 100 + k, True, 1.0) for k in range(8)]
                  + [(1, k, 500 + k, True, 1.0) for k in range(8)])
    out = flag_ambi_survivors(rows)
    assert dict(zip(out.seed_id, out.survived_ambi)) == {0: True, 1: True}
```

- [ ] **Step 2: Run to verify FAIL** (`ImportError: cannot import name 'select_ckf_branch'`)

- [ ] **Step 3: Write the implementation**

```python
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

    Branch hits = is_ckf_selected rows` cand_hit_id; branch chi2 = summed
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
```

- [ ] **Step 4: Run tests; all PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/winfail_uncensored.py tests/test_winfail_uncensored.py
git commit -m "feat: branch selection, seed purity, ambi-survivor replica"
```

---

### Task 4: Per-state reduction and tensor accumulation (parquet-only)

**Files:**
- Modify: `scripts/winfail_uncensored.py`
- Test: `tests/test_winfail_uncensored.py`

**Interfaces:**
- Consumes: Tasks 2-3.
- Produces:
  - `build_state_table(rows: pd.DataFrame) -> pd.DataFrame` - collapses candidate+hole rows of selected, majority-defined branches to one row per state with columns `(seed_id, branch_id, step_k, volume_id, state_theta, n_window, branch_majority_pid, on_surface, d_true, n_true_rows)`; `d_true` = min over true-hit rows of `max(|residual_l0|/√S00, |residual_l1|/√S11)` (l0 leg only where `is_1d`), NaN when the state has no true-hit row.
  - `accumulate_event(states: pd.DataFrame, pt_lut: pd.DataFrame) -> dict` - the tensor contract; `states` must carry a `survived_ambi` bool column (last tensor axis); `pt_lut` has columns `(particle_id, pt_gev)`; unmatched majority pids land in pT bin 0 and are counted in `n_pt_unmatched`.

- [ ] **Step 1: Write the failing tests**

```python
from winfail_uncensored import build_state_table, accumulate_event


def _prows(records):
    cols = ["seed_id", "branch_id", "step_k", "volume_id", "state_theta",
            "n_window", "cand_hit_id", "residual_l0", "residual_l1",
            "S00", "S11", "is_1d", "is_ckf_selected", "contrib_pids",
            "branch_majority_pid", "majority_undefined",
            "majority_true_hit_on_surface", "chi2_inc"]
    return pd.DataFrame(records, columns=cols)


HALF_PI = float(np.pi / 2)  # eta = 0


def test_build_state_table_taxonomy():
    rows = _prows([
        # state A: true-hit row at d = 4.0  (r0=0.8, S00=0.04; r1 small)
        (0, 0, 0, 17, HALF_PI, 3, 11, 0.8, 0.02, 0.04, 0.04, False, True, [7], 7, False, True),
        # ...and a wrong-hit row on the same state (must not affect d_true)
        (0, 0, 0, 17, HALF_PI, 3, 12, 0.1, 0.01, 0.04, 0.04, False, False, [8], 7, False, True),
        # state B: on surface, NO true-hit row -> escaped (d_true NaN)
        (1, 0, 0, 17, HALF_PI, 1, 13, 0.1, 0.01, 0.04, 0.04, False, True, [5], 9, False, True),
        # state C: hole row, majority not on surface -> module failure
        (2, 0, 0, 17, HALF_PI, 0, -1, np.nan, np.nan, np.nan, np.nan, False, False, None, 4, False, False),
    ])
    st = build_state_table(rows)
    assert len(st) == 3
    a = st[st.seed_id == 0].iloc[0]
    assert np.isclose(a.d_true, 4.0)
    b = st[st.seed_id == 1].iloc[0]
    assert b.on_surface and np.isnan(b.d_true)
    c = st[st.seed_id == 2].iloc[0]
    assert not c.on_surface


def test_build_state_table_min_over_two_true_rows_and_1d():
    rows = _prows([
        # two true-hit rows on one state (module overlap): d = 4.0 and d = 1.0
        (0, 0, 0, 29, HALF_PI, 2, 11, 0.8, np.nan, 0.04, np.nan, True, True, [7], 7, False, True),
        (0, 0, 0, 29, HALF_PI, 2, 12, 0.2, np.nan, 0.04, np.nan, True, False, [7], 7, False, True),
    ])
    st = build_state_table(rows)
    assert len(st) == 1
    assert np.isclose(st.iloc[0].d_true, 1.0)   # min wins; 1D uses l0 leg only
    assert st.iloc[0].n_true_rows == 2


def test_accumulate_event_window_and_module_counts():
    st = pd.DataFrame({
        "seed_id": [0, 1, 2], "branch_id": [0, 0, 0], "step_k": [0, 0, 0],
        "volume_id": [17, 17, 17], "state_theta": [HALF_PI] * 3,
        "n_window": [3, 1, 0], "branch_majority_pid": [7, 9, 4],
        "on_surface": [True, True, False],
        "d_true": [4.0, np.nan, np.nan], "n_true_rows": [1, 0, 0],
        "is_pure": [True, True, True],
        "survived_ambi": [True, False, True],
    })
    pt_lut = pd.DataFrame({"particle_id": [7, 9, 4], "pt_gev": [2.0, 2.0, 2.0]})
    out = accumulate_event(st, pt_lut)
    hi = 3  # pt bin [1.0, inf)
    assert out["mod_total"][:, :, :, :, hi, :].sum() == 3
    assert out["mod_fail"][:, :, :, :, hi, :].sum() == 1
    assert out["win_total"][:, :, :, :, hi, :].sum() == 2   # on-surface states
    # ambi axis: state A is on a surviving branch, state B is not
    assert out["win_total"][:, :, :, :, hi, 1].sum() == 1
    assert out["win_total"][:, :, :, :, hi, 0].sum() == 1
    # state A (d=4.0): fails n=3 only. state B (escaped): fails ALL n incl 10.
    n_idx = {n: i for i, n in enumerate((3.0, 5.0, 7.0, 10.0))}
    assert out["win_fail"][n_idx[3.0]].sum() == 2
    assert out["win_fail"][n_idx[5.0]].sum() == 1
    assert out["win_fail"][n_idx[10.0]].sum() == 1
    assert out["counters"]["n_escaped"] == 1


def test_accumulate_event_pt_binning_and_unmatched():
    st = pd.DataFrame({
        "seed_id": [0, 1], "branch_id": [0, 0], "step_k": [0, 0],
        "volume_id": [17, 17], "state_theta": [HALF_PI] * 2,
        "n_window": [1, 1], "branch_majority_pid": [7, 8],
        "on_surface": [True, True], "d_true": [1.0, 1.0],
        "n_true_rows": [1, 1], "is_pure": [False, False],
        "survived_ambi": [True, True],
    })
    pt_lut = pd.DataFrame({"particle_id": [7], "pt_gev": [0.75]})  # 8 unmatched
    out = accumulate_event(st, pt_lut)
    assert out["mod_total"][:, :, :, :, 1, :].sum() == 1   # [0.7, 0.9)
    assert out["mod_total"][:, :, :, :, 0, :].sum() == 1   # unmatched -> bin 0
    assert out["counters"]["n_pt_unmatched"] == 1
```

- [ ] **Step 2: Run to verify FAIL** (`ImportError`)

- [ ] **Step 3: Write the implementation**

```python
def build_state_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse candidate+hole rows to one row per (seed, branch, step).

    Input rows must already be restricted to selected, majority-defined
    branches. See module docstring for d_true.
    """
    maj = rows["branch_majority_pid"].to_numpy()
    is_true = np.fromiter(
        (
            (m in (c if c is not None else []))
            for m, c in zip(maj, rows["contrib_pids"])
        ),
        dtype=bool,
        count=len(rows),
    ) & (rows["cand_hit_id"].to_numpy() >= 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        d0 = np.abs(rows["residual_l0"].to_numpy()) / np.sqrt(
            rows["S00"].to_numpy()
        )
        d1 = np.abs(rows["residual_l1"].to_numpy()) / np.sqrt(
            rows["S11"].to_numpy()
        )
    is1d = rows["is_1d"].to_numpy().astype(bool)
    d = np.where(is1d, d0, np.maximum(d0, d1))

    work = rows[
        ["seed_id", "branch_id", "step_k", "volume_id", "state_theta",
         "n_window", "branch_majority_pid",
         "majority_true_hit_on_surface"]
    ].copy()
    work["_d"] = np.where(is_true & np.isfinite(d), d, np.nan)
    work["_is_true"] = is_true

    g = work.groupby(["seed_id", "branch_id", "step_k"], as_index=False)
    out = g.agg(
        volume_id=("volume_id", "first"),
        state_theta=("state_theta", "first"),
        n_window=("n_window", "first"),
        branch_majority_pid=("branch_majority_pid", "first"),
        on_surface=("majority_true_hit_on_surface", "first"),
        d_true=("_d", "min"),
        n_true_rows=("_is_true", "sum"),
    )
    out["on_surface"] = out["on_surface"].astype(bool)
    return out


def accumulate_event(states: pd.DataFrame, pt_lut: pd.DataFrame) -> dict:
    """Fill the failure tensors for one event. See module docstring."""
    n_eta, n_sen, n_pur, n_occ = len(ETA_BINS) - 1, 3, 2, len(OCC_EDGES)
    n_pt = len(PT_EDGES)
    shape = (n_eta, n_sen, n_pur, n_occ, n_pt, 2)  # trailing axis: survived_ambi
    mod_total = np.zeros(shape, np.int64)
    mod_fail = np.zeros(shape, np.int64)
    win_total = np.zeros(shape, np.int64)
    win_fail = np.zeros((len(N_VALUES),) + shape, np.int64)

    st = states.merge(
        pt_lut, left_on="branch_majority_pid", right_on="particle_id",
        how="left",
    )
    n_pt_unmatched = int(st["pt_gev"].isna().sum())

    theta = np.clip(st["state_theta"].to_numpy(), 1e-10, np.pi - 1e-10)
    eta = -np.log(np.tan(theta / 2.0))
    ei, si, oi = assign_strata(
        eta, st["volume_id"].to_numpy(), st["n_window"].to_numpy()
    )
    pi = st["is_pure"].to_numpy().astype(np.int64)
    ti = np.clip(
        np.digitize(np.nan_to_num(st["pt_gev"].to_numpy(), nan=0.0), PT_EDGES)
        - 1,
        0,
        n_pt - 1,
    )

    keep = (si >= 0) & np.isfinite(eta)
    n_vol20 = int((si < 0).sum())

    ai = st["survived_ambi"].to_numpy().astype(np.int64)

    def _at(tensor: np.ndarray, mask: np.ndarray) -> None:
        np.add.at(
            tensor,
            (ei[mask], si[mask], pi[mask], oi[mask], ti[mask], ai[mask]),
            1,
        )

    _at(mod_total, keep)
    on = st["on_surface"].to_numpy().astype(bool)
    _at(mod_fail, keep & ~on)

    w = keep & on
    _at(win_total, w)
    d = st["d_true"].to_numpy()
    escaped = w & ~np.isfinite(d)
    for ni, n in enumerate(N_VALUES):
        f = escaped | (w & (d > n))
        np.add.at(
            win_fail,
            (np.full(int(f.sum()), ni), ei[f], si[f], pi[f], oi[f], ti[f], ai[f]),
            1,
        )

    return {
        "mod_total": mod_total, "mod_fail": mod_fail,
        "win_total": win_total, "win_fail": win_fail,
        "counters": {
            "n_states": int(len(st)),
            "n_vol20": n_vol20,
            "n_escaped": int(escaped.sum()),
            "n_multi_true": int((st["n_true_rows"].to_numpy() > 1).sum()),
            "n_pt_unmatched": n_pt_unmatched,
            "n_branches": int(st["seed_id"].nunique()),
            "n_ambi_survivors": int(
                st.drop_duplicates("seed_id")["survived_ambi"].sum()
            ),
        },
    }
```

- [ ] **Step 4: Run tests; all PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/winfail_uncensored.py tests/test_winfail_uncensored.py
git commit -m "feat: parquet-only uncensored per-state reduction and accumulation"
```

---

### Task 5: pT lookup, per-event CLI, batch over 32 events

**Files:**
- Modify: `scripts/winfail_uncensored.py` (add `particle_pt_lookup` and `main()`)
- Create: `scripts/winfail_unc.sbatch`

**Interfaces:**
- Consumes: Tasks 2-4; `expansion.load_simhits`; `cckf.stage1_map.csv_dir_for`.
- Produces: `$SCRATCH/cckf/winfail_unc/winfail_unc_event{E:03d}.npz` per event: the tensor contract + `eta_bins, n_values, occ_edges, pt_edges` + counters as scalar arrays.

- [ ] **Step 1: Implement `particle_pt_lookup(csv_dir: str, event_id: int) -> pd.DataFrame`**

Contract (exact columns fixed by Task 1 recon): join simhit momentum to the custom-encoded `particle_id` **by hit id** (never by re-encoding barcodes), take each particle's earliest simhit, return `(particle_id, pt_gev)` with `pt_gev = hypot(tpx, tpy)`. The value is only consumed through `np.digitize` against `PT_EDGES`.

- [ ] **Step 2: Add `main()`**

```python
def main() -> None:
    import argparse
    import os

    import pyarrow.parquet as pq

    from cckf.stage1_map import csv_dir_for

    ap = argparse.ArgumentParser()
    ap.add_argument("event", type=int)
    ap.add_argument("scratch_base")
    args = ap.parse_args()
    ev, S = args.event, args.scratch_base

    pq_path = f"{S}/reexpanded/expanded_event{ev:09d}.parquet"
    cols = ["seed_id", "branch_id", "step_k", "volume_id", "state_theta",
            "n_window", "cand_hit_id", "residual_l0", "residual_l1",
            "S00", "S11", "is_1d", "is_ckf_selected", "contrib_pids",
            "branch_majority_pid", "majority_undefined",
            "majority_true_hit_on_surface"]

    pf = pq.ParquetFile(pq_path)
    partial_states: list[pd.DataFrame] = []
    cand_parts: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=2_000_000, columns=cols):
        df = batch.to_pandas()
        df = df[~df["majority_undefined"].astype(bool)]
        if not len(df):
            continue
        cand_parts.append(
            df[df["cand_hit_id"] >= 0][
                ["seed_id", "branch_id", "step_k", "cand_hit_id",
                 "is_ckf_selected", "contrib_pids", "branch_majority_pid",
                 "chi2_inc"]
            ]
        )
        partial_states.append(build_state_table(df))
    cand = pd.concat(cand_parts, ignore_index=True)
    del cand_parts

    selected = select_ckf_branch(cand)
    purity = branch_purity(cand, selected)
    survivors = flag_ambi_survivors(cand)
    del cand

    # A state split across two read batches yields two partial rows; rerun
    # the min/first reduction once on the (much smaller) concatenation.
    allp = pd.concat(partial_states, ignore_index=True)
    del partial_states
    g = allp.groupby(["seed_id", "branch_id", "step_k"], as_index=False)
    states = g.agg(
        volume_id=("volume_id", "first"),
        state_theta=("state_theta", "first"),
        n_window=("n_window", "first"),
        branch_majority_pid=("branch_majority_pid", "first"),
        on_surface=("on_surface", "first"),
        d_true=("d_true", "min"),
        n_true_rows=("n_true_rows", "sum"),
    )
    states = states.merge(selected, on=["seed_id", "branch_id"])
    states = states.merge(purity, on="seed_id", how="left")
    states["is_pure"] = states["is_pure"].fillna(False).astype(bool)
    states = states.merge(survivors, on="seed_id", how="left")
    states["survived_ambi"] = states["survived_ambi"].fillna(False).astype(bool)

    pt_lut = particle_pt_lookup(str(csv_dir_for(ev)), ev)
    out = accumulate_event(states, pt_lut)

    os.makedirs(f"{S}/winfail_unc", exist_ok=True)
    np.savez(
        f"{S}/winfail_unc/winfail_unc_event{ev:03d}.npz",
        eta_bins=ETA_BINS, n_values=np.array(N_VALUES),
        occ_edges=np.array(OCC_EDGES), pt_edges=np.array(PT_EDGES),
        **{k: v for k, v in out.items() if k != "counters"},
        **{f"counter_{k}": np.array(v) for k, v in out["counters"].items()},
    )
    c = out["counters"]
    print(
        f"event {ev}: states={c['n_states']:,} vol20={c['n_vol20']:,} "
        f"escaped={c['n_escaped']:,} multi_true={c['n_multi_true']:,} "
        f"pt_unmatched={c['n_pt_unmatched']:,} "
        f"branches={c['n_branches']:,} ambi_survivors={c['n_ambi_survivors']:,} "
        f"win_total={int(out['win_total'].sum()):,} "
        f"winfail_n10={int(out['win_fail'][-1].sum()):,}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Push, pull on CFS, smoke on event 4**

```bash
git push
ssh -i ~/.ssh/nersc -o IdentitiesOnly=yes mussonm@perlmutter.nersc.gov \
  'cd /global/cfs/cdirs/atlas/mussonm/cCKF && git pull && module load python && \
   CCKF_STAGE1_BASE=$SCRATCH/cckf/modal_backup/results \
   timeout 1500 python3 scripts/winfail_uncensored.py 4 $SCRATCH/cckf'
```

Expected: summary line prints; `winfail_n10 > 0` (equals `escaped` - the whole point of uncensoring); `pt_unmatched` small; `multi_true` small. If the login node is too slow, submit the same command as a single-event debug job - do not shrink the event.

- [ ] **Step 4: Cross-check against the censored analysis** - the uncensored n=3 rate must exceed the censored n=3 rate for the same event (its numerator gains the escaped states). Record both numbers in the commit message.

- [ ] **Step 5: Write and submit the batch script**

```bash
# scripts/winfail_unc.sbatch
#!/bin/bash
#SBATCH --job-name=winfail_unc
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
module load python
cd /global/cfs/cdirs/atlas/mussonm/cCKF
export CCKF_STAGE1_BASE=$SCRATCH/cckf/modal_backup/results
seq 0 31 | xargs -P 8 -I{} python3 scripts/winfail_uncensored.py {} $SCRATCH/cckf
echo "DONE $(ls $SCRATCH/cckf/winfail_unc/*.npz | wc -l)/32"
```

```bash
git add scripts/winfail_unc.sbatch && git commit -m "feat: winfail_unc batch job" && git push
ssh -i ~/.ssh/nersc -o IdentitiesOnly=yes mussonm@perlmutter.nersc.gov \
  'cd /global/cfs/cdirs/atlas/mussonm/cCKF && git pull && sbatch --parsable scripts/winfail_unc.sbatch'
```

- [ ] **Step 6: On completion verify 32 npz files exist; print the counters table across events. Data stays on scratch; commit nothing.**

---

### Task 6: Render script - four plot families

**Files:**
- Create: `scripts/plot_winfail_uncensored.py`
- Test: `tests/test_winfail_uncensored.py` (renderer smoke test on synthetic npz)

**Interfaces:**
- Consumes: the npz tensor contract; `wilson_interval`, `PT_EDGES` from `winfail_uncensored`.
- Produces: `render_all(npz_dir: str, out_dir: str, threshold_gev: float, branch_class: str) -> list[str]` writing 8 PNGs; `branch_class` is `"all"` (sum the ambi axis) or `"ambi"` (slice index 1). The CLI runs the 2x2 grid - thresholds {1.0, 0.9} x classes {all, ambi} - into `out/pt1p0_all/`, `out/pt1p0_ambi/`, `out/pt0p9_all/`, `out/pt0p9_ambi/` (32 PNGs total):
  - `winfail_vs_eta_pure.png`, `winfail_vs_eta_majority.png` - 3 sensor panels (pixel / short strip / long strip), lines n = 3/5/7/10 with Wilson bands.
  - `winfail_vs_eta_occupancy_n{3,5,7,10}.png` - one per n: 3 sensor panels, lines = 5 occupancy strata (purities pooled; noted in footer).
  - `modfail_vs_eta.png` - one panel, 3 sensor lines with Wilson bands.
  - `modfail_vs_occupancy.png` - x = 5 occupancy strata, 3 sensor lines, Wilson intervals as error bars.

- [ ] **Step 1: Write the failing renderer test**

```python
def test_render_all_produces_expected_files(tmp_path):
    import plot_winfail_uncensored as P
    shape = (140, 3, 2, 5, 4, 2)
    rng = np.random.default_rng(0)
    total = rng.integers(50, 100, size=shape)
    np.savez(
        tmp_path / "winfail_unc_event000.npz",
        eta_bins=np.linspace(-3.5, 3.5, 141),
        n_values=np.array([3.0, 5.0, 7.0, 10.0]),
        occ_edges=np.array([0.0, 2.0, 5.0, 10.0, 20.0]),
        pt_edges=np.array([0.0, 0.7, 0.9, 1.0]),
        mod_total=total, mod_fail=total // 10,
        win_total=total, win_fail=np.stack([total // (i + 2) for i in range(4)]),
        counter_n_states=np.array(1), counter_n_vol20=np.array(0),
        counter_n_escaped=np.array(0), counter_n_multi_true=np.array(0),
        counter_n_pt_unmatched=np.array(0),
    )
    made = P.render_all(str(tmp_path), str(tmp_path / "out"),
                        threshold_gev=1.0, branch_class="ambi")
    names = {p.split("/")[-1] for p in made}
    assert names == {
        "winfail_vs_eta_pure.png", "winfail_vs_eta_majority.png",
        "winfail_vs_eta_occupancy_n3.png", "winfail_vs_eta_occupancy_n5.png",
        "winfail_vs_eta_occupancy_n7.png", "winfail_vs_eta_occupancy_n10.png",
        "modfail_vs_eta.png", "modfail_vs_occupancy.png",
    }


def test_pt_slice_rejects_non_edge_threshold(tmp_path):
    import plot_winfail_uncensored as P
    import pytest
    with pytest.raises(ValueError):
        P.pt_slice(np.zeros((2, 4)), 0.8)
```

- [ ] **Step 2: Run to verify FAIL** (`ModuleNotFoundError`)

- [ ] **Step 3: Implement**

Core pieces (write exactly; the remaining figure assembly is plain matplotlib):

```python
# One brief line: only the statistics needed to reconstruct the denominator.
FOOTER_WIN = ("all surviving branches (pre-ambi, envelope: chi2 16.26/35.75, "
              "cap 5, terminal cuts off) · majority defined · denom: majority "
              "simhit on surface · majority pT > {pt} GeV · uncensored "
              "(escaped = fail; incl. undigitized) · vol 20 excluded")
FOOTER_MOD = ("all surviving branches (pre-ambi, envelope: chi2 16.26/35.75, "
              "cap 5, terminal cuts off) · majority defined · majority pT > "
              "{pt} GeV · fail = no majority simhit on surface · vol 20 excluded")
SENSOR_LABELS = ["Pixel", "Short strip", "Long strip"]
OCC_LABELS = ["0-1", "2-4", "5-9", "10-19", "20+"]


def pt_slice(tensor: np.ndarray, threshold_gev: float) -> np.ndarray:
    """Sum the pT-bin axis (last axis) over bins at/above threshold_gev.

    threshold_gev must be an edge of PT_EDGES; raises ValueError otherwise
    so an unsupported threshold fails loudly.
    """
    from winfail_uncensored import PT_EDGES

    if threshold_gev not in PT_EDGES:
        raise ValueError(f"{threshold_gev} not in PT_EDGES {PT_EDGES}")
    i = PT_EDGES.index(threshold_gev)
    return tensor[..., i:].sum(axis=-1)


def _load_sums(npz_dir: str) -> dict[str, np.ndarray]:
    import glob

    files = sorted(glob.glob(f"{npz_dir}/winfail_unc_event*.npz"))
    assert files, f"no npz files in {npz_dir}"
    keys = ["mod_total", "mod_fail", "win_total", "win_fail"]
    acc: dict[str, np.ndarray | None] = {k: None for k in keys}
    for f in files:
        z = np.load(f)
        for k in keys:
            acc[k] = z[k] if acc[k] is None else acc[k] + z[k]
    z = np.load(files[0])
    acc["eta_bins"] = z["eta_bins"]
    acc["n_values"] = z["n_values"]
    return acc


def _rate_with_band(ax, x, k, n, label, color):
    from winfail_uncensored import wilson_interval

    with np.errstate(invalid="ignore", divide="ignore"):
        rate = k / n
    lo, hi = wilson_interval(k, n)
    ok = n > 0
    ax.plot(x[ok], rate[ok], "-", lw=1.2, color=color, label=label)
    ax.fill_between(x[ok], lo[ok], hi[ok], alpha=0.18, color=color, lw=0)
```

Slicing rules - collapse the ambi axis FIRST (`t.sum(axis=-1)` for `"all"`, `t[..., 1]` for `"ambi"`), then apply `pt_slice(tensor, threshold_gev)` (collapsing the now-last pT axis to `wt/wf/mt/mf`), then:
- Family A (`winfail_vs_eta_{purity}.png`): purity p, sensor s: `k = wf[ni, :, s, p, :].sum(axis=-1)`, `n = wt[:, s, p, :].sum(axis=-1)`.
- Family B (`..._occupancy_n{n}.png`): n index ni, sensor s, occupancy o: `k = wf[ni, :, s, :, o].sum(axis=1)`, `n = wt[:, s, :, o].sum(axis=1)`.
- Family C (`modfail_vs_eta.png`): sensor s: `k = mf[:, s].sum(axis=(1, 2))`, `n = mt[:, s].sum(axis=(1, 2))`.
- Family D (`modfail_vs_occupancy.png`): sensor s: `k = mf[:, s].sum(axis=(0, 1))` (length 5), `ax.errorbar` with Wilson intervals.

Every figure: the footer is `FOOTER_XXX.format(pt=threshold_gev)` plus `" · ambi survivors (offline greedy replica, maxShared 3, nMeasMin 7)"` when `branch_class == "ambi"` else `" · all CKF-output branches"`, placed with `fig.text(0.99, 0.005, ..., ha="right", va="bottom", fontsize=7, color="0.35")` after `fig.tight_layout()`.

- [ ] **Step 4: Run tests; all PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_winfail_uncensored.py tests/test_winfail_uncensored.py
git commit -m "feat: uncensored winfail/modfail plot families"
```

- [ ] **Step 6: Render from real data and deliver** - scp the 32 npz result files to the local scratchpad (results, not source), run the CLI for 1.0 and 0.9 GeV, send the figures with a short comparison of uncensored vs censored rates.

---

### Task 7: pT-threshold annotation on the Pareto overlay

**Files:**
- Create: `scripts/plot_pareto_overlay.py` (repo copy of the session scratchpad renderer `overlay/render_overlay2.py`, plus the footer)

**Interfaces:**
- Consumes: `results/pareto_*.csv` sweep CSVs (`tau_g, tau_v, efficiency, fake_rate, source`).
- Produces: the overlay PNG with the truth-selection footer.

- [ ] **Step 1: Copy the scratchpad renderer into the repo and add the footer**

```python
FOOTER = ("efficiency/fake at truth selection: pT > 1 GeV, |η| < 3, "
          "≥6 meas, ≥3 pixel hits, charged · post-ambiguity (ACTS greedy) "
          "· 1 event (skip 4)")
# after fig.tight_layout():
fig.text(0.99, 0.005, FOOTER, ha="right", va="bottom", fontsize=7, color="0.35")
```

- [ ] **Step 2: Re-render, confirm the footer clears the legend, send the updated overlay to the user**

- [ ] **Step 3: Commit**

```bash
git add scripts/plot_pareto_overlay.py
git commit -m "feat: pareto overlay renderer with truth-selection annotation"
```

---

### Task 8 (OPTIONAL - not scheduled): escape distances via truth-measurement join

Only needed for (a) lines at n > 10, (b) splitting "digitized but escaped" from "simhit never digitized". Per event: `expansion.load_measurements` + `load_measurement_simhit_map` + `load_simhits` → (particle, surface) → the majority particle's own measurement position/variance; `load_predicted_cov` → P; d for escaped states from `r = true position − pred`, `S_ii = var_i + P_ii`. One residual per state - the majority particle's own hit - suffices even at n = 15; other hits on the module are occupancy, not failure. Do not build unless those two outputs are requested.

---

## Self-Review

1. **Spec coverage.** Dense η (140 ≥ 100) ✓; pixel/short/long ✓; pure vs majority separate plots ✓ (Family A); n = 3/5/7/10 ✓; GeV threshold marked ✓ (recoverable pT-bin axis, renders at 1.0 and 0.9, footers); occupancy stratification at each n ✓ (Family B); module failure vs η by sensor ✓ (Family C, parquet boolean only - no join, per user); module failure vs occupancy ✓ (Family D); binomial CIs ✓ (Wilson z=1); Pareto annotation ✓ (Task 7); "proper n=10 line" ✓ - escaped states counted as failures at every n, Task 5 smoke gates on `winfail_n10 > 0`.
1b. **Branch-class axis** - "all CKF-output branches" vs "ambi survivors" as the trailing tensor axis; survivor flag from the offline greedy replica (Task 3), rendered as separate figure sets. Pruned-in-flight branches remain impossible by construction and the plan says so wherever the axis is defined.
2. **Placeholder scan.** One open point by design: `particle_pt_lookup` column names await the Task 1 recon; its contract (join by hit id, earliest simhit, `(particle_id, pt_gev)`) is pinned and the barcode-re-encoding alternative is forbidden. Everything else is concrete.
3. **Type consistency.** `build_state_table` output columns match `accumulate_event` input and the Task 5 re-reduction; `select_ckf_branch`/`branch_purity` signatures match Tasks 3 and 5; tensor shapes `(140, 3, 2, 5, 4, 2)` consistent across Tasks 4-6; `pt_slice` consumes `PT_EDGES` from Task 2.

## Known limits (state on request, footnoted on figures)

- `majority_true_hit_on_surface` is simhit-level, so "escaped" conflates true-hit-outside-box with deposited-but-never-digitized. The escaped mass is therefore an upper bound on pure window escape (footer says "incl. undigitized"). Task 8 separates them if needed.
- Purity uses the CKF-selected candidate at the first 3 steps; the old Modal script used the first 3 candidate rows (can all sit on one surface). Small shifts vs the old 4-panel figure are a correction, not a regression.
- Branch inflation: with seed_id == track_nr there is no way to collapse multiple surviving branches of one physical seed from the parquet alone. If de-inflation is ever needed, join the stage-1 seeds CSV (`output_seeds_csv` was on) by the first three hit positions - an extension, not scheduled. The ambi-survivor slice removes most of it in practice (greedy evicts shared-hit duplicates).
- The ambi-survivor flag is a replica, not the chain's own output: its chi2 tie-break uses summed `chi2_inc` (shared-S approximation) instead of the fitter's track chi2, which can flip exact ties. The replica's selection matched a real greedy run's kept-track count exactly (e034 validation, this session); no stage-1 post-ambi reference exists to validate against directly.
- Chunked reading can split a state across batches; Task 5 re-reduces the concatenated partials (min of mins, sum of sums - both associative), so results are exact.
