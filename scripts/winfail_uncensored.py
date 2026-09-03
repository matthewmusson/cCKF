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

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import expansion  # noqa: E402

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


def _as_pid_list(c: object) -> list:
    """Normalize a `contrib_pids` cell to a membership-testable list.

    Parquet nulls in a list<int64> column can surface to pandas either as
    `None` or as a bare `float` NaN (arrow's null sentinel loses the list
    type on the empty case). Both mean "no contributors" here.
    """
    if c is None or isinstance(c, float):
        return []
    return c


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
        return all(maj in _as_pid_list(c) for c in g["contrib_pids"])

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


def build_state_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse candidate+hole rows to one row per (seed, branch, step).

    Input rows must already be restricted to selected, majority-defined
    branches. See module docstring for d_true.
    """
    maj = rows["branch_majority_pid"].to_numpy()
    is_true = np.fromiter(
        (
            (m in _as_pid_list(c))
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


def _pick_earliest_by_tt(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce (particle_id, pt_gev, tt) rows to one earliest row per particle.

    "Earliest" is minimum `tt`. Ties, and particles whose `tt` is entirely
    NaN (missing in some events), fall back to original row order - a
    stable sort with `na_position="last"` preserves input order among equal
    (including all-NaN) keys, which for hit rows built in file order is
    exactly "minimum hit_id". Pure and NERSC-independent so it is unit
    tested directly; `particle_pt_lookup` is the CSV-reading wrapper around
    it, exercised by the real-data smoke instead.

    Parameters
    ----------
    df : pd.DataFrame
        Columns ``particle_id, pt_gev, tt``, one row per simhit, in
        original (file) row order.

    Returns
    -------
    pd.DataFrame
        Columns ``particle_id, pt_gev``, one row per particle_id.
    """
    ordered = df.sort_values("tt", kind="stable", na_position="last")
    return (
        ordered.groupby("particle_id", as_index=False)
        .first()[["particle_id", "pt_gev"]]
    )


def particle_pt_lookup(csv_dir: str, event_id: int) -> pd.DataFrame:
    """(particle_id, pt_gev) for one event, from each particle's earliest simhit.

    Momentum (`tpx, tpy, tt`) lives only in the raw simhits CSV;
    `expansion.load_simhits` computes the custom-encoded `particle_id` but
    drops momentum. The two are joined **by row position** (hit_id), never
    by re-encoding barcodes: both read `event{event_id:09d}-simhits.csv` in
    file order with no row filtering, so row i means the same hit in both
    (verified full-file on event 4 - see Task 1 recon). The path is built
    from `event_id` directly, never globbed: each pilot `csv_dir` holds two
    events, so a glob-and-take-first would silently read the wrong one for
    odd event ids.
    """
    path = Path(csv_dir) / f"event{event_id:09d}-simhits.csv"
    raw = pd.read_csv(
        path,
        usecols=[
            SCHEMA["raw_simhit_tpx"],
            SCHEMA["raw_simhit_tpy"],
            SCHEMA["raw_simhit_tt"],
        ],
    ).reset_index(drop=True)

    sim = expansion.load_simhits(csv_dir, event_id)[
        [SCHEMA["simhit_hit_id"], SCHEMA["simhit_particle_id"]]
    ]
    pos = sim[SCHEMA["simhit_hit_id"]].to_numpy()  # positional index (arange)

    joined = pd.DataFrame({
        "particle_id": sim[SCHEMA["simhit_particle_id"]].to_numpy(),
        "pt_gev": np.hypot(
            raw[SCHEMA["raw_simhit_tpx"]].to_numpy()[pos],
            raw[SCHEMA["raw_simhit_tpy"]].to_numpy()[pos],
        ),
        "tt": raw[SCHEMA["raw_simhit_tt"]].to_numpy()[pos],
    })
    return _pick_earliest_by_tt(joined)


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
    # NOTE: chi2_inc added to the task-5 brief's column list -- it is
    # selected out of df below (for flag_ambi_survivors) but the brief's
    # iter_batches columns omitted it, which KeyErrors on real data.
    cols = ["seed_id", "branch_id", "step_k", "volume_id", "state_theta",
            "n_window", "cand_hit_id", "residual_l0", "residual_l1",
            "S00", "S11", "is_1d", "is_ckf_selected", "contrib_pids",
            "branch_majority_pid", "majority_undefined",
            "majority_true_hit_on_surface", "chi2_inc"]

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
