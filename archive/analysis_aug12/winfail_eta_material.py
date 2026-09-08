#!/usr/bin/env python3
"""Window failure rate vs eta, stratified by occupancy, with material overlay.

Cluster-agnostic: every path arrives via argv, nothing imports modal, and no
absolute path appears below. Run it on NERSC, on Modal, or on a laptop.

DEFINITION
----------
A row is a (seed, branch, surface, candidate-hit) tuple.  A row enters the
denominator when all of:

  * it is not a hole              (cand_hit_id != -1)
  * the branch has a majority particle  (majority_undefined is False)
  * the candidate hit was left by that majority particle
    (branch_majority_pid appears in contrib_pids)

It enters the numerator when that candidate falls OUTSIDE the CKF's n-sigma
search box.  The box is a bounding box on local coordinates with half-widths
dl0 = n*sqrt(S00), dl1 = n*sqrt(S11), so:

  pixel (2D measurement):  fail  <=>  |r0| > n*sqrt(S00)  OR  |r1| > n*sqrt(S11)
  strip (1D measurement):  fail  <=>  |r0| > n*sqrt(S00)

The strip case is the correction this script adds.  A strip has one measured
local coordinate, so S01 and S11 are NaN by construction and the l1 half of
the test is undefined, not failed.  Comparing a strip against the 2D null
is a category error.

Under a perfectly calibrated Gaussian S the expected failure rates differ:

  pixel:  1 - [erf(n/sqrt(2))]^2
  strip:  1 -  erf(n/sqrt(2))

This measurement is a LOWER BOUND on the true window failure rate.  It only
sees hits ACTS already surfaced as candidates; a true hit that fell outside
ACTS's own gathering window never becomes a row.  truth_residual_l0/l1 exist
in the schema but are 100% NaN in the re-expanded data (expansion.py:35 --
the truth residual needs SimTrackerHit information the expansion does not
carry), so the stronger definition is not available.

MATERIAL
--------
pathInX0_interval is per-step.  Material traversed to a given surface is the
cumulative sum along the branch, ordered by step_k.  Rows are globally sorted
by (seed_id, branch_id) with step_k monotone within a branch, so the cumsum is
computed per row group with the straddling branch's partial sum carried across
the boundary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------
ETA_EDGES = np.round(np.arange(-4.0, 4.0001, 0.05), 3)        # 160 bins
# 0.05 in eta. With ~12M usable rows this is ~75k rows/bin on average;
# sparse tail bins are dropped at plot time by --min-count rather than
# being silently drawn with meaningless error bars.
OCC_EDGES = np.array([1, 5, 10, 20, 10_000])                    # 4 strata
OCC_LABELS = ["n < 5", "5 <= n < 10", "10 <= n < 20", "n >= 20"]
N_SIGMAS = [1.0, 3.0, 5.0, 7.0, 10.0]   # swept in a single pass
SENSORS = ["strip", "pixel"]                                    # index by is_pixel

COLUMNS = [
    "cand_hit_id", "majority_undefined", "branch_majority_pid", "contrib_pids",
    "residual_l0", "residual_l1", "S00", "S11",
    "state_theta", "is_1d", "n_window", "pathInX0_interval",
    "seed_id", "branch_id", "step_k",
]


def _eta_from_theta(theta: np.ndarray) -> np.ndarray:
    """eta = -ln(tan(theta/2)), guarded against theta at 0 or pi."""
    t = np.clip(theta, 1e-9, np.pi - 1e-9)
    return -np.log(np.tan(t / 2.0))


def _label_same(table, n_rows: int, branch_maj: np.ndarray) -> np.ndarray:
    """True where branch_majority_pid appears in this row's contrib_pids.

    contrib_pids is a list column.  Flatten it once, compare every element
    against its own row's majority pid, then bincount the matches back to
    rows.  No Python loop over rows.
    """
    cp = table.column("contrib_pids").combine_chunks()
    offsets = cp.offsets.to_numpy().astype(np.int64)
    flat_arr = cp.flatten()
    if len(flat_arr) == 0:
        return np.zeros(n_rows, dtype=bool)
    flat = flat_arr.to_numpy(zero_copy_only=False).astype(np.int64)
    counts = np.diff(offsets)
    if counts.sum() == 0:
        return np.zeros(n_rows, dtype=bool)
    maj_rep = np.repeat(branch_maj, counts)
    row_idx = np.repeat(np.arange(n_rows, dtype=np.int64), counts)
    hits = flat == maj_rep
    matched = np.bincount(row_idx[hits], minlength=n_rows)
    return matched > 0


def _cumulative_x0(
    seed: np.ndarray, branch: np.ndarray, step: np.ndarray,
    interval: np.ndarray, carry: dict,
) -> np.ndarray:
    """Cumulative pathInX0 along each branch, carrying across row groups.

    Rows are sorted by (seed_id, branch_id) with step_k monotone inside a
    branch, so a plain cumsum reset at each branch boundary is correct.  The
    only wrinkle is the first branch of the row group, which may continue a
    branch that began in the previous row group.

    IMPORTANT: a row is a (branch, surface, candidate) tuple, and
    pathInX0_interval is a property of the SURFACE, repeated identically on
    every candidate row for that surface.  Summing it per row multiplies each
    surface's material by its candidate count, which inflates exactly the
    crowded high-|eta| regions and produces physically impossible totals
    (tens of radiation lengths).  The material must therefore be charged once
    per (seed, branch, step_k), on that surface's first row only.
    """
    x = np.nan_to_num(interval, nan=0.0)
    new_surface = np.empty(len(seed), dtype=bool)
    new_surface[0] = True
    new_surface[1:] = ((seed[1:] != seed[:-1])
                       | (branch[1:] != branch[:-1])
                       | (step[1:] != step[:-1]))
    x = np.where(new_surface, x, 0.0)
    total = np.cumsum(x)
    key = np.stack([seed, branch])
    new_branch = np.empty(len(seed), dtype=bool)
    new_branch[0] = True
    new_branch[1:] = (key[0, 1:] != key[0, :-1]) | (key[1, 1:] != key[1, :-1])
    # subtract the running total as it stood just before each branch started
    starts = np.where(new_branch)[0]
    base = np.zeros(len(seed))
    base_at_start = total[starts] - x[starts]
    base[starts] = base_at_start
    np.maximum.accumulate(base, out=base)
    cum = total - base

    # carry: if this row group opens mid-branch, add the previous partial sum
    first_key = (int(seed[0]), int(branch[0]))
    if carry.get("key") == first_key:
        end = starts[1] if len(starts) > 1 else len(seed)
        cum[:end] += carry["value"]

    last_key = (int(seed[-1]), int(branch[-1]))
    carry["key"] = last_key
    carry["value"] = float(cum[-1])
    return cum


def extract_event(path: Path, event_id: int, out_path: Path,
                  max_row_groups: int | None = None) -> int:
    """Write the usable rows for one event to Parquet, one row per candidate.

    Only rows passing the correct-hit filter survive, which is ~0.5% of the
    file, and only the six quantities the analysis needs are kept.  The whole
    32-event dataset lands at ~100 MB, so every downstream choice -- bin
    width, sigma values, occupancy strata, pixel/strip pooling -- becomes a
    local operation instead of a cluster round-trip.

    z0 and z1 are the normalised distances |r|/sqrt(S); the n-sigma test is
    just z > n, so storing z rather than (r, S) makes the stored data
    independent of which n we later choose to plot.
    """
    import pyarrow as pa

    pf = pq.ParquetFile(path)
    n_rg = pf.metadata.num_row_groups
    if max_row_groups is not None:
        n_rg = min(n_rg, max_row_groups)
    carry: dict = {}
    writer = None
    written = 0
    try:
        for rg in range(n_rg):
            t = pf.read_row_group(rg, columns=COLUMNS)
            n = t.num_rows
            if n == 0:
                continue
            g = lambda c: t.column(c).to_numpy(zero_copy_only=False)
            seed = g("seed_id").astype(np.int64)
            branch = g("branch_id").astype(np.int64)
            step = g("step_k").astype(np.int64)
            cum = _cumulative_x0(seed, branch, step,
                                 g("pathInX0_interval").astype("f8"), carry)
            cand = g("cand_hit_id")
            keep = ((cand != -1) & (~g("majority_undefined").astype(bool))
                    & _label_same(t, n, g("branch_majority_pid").astype(np.int64)))
            if not keep.any():
                continue
            r0 = g("residual_l0").astype("f8")[keep]
            r1 = g("residual_l1").astype("f8")[keep]
            s00 = g("S00").astype("f8")[keep]
            s11 = g("S11").astype("f8")[keep]
            d1 = g("is_1d").astype(bool)[keep]
            usable = np.isfinite(s00) & np.isfinite(r0)
            usable &= np.where(d1, True, np.isfinite(s11) & np.isfinite(r1))
            if not usable.any():
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                z0 = np.abs(r0[usable]) / np.sqrt(s00[usable])
                z1 = np.abs(r1[usable]) / np.sqrt(s11[usable])
            tab = pa.table({
                "event_id": pa.array(np.full(int(usable.sum()), event_id, "i2")),
                # seed/branch/step are needed to reduce per TRACK rather than
                # per candidate row -- the material budget is a property of a
                # trajectory, so it must be summed along a branch and then
                # averaged over branches, not averaged over candidate rows.
                "seed_id":  pa.array(seed[keep][usable].astype("i8")),
                "branch_id": pa.array(branch[keep][usable].astype("i4")),
                "step_k":   pa.array(step[keep][usable].astype("i2")),
                "eta":      pa.array(_eta_from_theta(g("state_theta").astype("f8"))[keep][usable].astype("f4")),
                "n_window": pa.array(np.nan_to_num(g("n_window").astype("f8")[keep][usable], nan=0).astype("i2")),
                "is_1d":    pa.array(d1[usable]),
                "z0":       pa.array(z0.astype("f4")),
                "z1":       pa.array(z1.astype("f4")),
                "cum_x0":   pa.array(cum[keep][usable].astype("f4")),
            })
            if writer is None:
                writer = pq.ParquetWriter(out_path, tab.schema, compression="zstd")
            writer.write_table(tab)
            written += tab.num_rows
    finally:
        if writer is not None:
            writer.close()
    return written


def summarise_branches(path: Path, event_id: int, out_path: Path,
                       max_row_groups: int | None = None) -> int:
    """One row per (seed, branch): the material it traversed and where it went.

    This is what the material overlay needs. The per-candidate extract
    truncates each branch at its last CORRECT hit, which understates the
    material; here the sum runs over every row of the branch, holes included.

    pathInX0_interval is a property of the surface, repeated on every candidate
    row for that surface, so it is charged once per (seed, branch, step_k).
    Summing per row instead multiplies each surface by its candidate count and
    inflates crowded regions specifically.

    Fully vectorised: rows are globally sorted by (seed_id, branch_id), so a
    branch occupies a contiguous span and reduceat can segment it. Only a
    branch straddling a row-group boundary needs carrying.
    """
    import pyarrow as pa

    pf = pq.ParquetFile(path)
    n_rg = pf.metadata.num_row_groups
    if max_row_groups is not None:
        n_rg = min(n_rg, max_row_groups)

    out = {k: [] for k in ("seed", "branch", "x0", "esum", "ecnt", "smax",
                           "htrue", "seedn", "seedp")}
    carry = None          # partial summary of the branch spanning the boundary

    for rg in range(n_rg):
        t = pf.read_row_group(rg, columns=COLUMNS)
        n = t.num_rows
        if n == 0:
            continue
        g = lambda c: t.column(c).to_numpy(zero_copy_only=False)
        sd = g("seed_id").astype(np.int64)
        br = g("branch_id").astype(np.int64)
        sk = g("step_k").astype(np.int64)
        px = np.nan_to_num(g("pathInX0_interval").astype("f8"), nan=0.0)
        eta = _eta_from_theta(g("state_theta").astype("f8"))
        nonhole = g("cand_hit_id") != -1
        same = (_label_same(t, n, g("branch_majority_pid").astype(np.int64))
                & (~g("majority_undefined").astype(bool)) & nonhole)

        new_surf = np.empty(n, bool); new_surf[0] = True
        new_surf[1:] = ((sd[1:] != sd[:-1]) | (br[1:] != br[:-1])
                        | (sk[1:] != sk[:-1]))
        x = np.where(new_surf, px, 0.0)

        new_br = np.empty(n, bool); new_br[0] = True
        new_br[1:] = (sd[1:] != sd[:-1]) | (br[1:] != br[:-1])
        st = np.where(new_br)[0]

        # Seed purity, following analyze_winfail_stratified.py: of the first
        # THREE NON-HOLE rows of a branch (ordered by step_k), how many carry a
        # hit from the branch's majority particle. 3/3 = pure, 2/3 = majority.
        # This needs the non-correct rows, which the per-candidate extract
        # drops, so it can only be computed here.
        #
        # Rows are already ordered by (seed, branch, step_k). Rank each
        # non-hole row within its branch, then keep ranks 0-2.
        c_nh = np.cumsum(nonhole).astype(np.int64)       # non-holes up to & incl.
        base_nh = np.zeros(n, dtype=np.int64)
        st_all = np.where(new_br)[0]
        # non-hole count BEFORE this branch begins
        base_nh[st_all] = c_nh[st_all] - nonhole[st_all].astype(np.int64)
        np.maximum.accumulate(base_nh, out=base_nh)
        rank = c_nh - base_nh - 1                        # 0-based within branch
        seed_slot = nonhole & (rank >= 0) & (rank < 3)
        seg_sn = np.add.reduceat(seed_slot.astype(np.int64), st)
        seg_sp = np.add.reduceat((seed_slot & same).astype(np.int64), st)

        ev_ok = nonhole & np.isfinite(eta)
        seg_x0 = np.add.reduceat(x, st)
        seg_es = np.add.reduceat(np.where(ev_ok, eta, 0.0), st)
        seg_ec = np.add.reduceat(ev_ok.astype(np.int64), st)
        seg_sm = np.maximum.reduceat(sk, st)
        seg_ht = np.maximum.reduceat(same.astype(np.int8), st) > 0
        seg_sd, seg_br = sd[st], br[st]

        if carry is not None:
            if carry[0] == seg_sd[0] and carry[1] == seg_br[0]:
                seg_x0[0] += carry[2]; seg_es[0] += carry[3]
                seg_ec[0] += carry[4]
                seg_sm[0] = max(seg_sm[0], carry[5])
                seg_ht[0] = seg_ht[0] or carry[6]
                # a branch split across row groups: the first three non-hole
                # rows may all lie in the earlier group, so combine and re-cap
                take = max(0, 3 - carry[7])
                seg_sp[0] = carry[8] + min(seg_sp[0], take)
                seg_sn[0] = min(3, carry[7] + seg_sn[0])
            else:
                for k, v in zip(out, carry):
                    out[k].append(v)
        # hold back the final branch: it may continue into the next row group
        carry = (seg_sd[-1], seg_br[-1], seg_x0[-1], seg_es[-1],
                 seg_ec[-1], seg_sm[-1], seg_ht[-1], seg_sn[-1], seg_sp[-1])
        if len(st) > 1:
            out["seed"].append(seg_sd[:-1]);   out["branch"].append(seg_br[:-1])
            out["x0"].append(seg_x0[:-1]);     out["esum"].append(seg_es[:-1])
            out["ecnt"].append(seg_ec[:-1]);   out["smax"].append(seg_sm[:-1])
            out["htrue"].append(seg_ht[:-1])
            out["seedn"].append(seg_sn[:-1]); out["seedp"].append(seg_sp[:-1])

    if carry is not None:
        for k, v in zip(out, carry):
            out[k].append(np.atleast_1d(v))

    cat = {k: (np.concatenate([np.atleast_1d(a) for a in v])
               if v else np.array([])) for k, v in out.items()}
    ec = cat["ecnt"].astype(float)
    mean_eta = np.divide(cat["esum"], ec, out=np.full(len(ec), np.nan),
                         where=ec > 0)
    tab = pa.table({
        "event_id":  pa.array(np.full(len(ec), event_id, "i2")),
        "seed_id":   pa.array(cat["seed"].astype("i8")),
        "branch_id": pa.array(cat["branch"].astype("i4")),
        "total_x0":  pa.array(cat["x0"].astype("f4")),
        "mean_eta":  pa.array(mean_eta.astype("f4")),
        "n_steps":   pa.array(cat["smax"].astype("i2")),
        "has_true":  pa.array(cat["htrue"].astype(bool)),
        # seed_n = how many of the first three non-hole rows exist (should be 3
        # for any branch that got going); seed_pure = how many of those came
        # from the majority particle. pure seed <=> seed_pure == 3.
        "seed_n":    pa.array(cat["seedn"].astype("i1")),
        "seed_pure": pa.array(cat["seedp"].astype("i1")),
    })
    pq.write_table(tab, out_path, compression="zstd")
    return tab.num_rows


def accumulate_event(path: Path, n_sigmas: list[float],
                     max_row_groups: int | None = None) -> dict:
    """One streaming pass over one event, sweeping every n in n_sigmas.

    The normalised distances z0 = |r0|/sqrt(S00) and z1 = |r1|/sqrt(S11) do not
    depend on n, so they are computed once per row group and compared against
    each threshold in turn.  Reading the file once per n would multiply 176 GB
    of I/O by five for no gain.
    """
    n_eta, n_occ, n_sen = len(ETA_EDGES) - 1, len(OCC_EDGES) - 1, len(SENSORS)
    n_n = len(n_sigmas)
    shape = (n_eta, n_occ, n_sen)
    tot = np.zeros(shape, dtype=np.int64)
    fail = np.zeros((n_n,) + shape, dtype=np.int64)
    # material overlay: mean cumulative X0 per eta bin, over the same rows
    mat_sum = np.zeros(n_eta, dtype=np.float64)
    mat_cnt = np.zeros(n_eta, dtype=np.int64)

    pf = pq.ParquetFile(path)
    carry: dict = {}
    n_rg = pf.metadata.num_row_groups
    if max_row_groups is not None:
        n_rg = min(n_rg, max_row_groups)
    for rg in range(n_rg):
        t = pf.read_row_group(rg, columns=COLUMNS)
        n = t.num_rows
        if n == 0:
            continue
        g = lambda c: t.column(c).to_numpy(zero_copy_only=False)

        seed = g("seed_id").astype(np.int64)
        branch = g("branch_id").astype(np.int64)
        step = g("step_k").astype(np.int64)
        cum_x0 = _cumulative_x0(seed, branch, step, g("pathInX0_interval").astype("f8"), carry)

        cand = g("cand_hit_id")
        maj_undef = g("majority_undefined").astype(bool)
        branch_maj = g("branch_majority_pid").astype(np.int64)
        same = _label_same(t, n, branch_maj)

        keep = (cand != -1) & (~maj_undef) & same
        if not keep.any():
            continue

        r0 = g("residual_l0").astype("f8")[keep]
        r1 = g("residual_l1").astype("f8")[keep]
        s00 = g("S00").astype("f8")[keep]
        s11 = g("S11").astype("f8")[keep]
        d1 = g("is_1d").astype(bool)[keep]
        eta = _eta_from_theta(g("state_theta").astype("f8"))[keep]
        occ = g("n_window").astype("f8")[keep]
        cx0 = cum_x0[keep]

        # a row is only usable where the half-widths its test needs are finite
        usable = np.isfinite(s00) & np.isfinite(r0)
        usable &= np.where(d1, True, np.isfinite(s11) & np.isfinite(r1))
        if not usable.any():
            continue

        d1u = d1[usable]
        # normalised distances, computed once and reused for every n
        with np.errstate(invalid="ignore", divide="ignore"):
            z0 = np.abs(r0[usable]) / np.sqrt(s00[usable])
            z1 = np.abs(r1[usable]) / np.sqrt(s11[usable])

        ei = np.digitize(eta[usable], ETA_EDGES) - 1
        oi = np.digitize(occ[usable], OCC_EDGES) - 1
        si = (~d1u).astype(np.int64)                 # 0 = strip, 1 = pixel
        cx = cx0[usable]

        ok = (ei >= 0) & (ei < n_eta) & (oi >= 0) & (oi < n_occ)
        ei, oi, si, cx = ei[ok], oi[ok], si[ok], cx[ok]
        z0, z1, d1u = z0[ok], z1[ok], d1u[ok]

        flat = (ei * n_occ + oi) * n_sen + si
        tot += np.bincount(flat, minlength=tot.size).reshape(shape)
        for j, nsig in enumerate(n_sigmas):
            out0 = z0 > nsig
            out1 = z1 > nsig          # False on strips: NaN comparisons are False
            failed = np.where(d1u, out0, out0 | out1)
            fail[j] += np.bincount(flat[failed], minlength=tot.size).reshape(shape)
        mat_sum += np.bincount(ei, weights=cx, minlength=n_eta)
        mat_cnt += np.bincount(ei, minlength=n_eta)

    return {"total": tot, "fail": fail, "mat_sum": mat_sum, "mat_cnt": mat_cnt}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--events", required=True,
                    help="comma list or a-b range, e.g. '0,1,2' or '0-31'")
    ap.add_argument("--n-sigmas", default=",".join(str(x) for x in N_SIGMAS),
                    help="comma list of window multipliers, swept in one pass")
    ap.add_argument("--max-row-groups", type=int, default=None,
                    help="smoke-test cap; omit for a full pass")
    ap.add_argument("--emit-branches", action="store_true",
                    help="write one row per (seed, branch) with its total "
                         "traversed material -- what the overlay needs")
    ap.add_argument("--emit-rows", action="store_true",
                    help="write filtered per-row Parquet instead of binned "
                         "accumulators, so binning stays a local decision")
    args = ap.parse_args()

    n_sigmas = [float(x) for x in args.n_sigmas.split(",")]
    spec = args.events
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        events = list(range(int(a), int(b) + 1))
    else:
        events = [int(x) for x in spec.split(",")]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ev in events:
        path = args.parquet_dir / f"expanded_event{ev:09d}.parquet"
        if not path.exists():
            print(f"[event {ev}] MISSING {path}", flush=True)
            continue
        import time
        t0 = time.time()
        if args.emit_branches:
            out = args.out_dir / f"branches_event{ev:09d}.parquet"
            nb_ = summarise_branches(path, ev, out, args.max_row_groups)
            print(f"[event {ev}] branches={nb_:,} secs={time.time()-t0:.0f} "
                  f"-> {out.name}", flush=True)
            continue
        if args.emit_rows:
            out = args.out_dir / f"rows_event{ev:09d}.parquet"
            nw = extract_event(path, ev, out, args.max_row_groups)
            mb = out.stat().st_size / 1e6 if out.exists() else 0.0
            print(f"[event {ev}] rows={nw:,} secs={time.time()-t0:.0f} "
                  f"size={mb:.1f}MB -> {out.name}", flush=True)
            continue
        acc = accumulate_event(path, n_sigmas, args.max_row_groups)
        out = args.out_dir / f"winfail_event{ev:09d}.npz"
        np.savez_compressed(out, **acc,
                            eta_edges=ETA_EDGES, occ_edges=OCC_EDGES,
                            n_sigmas=np.asarray(n_sigmas))
        n_tot = int(acc["total"].sum())
        rates = ", ".join(
            f"n={ns:g}: {acc['fail'][j].sum() / n_tot:.3%}" if n_tot else f"n={ns:g}: -"
            for j, ns in enumerate(n_sigmas)
        )
        print(f"[event {ev}] rows={n_tot:,} | {rates} | "
              f"secs={time.time()-t0:.0f} -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
