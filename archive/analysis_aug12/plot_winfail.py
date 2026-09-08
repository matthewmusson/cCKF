#!/usr/bin/env python3
"""Window failure rate vs eta with the material budget underlaid.

Reads the per-row extract (rows_event*.parquet), so binning, sigma values and
sensor pooling are all decided HERE, locally, in seconds. Nothing about this
script touches the 176 GB source.

LAYOUT
------
One panel. Mean accumulated X/X_0 per track is a filled grey underlay on the
LEFT axis; window failure rate is on the RIGHT axis, one curve per window
multiplier n. They share only the x-axis, so they get visually distinct
treatments: material is background context, the failure curves are the data.

WHAT THE FAILURE RATE IS
------------------------
A row is one (branch, surface, candidate-hit) tuple that survived the extract
filter, meaning: not a hole, the branch has a well-defined majority particle,
and this candidate hit was left by that particle -- so it is a genuinely
CORRECT hit for this branch.

It fails when that correct hit lies outside the CKF's n-sigma box. Writing
z = |r| / sqrt(S), the test is simply z > n:

    strip (1D): z0 > n
    pixel (2D): z0 > n  or  z1 > n

Plotted value per eta bin:

    (correct hits falling outside the n-sigma window) / (correct hits)

0.19 means a 5-sigma cut discards 19% of the hits that genuinely belong to
the track. Pixels and strips are pooled; each is still tested against the
criterion appropriate to its own dimensionality.

The measurement is right-censored: ACTS only records candidates gathered
inside its own 10-sigma box, so the n=10 curve is identically zero BY
CONSTRUCTION, not by measurement, and every rate below it is a lower bound.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds

CARDINAL, COOLGREY = "#8C1515", "#4D4F53"
N_COLORS = {3.0: "#2E9E8F", 5.0: "#E07B39", 7.0: "#7A6BA8", 10.0: "#D81B60"}


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.0):
    """Wilson score interval. Correct at the tiny rates in the tails, where
    the normal approximation would push the lower bound below zero."""
    k = k.astype(float); n = np.maximum(n.astype(float), 1e-9)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-values", default="3,5,7,10")
    ap.add_argument("--bin-width", type=float, default=0.1)
    ap.add_argument("--eta-max", type=float, default=4.0)
    ap.add_argument("--min-count", type=int, default=400)
    ap.add_argument("--figsize", default="13.0x5.6",
                    help="WxH in inches. Use a smaller canvas for the slide so "
                         "point-size fonts read larger relative to the figure.")
    ap.add_argument("--title", default=None, help="override or blank the title")
    ap.add_argument("--legend-cols", type=int, default=0)
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help="multiply all font sizes. Projected slides need ~1.8; "
                         "print posters are fine at 1.0")
    ap.add_argument("--sensor", choices=("both", "pixel", "strip"), default="both")
    ap.add_argument("--branches-dir", type=Path, default=None,
                    help="per-branch summary (branches_event*.parquet). When "
                         "given, the material overlay is computed from it: "
                         "each branch's total is summed over ALL its rows, "
                         "holes included, and charged once per surface. The "
                         "row extract keeps only correct-hit rows, so its "
                         "material truncates at the last correct hit.")
    ap.add_argument("--branch-min-steps", type=int, default=0)
    ap.add_argument("--pure-only", action="store_true",
                    help="restrict BOTH the failure curves and the material to "
                         "branches whose first three non-hole measurements all "
                         "came from the majority particle. NOTE: this is not "
                         "verified to be seed purity -- see the module docs.")
    ap.add_argument("--branch-has-true", action="store_true",
                    help="restrict the overlay to branches containing a hit "
                         "from their majority particle")
    ap.add_argument("--no-material", action="store_true",
                    help="suppress the X/X0 underlay and put the failure\n                          rate on the LEFT axis, for contexts where the\n                          material correlation is not being claimed")
    ap.add_argument("--material", choices=("per_track", "per_row"),
                    default="per_track",
                    help="per_track: sum pathInX0 along each (event, seed, "
                         "branch) and average those totals per eta bin -- the "
                         "material budget a trajectory actually traverses. "
                         "per_row: mean of the running accumulation over "
                         "candidate rows, which is a different and smaller "
                         "quantity.")
    args = ap.parse_args()

    dset = ds.dataset(str(args.rows_dir), format="parquet")
    cols = ["eta", "is_1d", "z0", "z1", "cum_x0",
            "event_id", "seed_id", "branch_id", "step_k"]
    tbl = dset.to_table(columns=cols)
    eta = tbl.column("eta").to_numpy(zero_copy_only=False).astype("f8")
    d1 = tbl.column("is_1d").to_numpy(zero_copy_only=False).astype(bool)
    z0 = tbl.column("z0").to_numpy(zero_copy_only=False).astype("f8")
    z1 = tbl.column("z1").to_numpy(zero_copy_only=False).astype("f8")
    cx = tbl.column("cum_x0").to_numpy(zero_copy_only=False).astype("f8")
    ev = tbl.column("event_id").to_numpy(zero_copy_only=False).astype("i8")
    sd = tbl.column("seed_id").to_numpy(zero_copy_only=False).astype("i8")
    br = tbl.column("branch_id").to_numpy(zero_copy_only=False).astype("i8")
    sk = tbl.column("step_k").to_numpy(zero_copy_only=False).astype("i8")
    del tbl

    if args.pure_only:
        if args.branches_dir is None:
            raise SystemExit("--pure-only needs --branches-dir")
        bt = ds.dataset(str(args.branches_dir), format="parquet").to_table(
            columns=["event_id", "seed_id", "branch_id", "seed_pure"])
        bp = bt.column("seed_pure").to_numpy(zero_copy_only=False)
        sel = bp == 3
        keys = set(zip(
            bt.column("event_id").to_numpy(zero_copy_only=False)[sel].tolist(),
            bt.column("seed_id").to_numpy(zero_copy_only=False)[sel].tolist(),
            bt.column("branch_id").to_numpy(zero_copy_only=False)[sel].tolist()))
        print(f"pure branches: {len(keys):,}")
        keep = np.fromiter(
            (k in keys for k in zip(ev.tolist(), sd.tolist(), br.tolist())),
            bool, len(ev))
        eta, d1, z0, z1, cx = eta[keep], d1[keep], z0[keep], z1[keep], cx[keep]
        ev, sd, br, sk = ev[keep], sd[keep], br[keep], sk[keep]
        print(f"rows after pure cut: {len(eta):,}")

    if args.sensor == "pixel":
        m = ~d1
    elif args.sensor == "strip":
        m = d1
    else:
        m = np.ones(len(eta), bool)
    eta, d1, z0, z1, cx = eta[m], d1[m], z0[m], z1[m], cx[m]
    ev, sd, br, sk = ev[m], sd[m], br[m], sk[m]
    print(f"rows={len(eta):,}  (pixel={np.sum(~d1):,}  strip={np.sum(d1):,})")

    edges = np.arange(-args.eta_max, args.eta_max + 1e-9, args.bin_width)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    bi = np.digitize(eta, edges) - 1
    ok = (bi >= 0) & (bi < len(ctr))
    nb = len(ctr)

    tot = np.bincount(bi[ok], minlength=nb).astype(np.int64)

    if args.material == "per_row":
        msum = np.bincount(bi[ok], weights=cx[ok], minlength=nb)
        mden = tot.astype(float)
    else:
        # Material is a property of a TRAJECTORY, so reduce per track and then
        # average the per-track totals. cum_x0 is the running accumulation, so
        # a track's total is its value at the largest step_k. Ordering by step
        # and taking the last entry per (event, seed, branch) picks that out.
        # Sort by (event, seed, branch, step) and find each group's last row.
        # The identifier columns are compared directly rather than packed into
        # one integer: branch_id has no guaranteed bit-width, so a shift-and-xor
        # key could collide and silently merge two distinct tracks.
        order = np.lexsort((sk, br, sd, ev))
        e_s, s_s, b_s = ev[order], sd[order], br[order]
        cs, bs = cx[order], bi[order]
        last = np.empty(len(order), bool)
        last[-1] = True
        last[:-1] = (e_s[1:] != e_s[:-1]) | (s_s[1:] != s_s[:-1]) | (b_s[1:] != b_s[:-1])
        # one entry per track: its total material and the eta bin it ended in
        t_bin, t_mat = bs[last], cs[last]
        good_t = (t_bin >= 0) & (t_bin < nb)
        msum = np.bincount(t_bin[good_t], weights=t_mat[good_t], minlength=nb)
        mden = np.bincount(t_bin[good_t], minlength=nb).astype(float)
        print(f"material: {int(good_t.sum()):,} tracks "
              f"(from {len(eta):,} candidate rows)")

    if args.branches_dir is not None:
        bt = ds.dataset(str(args.branches_dir), format="parquet").to_table(
            columns=["total_x0", "mean_eta", "n_steps", "has_true",
                     "seed_pure"])
        bx = bt.column("total_x0").to_numpy(zero_copy_only=False).astype("f8")
        bm = bt.column("mean_eta").to_numpy(zero_copy_only=False).astype("f8")
        bs = bt.column("n_steps").to_numpy(zero_copy_only=False).astype("i8")
        bh = bt.column("has_true").to_numpy(zero_copy_only=False).astype(bool)
        bt2 = bt.column("seed_pure").to_numpy(zero_copy_only=False).astype("i8")
        keep = np.isfinite(bm) & np.isfinite(bx) & (bs >= args.branch_min_steps)
        if args.branch_has_true:
            keep &= bh
        if args.pure_only:
            keep &= (bt2 == 3)
        bbi = np.digitize(bm[keep], edges) - 1
        bok = (bbi >= 0) & (bbi < nb)
        msum = np.bincount(bbi[bok], weights=bx[keep][bok], minlength=nb)
        mden = np.bincount(bbi[bok], minlength=nb).astype(float)
        print(f"material: {int(keep.sum()):,} branches from branch summary")

    mat = np.divide(msum, mden, out=np.full(nb, np.nan), where=mden > 0)

    fs = args.font_scale
    if fs != 1.0:
        plt.rcParams.update({
            "font.size": 10 * fs, "axes.labelsize": 11 * fs,
            "xtick.labelsize": 10 * fs, "ytick.labelsize": 10 * fs,
            "legend.fontsize": 10 * fs, "lines.linewidth": 1.4 * fs,
            "lines.markersize": 3.0 * fs, "axes.linewidth": 0.8 * fs,
        })
    fw, fh = (float(v) for v in args.figsize.lower().split("x"))
    fig, axm = plt.subplots(figsize=(fw, fh))

    # ---- material underlay, LEFT axis (unless suppressed) ----
    if args.no_material:
        # No second quantity, so the failure rate takes the primary axis and
        # there is no twin. Used where the material correlation is not being
        # claimed, so the figure must not imply one.
        good = tot >= args.min_count
        axr = axm
    else:
        good = (tot >= args.min_count) & np.isfinite(mat)
        axm.fill_between(ctr[good], 0, mat[good], color="#DCDCDC", alpha=0.9,
                         lw=0, zorder=0)
        axm.plot(ctr[good], mat[good], color=COOLGREY, lw=1.7, zorder=1,
                 label=r"$X/X_0$")
        axm.set_ylabel(r"mean $X/X_0$ per track")
        axm.set_ylim(0, np.nanmax(mat[good]) * 1.05)
        axr = axm.twinx()
        axr.set_zorder(axm.get_zorder() + 1)
        axr.patch.set_visible(False)

    axm.set_xlabel(r"$\eta$")
    axm.set_xlim(-args.eta_max, args.eta_max)
    axm.axvline(0, color="#BBBBBB", ls=":", lw=1.0, zorder=0)
    axm.grid(alpha=0.15)

    hi_all = 0.0
    for nv in [float(x) for x in args.n_values.split(",")]:
        failed = (z0 > nv) | ((~d1) & (z1 > nv))   # NaN z1 on strips -> False
        k = np.bincount(bi[ok & failed], minlength=nb).astype(np.int64)
        p = np.divide(k, np.maximum(tot, 1), dtype=float)
        lo, hi = wilson(k, tot)
        # The Wilson interval is not centred on p -- it is pulled toward 0.5 --
        # so p - lo can be very slightly negative at small p. errorbar() takes
        # offsets, which cannot be negative, so clip at zero. The asymmetry
        # this hides is far below a pixel at these bin populations.
        yerr = np.clip(np.vstack([p - lo, hi - p]), 0.0, None)
        axr.errorbar(ctr[good], p[good], yerr=yerr[:, good], fmt="-o", ms=3.0,
                     lw=1.4, elinewidth=1.0, capsize=1.6,
                     color=N_COLORS.get(nv, CARDINAL),
                     label=f"$n={nv:g}$", zorder=3)
        hi_all = max(hi_all, np.nanmax(hi[good]))
        print(f"  n={nv:>4g}: {k.sum()/tot.sum():.4%}")

    if args.no_material:
        axr.set_ylabel("window failure rate")
        axr.set_ylim(0, hi_all * 1.08)
    else:
        axr.set_ylabel("window failure rate", color=COOLGREY)
        axr.set_ylim(-0.02 * hi_all, hi_all * 1.08)
        axr.tick_params(axis="y", colors=COOLGREY)

    h1, l1 = axm.get_legend_handles_labels()
    h2, l2 = ([], []) if args.no_material else axr.get_legend_handles_labels()
    ncol = args.legend_cols or len(l1 + l2)
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=ncol,
               frameon=True, framealpha=0.95, fontsize=10 * fs,
               bbox_to_anchor=(0.5, -0.015))
    ttl = (r"Window failure rate vs material budget "
           r"(ODD, $\mu=200$ $t\bar{t}$)") if args.title is None else args.title
    if ttl:
        axm.set_title(ttl, color=CARDINAL, fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    print(f"eta bins={nb} (width {args.bin_width})  wrote {args.out}")


if __name__ == "__main__":
    main()
