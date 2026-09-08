"""
Expanded-parquet training distribution analysis.

Runs on Modal (data lives on the surp-acts-data volume). Produces:
  - Per-event summary CSV
  - PNG plots for row composition, positive fraction, occupancy,
    chi-squared, and config-pruning impact
  - Stdout text summary

Usage:
    python3.12 -m modal run analyze_training_dist.py
"""

import modal

app = modal.App("surp-training-analysis")
data_vol = modal.Volume.from_name("surp-acts-data")
DATA_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyarrow", "pandas", "numpy", "matplotlib", "scipy")
)

HEALTHY_EVENTS = [
    0, 1, 2, 3, 4, 7, 8, 9, 11, 12, 13,
    16, 19, 20, 21, 23, 24, 25, 26, 27, 31,
]

PARQUET_DIR = f"{DATA_PATH}/results/train32/expanded"
OUTPUT_DIR = f"{DATA_PATH}/results/train32/analysis"

# Three CKF operating points from Optuna joint optimization
# (experiments/LOG.md, configs/tight_t79.yaml, medium_t70.yaml, fast_t331.json)
CONFIGS = {
    "tight_t79": {
        "chi2_max": 16.26,
        "max_holes_and_outliers": 1,
        "n_meas_min": 9,
        "branch_cap": 3,
        "pt_min": 0.46,
    },
    "medium_t70": {
        "chi2_max": 12.04,
        "max_holes_and_outliers": 1,
        "n_meas_min": 7,
        "branch_cap": 2,
        "pt_min": 0.594,
    },
    "fast_t331": {
        "chi2_max": 15.40,
        "max_holes_and_outliers": 1,
        "n_meas_min": 8,
        "branch_cap": 5,
        "pt_min": 0.622,
    },
}

WINDOW_THRESHOLDS = [3, 5, 7, 10]
MAX_OCCUPANCY_BIN = 200


def _compute_event_stats(event_id: int) -> dict:
    """Read one expanded parquet and return per-event statistics dict.

    Reads the full event with column projection, computes all statistics
    with vectorized numpy, then frees memory before returning.
    """
    import numpy as np
    import pyarrow.parquet as pq

    path = f"{PARQUET_DIR}/expanded_event{event_id:09d}.parquet"

    needed_cols = [
        "cand_hit_id", "majority_undefined", "branch_majority_pid",
        "contrib_pids",
        "residual_l0", "residual_l1", "S00", "S11",
        "seed_id", "branch_id", "step_k",
        "is_pixel", "chi2_inc", "n_window",
        "n_hits", "n_holes", "n_seq_holes", "state_qop",
    ]
    table = pq.read_table(path, columns=needed_cols)
    n_rows = len(table)

    # ── Extract columns to numpy ────────────────────────────────────────
    cand_hit = table.column("cand_hit_id").to_numpy()
    maj_undef = table.column("majority_undefined").to_numpy().astype(bool)
    branch_maj = table.column("branch_majority_pid").to_numpy()
    chi2 = table.column("chi2_inc").to_numpy()
    n_win = table.column("n_window").to_numpy()
    is_pix_raw = table.column("is_pixel").to_numpy()
    is_pix = np.nan_to_num(is_pix_raw, nan=0.0).astype(bool)
    r0 = table.column("residual_l0").to_numpy()
    r1 = table.column("residual_l1").to_numpy()
    s00 = table.column("S00").to_numpy()
    s11 = table.column("S11").to_numpy()
    seed = table.column("seed_id").to_numpy().astype(np.int64)
    branch = table.column("branch_id").to_numpy().astype(np.int64)
    step = table.column("step_k").to_numpy().astype(np.int64)
    n_hits_col = table.column("n_hits").to_numpy()
    n_holes_col = table.column("n_holes").to_numpy()
    n_seq_col = table.column("n_seq_holes").to_numpy()
    qop = table.column("state_qop").to_numpy()

    # ── label_same_particle via vectorized list membership ──────────────
    cp_chunked = table.column("contrib_pids")
    cp_col = cp_chunked.combine_chunks()  # ListArray (single chunk)
    del cp_chunked
    cp_offsets = cp_col.offsets.to_numpy().astype(np.int64)
    cp_flat_arr = cp_col.flatten()
    cp_flat = cp_flat_arr.to_numpy().astype(np.int64) if len(cp_flat_arr) > 0 else np.array([], dtype=np.int64)
    del table, cp_col, cp_flat_arr

    is_hole = cand_hit == -1
    nonhole = ~is_hole
    nonhole_defined = nonhole & (~maj_undef)

    diffs = np.diff(cp_offsets)
    if len(cp_flat) > 0 and diffs.sum() > 0:
        majority_repeated = np.repeat(branch_maj, diffs)
        element_matches = cp_flat == majority_repeated
        row_idx = np.repeat(np.arange(n_rows, dtype=np.int64), diffs)
        match_counts = np.bincount(row_idx[element_matches], minlength=n_rows)
        label_same = (match_counts > 0) & nonhole_defined
        del majority_repeated, element_matches, row_idx, match_counts
    else:
        label_same = np.zeros(n_rows, dtype=bool)
    del cp_flat, cp_offsets, diffs

    # ── §1-2: Basic stats ───────────────────────────────────────────────
    total_rows = n_rows
    n_hole_count = int(is_hole.sum())
    n_maj_undef = int(maj_undef.sum())
    n_nhd = int(nonhole_defined.sum())
    n_pos = int(label_same.sum())

    # ── §4: Chi-squared histogram ───────────────────────────────────────
    chi2_bin_edges = np.logspace(-3, 3, 201)
    chi2_nh = chi2[nonhole]
    label_nh = label_same[nonhole]
    valid_chi2 = np.isfinite(chi2_nh) & (chi2_nh > 0)
    chi2_pos_bins = np.histogram(chi2_nh[valid_chi2 & label_nh], bins=chi2_bin_edges)[0]
    chi2_neg_bins = np.histogram(chi2_nh[valid_chi2 & ~label_nh], bins=chi2_bin_edges)[0]
    median_chi2 = float(np.median(chi2_nh[valid_chi2])) if valid_chi2.any() else float("nan")
    del chi2_nh, valid_chi2

    # ── §3: Occupancy recomputation ─────────────────────────────────────
    valid_S = nonhole & (s00 > 0) & (s11 > 0) & np.isfinite(s00) & np.isfinite(s11)
    n_needed = np.full(n_rows, np.inf)
    n_needed[valid_S] = np.maximum(
        np.abs(r0[valid_S]) / np.sqrt(s00[valid_S]),
        np.abs(r1[valid_S]) / np.sqrt(s11[valid_S]),
    )
    del r0, r1, s00, s11

    max_step = int(step.max()) + 1 if n_rows > 0 else 1
    max_branch = int(branch.max()) + 1 if n_rows > 0 else 1

    # compound group key: (seed, branch, step) → single int64
    group_key = seed * (max_branch * max_step) + branch * max_step + step

    nw_hists = {}
    pf_by_nw = {}

    for n_thresh in WINDOW_THRESHOLDS:
        in_win = nonhole & (n_needed <= n_thresh)
        if not in_win.any():
            nw_hists[n_thresh] = {
                "pixel": np.zeros(MAX_OCCUPANCY_BIN, dtype=np.int64),
                "strip": np.zeros(MAX_OCCUPANCY_BIN, dtype=np.int64),
            }
            pf_by_nw[n_thresh] = {
                "count": np.zeros(MAX_OCCUPANCY_BIN, dtype=np.int64),
                "pos": np.zeros(MAX_OCCUPANCY_BIN, dtype=np.int64),
            }
            continue

        filt_keys = group_key[in_win]
        filt_pix = is_pix[in_win]
        filt_label = label_same[in_win]

        # sort by key for efficient grouping
        sort_idx = np.argsort(filt_keys)
        sk = filt_keys[sort_idx]
        sp = filt_pix[sort_idx]
        sl = filt_label[sort_idx]

        # group boundaries
        boundaries = np.concatenate([[0], np.where(sk[1:] != sk[:-1])[0] + 1, [len(sk)]])
        group_sizes = np.diff(boundaries)
        group_is_pixel = sp[boundaries[:-1]]
        group_pos_count = np.add.reduceat(sl.astype(np.int64), boundaries[:-1])

        clipped = np.clip(group_sizes, 0, MAX_OCCUPANCY_BIN - 1)

        pixel_hist = np.zeros(MAX_OCCUPANCY_BIN, dtype=np.int64)
        strip_hist = np.zeros(MAX_OCCUPANCY_BIN, dtype=np.int64)
        np.add.at(pixel_hist, clipped[group_is_pixel], 1)
        np.add.at(strip_hist, clipped[~group_is_pixel], 1)
        nw_hists[n_thresh] = {"pixel": pixel_hist, "strip": strip_hist}

        cnt_arr = np.zeros(MAX_OCCUPANCY_BIN, dtype=np.int64)
        pos_arr = np.zeros(MAX_OCCUPANCY_BIN, dtype=np.int64)
        np.add.at(cnt_arr, clipped, group_sizes)
        np.add.at(pos_arr, clipped, group_pos_count)
        pf_by_nw[n_thresh] = {"count": cnt_arr, "pos": pos_arr}

        del filt_keys, sort_idx, sk, sp, sl, boundaries, group_sizes

    # ── §5: Terminal rows for pruning analysis ──────────────────────────
    branch_key = seed * max_branch + branch  # unique per (seed, branch)
    sort_idx = np.lexsort((-step, branch_key))
    sorted_bk = branch_key[sort_idx]
    first_mask = np.concatenate([[True], sorted_bk[1:] != sorted_bk[:-1]])
    terminal_idx = sort_idx[first_mask]

    # count rows per branch
    _, bk_counts = np.unique(branch_key, return_counts=True)

    terminal_n_hits = n_hits_col[terminal_idx]
    terminal_n_holes = n_holes_col[terminal_idx]
    terminal_n_seq = n_seq_col[terminal_idx]
    terminal_qop = qop[terminal_idx]
    terminal_n_rows = bk_counts

    del sort_idx, sorted_bk, first_mask, branch_key, group_key, n_needed

    median_nw = float(np.median(n_win[nonhole])) if nonhole.any() else float("nan")

    return {
        "event_id": event_id,
        "total_rows": total_rows,
        "n_holes": n_hole_count,
        "hole_fraction": n_hole_count / total_rows if total_rows > 0 else 0.0,
        "n_majority_undef": n_maj_undef,
        "majority_undef_fraction": n_maj_undef / total_rows if total_rows > 0 else 0.0,
        "n_nonhole_defined": n_nhd,
        "n_positive": n_pos,
        "positive_fraction": n_pos / n_nhd if n_nhd > 0 else 0.0,
        "median_chi2": median_chi2,
        "median_n_window": median_nw,
        "chi2_pos_bins": chi2_pos_bins.tolist(),
        "chi2_neg_bins": chi2_neg_bins.tolist(),
        "chi2_bin_edges": chi2_bin_edges.tolist(),
        "nw_hists": {
            str(n): {k: v.tolist() for k, v in d.items()}
            for n, d in nw_hists.items()
        },
        "pf_by_nw": {
            str(n): {k: v.tolist() for k, v in d.items()}
            for n, d in pf_by_nw.items()
        },
        "terminal_n_hits": terminal_n_hits.tolist(),
        "terminal_n_holes": terminal_n_holes.tolist(),
        "terminal_n_seq": terminal_n_seq.tolist(),
        "terminal_qop": terminal_qop.tolist(),
        "terminal_n_rows": terminal_n_rows.tolist(),
    }


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    memory=65536,
    cpu=8,
    timeout=7200,
)
def run_analysis():
    """Process all healthy events and generate outputs."""
    import os
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Process all events ──────────────────────────────────────────────
    all_stats = []
    for i, eid in enumerate(HEALTHY_EVENTS):
        print(f"  [{i+1}/{len(HEALTHY_EVENTS)}] Event {eid}...", flush=True)
        stats = _compute_event_stats(eid)
        all_stats.append(stats)
        print(f"    {stats['total_rows']:>12,} rows  "
              f"hole={stats['hole_fraction']:.3f}  "
              f"pos={stats['positive_fraction']:.5f}", flush=True)

    # ── §6: Per-event summary table ─────────────────────────────────────
    summary_rows = []
    for s in all_stats:
        summary_rows.append({
            "event_id": s["event_id"],
            "total_rows": s["total_rows"],
            "hole_fraction": round(s["hole_fraction"], 5),
            "majority_undefined_fraction": round(s["majority_undef_fraction"], 5),
            "positive_fraction": round(s["positive_fraction"], 6),
            "median_n_window": round(s["median_n_window"], 1),
            "mean_chi2_inc": round(s["median_chi2"], 2),
        })
    summary_df = pd.DataFrame(summary_rows)

    numeric_cols = ["total_rows", "hole_fraction", "majority_undefined_fraction",
                    "positive_fraction", "median_n_window", "mean_chi2_inc"]
    flags = []
    for _, row in summary_df.iterrows():
        outlier_cols = []
        for col in numeric_cols:
            vals = summary_df[col].dropna()
            if len(vals) > 2:
                mean, std = vals.mean(), vals.std()
                if std > 0 and abs(row[col] - mean) > 2 * std:
                    outlier_cols.append(col)
        flags.append(", ".join(outlier_cols) if outlier_cols else "")
    summary_df["outlier_flags"] = flags
    summary_df.to_csv(f"{OUTPUT_DIR}/per_event_summary.csv", index=False)

    # ── Overall statistics ──────────────────────────────────────────────
    total_all = sum(s["total_rows"] for s in all_stats)
    holes_all = sum(s["n_holes"] for s in all_stats)
    mundef_all = sum(s["n_majority_undef"] for s in all_stats)
    nhd_all = sum(s["n_nonhole_defined"] for s in all_stats)
    npos_all = sum(s["n_positive"] for s in all_stats)

    print(f"\n{'='*60}")
    print(f"Overall ({len(HEALTHY_EVENTS)} events)")
    print(f"{'='*60}")
    print(f"Total rows:          {total_all:>15,}")
    print(f"Hole fraction:       {holes_all/total_all:.5f}  "
          f"(mean±std: {np.mean([s['hole_fraction'] for s in all_stats]):.5f} "
          f"± {np.std([s['hole_fraction'] for s in all_stats]):.5f})")
    print(f"Majority undefined:  {mundef_all/total_all:.5f}  "
          f"(mean±std: {np.mean([s['majority_undef_fraction'] for s in all_stats]):.5f} "
          f"± {np.std([s['majority_undef_fraction'] for s in all_stats]):.5f})")
    print(f"Positive fraction:   {npos_all/nhd_all:.6f}  "
          f"(mean±std: {np.mean([s['positive_fraction'] for s in all_stats]):.6f} "
          f"± {np.std([s['positive_fraction'] for s in all_stats]):.6f})")
    print(f"  ({npos_all:,} positives / {nhd_all:,} non-hole, defined)")

    outlier_events = summary_df[summary_df["outlier_flags"] != ""]
    if len(outlier_events) > 0:
        print(f"\nOutlier events (>2 sigma):")
        for _, row in outlier_events.iterrows():
            print(f"  Event {int(row['event_id'])}: {row['outlier_flags']}")

    print(f"\nPer-event summary:")
    print(summary_df.to_string(index=False))

    # ── Aggregate histograms ────────────────────────────────────────────
    chi2_pos = np.zeros(200, dtype=np.int64)
    chi2_neg = np.zeros(200, dtype=np.int64)
    chi2_edges = np.array(all_stats[0]["chi2_bin_edges"])

    nw_total = {n: {"pixel": np.zeros(MAX_OCCUPANCY_BIN, dtype=np.int64),
                     "strip": np.zeros(MAX_OCCUPANCY_BIN, dtype=np.int64)}
                for n in WINDOW_THRESHOLDS}
    pf_total = {n: {"count": np.zeros(MAX_OCCUPANCY_BIN, dtype=np.int64),
                     "pos": np.zeros(MAX_OCCUPANCY_BIN, dtype=np.int64)}
                for n in WINDOW_THRESHOLDS}

    all_term_nhits = []
    all_term_nholes = []
    all_term_nseq = []
    all_term_qop = []
    all_term_nrows = []

    for s in all_stats:
        chi2_pos += np.array(s["chi2_pos_bins"])
        chi2_neg += np.array(s["chi2_neg_bins"])
        for n in WINDOW_THRESHOLDS:
            nk = str(n)
            for k in ("pixel", "strip"):
                nw_total[n][k] += np.array(s["nw_hists"][nk][k])
            for k in ("count", "pos"):
                pf_total[n][k] += np.array(s["pf_by_nw"][nk][k])
        all_term_nhits.extend(s["terminal_n_hits"])
        all_term_nholes.extend(s["terminal_n_holes"])
        all_term_nseq.extend(s["terminal_n_seq"])
        all_term_qop.extend(s["terminal_qop"])
        all_term_nrows.extend(s["terminal_n_rows"])

    term_nholes = np.array(all_term_nholes)
    term_nseq = np.array(all_term_nseq)
    term_nhits = np.array(all_term_nhits)
    term_nrows = np.array(all_term_nrows)
    n_branches = len(term_nholes)
    total_branch_rows = int(term_nrows.sum())

    # ── Plotting ────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })

    # ── Plot 1: Chi-squared distribution ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    centers = np.sqrt(chi2_edges[:-1] * chi2_edges[1:])
    widths = chi2_edges[1:] - chi2_edges[:-1]
    ax.bar(centers, chi2_pos, width=widths, alpha=0.6,
           label="Same particle (positive)", color="#4C72B0", log=True)
    ax.bar(centers, chi2_neg, width=widths, alpha=0.6,
           label="Different particle (negative)", color="#DD5555", log=True)
    cfg_colors = {"tight_t79": "#2ca02c", "medium_t70": "#ff7f0e", "fast_t331": "#9467bd"}
    for name, cfg in CONFIGS.items():
        ax.axvline(cfg["chi2_max"], color=cfg_colors[name], linestyle="--",
                   alpha=0.8, linewidth=2, label=f"{name} chi2_max={cfg['chi2_max']:.1f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\chi^2_{\mathrm{inc}}$")
    ax.set_ylabel("Count")
    ax.set_title(r"$\chi^2$ increment: positive vs negative candidates")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/chi2_distribution.png", dpi=150)
    plt.close(fig)
    print("\nSaved chi2_distribution.png")

    # ── Plot 2: Occupancy histograms ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_idx, n_thresh in enumerate(WINDOW_THRESHOLDS):
        ax = axes.flat[ax_idx]
        pixel_h = nw_total[n_thresh]["pixel"]
        strip_h = nw_total[n_thresh]["strip"]
        combined = pixel_h + strip_h
        mask = combined > 0
        xmax = min(int(np.max(np.where(mask))) + 5, 100) if mask.any() else 50
        x = np.arange(xmax)
        ax.bar(x, pixel_h[:xmax], alpha=0.7, label="Pixel", color="#4C72B0", width=0.8)
        ax.bar(x, strip_h[:xmax], alpha=0.7, label="Strip", color="#DD8452",
               bottom=pixel_h[:xmax], width=0.8)
        ax.set_xlabel("Recomputed n_window")
        ax.set_ylabel("# (branch, surface) groups")
        ax.set_title(f"Window multiplier n = {n_thresh}")
        ax.legend(fontsize=9)
        ax.set_yscale("log")
    fig.suptitle("Occupancy distribution at different window sizes", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/occupancy_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved occupancy_histograms.png")

    # ── Plot 3: Positive fraction vs n_window ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(WINDOW_THRESHOLDS)))
    for i, n_thresh in enumerate(WINDOW_THRESHOLDS):
        counts = pf_total[n_thresh]["count"]
        pos = pf_total[n_thresh]["pos"]
        valid = counts > 100
        x = np.arange(MAX_OCCUPANCY_BIN)[valid]
        frac = pos[valid].astype(float) / counts[valid].astype(float)
        ax.plot(x, frac, "o-", color=cmap[i], label=f"n = {n_thresh}",
                markersize=3, alpha=0.8)
    ax.set_xlabel("n_window (recomputed at given n)")
    ax.set_ylabel("Positive fraction")
    ax.set_title("Positive fraction vs occupancy at different window sizes")
    ax.legend()
    ax.set_xlim(0, 60)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/pos_fraction_vs_occupancy.png", dpi=150)
    plt.close(fig)
    print("Saved pos_fraction_vs_occupancy.png")

    # ── Plot 4: Per-event composition ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    eids = [s["event_id"] for s in all_stats]

    ax = axes[0]
    ax.bar(range(len(eids)), [s["total_rows"] / 1e6 for s in all_stats], color="#4C72B0")
    ax.set_xticks(range(len(eids)))
    ax.set_xticklabels(eids, rotation=45, fontsize=7)
    ax.set_ylabel("Rows (millions)")
    ax.set_title("Row count per event")

    ax = axes[1]
    ax.bar(range(len(eids)), [s["hole_fraction"] for s in all_stats], color="#DD8452")
    ax.set_xticks(range(len(eids)))
    ax.set_xticklabels(eids, rotation=45, fontsize=7)
    ax.set_ylabel("Fraction")
    ax.set_title("Hole fraction per event")

    ax = axes[2]
    ax.bar(range(len(eids)), [s["positive_fraction"] for s in all_stats], color="#55A868")
    ax.set_xticks(range(len(eids)))
    ax.set_xticklabels(eids, rotation=45, fontsize=7)
    ax.set_ylabel("Fraction")
    ax.set_title("Positive fraction per event")

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/per_event_composition.png", dpi=150)
    plt.close(fig)
    print("Saved per_event_composition.png")

    # ── §5: Config pruning impact ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Config pruning impact ({n_branches:,} total branches)")
    print(f"{'='*60}")
    print(f"{'Config':<14} {'Branch surv':<16} {'Row surv':<16} "
          f"{'Note'}")

    pruning_data = {}
    for name, cfg in CONFIGS.items():
        surv = term_nholes <= cfg["max_holes_and_outliers"]
        n_surv = int(surv.sum())
        rows_surv = int(term_nrows[surv].sum())
        branch_frac = n_surv / n_branches if n_branches > 0 else 0
        row_frac = rows_surv / total_branch_rows if total_branch_rows > 0 else 0
        pruning_data[name] = {"branch": branch_frac, "row": row_frac}
        print(f"{name:<14} {branch_frac:.4f} ({n_surv:>10,}/{n_branches:,})  "
              f"{row_frac:.4f}  hole-cut only (upper bound)")

    # Plot 5: Config pruning
    fig, ax = plt.subplots(figsize=(8, 5))
    config_names = list(CONFIGS.keys())
    x = np.arange(len(config_names))
    ax.bar(x - 0.15, [pruning_data[n]["branch"] for n in config_names],
           0.3, label="Branch survival", color="#4C72B0")
    ax.bar(x + 0.15, [pruning_data[n]["row"] for n in config_names],
           0.3, label="Row survival", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, fontsize=10)
    ax.set_ylabel("Survival fraction")
    ax.set_title("Config pruning impact (hole cut only — upper bound)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    for i, name in enumerate(config_names):
        ax.text(i - 0.15, pruning_data[name]["branch"] + 0.02,
                f"{pruning_data[name]['branch']:.2%}", ha="center", fontsize=8)
        ax.text(i + 0.15, pruning_data[name]["row"] + 0.02,
                f"{pruning_data[name]['row']:.2%}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/config_pruning.png", dpi=150)
    plt.close(fig)
    print("Saved config_pruning.png")

    data_vol.commit()
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")

    return {
        "total_rows": total_all,
        "n_events": len(HEALTHY_EVENTS),
        "hole_fraction": holes_all / total_all,
        "majority_undef_fraction": mundef_all / total_all,
        "positive_fraction": npos_all / nhd_all,
        "output_dir": OUTPUT_DIR,
    }


@app.local_entrypoint()
def main():
    print(f"Analyzing {len(HEALTHY_EVENTS)} healthy events...")
    result = run_analysis.remote()
    print(f"\n{'='*60}")
    print(f"Analysis complete.")
    print(f"  Events:            {result['n_events']}")
    print(f"  Total rows:        {result['total_rows']:,}")
    print(f"  Hole fraction:     {result['hole_fraction']:.5f}")
    print(f"  Majority undef:    {result['majority_undef_fraction']:.5f}")
    print(f"  Positive fraction: {result['positive_fraction']:.6f}")
    print(f"  Outputs at:        {result['output_dir']}")
