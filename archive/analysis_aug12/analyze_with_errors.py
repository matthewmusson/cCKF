"""
Replot three figures with binomial error bars:
  1. Window failure rate vs |eta| (from pilot data)
  2. Positive fraction vs occupancy at different n (log scale, original unstratified)
  3. Positive fraction vs occupancy by |eta| (log scale)

Usage:
    python3.12 -m modal run analyze_with_errors.py
"""

import modal

app = modal.App("surp-plots-with-errors")
data_vol = modal.Volume.from_name("surp-acts-data")
DATA_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyarrow", "numpy", "matplotlib")
)

HEALTHY_EVENTS = [
    0, 1, 2, 3, 4, 7, 8, 9, 11, 12, 13,
    16, 19, 20, 21, 23, 24, 25, 26, 27, 31,
]

PARQUET_DIR = f"{DATA_PATH}/results/train32/expanded"
OUTPUT_DIR = f"{DATA_PATH}/results/train32/analysis"

ETA_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
N_ETA = len(ETA_BINS) - 1

WINDOW_THRESHOLDS = [3, 5, 7, 10]
MAX_OCC = 100

# Window multiplier values for failure rate plot
FAILURE_N_VALUES = [3, 4, 5, 6, 7, 8, 9, 10]
N_FAIL = len(FAILURE_N_VALUES)


def _accumulate_event(event_id: int) -> dict:
    """Accumulate all needed histograms for one event."""
    import numpy as np
    import pyarrow.parquet as pq

    path = f"{PARQUET_DIR}/expanded_event{event_id:09d}.parquet"
    needed = [
        "cand_hit_id", "majority_undefined", "branch_majority_pid",
        "contrib_pids", "is_pixel",
        "residual_l0", "residual_l1", "S00", "S11",
        "seed_id", "branch_id", "step_k", "state_theta",
    ]
    table = pq.read_table(path, columns=needed)
    n_rows = len(table)

    cand_hit = table.column("cand_hit_id").to_numpy()
    maj_undef = table.column("majority_undefined").to_numpy().astype(bool)
    branch_maj = table.column("branch_majority_pid").to_numpy()
    is_pix = np.nan_to_num(table.column("is_pixel").to_numpy(), nan=0.0).astype(bool)
    r0 = table.column("residual_l0").to_numpy()
    r1 = table.column("residual_l1").to_numpy()
    s00 = table.column("S00").to_numpy()
    s11 = table.column("S11").to_numpy()
    seed = table.column("seed_id").to_numpy().astype(np.int64)
    branch = table.column("branch_id").to_numpy().astype(np.int64)
    step = table.column("step_k").to_numpy().astype(np.int64)
    theta = table.column("state_theta").to_numpy()

    # label
    cp_col = table.column("contrib_pids").combine_chunks()
    del table
    cp_offsets = cp_col.offsets.to_numpy().astype(np.int64)
    cp_flat_arr = cp_col.flatten()
    cp_flat = cp_flat_arr.to_numpy().astype(np.int64) if len(cp_flat_arr) > 0 else np.array([], dtype=np.int64)
    del cp_col, cp_flat_arr

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

    # eta
    half_theta = np.clip(theta / 2.0, 1e-10, np.pi - 1e-10)
    eta = -np.log(np.tan(half_theta))
    abs_eta = np.abs(eta)
    eta_valid = np.isfinite(abs_eta)
    eta_bin_idx = np.clip(np.digitize(abs_eta, ETA_BINS) - 1, 0, N_ETA - 1)

    # n_needed
    valid_S = nonhole & (s00 > 0) & (s11 > 0) & np.isfinite(s00) & np.isfinite(s11)
    n_needed = np.full(n_rows, np.inf)
    n_needed[valid_S] = np.maximum(
        np.abs(r0[valid_S]) / np.sqrt(s00[valid_S]),
        np.abs(r1[valid_S]) / np.sqrt(s11[valid_S]),
    )
    del r0, r1, s00, s11

    # compound group key
    max_step = int(step.max()) + 1 if n_rows > 0 else 1
    max_branch = int(branch.max()) + 1 if n_rows > 0 else 1
    group_key = seed * (max_branch * max_step) + branch * max_step + step

    # ── Plot 1: Window failure rate vs eta ──────────────────────────────
    # For each (eta_bin, n_value): count true hits (label_same & nonhole_defined)
    # and how many fall outside the window (n_needed > n_value)
    # failure_total[ei, ni] = number of true hits in that eta bin
    # failure_outside[ei, ni] = number of those outside the n-sigma window
    failure_total = np.zeros((N_ETA, N_FAIL), dtype=np.int64)
    failure_outside = np.zeros((N_ETA, N_FAIL), dtype=np.int64)

    true_hit_mask = label_same & nonhole_defined & eta_valid & np.isfinite(n_needed)
    if true_hit_mask.any():
        true_eta = eta_bin_idx[true_hit_mask]
        true_n_needed = n_needed[true_hit_mask]
        for ni, n_val in enumerate(FAILURE_N_VALUES):
            outside = true_n_needed > n_val
            for ei in range(N_ETA):
                emask = true_eta == ei
                failure_total[ei, ni] += int(emask.sum())
                failure_outside[ei, ni] += int((emask & outside).sum())

    # ── Plots 2 & 3: Positive fraction vs occupancy ────────────────────
    # Unstratified (plot 2): shape (len(WINDOW_THRESHOLDS), MAX_OCC)
    pf_count = np.zeros((len(WINDOW_THRESHOLDS), MAX_OCC), dtype=np.int64)
    pf_pos = np.zeros((len(WINDOW_THRESHOLDS), MAX_OCC), dtype=np.int64)

    # Stratified by eta (plot 3): shape (N_ETA, len(WINDOW_THRESHOLDS), MAX_OCC)
    pf_eta_count = np.zeros((N_ETA, len(WINDOW_THRESHOLDS), MAX_OCC), dtype=np.int64)
    pf_eta_pos = np.zeros((N_ETA, len(WINDOW_THRESHOLDS), MAX_OCC), dtype=np.int64)

    for wi, n_thresh in enumerate(WINDOW_THRESHOLDS):
        in_win = nonhole & (n_needed <= n_thresh)
        if not in_win.any():
            continue

        filt_keys = group_key[in_win]
        filt_label = label_same[in_win]
        filt_eta = eta_bin_idx[in_win]

        sort_idx = np.argsort(filt_keys)
        sk = filt_keys[sort_idx]
        sl = filt_label[sort_idx]
        se = filt_eta[sort_idx]

        boundaries = np.concatenate([[0], np.where(sk[1:] != sk[:-1])[0] + 1, [len(sk)]])
        group_sizes = np.diff(boundaries)
        group_eta = se[boundaries[:-1]]
        group_pos = np.add.reduceat(sl.astype(np.int64), boundaries[:-1])

        clipped = np.clip(group_sizes, 0, MAX_OCC - 1)

        # unstratified
        np.add.at(pf_count[wi], clipped, group_sizes)
        np.add.at(pf_pos[wi], clipped, group_pos)

        # stratified by eta
        for ei in range(N_ETA):
            emask = group_eta == ei
            if emask.any():
                np.add.at(pf_eta_count[ei, wi], clipped[emask], group_sizes[emask])
                np.add.at(pf_eta_pos[ei, wi], clipped[emask], group_pos[emask])

        del filt_keys, sort_idx, sk, sl, se, boundaries, group_sizes

    return {
        "failure_total": failure_total.tolist(),
        "failure_outside": failure_outside.tolist(),
        "pf_count": pf_count.tolist(),
        "pf_pos": pf_pos.tolist(),
        "pf_eta_count": pf_eta_count.tolist(),
        "pf_eta_pos": pf_eta_pos.tolist(),
        "n_rows": n_rows,
    }


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    memory=65536,
    cpu=8,
    timeout=7200,
)
def run_error_plots():
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # accumulate
    tot_fail_total = np.zeros((N_ETA, N_FAIL), dtype=np.int64)
    tot_fail_outside = np.zeros((N_ETA, N_FAIL), dtype=np.int64)
    tot_pf_count = np.zeros((len(WINDOW_THRESHOLDS), MAX_OCC), dtype=np.int64)
    tot_pf_pos = np.zeros((len(WINDOW_THRESHOLDS), MAX_OCC), dtype=np.int64)
    tot_pf_eta_count = np.zeros((N_ETA, len(WINDOW_THRESHOLDS), MAX_OCC), dtype=np.int64)
    tot_pf_eta_pos = np.zeros((N_ETA, len(WINDOW_THRESHOLDS), MAX_OCC), dtype=np.int64)

    for i, eid in enumerate(HEALTHY_EVENTS):
        print(f"  [{i+1}/{len(HEALTHY_EVENTS)}] Event {eid}...", flush=True)
        r = _accumulate_event(eid)
        tot_fail_total += np.array(r["failure_total"])
        tot_fail_outside += np.array(r["failure_outside"])
        tot_pf_count += np.array(r["pf_count"])
        tot_pf_pos += np.array(r["pf_pos"])
        tot_pf_eta_count += np.array(r["pf_eta_count"])
        tot_pf_eta_pos += np.array(r["pf_eta_pos"])
        print(f"    {r['n_rows']:>12,} rows", flush=True)

    def binom_err(k, n):
        """Binomial standard error: sqrt(p_hat * (1 - p_hat) / n)."""
        p = np.where(n > 0, k.astype(float) / n.astype(float), 0.0)
        return np.where(n > 0, np.sqrt(p * (1 - p) / n), 0.0)

    eta_labels = [f"|η|∈[{ETA_BINS[i]:.1f},{ETA_BINS[i+1]:.1f})"
                  for i in range(N_ETA)]
    eta_centers = [(ETA_BINS[i] + ETA_BINS[i+1]) / 2 for i in range(N_ETA)]

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })

    # ── Plot 1: Window failure rate vs |eta| ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, N_FAIL))

    for ni, n_val in enumerate(FAILURE_N_VALUES):
        total = tot_fail_total[:, ni].astype(float)
        outside = tot_fail_outside[:, ni].astype(float)
        valid = total > 0
        rate = np.where(valid, outside / total, 0.0)
        err = np.where(valid, binom_err(tot_fail_outside[:, ni], tot_fail_total[:, ni]), 0.0)

        x = np.array(eta_centers)
        ax.errorbar(x[valid], rate[valid], yerr=err[valid],
                    fmt="o-", color=cmap[ni], label=f"n = {n_val}",
                    markersize=5, capsize=3, linewidth=1.5, alpha=0.85)

    ax.set_xlabel("|η|")
    ax.set_ylabel("Window failure rate (true hits outside window)")
    ax.set_title("Window failure rate vs |η| at different multipliers")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3.7)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/window_failure_vs_eta_errors.png", dpi=150)
    plt.close(fig)
    print("\nSaved window_failure_vs_eta_errors.png")

    # ── Plot 2: Positive fraction vs occupancy (unstratified, log) ─────
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap2 = plt.cm.viridis(np.linspace(0.2, 0.9, len(WINDOW_THRESHOLDS)))

    for wi, n_thresh in enumerate(WINDOW_THRESHOLDS):
        c = tot_pf_count[wi]
        p = tot_pf_pos[wi]
        valid = (c > 100) & (p > 0)
        x = np.arange(MAX_OCC)[valid]
        frac = p[valid].astype(float) / c[valid].astype(float)
        err = binom_err(p[valid], c[valid])

        ax.errorbar(x, frac, yerr=err,
                    fmt="o-", color=cmap2[wi], label=f"n = {n_thresh}",
                    markersize=2.5, capsize=2, linewidth=1.5, alpha=0.85)

    ax.set_xlabel("n_window (recomputed at given n)")
    ax.set_ylabel("Positive fraction")
    ax.set_title("Positive fraction vs occupancy at different window sizes (log scale)")
    ax.set_yscale("log")
    ax.set_xlim(0, 60)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/pos_fraction_vs_occ_log_errors.png", dpi=150)
    plt.close(fig)
    print("Saved pos_fraction_vs_occ_log_errors.png")

    # ── Plot 3: Positive fraction vs occupancy by |eta| (log) ──────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cmap3 = plt.cm.coolwarm(np.linspace(0.0, 1.0, N_ETA))

    for wi, n_thresh in enumerate(WINDOW_THRESHOLDS):
        ax = axes.flat[wi]
        for ei in range(N_ETA):
            c = tot_pf_eta_count[ei, wi]
            p = tot_pf_eta_pos[ei, wi]
            valid = (c > 100) & (p > 0)
            x = np.arange(MAX_OCC)[valid]
            frac = p[valid].astype(float) / c[valid].astype(float)
            err = binom_err(p[valid], c[valid])

            if len(x) > 0:
                ax.errorbar(x, frac, yerr=err,
                            fmt="o-", color=cmap3[ei], label=eta_labels[ei],
                            markersize=2, capsize=1.5, linewidth=1.2, alpha=0.8)

        ax.set_xlabel("n_window (recomputed)")
        ax.set_ylabel("Positive fraction")
        ax.set_title(f"Window multiplier n = {n_thresh}", fontsize=12)
        ax.set_xlim(0, 60)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Positive fraction vs occupancy by |η| (log scale, binomial errors)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/pos_fraction_by_eta_log_errors.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved pos_fraction_by_eta_log_errors.png")

    data_vol.commit()
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


@app.local_entrypoint()
def main():
    print(f"Generating plots with binomial errors across {len(HEALTHY_EVENTS)} events...")
    run_error_plots.remote()
    print("Done.")
