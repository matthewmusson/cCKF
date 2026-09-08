"""
Window failure rate vs signed eta, stratified by:
  - Pixel vs strip sensor type
  - Pure seed (3/3 seed hits match majority) vs majority seed (2/3)

Fine eta bins (0.2 width, -4 to +4).

Usage:
    python3.12 -m modal run analyze_winfail_stratified.py
"""

import modal

app = modal.App("surp-winfail-stratified")
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

ETA_BINS = [round(-4.0 + i * 0.2, 1) for i in range(41)]  # 40 bins
N_ETA = len(ETA_BINS) - 1

FAILURE_N_VALUES = [3, 4, 5, 6, 7, 8, 9, 10]
N_FAIL = len(FAILURE_N_VALUES)

# Stratification: (sensor, seed_purity) = (2, 2) = 4 combos
# sensor: 0=strip, 1=pixel
# purity: 0=majority (2/3), 1=pure (3/3)
N_SENSOR = 2
N_PURITY = 2


def _accumulate_event(event_id: int) -> dict:
    import numpy as np
    import pyarrow.parquet as pq

    path = f"{PARQUET_DIR}/expanded_event{event_id:09d}.parquet"
    needed = [
        "cand_hit_id", "majority_undefined", "branch_majority_pid",
        "contrib_pids", "residual_l0", "residual_l1", "S00", "S11",
        "state_theta", "is_pixel",
        "seed_id", "branch_id", "step_k",
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
    theta = table.column("state_theta").to_numpy()
    seed = table.column("seed_id").to_numpy().astype(np.int64)
    branch = table.column("branch_id").to_numpy().astype(np.int64)
    step = table.column("step_k").to_numpy().astype(np.int64)

    # label via vectorized list membership
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

    # ── Compute seed purity per branch ─────────────────────────────────
    # For each (seed, branch), look at the first 3 non-hole rows by step_k.
    # Count how many have label_same=True. 3/3 = pure, else majority.
    max_step = int(step.max()) + 1 if n_rows > 0 else 1
    max_branch = int(branch.max()) + 1 if n_rows > 0 else 1
    branch_key = seed * max_branch + branch

    # We need to find the first 3 non-hole rows per branch and check label_same.
    # Strategy: sort by (branch_key, step_k), then for each branch take first 3 non-holes.
    sort_idx = np.lexsort((step, branch_key))
    sorted_bk = branch_key[sort_idx]
    sorted_nonhole = nonhole[sort_idx]
    sorted_label = label_same[sort_idx]

    # find branch boundaries in sorted array
    bk_change = np.concatenate([[True], sorted_bk[1:] != sorted_bk[:-1]])
    bk_starts = np.where(bk_change)[0]
    n_branches = len(bk_starts)
    bk_ends = np.concatenate([bk_starts[1:], [len(sorted_bk)]])
    unique_bk = sorted_bk[bk_starts]

    # for each branch, count label_same among first 3 non-hole rows
    seed_match_count = np.zeros(n_branches, dtype=np.int32)
    seed_total_count = np.zeros(n_branches, dtype=np.int32)

    for bi in range(n_branches):
        start = bk_starts[bi]
        end = bk_ends[bi]
        count = 0
        matches = 0
        for j in range(start, end):
            if sorted_nonhole[j]:
                if sorted_label[j]:
                    matches += 1
                count += 1
                if count >= 3:
                    break
        seed_match_count[bi] = matches
        seed_total_count[bi] = count

    # pure = all seed hits match (3/3), majority = less than all but >= 2/3
    branch_is_pure = seed_match_count == seed_total_count  # all match
    del sort_idx, sorted_bk, sorted_nonhole, sorted_label, bk_change

    # map branch purity back to per-row via searchsorted (unique_bk is already sorted)
    row_branch_idx = np.searchsorted(unique_bk, branch_key)
    row_is_pure = branch_is_pure[row_branch_idx]
    del branch_is_pure, unique_bk, branch_key

    # signed eta
    half_theta = np.clip(theta / 2.0, 1e-10, np.pi - 1e-10)
    eta = -np.log(np.tan(half_theta))
    eta_bin_idx = np.clip(np.digitize(eta, ETA_BINS) - 1, 0, N_ETA - 1)

    # n_needed
    valid_S = nonhole & (s00 > 0) & (s11 > 0) & np.isfinite(s00) & np.isfinite(s11)
    n_needed = np.full(n_rows, np.inf)
    n_needed[valid_S] = np.maximum(
        np.abs(r0[valid_S]) / np.sqrt(s00[valid_S]),
        np.abs(r1[valid_S]) / np.sqrt(s11[valid_S]),
    )
    del r0, r1, s00, s11

    # ── Accumulate ─────────────────────────────────────────────────────
    # shape: (N_ETA, N_FAIL, N_SENSOR, N_PURITY) for total and outside
    failure_total = np.zeros((N_ETA, N_FAIL, N_SENSOR, N_PURITY), dtype=np.int64)
    failure_outside = np.zeros((N_ETA, N_FAIL, N_SENSOR, N_PURITY), dtype=np.int64)

    true_mask = label_same & nonhole_defined & np.isfinite(eta) & np.isfinite(n_needed)
    if true_mask.any():
        true_eta = eta_bin_idx[true_mask]
        true_n_needed = n_needed[true_mask]
        true_pix = is_pix[true_mask].astype(np.int32)  # 0=strip, 1=pixel
        true_pure = row_is_pure[true_mask].astype(np.int32)  # 0=majority, 1=pure

        for ni, n_val in enumerate(FAILURE_N_VALUES):
            outside = true_n_needed > n_val
            for si in range(N_SENSOR):
                smask = true_pix == si
                for pi in range(N_PURITY):
                    pmask = true_pure == pi
                    combined = smask & pmask
                    if not combined.any():
                        continue
                    for ei in range(N_ETA):
                        emask = true_eta == ei
                        sel = combined & emask
                        failure_total[ei, ni, si, pi] += int(sel.sum())
                        failure_outside[ei, ni, si, pi] += int((sel & outside).sum())

    return {
        "failure_total": failure_total.tolist(),
        "failure_outside": failure_outside.tolist(),
        "n_rows": n_rows,
        "n_branches": n_branches,
        "n_pure": int((seed_match_count == seed_total_count).sum()),
    }


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    memory=131072,
    cpu=8,
    timeout=10800,
)
def run_stratified():
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tot = np.zeros((N_ETA, N_FAIL, N_SENSOR, N_PURITY), dtype=np.int64)
    tot_out = np.zeros((N_ETA, N_FAIL, N_SENSOR, N_PURITY), dtype=np.int64)
    total_branches = 0
    total_pure = 0

    for i, eid in enumerate(HEALTHY_EVENTS):
        print(f"  [{i+1}/{len(HEALTHY_EVENTS)}] Event {eid}...", flush=True)
        r = _accumulate_event(eid)
        tot += np.array(r["failure_total"])
        tot_out += np.array(r["failure_outside"])
        total_branches += r["n_branches"]
        total_pure += r["n_pure"]
        print(f"    {r['n_rows']:>12,} rows  {r['n_branches']:>10,} branches  "
              f"{r['n_pure']:>8,} pure", flush=True)

    print(f"\nTotal: {total_branches:,} branches, {total_pure:,} pure "
          f"({100*total_pure/total_branches:.1f}%)")

    def binom_err(k, n):
        p = np.where(n > 0, k.astype(float) / n.astype(float), 0.0)
        return np.where(n > 0, np.sqrt(p * (1 - p) / n), 0.0)

    eta_centers = [(ETA_BINS[i] + ETA_BINS[i+1]) / 2 for i in range(N_ETA)]
    x = np.array(eta_centers)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })

    sensor_names = ["Strip", "Pixel"]
    purity_names = ["Majority seed", "Pure seed"]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, N_FAIL))

    # ── Plot 1: 2x2 grid (sensor x purity), all n curves ──────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)

    for si in range(N_SENSOR):
        for pi in range(N_PURITY):
            ax = axes[si, pi]
            for ni, n_val in enumerate(FAILURE_N_VALUES):
                total_arr = tot[:, ni, si, pi].astype(float)
                outside_arr = tot_out[:, ni, si, pi].astype(float)
                valid = total_arr > 50
                rate = np.where(valid, outside_arr / total_arr, np.nan)
                err = np.where(valid,
                               binom_err(tot_out[:, ni, si, pi], tot[:, ni, si, pi]),
                               0.0)

                ax.errorbar(x[valid], rate[valid], yerr=err[valid],
                            fmt="o-", color=cmap[ni], label=f"n={n_val}",
                            markersize=3, capsize=2, linewidth=1.3, alpha=0.85)

            ax.set_title(f"{sensor_names[si]} — {purity_names[pi]}", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axvline(0, color="gray", linestyle=":", alpha=0.4)
            if si == 1:
                ax.set_xlabel("η")
            if pi == 0:
                ax.set_ylabel("Failure rate")

    axes[0, 0].legend(fontsize=7, ncol=4, loc="upper center")
    fig.suptitle("Window failure rate vs η — stratified by sensor type and seed purity",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/winfail_stratified_grid.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("\nSaved winfail_stratified_grid.png")

    # ── Plot 2: All four strata overlaid, one subplot per n ────────────
    # Pick 4 representative n values
    rep_n = [3, 5, 7, 10]
    strata_colors = ["#4C72B0", "#DD5555", "#55A868", "#C49C94"]
    strata_labels = ["Pixel pure", "Pixel majority", "Strip pure", "Strip majority"]
    strata_idx = [(1, 1), (1, 0), (0, 1), (0, 0)]  # (sensor, purity)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

    for ai, n_val in enumerate(rep_n):
        ax = axes.flat[ai]
        ni = FAILURE_N_VALUES.index(n_val)

        for ci, (si, pi) in enumerate(strata_idx):
            total_arr = tot[:, ni, si, pi].astype(float)
            outside_arr = tot_out[:, ni, si, pi].astype(float)
            valid = total_arr > 50
            rate = np.where(valid, outside_arr / total_arr, np.nan)
            err = np.where(valid,
                           binom_err(tot_out[:, ni, si, pi], tot[:, ni, si, pi]),
                           0.0)

            ax.errorbar(x[valid], rate[valid], yerr=err[valid],
                        fmt="o-", color=strata_colors[ci], label=strata_labels[ci],
                        markersize=3, capsize=2, linewidth=1.3, alpha=0.8)

        ax.set_title(f"n = {n_val}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.4)
        if ai >= 2:
            ax.set_xlabel("η")
        ax.set_ylabel("Failure rate")
        ax.legend(fontsize=8)

    fig.suptitle("Window failure rate vs η — all strata compared",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/winfail_strata_overlay.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved winfail_strata_overlay.png")

    # ── Plot 3: Unstratified high-res (pixel+strip, pure+majority) ────
    fig, ax = plt.subplots(figsize=(12, 6))
    for ni, n_val in enumerate(FAILURE_N_VALUES):
        total_arr = tot[:, ni, :, :].sum(axis=(1, 2)).astype(float)
        outside_arr = tot_out[:, ni, :, :].sum(axis=(1, 2)).astype(float)
        valid = total_arr > 50
        rate = np.where(valid, outside_arr / total_arr, np.nan)
        total_int = tot[:, ni, :, :].sum(axis=(1, 2))
        outside_int = tot_out[:, ni, :, :].sum(axis=(1, 2))
        err = np.where(valid, binom_err(outside_int, total_int), 0.0)

        ax.errorbar(x[valid], rate[valid], yerr=err[valid],
                    fmt="o-", color=cmap[ni], label=f"n = {n_val}",
                    markersize=3.5, capsize=2, linewidth=1.5, alpha=0.85)

    ax.set_xlabel("η")
    ax.set_ylabel("Window failure rate")
    ax.set_title("Window failure rate vs η (high resolution, 0.2 bins)")
    ax.legend(fontsize=9, loc="upper center", ncol=4)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(0, None)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/winfail_signed_eta_hires.png", dpi=150)
    plt.close(fig)
    print("Saved winfail_signed_eta_hires.png")

    data_vol.commit()
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


@app.local_entrypoint()
def main():
    print(f"Stratified window failure analysis across {len(HEALTHY_EVENTS)} events...")
    run_stratified.remote()
    print("Done.")
