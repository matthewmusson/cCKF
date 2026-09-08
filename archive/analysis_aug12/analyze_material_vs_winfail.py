"""
Dual-axis plot: window failure rate vs eta overlaid with
accumulated material budget (X/X₀) vs eta.

Uses pathInX0_interval from expanded parquets to compute
average total X/X₀ traversed by true tracks at each eta.

Usage:
    python3.12 -m modal run analyze_material_vs_winfail.py
"""

import modal

app = modal.App("surp-material-vs-winfail")
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

ETA_BINS = [round(-4.0 + i * 0.1, 1) for i in range(81)]
N_ETA = len(ETA_BINS) - 1

FAILURE_N_VALUES = [3, 5, 7, 10]
N_FAIL = len(FAILURE_N_VALUES)


def _accumulate_event(event_id: int) -> dict:
    import numpy as np
    import pyarrow.parquet as pq

    path = f"{PARQUET_DIR}/expanded_event{event_id:09d}.parquet"
    needed = [
        "cand_hit_id", "majority_undefined", "branch_majority_pid",
        "contrib_pids", "residual_l0", "residual_l1", "S00", "S11",
        "state_theta", "pathInX0_interval",
        "seed_id", "branch_id", "step_k",
    ]
    table = pq.read_table(path, columns=needed)
    n_rows = len(table)

    cand_hit = table.column("cand_hit_id").to_numpy()
    maj_undef = table.column("majority_undefined").to_numpy().astype(bool)
    branch_maj = table.column("branch_majority_pid").to_numpy()
    r0 = table.column("residual_l0").to_numpy()
    r1 = table.column("residual_l1").to_numpy()
    s00 = table.column("S00").to_numpy()
    s11 = table.column("S11").to_numpy()
    theta = table.column("state_theta").to_numpy()
    path_x0 = table.column("pathInX0_interval").to_numpy()
    seed = table.column("seed_id").to_numpy().astype(np.int64)
    branch = table.column("branch_id").to_numpy().astype(np.int64)
    step = table.column("step_k").to_numpy().astype(np.int64)

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

    # ── Window failure rate accumulation ──────────────────────────────
    failure_total = np.zeros((N_ETA, N_FAIL), dtype=np.int64)
    failure_outside = np.zeros((N_ETA, N_FAIL), dtype=np.int64)

    true_mask = label_same & nonhole_defined & np.isfinite(eta) & np.isfinite(n_needed)
    if true_mask.any():
        true_eta = eta_bin_idx[true_mask]
        true_n_needed = n_needed[true_mask]
        for ni, n_val in enumerate(FAILURE_N_VALUES):
            outside = true_n_needed > n_val
            for ei in range(N_ETA):
                emask = true_eta == ei
                failure_total[ei, ni] += int(emask.sum())
                failure_outside[ei, ni] += int((emask & outside).sum())

    # ── Material budget: accumulated X/X₀ per terminal branch ────────
    # For each (seed, branch), sum pathInX0_interval along the branch.
    # Then average the total X/X₀ across branches in each eta bin.
    # Use the branch's mean eta (from the branch's measurements).
    max_branch = int(branch.max()) + 1 if n_rows > 0 else 1
    branch_key = seed * max_branch + branch

    valid_x0 = np.isfinite(path_x0) & (path_x0 >= 0)
    valid_eta_mask = np.isfinite(eta)

    # Use np.unique to get branch keys and indices efficiently
    unique_bk, inverse_idx = np.unique(branch_key, return_inverse=True)
    n_branches = len(unique_bk)

    # Accumulate total X/X₀ per branch
    branch_x0_sum = np.zeros(n_branches, dtype=np.float64)
    np.add.at(branch_x0_sum, inverse_idx[valid_x0], path_x0[valid_x0])

    # Mean eta per branch (weighted by valid eta measurements)
    branch_eta_sum = np.zeros(n_branches, dtype=np.float64)
    branch_eta_count = np.zeros(n_branches, dtype=np.int64)
    valid_both = valid_eta_mask & nonhole
    np.add.at(branch_eta_sum, inverse_idx[valid_both], eta[valid_both])
    np.add.at(branch_eta_count, inverse_idx[valid_both], 1)

    has_eta = branch_eta_count > 0
    branch_mean_eta = np.where(has_eta, branch_eta_sum / branch_eta_count, np.nan)
    branch_eta_bin = np.clip(
        np.digitize(branch_mean_eta, ETA_BINS) - 1, 0, N_ETA - 1
    )

    # Only consider branches with real tracks (at least one true hit)
    branch_has_true = np.zeros(n_branches, dtype=bool)
    np.maximum.at(branch_has_true, inverse_idx[label_same], True)

    # Accumulate per eta bin: sum of X/X₀ and count of branches
    material_sum = np.zeros(N_ETA, dtype=np.float64)
    material_count = np.zeros(N_ETA, dtype=np.int64)

    valid_branches = has_eta & branch_has_true & (branch_x0_sum > 0)
    for bi in range(n_branches):
        if valid_branches[bi]:
            material_sum[branch_eta_bin[bi]] += branch_x0_sum[bi]
            material_count[branch_eta_bin[bi]] += 1

    return {
        "failure_total": failure_total.tolist(),
        "failure_outside": failure_outside.tolist(),
        "material_sum": material_sum.tolist(),
        "material_count": material_count.tolist(),
        "n_rows": n_rows,
    }


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    memory=131072,
    cpu=8,
    timeout=7200,
)
def run_material_comparison():
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tot_total = np.zeros((N_ETA, N_FAIL), dtype=np.int64)
    tot_outside = np.zeros((N_ETA, N_FAIL), dtype=np.int64)
    mat_sum = np.zeros(N_ETA, dtype=np.float64)
    mat_count = np.zeros(N_ETA, dtype=np.int64)

    for i, eid in enumerate(HEALTHY_EVENTS):
        print(f"  [{i+1}/{len(HEALTHY_EVENTS)}] Event {eid}...", flush=True)
        r = _accumulate_event(eid)
        tot_total += np.array(r["failure_total"])
        tot_outside += np.array(r["failure_outside"])
        mat_sum += np.array(r["material_sum"])
        mat_count += np.array(r["material_count"])
        print(f"    {r['n_rows']:>12,} rows", flush=True)

    # Compute averages
    eta_centers = [(ETA_BINS[i] + ETA_BINS[i+1]) / 2 for i in range(N_ETA)]
    x = np.array(eta_centers)

    avg_x0 = np.where(mat_count > 0, mat_sum / mat_count, np.nan)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

    # ── Main plot: dual y-axis ────────────────────────────────────────
    fail_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    # Material budget on left axis (filled area)
    valid_mat = mat_count > 50
    ax1.fill_between(x[valid_mat], avg_x0[valid_mat], alpha=0.15, color="#888888")
    ax1.plot(x[valid_mat], avg_x0[valid_mat], "k-", linewidth=2, alpha=0.6,
             label="Mean total X/X₀")
    ax1.set_ylabel("Mean accumulated X/X₀ per track", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Window failure rate on right axis
    def binom_err(k, n):
        p = np.where(n > 0, k.astype(float) / n.astype(float), 0.0)
        return np.where(n > 0, np.sqrt(p * (1 - p) / n), 0.0)

    for ni, n_val in enumerate(FAILURE_N_VALUES):
        total = tot_total[:, ni].astype(float)
        outside = tot_outside[:, ni].astype(float)
        valid = total > 50
        rate = np.where(valid, outside / total, np.nan)
        err = binom_err(tot_outside[:, ni], tot_total[:, ni])

        ax2.errorbar(x[valid], rate[valid], yerr=err[valid],
                     fmt="o-", color=fail_colors[ni], label=f"Failure rate n={n_val}",
                     markersize=4, capsize=2, linewidth=1.5, alpha=0.85)

    ax2.set_ylabel("Window failure rate (true hits outside window)", color="#7570b3")
    ax2.tick_params(axis="y", labelcolor="#7570b3")
    ax2.set_ylim(0, None)

    ax1.set_xlabel("η")
    ax1.set_xlim(-4.0, 4.0)
    ax1.axvline(0, color="gray", linestyle=":", alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
               loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.12))

    fig.suptitle("Window failure rate vs material budget (ODD, μ=200 ttbar)", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(f"{OUTPUT_DIR}/winfail_vs_material.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved winfail_vs_material.png")

    # ── Two-panel version ─────────────────────────────────────────────
    fig, (ax_mat, ax_fail) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                           gridspec_kw={"height_ratios": [1, 1.5]})

    # Top panel: material budget
    ax_mat.fill_between(x[valid_mat], avg_x0[valid_mat], alpha=0.2, color="#555555")
    ax_mat.plot(x[valid_mat], avg_x0[valid_mat], "k-", linewidth=2)
    ax_mat.set_ylabel("Mean total X/X₀")
    ax_mat.set_title("Accumulated material budget per track", fontsize=12)
    ax_mat.grid(True, alpha=0.3)
    ax_mat.axvline(0, color="gray", linestyle=":", alpha=0.3)

    # Bottom panel: failure rate
    for ni, n_val in enumerate(FAILURE_N_VALUES):
        total = tot_total[:, ni].astype(float)
        outside = tot_outside[:, ni].astype(float)
        valid = total > 50
        rate = np.where(valid, outside / total, np.nan)
        err = binom_err(tot_outside[:, ni], tot_total[:, ni])

        ax_fail.errorbar(x[valid], rate[valid], yerr=err[valid],
                         fmt="o-", color=fail_colors[ni], label=f"n = {n_val}",
                         markersize=4, capsize=2, linewidth=1.5, alpha=0.85)

    ax_fail.set_xlabel("η")
    ax_fail.set_ylabel("Window failure rate")
    ax_fail.set_title("True-hit window failure rate", fontsize=12)
    ax_fail.legend(fontsize=9, ncol=4, loc="upper center")
    ax_fail.grid(True, alpha=0.3)
    ax_fail.set_xlim(-4.0, 4.0)
    ax_fail.set_ylim(0, None)
    ax_fail.axvline(0, color="gray", linestyle=":", alpha=0.3)

    fig.suptitle("Material budget and window failure rate vs η (ODD, μ=200 ttbar)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/winfail_vs_material_panels.png", dpi=150)
    plt.close(fig)
    print("Saved winfail_vs_material_panels.png")

    # Print summary
    print(f"\n{'Eta bin':<16}  avg X/X₀  branches", end="")
    for n_val in FAILURE_N_VALUES:
        print(f"  fail@n={n_val}", end="")
    print()
    print("-" * 90)
    for ei in range(N_ETA):
        eta_label = f"[{ETA_BINS[ei]:.1f},{ETA_BINS[ei+1]:.1f})"
        if mat_count[ei] > 0:
            print(f"{eta_label:<16}  {avg_x0[ei]:7.3f}  {mat_count[ei]:>8}", end="")
        else:
            print(f"{eta_label:<16}     N/A       N/A", end="")
        for ni in range(N_FAIL):
            t = tot_total[ei, ni]
            o = tot_outside[ei, ni]
            if t > 0:
                print(f"     {100*o/t:5.1f}%", end="")
            else:
                print(f"       N/A", end="")
        print()

    data_vol.commit()
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


@app.local_entrypoint()
def main():
    print(f"Material budget + window failure analysis across {len(HEALTHY_EVENTS)} events...")
    run_material_comparison.remote()
    print("Done.")
