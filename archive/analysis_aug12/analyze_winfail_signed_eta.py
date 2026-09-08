"""
Window failure rate vs signed eta (full range -4 to 4).

Usage:
    python3.12 -m modal run analyze_winfail_signed_eta.py
"""

import modal

app = modal.App("surp-winfail-signed-eta")
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

ETA_BINS = [round(-4.0 + i * 0.2, 1) for i in range(41)]  # -4.0 to +4.0 in 0.2 steps
N_ETA = len(ETA_BINS) - 1

FAILURE_N_VALUES = [3, 4, 5, 6, 7, 8, 9, 10]
N_FAIL = len(FAILURE_N_VALUES)


def _accumulate_event(event_id: int) -> dict:
    import numpy as np
    import pyarrow.parquet as pq

    path = f"{PARQUET_DIR}/expanded_event{event_id:09d}.parquet"
    needed = [
        "cand_hit_id", "majority_undefined", "branch_majority_pid",
        "contrib_pids", "residual_l0", "residual_l1", "S00", "S11",
        "state_theta",
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

    # n_needed
    valid_S = nonhole & (s00 > 0) & (s11 > 0) & np.isfinite(s00) & np.isfinite(s11)
    n_needed = np.full(n_rows, np.inf)
    n_needed[valid_S] = np.maximum(
        np.abs(r0[valid_S]) / np.sqrt(s00[valid_S]),
        np.abs(r1[valid_S]) / np.sqrt(s11[valid_S]),
    )
    del r0, r1, s00, s11

    eta_bin_idx = np.clip(np.digitize(eta, ETA_BINS) - 1, 0, N_ETA - 1)

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

    return {
        "failure_total": failure_total.tolist(),
        "failure_outside": failure_outside.tolist(),
        "n_rows": n_rows,
    }


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    memory=65536,
    cpu=8,
    timeout=7200,
)
def run_signed_eta():
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tot_total = np.zeros((N_ETA, N_FAIL), dtype=np.int64)
    tot_outside = np.zeros((N_ETA, N_FAIL), dtype=np.int64)

    for i, eid in enumerate(HEALTHY_EVENTS):
        print(f"  [{i+1}/{len(HEALTHY_EVENTS)}] Event {eid}...", flush=True)
        r = _accumulate_event(eid)
        tot_total += np.array(r["failure_total"])
        tot_outside += np.array(r["failure_outside"])
        print(f"    {r['n_rows']:>12,} rows", flush=True)

    def binom_err(k, n):
        p = np.where(n > 0, k.astype(float) / n.astype(float), 0.0)
        return np.where(n > 0, np.sqrt(p * (1 - p) / n), 0.0)

    eta_centers = [(ETA_BINS[i] + ETA_BINS[i+1]) / 2 for i in range(N_ETA)]

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, N_FAIL))

    for ni, n_val in enumerate(FAILURE_N_VALUES):
        total = tot_total[:, ni].astype(float)
        outside = tot_outside[:, ni].astype(float)
        valid = total > 0
        rate = np.where(valid, outside / total, 0.0)
        err = np.where(valid, binom_err(tot_outside[:, ni], tot_total[:, ni]), 0.0)

        x = np.array(eta_centers)
        ax.errorbar(x[valid], rate[valid], yerr=err[valid],
                    fmt="o-", color=cmap[ni], label=f"n = {n_val}",
                    markersize=5, capsize=3, linewidth=1.5, alpha=0.85)

    ax.set_xlabel("η")
    ax.set_ylabel("Window failure rate (true hits outside window)")
    ax.set_title("Window failure rate vs η at different multipliers")
    ax.legend(fontsize=9, loc="upper center", ncol=4)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(0, None)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/window_failure_vs_signed_eta.png", dpi=150)
    plt.close(fig)
    print("\nSaved window_failure_vs_signed_eta.png")

    # print summary
    eta_labels = [f"[{ETA_BINS[i]:.1f},{ETA_BINS[i+1]:.1f})"
                  for i in range(N_ETA)]
    print(f"\n{'Eta bin':<16}", end="")
    for n_val in FAILURE_N_VALUES:
        print(f"  n={n_val:>2}", end="")
    print()
    print("-" * (16 + 8 * N_FAIL))
    for ei in range(N_ETA):
        print(f"{eta_labels[ei]:<16}", end="")
        for ni in range(N_FAIL):
            t = tot_total[ei, ni]
            o = tot_outside[ei, ni]
            if t > 0:
                print(f"  {100*o/t:5.1f}%", end="")
            else:
                print(f"    N/A", end="")
        print()

    data_vol.commit()
    print(f"\nOutput saved to {OUTPUT_DIR}/")


@app.local_entrypoint()
def main():
    print(f"Window failure rate vs signed eta across {len(HEALTHY_EVENTS)} events...")
    run_signed_eta.remote()
    print("Done.")
