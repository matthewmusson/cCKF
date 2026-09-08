"""
Positive fraction vs occupancy, stratified by |eta| bin.

One subplot per window multiplier n ∈ {3, 5, 7, 10}, each with curves
colored by |eta| bin showing positive fraction vs recomputed n_window.

Usage:
    python3.12 -m modal run analyze_posfrac_eta.py
"""

import modal

app = modal.App("surp-posfrac-eta")
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


def _accumulate_event(event_id: int) -> dict:
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
    eta_bin_idx = np.clip(np.digitize(abs_eta, ETA_BINS) - 1, 0, N_ETA - 1)

    # n_needed for window recomputation
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

    # result: for each (eta_bin, window_threshold),
    # count and pos arrays of length MAX_OCC
    # shape: (N_ETA, len(WINDOW_THRESHOLDS), MAX_OCC) for count and pos
    counts = np.zeros((N_ETA, len(WINDOW_THRESHOLDS), MAX_OCC), dtype=np.int64)
    positives = np.zeros((N_ETA, len(WINDOW_THRESHOLDS), MAX_OCC), dtype=np.int64)

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
        # for each group, take the eta bin of the first element (all same within group)
        group_eta = se[boundaries[:-1]]
        group_pos = np.add.reduceat(sl.astype(np.int64), boundaries[:-1])

        clipped = np.clip(group_sizes, 0, MAX_OCC - 1)

        for ei in range(N_ETA):
            emask = group_eta == ei
            if not emask.any():
                continue
            np.add.at(counts[ei, wi], clipped[emask], group_sizes[emask])
            np.add.at(positives[ei, wi], clipped[emask], group_pos[emask])

        del filt_keys, sort_idx, sk, sl, se, boundaries, group_sizes

    return {
        "counts": counts.tolist(),
        "positives": positives.tolist(),
        "n_rows": n_rows,
    }


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    memory=65536,
    cpu=8,
    timeout=7200,
)
def run_posfrac_eta():
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_counts = np.zeros((N_ETA, len(WINDOW_THRESHOLDS), MAX_OCC), dtype=np.int64)
    total_pos = np.zeros((N_ETA, len(WINDOW_THRESHOLDS), MAX_OCC), dtype=np.int64)

    for i, eid in enumerate(HEALTHY_EVENTS):
        print(f"  [{i+1}/{len(HEALTHY_EVENTS)}] Event {eid}...", flush=True)
        result = _accumulate_event(eid)
        total_counts += np.array(result["counts"])
        total_pos += np.array(result["positives"])
        print(f"    {result['n_rows']:>12,} rows", flush=True)

    eta_labels = [f"|η|∈[{ETA_BINS[i]:.1f},{ETA_BINS[i+1]:.1f})"
                  for i in range(N_ETA)]

    # ── Plot: 2x2 grid, one subplot per window threshold ───────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cmap = plt.cm.coolwarm(np.linspace(0.0, 1.0, N_ETA))

    for wi, n_thresh in enumerate(WINDOW_THRESHOLDS):
        ax = axes.flat[wi]
        for ei in range(N_ETA):
            c = total_counts[ei, wi]
            p = total_pos[ei, wi]
            valid = c > 100
            x = np.arange(MAX_OCC)[valid]
            frac = p[valid].astype(float) / c[valid].astype(float)
            if len(x) > 0:
                ax.plot(x, frac, "o-", color=cmap[ei], label=eta_labels[ei],
                        markersize=2.5, alpha=0.85, linewidth=1.5)

        ax.set_xlabel("n_window (recomputed)")
        ax.set_ylabel("Positive fraction")
        ax.set_title(f"Window multiplier n = {n_thresh}", fontsize=12)
        ax.set_xlim(0, 60)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Positive fraction vs occupancy, stratified by |η|",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/pos_fraction_by_eta.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("\nSaved pos_fraction_by_eta.png")

    # ── Also make a version with log y-axis ────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for wi, n_thresh in enumerate(WINDOW_THRESHOLDS):
        ax = axes.flat[wi]
        for ei in range(N_ETA):
            c = total_counts[ei, wi]
            p = total_pos[ei, wi]
            valid = (c > 100) & (p > 0)
            x = np.arange(MAX_OCC)[valid]
            frac = p[valid].astype(float) / c[valid].astype(float)
            if len(x) > 0:
                ax.plot(x, frac, "o-", color=cmap[ei], label=eta_labels[ei],
                        markersize=2.5, alpha=0.85, linewidth=1.5)

        ax.set_xlabel("n_window (recomputed)")
        ax.set_ylabel("Positive fraction")
        ax.set_title(f"Window multiplier n = {n_thresh}", fontsize=12)
        ax.set_xlim(0, 60)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Positive fraction vs occupancy by |η| (log scale)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/pos_fraction_by_eta_log.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved pos_fraction_by_eta_log.png")

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'Eta bin':<22} {'n':>4} {'Occ=1 pos%':>12} {'Occ=5 pos%':>12} {'Occ=10 pos%':>12} {'Occ=20 pos%':>12}")
    print("-" * 78)
    for ei in range(N_ETA):
        for wi, n_thresh in enumerate(WINDOW_THRESHOLDS):
            parts = []
            for occ in [1, 5, 10, 20]:
                c = total_counts[ei, wi, occ]
                p = total_pos[ei, wi, occ]
                if c > 0:
                    parts.append(f"{100*p/c:>10.4f}%")
                else:
                    parts.append(f"{'N/A':>11}")
            print(f"{eta_labels[ei]:<22} {n_thresh:>4} {''.join(parts)}")

    data_vol.commit()
    print(f"\nOutputs saved to {OUTPUT_DIR}/")


@app.local_entrypoint()
def main():
    print(f"Analyzing positive fraction by eta across {len(HEALTHY_EVENTS)} events...")
    run_posfrac_eta.remote()
    print("Done.")
