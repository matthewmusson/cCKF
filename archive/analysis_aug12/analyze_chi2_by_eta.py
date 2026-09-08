"""
Chi-squared distribution stratified by sensor type (pixel/strip) and
pseudorapidity eta bins.

Usage:
    python3.12 -m modal run analyze_chi2_by_eta.py
"""

import modal

app = modal.App("surp-chi2-by-eta")
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
N_ETA_BINS = len(ETA_BINS) - 1

CONFIGS = {
    "tight_t79":  {"chi2_max": 16.26},
    "medium_t70": {"chi2_max": 12.04},
    "fast_t331":  {"chi2_max": 15.40},
}

CHI2_BIN_EDGES_LOG = None  # computed once at runtime


def _accumulate_event(event_id: int) -> dict:
    """Return per-(eta_bin, sensor, label) chi2 histograms for one event."""
    import numpy as np
    import pyarrow.parquet as pq

    path = f"{PARQUET_DIR}/expanded_event{event_id:09d}.parquet"
    needed = [
        "cand_hit_id", "majority_undefined", "branch_majority_pid",
        "contrib_pids", "chi2_inc", "is_pixel", "state_theta",
    ]
    table = pq.read_table(path, columns=needed)
    n_rows = len(table)

    cand_hit = table.column("cand_hit_id").to_numpy()
    maj_undef = table.column("majority_undefined").to_numpy().astype(bool)
    branch_maj = table.column("branch_majority_pid").to_numpy()
    chi2 = table.column("chi2_inc").to_numpy()
    is_pix_raw = table.column("is_pixel").to_numpy()
    is_pix = np.nan_to_num(is_pix_raw, nan=0.0).astype(bool)
    theta = table.column("state_theta").to_numpy()

    # label_same_particle via vectorized list membership
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

    # eta from theta
    half_theta = np.clip(theta / 2.0, 1e-10, np.pi - 1e-10)
    eta = -np.log(np.tan(half_theta))
    abs_eta = np.abs(eta)

    # chi2 histogram bins (log-spaced)
    chi2_edges = np.logspace(-3, 3, 201)

    # valid rows for histogramming: non-hole, finite chi2 > 0
    valid = nonhole & np.isfinite(chi2) & (chi2 > 0) & np.isfinite(abs_eta)

    # pre-digitize eta bins for valid rows
    eta_bin_idx = np.digitize(abs_eta[valid], ETA_BINS) - 1  # 0-indexed
    eta_bin_idx = np.clip(eta_bin_idx, 0, N_ETA_BINS - 1)

    chi2_valid = chi2[valid]
    pix_valid = is_pix[valid]
    label_valid = label_same[valid]

    # accumulate: shape (n_eta_bins, 2 sensors, 2 labels, 200 chi2 bins)
    hists = np.zeros((N_ETA_BINS, 2, 2, 200), dtype=np.int64)

    for ei in range(N_ETA_BINS):
        eta_mask = eta_bin_idx == ei
        for si, sensor_mask in enumerate([~pix_valid, pix_valid]):  # 0=strip, 1=pixel
            for li, label_mask in enumerate([~label_valid, label_valid]):  # 0=neg, 1=pos
                combined = eta_mask & sensor_mask & label_mask
                if combined.any():
                    h, _ = np.histogram(chi2_valid[combined], bins=chi2_edges)
                    hists[ei, si, li] = h

    return {
        "hists": hists.tolist(),
        "chi2_edges": chi2_edges.tolist(),
        "n_rows": n_rows,
    }


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    memory=65536,
    cpu=8,
    timeout=7200,
)
def run_chi2_eta_analysis():
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # accumulate across all events
    total_hists = np.zeros((N_ETA_BINS, 2, 2, 200), dtype=np.int64)
    chi2_edges = None

    for i, eid in enumerate(HEALTHY_EVENTS):
        print(f"  [{i+1}/{len(HEALTHY_EVENTS)}] Event {eid}...", flush=True)
        result = _accumulate_event(eid)
        total_hists += np.array(result["hists"])
        if chi2_edges is None:
            chi2_edges = np.array(result["chi2_edges"])
        print(f"    {result['n_rows']:>12,} rows", flush=True)

    centers = np.sqrt(chi2_edges[:-1] * chi2_edges[1:])
    widths = chi2_edges[1:] - chi2_edges[:-1]

    eta_labels = [f"|η| ∈ [{ETA_BINS[i]:.1f}, {ETA_BINS[i+1]:.1f})"
                  for i in range(N_ETA_BINS)]

    # ── Plot: 2 columns (pixel/strip) x N_ETA_BINS rows ───────────────
    fig, axes = plt.subplots(N_ETA_BINS, 2, figsize=(14, 3.2 * N_ETA_BINS),
                             sharex=True)

    sensor_names = ["Strip", "Pixel"]
    cfg_colors = {"tight_t79": "#2ca02c", "medium_t70": "#ff7f0e", "fast_t331": "#9467bd"}

    for ei in range(N_ETA_BINS):
        for si in range(2):
            ax = axes[ei, si]
            neg = total_hists[ei, si, 0]
            pos = total_hists[ei, si, 1]

            ax.bar(centers, neg, width=widths, alpha=0.55,
                   color="#DD5555", label="Negative" if ei == 0 else None)
            ax.bar(centers, pos, width=widths, alpha=0.55,
                   color="#4C72B0", label="Positive" if ei == 0 else None)

            for name, cfg in CONFIGS.items():
                ax.axvline(cfg["chi2_max"], color=cfg_colors[name],
                           linestyle="--", alpha=0.7, linewidth=1.5,
                           label=f"{name} ({cfg['chi2_max']})" if ei == 0 else None)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1, None)

            # row labels on left column, column labels on top row
            if si == 0:
                ax.set_ylabel(eta_labels[ei], fontsize=10)
            if ei == 0:
                ax.set_title(sensor_names[si], fontsize=13, fontweight="bold")
            if ei == N_ETA_BINS - 1:
                ax.set_xlabel(r"$\chi^2_{\mathrm{inc}}$")

            # stats annotation
            n_pos = int(pos.sum())
            n_neg = int(neg.sum())
            ratio = n_neg / n_pos if n_pos > 0 else float("inf")
            ax.text(0.97, 0.95,
                    f"pos={n_pos:,}\nneg={n_neg:,}\nratio=1:{ratio:.0f}",
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # shared legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 1].get_legend_handles_labels()
    for h, l in zip(h2, l2):
        if l not in labels:
            handles.append(h)
            labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(r"$\chi^2$ increment by sensor type and $|\eta|$",
                 fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/chi2_by_eta_sensor.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("\nSaved chi2_by_eta_sensor.png")

    # ── Print summary table ────────────────────────────────────────────
    print(f"\n{'Eta bin':<22} {'Sensor':<8} {'Positive':>12} {'Negative':>12} {'Ratio':>10}")
    print("-" * 66)
    for ei in range(N_ETA_BINS):
        for si in range(2):
            n_pos = int(total_hists[ei, si, 1].sum())
            n_neg = int(total_hists[ei, si, 0].sum())
            ratio = f"1:{n_neg/n_pos:.0f}" if n_pos > 0 else "N/A"
            print(f"{eta_labels[ei]:<22} {sensor_names[si]:<8} "
                  f"{n_pos:>12,} {n_neg:>12,} {ratio:>10}")

    data_vol.commit()
    print(f"\nOutputs saved to {OUTPUT_DIR}/")


@app.local_entrypoint()
def main():
    print(f"Analyzing chi2 by eta and sensor type across {len(HEALTHY_EVENTS)} events...")
    run_chi2_eta_analysis.remote()
    print("Done.")
