"""Plot V^{pi-dagger} target distributions from analyze_value_targets output.

Usage
-----
    python scripts/plot_value_targets.py value_target_distributions.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})

COLORS = {"t1": "#2196F3", "t2": "#FF5722"}
PURITY_STYLE = {"all": "-", "pure": "-", "majority": "--"}


def _bin_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def _plot_vs_var(ax, data, xvar, cat, xlabel, title_suffix=""):
    edges = data[f"{cat}_{xvar}_edges"]
    mean_t1 = data[f"{cat}_{xvar}_mean_t1"]
    mean_t2 = data[f"{cat}_{xvar}_mean_t2"]
    counts = data[f"{cat}_{xvar}_counts"]
    centers = _bin_centers(edges)

    mask = counts > 100
    ax.plot(centers[mask], mean_t1[mask], color=COLORS["t1"], label="Tier 1 (V*_truth)")
    ax.plot(centers[mask], mean_t2[mask], color=COLORS["t2"], label="Tier 2 (training target)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean V*")
    title = f"V* vs {xlabel}"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.bar(centers, counts, width=np.diff(edges), alpha=0.08, color="gray")
    ax2.set_ylabel("Count", color="gray", alpha=0.5)
    ax2.tick_params(axis="y", colors="gray", labelcolor="gray")


def _plot_stratified(axes, data, xvar, xlabel):
    """2-panel: pure seeds (left) vs majority seeds (right)."""
    for ax, cat, label in zip(axes, ["pure", "majority"], ["Pure triplet (3/3)", "Majority (2/3)"]):
        _plot_vs_var(ax, data, xvar, cat, xlabel, title_suffix=label)


def main(npz_path: str) -> None:
    data = dict(np.load(npz_path, allow_pickle=True))

    # --- Figure 1: V* vs eta, stratified by seed purity ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    _plot_stratified(axes1, data, "eta", "η")
    fig1.suptitle("Value Target vs Pseudorapidity", fontsize=14, y=1.02)
    fig1.tight_layout()

    # --- Figure 2: V* vs occupancy, stratified by seed purity ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    _plot_stratified(axes2, data, "occ", "n_window (occupancy)")
    fig2.suptitle("Value Target vs Occupancy", fontsize=14, y=1.02)
    fig2.tight_layout()

    # --- Figure 3: V* vs step_k ---
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
    for cat, ls, label in [("all", "-", "All seeds"),
                            ("pure", "--", "Pure (3/3)"),
                            ("majority", ":", "Majority (2/3)")]:
        edges = data[f"{cat}_step_edges"]
        mean_t1 = data[f"{cat}_step_mean_t1"]
        mean_t2 = data[f"{cat}_step_mean_t2"]
        counts = data[f"{cat}_step_counts"]
        centers = _bin_centers(edges)
        mask = counts > 100
        ax3.plot(centers[mask], mean_t1[mask], color=COLORS["t1"], ls=ls,
                 label=f"Tier 1 — {label}")
        ax3.plot(centers[mask], mean_t2[mask], color=COLORS["t2"], ls=ls,
                 label=f"Tier 2 — {label}")
    ax3.set_xlabel("Step k (CKF layer)")
    ax3.set_ylabel("Mean V*")
    ax3.set_title("Value Target vs CKF Step")
    ax3.legend(ncol=2, fontsize=8)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()

    # --- Figure 4: Volume/layer breakdown table ---
    if "vol_layer_volume_id" in data:
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.axis("off")
        vids = data["vol_layer_volume_id"].astype(int)
        lids = data["vol_layer_layer_id"].astype(int)
        mt1 = data["vol_layer_mean_t1"]
        mt2 = data["vol_layer_mean_t2"]
        cnt = data["vol_layer_count"].astype(int)
        m_eta = data["vol_layer_mean_eta"]
        m_occ = data["vol_layer_mean_occ"]

        order = np.lexsort((lids, vids))
        rows = []
        for i in order:
            rows.append([
                f"{vids[i]}", f"{lids[i]}", f"{cnt[i]:,}",
                f"{mt1[i]:.4f}", f"{mt2[i]:.4f}", f"{mt1[i]-mt2[i]:.4f}",
                f"{m_eta[i]:.2f}", f"{m_occ[i]:.1f}",
            ])
        cols = ["Volume", "Layer", "States", "Mean T1", "Mean T2", "Gap", "Mean |η|", "Mean Occ"]
        table = ax4.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)
        ax4.set_title("V* by Volume / Layer", fontsize=14, pad=20)
        fig4.tight_layout()

    out_dir = Path(npz_path).parent
    for i, fig in enumerate([fig1, fig2, fig3] + ([fig4] if "vol_layer_volume_id" in data else []), 1):
        path = out_dir / f"value_target_fig{i}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"saved {path}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_value_targets.py <path-to-npz>")
        sys.exit(1)
    main(sys.argv[1])
