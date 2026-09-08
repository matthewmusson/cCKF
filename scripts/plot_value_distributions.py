"""Plot V* target distributions from analysis npz.

Reads the binned statistics produced by analyze_value_targets.py and creates:
  1. V* vs eta (tier 1 + tier 2, with count histogram)
  2. V* vs occupancy (n_window)
  3. V* vs step_k
  4. Volume/layer heatmap table
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
})

C_T1 = "#2563EB"
C_T2 = "#DC2626"
C_COUNT = "#94A3B8"


def _centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def plot_vs_var(edges, mean_t1, mean_t2, counts, xlabel, title, ax, ax2=None):
    c = _centers(edges)
    mask = counts > 50

    if ax2 is None:
        ax2 = ax.twinx()

    ax2.bar(c, counts, width=np.diff(edges), alpha=0.08, color=C_COUNT,
            label="Count", zorder=1)
    ax2.set_ylabel("Count", color=C_COUNT, alpha=0.6)
    ax2.tick_params(axis="y", colors=C_COUNT, labelcolor=C_COUNT)

    ax.plot(c[mask], mean_t1[mask], "-", color=C_T1, lw=1.8,
            label="Tier 1 (all simhits)", zorder=3)
    ax.plot(c[mask], mean_t2[mask], "-", color=C_T2, lw=1.8,
            label="Tier 2 (visited surfaces)", zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean V*")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)


def main(npz_path: str, out_dir: str = ".") -> None:
    d = np.load(npz_path, allow_pickle=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: V* vs eta ---
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    plot_vs_var(
        d["all_eta_edges"], d["all_eta_mean_t1"], d["all_eta_mean_t2"],
        d["all_eta_counts"], r"$\eta$",
        r"Mean $V^{\pi\dagger}$ vs $\eta$  (32 events, 27M states)",
        ax1,
    )
    ax1.set_xlim(-4, 4)
    fig1.tight_layout()
    fig1.savefig(out / "vstar_vs_eta.png", bbox_inches="tight", dpi=150)
    print(f"saved {out / 'vstar_vs_eta.png'}")

    # --- Figure 2: V* vs occupancy ---
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    plot_vs_var(
        d["all_occ_edges"].astype(float), d["all_occ_mean_t1"], d["all_occ_mean_t2"],
        d["all_occ_counts"], r"$n_{\mathrm{window}}$ (occupancy)",
        r"Mean $V^{\pi\dagger}$ vs Occupancy  (32 events)",
        ax2,
    )
    ax2.set_xlim(-1, 60)
    fig2.tight_layout()
    fig2.savefig(out / "vstar_vs_occ.png", bbox_inches="tight", dpi=150)
    print(f"saved {out / 'vstar_vs_occ.png'}")

    # --- Figure 3: V* vs step_k ---
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    plot_vs_var(
        d["all_step_edges"], d["all_step_mean_t1"], d["all_step_mean_t2"],
        d["all_step_counts"],
        r"CKF step $k$",
        r"Mean $V^{\pi\dagger}$ vs CKF Step  (32 events)",
        ax3,
    )
    ax3.set_xlim(-1, 36)
    fig3.tight_layout()
    fig3.savefig(out / "vstar_vs_step.png", bbox_inches="tight", dpi=150)
    print(f"saved {out / 'vstar_vs_step.png'}")

    # --- Figure 4: Volume/layer table ---
    vol_ids = d["vol_layer_volume_id"].astype(int)
    lay_ids = d["vol_layer_layer_id"].astype(int)
    mt1 = d["vol_layer_mean_t1"]
    mt2 = d["vol_layer_mean_t2"]
    counts = d["vol_layer_count"]
    mean_eta = d["vol_layer_mean_eta"]
    mean_occ = d["vol_layer_mean_occ"]

    vol_names = {
        16: "NegEndcapPixel", 17: "BarrelPixel", 18: "PosEndcapPixel",
        20: "BeamPipe", 23: "NegEndcapStrip", 24: "BarrelStrip",
        25: "PosEndcapStrip", 28: "NegEndcapLStrip", 29: "BarrelLStrip",
        30: "PosEndcapLStrip", 31: "PSHT",
    }

    order = np.argsort(mean_eta)
    fig4, ax4 = plt.subplots(figsize=(16, max(6, len(vol_ids) * 0.28)))
    ax4.axis("off")

    headers = ["Volume", "Layer", "Mean V*_T1", "Mean V*_T2", "Gap",
               "Count", "Mean eta", "Mean Occ"]
    col_widths = [0.18, 0.07, 0.1, 0.1, 0.1, 0.12, 0.1, 0.1]
    col_x = [sum(col_widths[:i]) for i in range(len(col_widths))]

    y = 0.97
    dy = 0.025
    for j, h in enumerate(headers):
        ax4.text(col_x[j] + col_widths[j] / 2, y, h,
                 ha="center", va="center", fontweight="bold", fontsize=9,
                 transform=ax4.transAxes)
    y -= dy * 1.5

    for i in order:
        vname = vol_names.get(vol_ids[i], f"Vol{vol_ids[i]}")
        gap = mt1[i] - mt2[i]
        row = [
            vname, str(lay_ids[i]),
            f"{mt1[i]:.3f}", f"{mt2[i]:.3f}", f"{gap:.3f}",
            f"{int(counts[i]):,}", f"{mean_eta[i]:.2f}", f"{mean_occ[i]:.1f}",
        ]
        t2_color = C_T2 if mt2[i] < 0.05 else ("#b45309" if mt2[i] < 0.1 else "#000")
        colors = ["#000", "#000", C_T1, t2_color, "#000", C_COUNT, "#000", "#000"]
        for j, (txt, col) in enumerate(zip(row, colors)):
            ax4.text(col_x[j] + col_widths[j] / 2, y, txt,
                     ha="center", va="center", fontsize=8, color=col,
                     fontfamily="monospace", transform=ax4.transAxes)
        y -= dy

    ax4.set_title("V* by Volume / Layer (sorted by mean eta)", fontsize=13, pad=20)
    fig4.tight_layout()
    fig4.savefig(out / "vstar_vol_layer.png", bbox_inches="tight", dpi=150)
    print(f"saved {out / 'vstar_vol_layer.png'}")

    # --- Figure 5: V* vs eta, stratified by seed purity ---
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    all_eta_edges = d["all_eta_edges"]
    all_eta_counts = d["all_eta_counts"]
    for ax, tier_key, tier_label, color in [
        (ax5a, "mean_t1", "Tier 1", C_T1),
        (ax5b, "mean_t2", "Tier 2", C_T2),
    ]:
        ax2 = ax.twinx()
        ax2.bar(_centers(all_eta_edges), all_eta_counts,
                width=np.diff(all_eta_edges), alpha=0.08, color=C_COUNT,
                label="Count", zorder=1)
        ax2.set_ylabel("Count", color=C_COUNT, alpha=0.6)
        ax2.tick_params(axis="y", colors=C_COUNT, labelcolor=C_COUNT)
        for cat, ls, lbl in [("pure", "-", "Pure (3/3)"), ("majority", "--", "Majority (2/3)")]:
            edges = d[f"{cat}_eta_edges"]
            c = _centers(edges)
            vals = d[f"{cat}_eta_{tier_key}"]
            counts = d[f"{cat}_eta_counts"]
            mask = counts > 50
            ax.plot(c[mask], vals[mask], ls, color=color, lw=1.8 if cat == "pure" else 1.2,
                    alpha=1.0 if cat == "pure" else 0.6, label=lbl, zorder=3)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(f"Mean V* ({tier_label})")
        ax.set_title(f"{tier_label} by Seed Purity")
        ax.legend()
        ax.set_xlim(-4, 4)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.2, zorder=0)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
    fig5.suptitle(r"$V^{\pi\dagger}$ vs $\eta$ — Pure vs Majority Seeds", fontsize=14, y=1.02)
    fig5.tight_layout()
    fig5.savefig(out / "vstar_vs_eta_purity.png", bbox_inches="tight", dpi=150)
    print(f"saved {out / 'vstar_vs_eta_purity.png'}")

    # --- Figure 6: V* vs occupancy, stratified by seed purity ---
    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    all_occ_edges = d["all_occ_edges"].astype(float)
    all_occ_counts = d["all_occ_counts"]
    for ax, tier_key, tier_label, color in [
        (ax6a, "mean_t1", "Tier 1", C_T1),
        (ax6b, "mean_t2", "Tier 2", C_T2),
    ]:
        ax2 = ax.twinx()
        ax2.bar(_centers(all_occ_edges), all_occ_counts,
                width=np.diff(all_occ_edges), alpha=0.08, color=C_COUNT,
                label="Count", zorder=1)
        ax2.set_ylabel("Count", color=C_COUNT, alpha=0.6)
        ax2.tick_params(axis="y", colors=C_COUNT, labelcolor=C_COUNT)
        for cat, ls, lbl in [("pure", "-", "Pure (3/3)"), ("majority", "--", "Majority (2/3)")]:
            edges = d[f"{cat}_occ_edges"].astype(float)
            c = _centers(edges)
            vals = d[f"{cat}_occ_{tier_key}"]
            counts = d[f"{cat}_occ_counts"]
            mask = counts > 50
            ax.plot(c[mask], vals[mask], ls, color=color, lw=1.8 if cat == "pure" else 1.2,
                    alpha=1.0 if cat == "pure" else 0.6, label=lbl, zorder=3)
        ax.set_xlabel(r"$n_{\mathrm{window}}$")
        ax.set_ylabel(f"Mean V* ({tier_label})")
        ax.set_title(f"{tier_label} by Seed Purity")
        ax.legend()
        ax.set_xlim(-1, 60)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.2, zorder=0)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
    fig6.suptitle(r"$V^{\pi\dagger}$ vs Occupancy — Pure vs Majority Seeds", fontsize=14, y=1.02)
    fig6.tight_layout()
    fig6.savefig(out / "vstar_vs_occ_purity.png", bbox_inches="tight", dpi=150)
    print(f"saved {out / 'vstar_vs_occ_purity.png'}")

    # --- Figure 7: V* vs step_k, stratified by seed purity ---
    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    all_step_edges = d["all_step_edges"]
    all_step_counts = d["all_step_counts"]
    for ax, tier_key, tier_label, color in [
        (ax7a, "mean_t1", "Tier 1", C_T1),
        (ax7b, "mean_t2", "Tier 2", C_T2),
    ]:
        ax2 = ax.twinx()
        ax2.bar(_centers(all_step_edges), all_step_counts,
                width=np.diff(all_step_edges), alpha=0.08, color=C_COUNT,
                label="Count", zorder=1)
        ax2.set_ylabel("Count", color=C_COUNT, alpha=0.6)
        ax2.tick_params(axis="y", colors=C_COUNT, labelcolor=C_COUNT)
        for cat, ls, lbl in [("pure", "-", "Pure (3/3)"), ("majority", "--", "Majority (2/3)")]:
            edges = d[f"{cat}_step_edges"]
            c = _centers(edges)
            vals = d[f"{cat}_step_{tier_key}"]
            counts = d[f"{cat}_step_counts"]
            mask = counts > 50
            ax.plot(c[mask], vals[mask], ls, color=color, lw=1.8 if cat == "pure" else 1.2,
                    alpha=1.0 if cat == "pure" else 0.6, label=lbl, zorder=3)
        ax.set_xlabel(r"CKF step $k$")
        ax.set_ylabel(f"Mean V* ({tier_label})")
        ax.set_title(f"{tier_label} by Seed Purity")
        ax.legend()
        ax.set_xlim(-1, 36)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.2, zorder=0)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
    fig7.suptitle(r"$V^{\pi\dagger}$ vs CKF Step — Pure vs Majority Seeds", fontsize=14, y=1.02)
    fig7.tight_layout()
    fig7.savefig(out / "vstar_vs_step_purity.png", bbox_inches="tight", dpi=150)
    print(f"saved {out / 'vstar_vs_step_purity.png'}")

    # Print summary stats
    total = d["all_eta_counts"].sum()
    n_pure = d["pure_eta_counts"].sum()
    n_maj = d["majority_eta_counts"].sum()
    print(f"\nTotal states: {total:,}")
    print(f"  Pure (3/3): {n_pure:,} ({100*n_pure/total:.1f}%)")
    print(f"  Majority (2/3): {n_maj:,} ({100*n_maj/total:.1f}%)")
    print(f"Overall mean V*_T1: {np.nanmean(d['all_eta_mean_t1'][d['all_eta_counts'] > 0]):.4f}")
    print(f"Overall mean V*_T2: {np.nanmean(d['all_eta_mean_t2'][d['all_eta_counts'] > 0]):.4f}")

    step_c = d["all_step_counts"]
    step_t2 = d["all_step_mean_t2"]
    print(f"\nStep_k with max mean V*_T2: k={np.nanargmax(step_t2)} (V*={np.nanmax(step_t2):.4f})")
    print(f"Step_k with max count: k={np.argmax(step_c)} (n={step_c.max():,})")
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_value_distributions.py <npz-path> [out-dir]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
