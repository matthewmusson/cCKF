"""Generate χ² reliability diagrams from slim chi2_calib_rows Parquet files.

Reads the existing slim schema (expand_trackstates_to_chi2_rows output).
Produces reliability diagrams (Λ_χ² vs observed true-hit fraction) stratified
by occupancy (n_in_ellipse: candidates within the χ² acceptance gate) and η.

Usage:
    python plot_reliability_diagrams.py /path/to/run_dir/ --output-dir output/plots/

    where run_dir contains event_*/chi2_calib_rows.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

# --- Constants ---
CHI2_MAX = 12.04  # Medium chi2CutOffMeasurement

# --- Style ---
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)
COLORS = plt.cm.tab10.colors
DIAGONAL_STYLE = dict(color="black", linestyle="--", linewidth=1, zorder=0)


def load_data(run_dir: Path, max_events: int = 32) -> pd.DataFrame:
    """Load and filter slim Parquets from run directory."""
    paths = sorted(run_dir.glob("event_*/chi2_calib_rows.parquet"))[:max_events]
    if not paths:
        raise FileNotFoundError(f"No chi2_calib_rows.parquet under {run_dir}")

    frames = []
    for p in paths:
        table = pq.read_table(p)
        frames.append(table.to_pandas())
    df = pd.concat(frames, ignore_index=True)

    # Filter: exclude majority_undefined rows (fake tracks with no clear owner)
    df = df[df["majority_undefined"].astype(int) == 0].copy()
    # Filter: finite chi2
    df = df[np.isfinite(df["chi2_inc"]) & (df["chi2_inc"] >= 0)].copy()

    # Derive n_in_ellipse: candidates passing the actual χ² gate per track state
    df["n_in_ellipse"] = df.groupby(["event_id", "seed_id", "step_k"])[
        "chi2_inc"
    ].transform(lambda x: (x <= CHI2_MAX).sum())

    return df


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    z = norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_lambda(chi2: np.ndarray) -> np.ndarray:
    """Λ_χ² = exp(−χ²/2), the survival function of χ²₂."""
    return np.exp(-chi2 / 2.0)


def compute_bins(
    predicted: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Quantile-binned reliability diagram data."""
    edges = np.quantile(predicted, np.linspace(0, 1, n_bins + 1))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    bin_idx = np.digitize(predicted, edges[1:-1])
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    results = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        k = int(labels[mask].sum()) if n > 0 else 0
        mean_pred = float(predicted[mask].mean()) if n > 0 else float(edges[b])
        obs_frac = k / n if n > 0 else 0.0
        ci_lo, ci_hi = wilson_ci(k, n)
        results.append(
            {
                "mean_predicted": mean_pred,
                "observed_fraction": obs_frac,
                "bin_count": n,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
            }
        )

    return {"bins": results, "edges": edges.tolist()}


def compute_ece(bins: list[dict], total_n: int) -> float:
    """Expected Calibration Error (spec §6.2)."""
    ece = 0.0
    for b in bins:
        weight = b["bin_count"] / total_n
        ece += weight * abs(b["observed_fraction"] - b["mean_predicted"])
    return ece


def plot_reliability_curve(
    bins_data: dict,
    ax: plt.Axes,
    color=None,
    label: str = None,
) -> None:
    """Plot one reliability curve on given axes."""
    bins = bins_data["bins"]
    x = [b["mean_predicted"] for b in bins]
    y = [b["observed_fraction"] for b in bins]
    ci_lo = [b["ci_lower"] for b in bins]
    ci_hi = [b["ci_upper"] for b in bins]

    c = color or COLORS[0]
    ax.plot(x, y, "o-", color=c, markersize=4, label=label)
    ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=c)


def plot_overall(df: pd.DataFrame, lam: np.ndarray, output_dir: Path) -> dict:
    """Plot 1: Overall reliability diagram."""
    labels = df["label"].to_numpy().astype(bool)
    bins_data = compute_bins(lam, labels, n_bins=15)
    ece = compute_ece(bins_data["bins"], len(df))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], **DIAGONAL_STYLE)
    plot_reliability_curve(bins_data, ax, label=f"CKF Medium (ECE = {ece:.4f})")
    ax.set_xlabel(r"Predicted probability ($\Lambda_{\chi^2}$)")
    ax.set_ylabel("Observed fraction (true hits)")
    ax.set_title(r"CKF $\chi^2$ Reliability Diagram — Medium OP")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"reliability_overall.{ext}", bbox_inches="tight")
    plt.close(fig)

    return {"overall_ece": ece, "overall_bins": bins_data}


def plot_occupancy_stratified(
    df: pd.DataFrame, lam: np.ndarray, output_dir: Path
) -> dict:
    """Plot 2: Occupancy-stratified reliability diagram (4 quartiles of n_in_ellipse)."""
    occ = df["n_in_ellipse"].to_numpy(dtype=np.float64)
    quartiles = np.quantile(occ, [0.25, 0.5, 0.75])
    edges = [occ.min(), quartiles[0], quartiles[1], quartiles[2], occ.max() + 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], **DIAGONAL_STYLE)

    stratum_results = {}
    stratum_names = ["Q1 (sparse)", "Q2", "Q3", "Q4 (dense)"]
    labels = df["label"].to_numpy().astype(bool)
    for i in range(4):
        mask = (occ >= edges[i]) & (occ < edges[i + 1])
        n_sub = int(mask.sum())
        if n_sub < 100:
            continue
        pred_sub = lam[mask]
        lab_sub = labels[mask]
        n_bins = 15 if n_sub >= 50000 else 10
        bins_data = compute_bins(pred_sub, lab_sub, n_bins=n_bins)
        ece = compute_ece(bins_data["bins"], n_sub)
        plot_reliability_curve(
            bins_data,
            ax,
            color=COLORS[i],
            label=f"{stratum_names[i]} (ECE={ece:.4f}, n={n_sub:,})",
        )
        stratum_results[stratum_names[i]] = {"ece": ece, "n": n_sub}

    ax.set_xlabel(r"Predicted probability ($\Lambda_{\chi^2}$)")
    ax.set_ylabel("Observed fraction (true hits)")
    ax.set_title(r"Reliability by Occupancy Quartile ($n_{\mathrm{in\,ellipse}}$)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"reliability_occupancy.{ext}", bbox_inches="tight")
    plt.close(fig)

    return {
        "occupancy_quartile_edges": [float(e) for e in edges],
        "strata": stratum_results,
    }


def plot_eta_stratified(df: pd.DataFrame, lam: np.ndarray, output_dir: Path) -> dict:
    """Plot 3: η-stratified reliability diagram."""
    eta = df["eta"].to_numpy()
    labels = df["label"].to_numpy().astype(bool)
    eta_defs = [
        ("Central (|η|<1)", np.abs(eta) < 1.0),
        ("Transition (1≤|η|<2)", (np.abs(eta) >= 1.0) & (np.abs(eta) < 2.0)),
        ("Forward (|η|≥2)", np.abs(eta) >= 2.0),
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], **DIAGONAL_STYLE)

    stratum_results = {}
    for i, (name, mask) in enumerate(eta_defs):
        n_sub = int(mask.sum())
        if n_sub < 100:
            continue
        pred_sub = lam[mask]
        lab_sub = labels[mask]
        n_bins = 15 if n_sub >= 50000 else 10
        bins_data = compute_bins(pred_sub, lab_sub, n_bins=n_bins)
        ece = compute_ece(bins_data["bins"], n_sub)
        plot_reliability_curve(
            bins_data,
            ax,
            color=COLORS[i],
            label=f"{name} (ECE={ece:.4f}, n={n_sub:,})",
        )
        stratum_results[name] = {"ece": ece, "n": n_sub}

    ax.set_xlabel(r"Predicted probability ($\Lambda_{\chi^2}$)")
    ax.set_ylabel("Observed fraction (true hits)")
    ax.set_title("Reliability by η Region")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"reliability_eta.{ext}", bbox_inches="tight")
    plt.close(fig)

    return {"strata": stratum_results}


def plot_combined_panel(df: pd.DataFrame, lam: np.ndarray, output_dir: Path) -> dict:
    """Plot 4: 4×3 grid (occupancy × η)."""
    occ = df["n_in_ellipse"].to_numpy(dtype=np.float64)
    eta = df["eta"].to_numpy()
    labels = df["label"].to_numpy().astype(bool)

    quartiles = np.quantile(occ, [0.25, 0.5, 0.75])
    occ_edges = [occ.min(), quartiles[0], quartiles[1], quartiles[2], occ.max() + 1]
    occ_names = ["Q1", "Q2", "Q3", "Q4"]

    eta_defs = [
        ("Central", np.abs(eta) < 1.0),
        ("Transition", (np.abs(eta) >= 1.0) & (np.abs(eta) < 2.0)),
        ("Forward", np.abs(eta) >= 2.0),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(12, 16), sharex=True, sharey=True)
    panel_info = {}

    for row in range(4):
        occ_mask = (occ >= occ_edges[row]) & (occ < occ_edges[row + 1])
        for col, (eta_name, eta_mask) in enumerate(eta_defs):
            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], **DIAGONAL_STYLE)
            combined_mask = occ_mask & eta_mask
            n_sub = int(combined_mask.sum())
            panel_key = f"{occ_names[row]}_{eta_name}"

            if n_sub < 100:
                ax.text(
                    0.5,
                    0.5,
                    f"n={n_sub}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                panel_info[panel_key] = {"n": n_sub, "ece": None}
            else:
                pred_sub = lam[combined_mask]
                lab_sub = labels[combined_mask]
                bins_data = compute_bins(pred_sub, lab_sub, n_bins=10)
                ece = compute_ece(bins_data["bins"], n_sub)
                plot_reliability_curve(bins_data, ax, color=COLORS[row])
                ax.text(
                    0.05,
                    0.95,
                    f"ECE={ece:.3f}\nn={n_sub:,}",
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                panel_info[panel_key] = {"n": n_sub, "ece": ece}

            if row == 0:
                ax.set_title(eta_name)
            if col == 0:
                ax.set_ylabel(f"{occ_names[row]}\nObserved frac.")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.2)

    fig.supxlabel(r"Predicted probability ($\Lambda_{\chi^2}$)", fontsize=12)
    fig.suptitle(r"Reliability: $n_{\mathrm{in\,ellipse}}$ × η", fontsize=14, y=0.995)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"reliability_combined_4x3.{ext}", bbox_inches="tight")
    plt.close(fig)

    return panel_info


def plot_lambda_histogram(df: pd.DataFrame, lam: np.ndarray, output_dir: Path) -> None:
    """Plot 5: Diagnostic — Λ_χ² distribution."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(lam, bins=100, log=True, color=COLORS[0], edgecolor="none", alpha=0.8)
    ax.set_xlabel(r"$\Lambda_{\chi^2} = \exp(-\chi^2/2)$")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Distribution of Predicted Probabilities (all candidates)")
    ax.axvline(
        np.exp(-CHI2_MAX / 2),
        color="red",
        linestyle="--",
        label=f"Λ at χ²_max={CHI2_MAX}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"diag_lambda_histogram.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_occupancy_histogram(df: pd.DataFrame, output_dir: Path) -> dict:
    """Plot 6: Diagnostic — n_in_ellipse distribution with quartile boundaries."""
    occ = df["n_in_ellipse"].to_numpy(dtype=np.float64)
    quartiles = np.quantile(occ, [0.25, 0.5, 0.75])

    fig, ax = plt.subplots(figsize=(7, 5))
    max_val = int(min(occ.max(), np.quantile(occ, 0.99) * 2))
    bins = np.arange(0, max_val + 2, 1)
    ax.hist(
        np.clip(occ, 0, max_val),
        bins=bins,
        log=True,
        color=COLORS[0],
        edgecolor="none",
        alpha=0.8,
    )
    for i, (q_val, q_name) in enumerate(zip(quartiles, ["25th", "50th", "75th"])):
        ax.axvline(q_val, color="red", linestyle="--", alpha=0.7)
        ax.text(
            q_val + 0.3,
            ax.get_ylim()[1] * 0.3,
            f"{q_name}={q_val:.0f}",
            rotation=90,
            fontsize=9,
            color="red",
        )
    ax.set_xlabel(r"$n_{\mathrm{in\,ellipse}}$ (candidates within $\chi^2$ gate)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(r"Occupancy Distribution ($n_{\mathrm{in\,ellipse}}$)")
    ax.grid(True, alpha=0.3)

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"diag_occupancy_histogram.{ext}", bbox_inches="tight")
    plt.close(fig)

    return {
        "quartiles": {
            f"q{int(q * 100)}": float(v) for q, v in zip([0.25, 0.5, 0.75], quartiles)
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory with event_*/chi2_calib_rows.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/plots"))
    parser.add_argument("--max-events", type=int, default=32)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.run_dir}...")
    df = load_data(args.run_dir, max_events=args.max_events)
    print(
        f"  {len(df)} rows, {df['event_id'].nunique()} events "
        f"(after majority_undefined filter)"
    )

    # Compute Λ_χ²
    lam = compute_lambda(df["chi2_inc"].to_numpy())

    print("Generating plots...")
    overall = plot_overall(df, lam, args.output_dir)
    print(f"  Overall ECE: {overall['overall_ece']:.4f}")

    occupancy = plot_occupancy_stratified(df, lam, args.output_dir)
    eta = plot_eta_stratified(df, lam, args.output_dir)
    panels = plot_combined_panel(df, lam, args.output_dir)
    plot_lambda_histogram(df, lam, args.output_dir)
    occ_hist = plot_occupancy_histogram(df, args.output_dir)

    # Worst-stratum ECE
    all_eces = [v["ece"] for v in occupancy["strata"].values()] + [
        v["ece"] for v in eta["strata"].values()
    ]
    worst_ece = max(all_eces) if all_eces else 0.0

    # Summary JSON
    summary = {
        "n_rows": len(df),
        "n_events": int(df["event_id"].nunique()),
        "config": "medium_t70",
        "chi2_max": CHI2_MAX,
        "overall_ece": overall["overall_ece"],
        "worst_stratum_ece": worst_ece,
        "true_hit_fraction": float(df["label"].mean()),
        "lambda_range": [float(lam.min()), float(lam.max())],
        "n_in_ellipse_range": [
            int(df["n_in_ellipse"].min()),
            int(df["n_in_ellipse"].max()),
        ],
        "occupancy_quartile_edges": occupancy["occupancy_quartile_edges"],
        "occupancy_strata_ece": {k: v["ece"] for k, v in occupancy["strata"].items()},
        "eta_strata_ece": {k: v["ece"] for k, v in eta["strata"].items()},
        "panel_info": panels,
        "occupancy_quartiles": occ_hist.get("quartiles", {}),
    }

    summary_path = args.output_dir.parent / "calibration_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print(f"Summary written to {summary_path}")
    print(f"  Overall ECE:       {summary['overall_ece']:.4f}")
    print(f"  Worst stratum ECE: {summary['worst_stratum_ece']:.4f}")
    print(f"  True-hit fraction: {summary['true_hit_fraction']:.4f}")
    print(f"  Λ range:           [{lam.min():.6f}, {lam.max():.6f}]")
    print(f"  Rows:              {len(df):,}")
    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
