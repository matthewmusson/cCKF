"""Pareto overlay renderer with truth-selection annotation.

Reads four sweep CSVs from a specified directory and renders the Pareto curve
overlay with classical CKF tuned operating points and a pT-selection footer.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


def load(csv_dir: Path, name: str) -> pd.DataFrame:
    """Load a sweep CSV and filter out invalid points.

    Args:
        csv_dir: Directory containing the CSV files.
        name: Filename of the sweep CSV.

    Returns:
        DataFrame with efficiency > 0 and fake_rate < 1.0.
    """
    df = pd.read_csv(csv_dir / name)
    return df[(df.efficiency > 0) & (df.fake_rate < 1.0)]


def render_overlay(csv_dir: Path, output_path: Path | None = None) -> None:
    """Render the Pareto overlay with classical CKF points and footer.

    Args:
        csv_dir: Directory containing the sweep CSVs.
        output_path: Path to save the output PNG. Defaults to
            pareto_overlay_classical.png in csv_dir.
    """
    if output_path is None:
        output_path = csv_dir / "pareto_overlay_classical.png"

    series = [
        ("pareto_maj_n3_dense.csv", "n = 3", "#1a7a3a", "o"),
        ("pareto_maj_n5_dense.csv", "n = 5", "#1f6fb5", "s"),
        ("pareto_maj_dense.csv", "n = 10", "#c2452c", "^"),
        (
            "pareto_sweep_baseline.csv",
            "pre-retrain weights, n = 10 (Aug 25)",
            "#8a8a8a",
            "x",
        ),
    ]

    # Tuned classical CKF operating points (joint Optuna, balanced column, 8-event Modal eval)
    classical = [
        ("Tight", 0.895, 0.0063, "#6a3d9a", "D"),
        ("Medium", 0.939, 0.0077, "#b15928", "P"),
        ("Loose", 0.936, 0.0027, "#e31a8d", "*"),
    ]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for name, label, color, marker in series:
        df = load(csv_dir, name)
        ax.scatter(
            df.fake_rate,
            df.efficiency,
            s=34,
            alpha=0.75,
            color=color,
            marker=marker,
            label=f"{label} ({len(df)} evals)",
            edgecolors="none",
        )

    label_pos = {
        "Tight": (0.045, 0.845),
        "Medium": (0.045, 0.968),
        "Loose": (0.045, 0.905),
    }
    for label, eff, fake, color, marker in classical:
        ax.scatter(
            [fake],
            [eff],
            marker=marker,
            s=190 if marker == "*" else 110,
            color=color,
            zorder=5,
            label=f"classical CKF, {label} (tuned)",
        )
        ax.annotate(
            label,
            xy=(fake, eff),
            xytext=label_pos[label],
            fontsize=9,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.6),
        )

    ax.set_xlabel("Fake rate f_DM  [% of reconstructed tracks failing DM matching]")
    ax.set_ylabel("Efficiency ε_DM  [% of matchable particles DM-matched]")
    ax.set_title("cCKF threshold sweeps by window multiplier n vs tuned classical CKF")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlim(-0.02, 0.62)
    ax.set_ylim(0.15, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    # Add footer with truth-selection annotation
    footer = (
        "efficiency/fake at truth selection: pT > 1 GeV, |η| < 3, "
        "≥6 meas, ≥3 pixel hits, charged · post-ambiguity (ACTS greedy) "
        "· 1 event (skip 4)"
    )
    fig.text(0.99, 0.005, footer, ha="right", va="bottom", fontsize=7, color="0.35")

    # Reserve bottom margin to ensure footer doesn't collide with x label
    fig.subplots_adjust(bottom=0.12)

    fig.savefig(output_path, dpi=160)
    print(f"saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render Pareto overlay with classical CKF comparison."
    )
    parser.add_argument(
        "csv_dir",
        type=Path,
        help="Directory containing the sweep CSV files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (defaults to pareto_overlay_classical.png in csv_dir).",
    )
    args = parser.parse_args()

    render_overlay(args.csv_dir, args.out)
