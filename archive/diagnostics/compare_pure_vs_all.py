"""Compare pure-seed vs all-seed gate and value function models.

We train the gate g_ψ and value function V_φ on two label sources: "all seeds"
(existing pipeline, seeds may include a hit from a non-majority particle) and
"pure seeds" (new — seed uses exactly 3/3 hits from the majority particle).
This script loads both models' saved metrics / validation predictions and
produces side-by-side comparison plots + a metrics summary, to answer whether
pure-seed training measurably helps calibration and predictive quality.

Inputs (per model dir, already downloaded from the Modal volume):
  gate dir:  gate_model.pt, gate_metrics.json
  value dir: value_metrics.json, value_val_predictions.npz
    - value_val_predictions.npz arrays: pred (float32), target (float32),
      aux (float32, shape (N, 3) — columns [vstar_t1, step_k, eta])

Usage
-----
    python scripts/compare_pure_vs_all.py \\
        --all-gate-dir /data/models/gate_A \\
        --pure-gate-dir /data/models/gate_pure_A \\
        --all-value-dir /data/models/value_v0 \\
        --pure-value-dir /data/models/value_pure_v0 \\
        --out-dir analysis/plots/pure_comparison

Either the gate pair or the value pair (or both) may be omitted; the
corresponding comparison is skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

C_ALL = "#2196F3"
C_PURE = "#FF5722"
C_COUNT = "#94A3B8"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def print_table(
    title: str, rows: dict[str, tuple[float | None, float | None]]
) -> None:
    """Print a metric x {all-seed, pure-seed, delta} table to stdout."""
    print(f"\n{title}")
    print(f"{'metric':<12}{'all-seed':>14}{'pure-seed':>14}{'delta':>14}")
    print("-" * 54)
    for name, (a, p) in rows.items():
        if a is None or p is None:
            print(f"{name:<12}{'N/A':>14}{'N/A':>14}{'N/A':>14}")
            continue
        print(f"{name:<12}{a:>14.4f}{p:>14.4f}{p - a:>+14.4f}")


def rows_to_json(rows: dict[str, tuple[float | None, float | None]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (a, p) in rows.items():
        out[name] = {
            "all_seed": a,
            "pure_seed": p,
            "delta_pure_minus_all": (p - a) if (a is not None and p is not None) else None,
        }
    return out


# --------------------------------------------------------------------------
# Gate comparison
# --------------------------------------------------------------------------


def run_gate_comparison(all_dir: Path, pure_dir: Path, out_dir: Path) -> dict[str, Any]:
    all_m = load_json(all_dir / "gate_metrics.json")
    pure_m = load_json(pure_dir / "gate_metrics.json")

    def ap(m: dict[str, Any]) -> float | None:
        # Existing gate training writes "auc_pr"; the task spec calls it "ap" —
        # accept either key so this script works against either naming.
        return m.get("auc_pr", m.get("ap"))

    rows: dict[str, tuple[float | None, float | None]] = {
        "val_bce": (all_m.get("val_bce"), pure_m.get("val_bce")),
        "auc_roc": (all_m.get("auc_roc"), pure_m.get("auc_roc")),
        "ap": (ap(all_m), ap(pure_m)),
    }
    print_table("Gate g_ψ: all-seed vs pure-seed", rows)

    table = rows_to_json(rows)
    metrics_path = out_dir / "gate_comparison_metrics.json"
    metrics_path.write_text(json.dumps(table, indent=2))
    print(f"saved {metrics_path}")

    # Bonus visual: validation-loss curves side by side. Both metrics files
    # carry the same trainer's `history` dict, so no extra data is needed.
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, color, label in [(all_m, C_ALL, "all-seed"), (pure_m, C_PURE, "pure-seed")]:
        val_loss = m.get("history", {}).get("val_loss")
        if val_loss:
            ax.plot(range(len(val_loss)), val_loss, color=color, lw=1.8, label=f"{label} (val BCE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Gate g_ψ Validation Loss — All-seed vs Pure-seed")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "gate_val_loss_comparison.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {fig_path}")

    return table


# --------------------------------------------------------------------------
# Value comparison
# --------------------------------------------------------------------------


def load_value_predictions(model_dir: Path) -> dict[str, np.ndarray]:
    npz = np.load(model_dir / "value_val_predictions.npz")
    aux = npz["aux"]
    return {
        "pred": npz["pred"],
        "target": npz["target"],
        "vstar_t1": aux[:, 0],
        "step_k": aux[:, 1],
        "eta": aux[:, 2],
    }


def _centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def bin_stats(
    x: np.ndarray, y: np.ndarray, edges: np.ndarray, min_count: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of `y` and count of rows per bin of `x`, using the given `edges`.

    Bins with fewer than `min_count` rows are set to NaN in the mean so that
    sparse tails do not draw noisy line segments (matches the convention used
    in plot_value_training.py / plot_value_distributions.py).
    """
    n_bins = len(edges) - 1
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)

    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    np.add.at(sums, idx, y)
    np.add.at(counts, idx, 1)

    means = np.full(n_bins, np.nan)
    mask = counts >= min_count
    means[mask] = sums[mask] / counts[mask]
    return means, counts


def plot_vs_var_compare(
    edges: np.ndarray,
    mean_all: np.ndarray,
    mean_pure: np.ndarray,
    counts_all: np.ndarray,
    xlabel: str,
    title: str,
    ax: plt.Axes,
) -> None:
    """All-seed vs pure-seed mean-prediction curves, with a count histogram."""
    c = _centers(edges)
    mask_all = ~np.isnan(mean_all)
    mask_pure = ~np.isnan(mean_pure)

    ax2 = ax.twinx()
    ax2.bar(c, counts_all, width=np.diff(edges), alpha=0.08, color=C_COUNT, zorder=1)
    ax2.set_ylabel("Count (all-seed val set)", color=C_COUNT, alpha=0.6)
    ax2.tick_params(axis="y", colors=C_COUNT, labelcolor=C_COUNT)

    ax.plot(c[mask_all], mean_all[mask_all], "-", color=C_ALL, lw=1.8, label="All-seed", zorder=3)
    ax.plot(c[mask_pure], mean_pure[mask_pure], "-", color=C_PURE, lw=1.8, label="Pure-seed", zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Mean predicted $V_\varphi$")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)


def plot_reliability_compare(
    ax: plt.Axes,
    pred_all: np.ndarray,
    target_all: np.ndarray,
    pred_pure: np.ndarray,
    target_pure: np.ndarray,
    n_bins: int = 20,
) -> None:
    """Reliability diagram: predicted V_φ vs observed V*, both models overlaid."""
    edges = np.linspace(0, 1, n_bins + 1)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    for pred, target, color, label in [
        (pred_all, target_all, C_ALL, "All-seed"),
        (pred_pure, target_pure, C_PURE, "Pure-seed"),
    ]:
        mean_pred, counts = bin_stats(pred, pred, edges)
        mean_actual, _ = bin_stats(pred, target, edges)
        mask = ~np.isnan(mean_pred)
        ax.plot(
            mean_pred[mask], mean_actual[mask], "o-", color=color, ms=5, lw=1.5,
            label=f"{label} (n={int(counts.sum()):,})",
        )


def run_value_comparison(all_dir: Path, pure_dir: Path, out_dir: Path) -> dict[str, Any]:
    all_m = load_json(all_dir / "value_metrics.json")
    pure_m = load_json(pure_dir / "value_metrics.json")

    rows: dict[str, tuple[float | None, float | None]] = {
        "val_bce": (all_m.get("val_bce"), pure_m.get("val_bce")),
        "val_mse": (all_m.get("val_mse"), pure_m.get("val_mse")),
        "auc_roc": (all_m.get("auc_roc"), pure_m.get("auc_roc")),
    }
    print_table("Value V_φ: all-seed vs pure-seed", rows)

    table = rows_to_json(rows)
    metrics_path = out_dir / "value_comparison_metrics.json"
    metrics_path.write_text(json.dumps(table, indent=2))
    print(f"saved {metrics_path}")

    all_data = load_value_predictions(all_dir)
    pure_data = load_value_predictions(pure_dir)

    # --- V_φ prediction vs eta ---
    eta_edges = np.linspace(-4, 4, 41)
    mean_all, counts_all = bin_stats(all_data["eta"], all_data["pred"], eta_edges)
    mean_pure, _ = bin_stats(pure_data["eta"], pure_data["pred"], eta_edges)
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_vs_var_compare(
        eta_edges, mean_all, mean_pure, counts_all,
        r"$\eta$", r"Predicted $V_\varphi$ vs $\eta$ — All-seed vs Pure-seed", ax,
    )
    ax.set_xlim(-4, 4)
    fig.tight_layout()
    fig_path = out_dir / "value_pred_vs_eta.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {fig_path}")

    # --- V_φ prediction vs step_k ---
    max_step = int(max(all_data["step_k"].max(), pure_data["step_k"].max()))
    step_edges = np.arange(0, max_step + 3) - 0.5
    mean_all_s, counts_all_s = bin_stats(all_data["step_k"], all_data["pred"], step_edges)
    mean_pure_s, _ = bin_stats(pure_data["step_k"], pure_data["pred"], step_edges)
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_vs_var_compare(
        step_edges, mean_all_s, mean_pure_s, counts_all_s,
        "CKF step $k$", r"Predicted $V_\varphi$ vs CKF Step — All-seed vs Pure-seed", ax,
    )
    ax.set_xlim(-1, max_step + 1)
    fig.tight_layout()
    fig_path = out_dir / "value_pred_vs_step.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {fig_path}")

    # --- Reliability diagram: predicted V_φ vs observed V*, 20 bins ---
    fig, ax = plt.subplots(figsize=(7, 6.5))
    plot_reliability_compare(
        ax, all_data["pred"], all_data["target"], pure_data["pred"], pure_data["target"],
        n_bins=20,
    )
    ax.set_xlabel(r"Predicted $V_\varphi$")
    ax.set_ylabel(r"Observed $V^{\pi\dagger}$ (mean in bin)")
    ax.set_title("Value Function Reliability — All-seed vs Pure-seed")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "value_reliability_comparison.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {fig_path}")

    return table


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-gate-dir", type=Path, default=None)
    parser.add_argument("--pure-gate-dir", type=Path, default=None)
    parser.add_argument("--all-value-dir", type=Path, default=None)
    parser.add_argument("--pure-value-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    do_gate = args.all_gate_dir is not None and args.pure_gate_dir is not None
    do_value = args.all_value_dir is not None and args.pure_value_dir is not None

    if args.all_gate_dir is not None and args.pure_gate_dir is None:
        print("warning: --all-gate-dir given without --pure-gate-dir; skipping gate comparison")
    if args.pure_gate_dir is not None and args.all_gate_dir is None:
        print("warning: --pure-gate-dir given without --all-gate-dir; skipping gate comparison")
    if args.all_value_dir is not None and args.pure_value_dir is None:
        print("warning: --all-value-dir given without --pure-value-dir; skipping value comparison")
    if args.pure_value_dir is not None and args.all_value_dir is None:
        print("warning: --pure-value-dir given without --all-value-dir; skipping value comparison")

    if not do_gate and not do_value:
        raise SystemExit(
            "Nothing to compare: provide --all-gate-dir & --pure-gate-dir, "
            "and/or --all-value-dir & --pure-value-dir."
        )

    summary: dict[str, Any] = {}
    if do_gate:
        summary["gate"] = run_gate_comparison(args.all_gate_dir, args.pure_gate_dir, args.out_dir)
    if do_value:
        summary["value"] = run_value_comparison(args.all_value_dir, args.pure_value_dir, args.out_dir)

    summary_path = args.out_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
