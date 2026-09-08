"""Plot value function training diagnostics: loss curves, ROC, PR, reliability.

Reads from Modal volume:
  - /data/models/value_v0/value_metrics.json  (loss history)
  - /data/models/value_v0/value_val_predictions.npz  (pred, target, aux)

aux columns: [vstar_t1, step_k, eta]

Usage
-----
    # Download from volume first:
    modal volume get surp-acts-data models/value_v0/ ./value_v0/
    python scripts/plot_value_training.py ./value_v0/
"""

from __future__ import annotations

import json
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
})


def plot_loss_curves(metrics: dict, ax: plt.Axes) -> None:
    history = metrics["history"]
    epochs = range(len(history["train_loss"]))
    ax.plot(epochs, history["train_loss"], label="Train BCE", color="#2196F3")
    ax.plot(epochs, history["val_loss"], label="Val BCE", color="#FF5722")
    best_epoch = np.argmin(history["val_loss"])
    ax.axvline(best_epoch, color="gray", ls="--", alpha=0.5, label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title(f"Training Loss — val BCE {metrics['val_bce']:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_roc(pred: np.ndarray, target_binary: np.ndarray, ax: plt.Axes) -> None:
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(target_binary, pred)
    auc = roc_auc_score(target_binary, pred)
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")


def plot_pr(pred: np.ndarray, target_binary: np.ndarray, ax: plt.Axes) -> None:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(target_binary, pred)
    ap = average_precision_score(target_binary, pred)
    ax.plot(rec, prec, color="#FF5722", lw=2, label=f"AP = {ap:.4f}")
    prevalence = target_binary.mean()
    ax.axhline(prevalence, color="gray", ls="--", alpha=0.5, label=f"Prevalence = {prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")


def plot_reliability(pred: np.ndarray, target: np.ndarray, ax: plt.Axes,
                     n_bins: int = 20, label: str = "", color: str = "#2196F3") -> None:
    """Reliability diagram: mean predicted vs mean actual in equal-width bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(pred, edges) - 1, 0, n_bins - 1)

    mean_pred = np.zeros(n_bins)
    mean_actual = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    np.add.at(mean_pred, idx, pred)
    np.add.at(mean_actual, idx, target)
    np.add.at(counts, idx, 1)

    mask = counts > 50
    mean_pred[mask] /= counts[mask]
    mean_actual[mask] /= counts[mask]

    ax.plot(mean_pred[mask], mean_actual[mask], "o-", color=color, ms=5, lw=1.5, label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")

    ax2 = ax.twinx()
    ax2.bar(centers, counts, width=np.diff(edges), alpha=0.08, color="gray")
    ax2.set_ylabel("Count", color="gray", alpha=0.5)
    ax2.tick_params(axis="y", colors="gray", labelcolor="gray")


def main(model_dir: str) -> None:
    d = Path(model_dir)
    metrics = json.loads((d / "value_metrics.json").read_text())
    preds = np.load(d / "value_val_predictions.npz")

    pred = preds["pred"]       # sigmoid output ∈ [0, 1]
    target_t2 = preds["target"]  # vstar_t2 (training target)
    aux = preds["aux"]         # [vstar_t1, step_k, eta]
    target_t1 = aux[:, 0]

    # Binary label for ROC/PR: V* >= 0.5 ↔ DM-matchable
    y_binary = (target_t2 >= 0.5).astype(np.uint8)

    # --- Figure 1: Loss curves + ROC + PR ---
    fig1, (ax_loss, ax_roc, ax_pr) = plt.subplots(1, 3, figsize=(18, 5))
    plot_loss_curves(metrics, ax_loss)
    plot_roc(pred, y_binary, ax_roc)
    plot_pr(pred, y_binary, ax_pr)
    fig1.suptitle("V_φ Training Diagnostics", fontsize=14, y=1.02)
    fig1.tight_layout()

    # --- Figure 2: Reliability vs Tier 2 (training target) ---
    fig2, ax_rel2 = plt.subplots(figsize=(7, 6))
    plot_reliability(pred, target_t2, ax_rel2,
                     label="V_φ vs Tier 2 (training target)", color="#2196F3")
    ax_rel2.set_xlabel("Predicted V")
    ax_rel2.set_ylabel("Actual V* (Tier 2)")
    ax_rel2.set_title("Reliability — V_φ vs Tier 2 Target")
    ax_rel2.legend(loc="upper left")
    ax_rel2.set_xlim(0, 1)
    ax_rel2.set_ylim(0, 1.05)
    ax_rel2.grid(True, alpha=0.3)
    fig2.tight_layout()

    # --- Figure 3: Reliability vs Tier 1 ---
    fig3, ax_rel1 = plt.subplots(figsize=(7, 6))
    plot_reliability(pred, target_t1, ax_rel1,
                     label="V_φ vs Tier 1 (V*_truth)", color="#FF5722")
    ax_rel1.set_xlabel("Predicted V")
    ax_rel1.set_ylabel("Actual V* (Tier 1)")
    ax_rel1.set_title("Reliability — V_φ vs Tier 1 Target")
    ax_rel1.legend(loc="upper left")
    ax_rel1.set_xlim(0, 1)
    ax_rel1.set_ylim(0, 1.05)
    ax_rel1.grid(True, alpha=0.3)
    fig3.tight_layout()

    # Print summary
    print(f"Val BCE:   {metrics['val_bce']:.4f}")
    print(f"Val MSE:   {metrics['val_mse']:.4f}")
    print(f"AUC-ROC:   {metrics.get('auc_roc', 'N/A')}")
    print(f"Tier gap:  {metrics['tier1_minus_tier2_mean']:.4f}")
    print(f"Stopped:   epoch {metrics['stopped_epoch']}")
    if metrics.get("red_flags"):
        for w in metrics["red_flags"]:
            print(f"RED FLAG:  {w}")

    for i, fig in enumerate([fig1, fig2, fig3], 1):
        path = d / f"value_training_fig{i}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"saved {path}")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_value_training.py <model-dir>")
        sys.exit(1)
    main(sys.argv[1])
