"""Fit Platt scaling on the calibration split and run the §4.2 audit.

Produces figure G3: the χ²-implied probability Λ = exp(−χ²/2) and the
Platt-calibrated gate, on the same reliability axes, stratified. That side-by-side
is the headline claim — χ² is not a calibrated probability and is miscalibrated
*differently* across η and occupancy, while the learned gate is calibrated
uniformly.

The calibration split is used here and nowhere else. It is never trained on and
never used to pick a hyperparameter.

Usage
-----
    python scripts/calibrate_and_audit.py \\
        --model /data/models/gate_A/gate_model.pt \\
        --cal-cache /data/cache/gate/cal \\
        --out-dir /data/audit/gate_A
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cckf import cache, calibration, features, metrics, models, train

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})
COLORS = plt.cm.tab10.colors
DIAGONAL = dict(color="black", linestyle="--", linewidth=1, zorder=0)

#: spec §4.2 targets
ECE_TARGET_OVERALL = 0.02
ECE_TARGET_STRATUM = 0.05

#: One fixed log-odds-uniform x-axis for every curve in every figure. Shared so
#: that χ² and the gate are directly comparable in G3, and so that strata are
#: comparable to each other and across runs.
EDGES = metrics.logit_bin_edges(n_bins=30)


def _plot_curve(ax, data: dict, color, label: str | None) -> None:
    """Draw one reliability curve, omitting sparse bins from the line."""
    bins = [b for b in data["bins"] if b["count"] > 0 and not b["sparse"]]
    if not bins:
        return
    x = [b["mean_predicted"] for b in bins]
    y = [b["observed_fraction"] for b in bins]
    ax.plot(x, y, "o-", color=color, markersize=4, label=label)
    ax.fill_between(
        x, [b["ci_lower"] for b in bins], [b["ci_upper"] for b in bins],
        alpha=0.2, color=color,
    )
    # Sparse bins are still shown, as bare Wilson intervals with no marker, so a
    # thin region reads as "uncertain" rather than silently absent.
    thin = [b for b in data["bins"] if 0 < b["count"] and b["sparse"]]
    for b in thin:
        ax.plot(
            [b["mean_predicted"]] * 2, [b["ci_lower"], b["ci_upper"]],
            color=color, alpha=0.35, linewidth=1,
        )


def _style(ax, xlabel: str, title: str) -> None:
    """Log-scaled reliability axes.

    Both axes are log: the gate spans several orders of magnitude in probability
    and a linear axis would compress everything below p≈0.05 into the origin,
    which is precisely the region the τ sweep operates in.
    """
    lo = float(EDGES[0])
    ax.plot([lo, 1], [lo, 1], **DIAGONAL)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Observed fraction (true hits)")
    ax.set_title(title)
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(lo, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, which="both")
    # Shade the τ sweep range: the only region where calibration affects a
    # reachable operating point.
    ax.axvspan(*metrics.DECISION_REGION, color="grey", alpha=0.10, zorder=0)


def _stratified(ax, pred, y, strata: dict[str, np.ndarray], xlabel, title) -> dict:
    """Overlay one reliability curve per stratum on the shared axis."""
    out = {}
    for i, (name, mask) in enumerate(strata.items()):
        n = int(mask.sum())
        if n < 1000:
            out[name] = {"ece": None, "n": n, "skipped": "fewer than 1000 rows"}
            continue
        data = metrics.reliability_bins(pred[mask], y[mask], edges=EDGES)
        ece = metrics.expected_calibration_error(pred[mask], y[mask], edges=EDGES)
        dre = metrics.decision_region_ece(pred[mask], y[mask])
        mce = metrics.max_calibration_error(pred[mask], y[mask], edges=EDGES)
        _plot_curve(ax, data, COLORS[i % 10], f"{name} (ECE={ece:.4f}, n={n:,})")
        out[name] = {
            "ece": ece,
            "decision_region_ece": dre["ece"],
            "decision_region_n": dre["n_rows"],
            "mce": mce,
            "n": n,
        }
    _style(ax, xlabel, title)
    ax.legend(loc="lower right")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cal-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--value-predictions", default="",
                        help="optional value_val_predictions.npz for the V_φ diagonal plot")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = models.GateMLP(
        n_features=ckpt["n_features"], width=ckpt["width"], depth=ckpt["depth"]
    )
    model.load_state_dict(ckpt["state_dict"])

    cal = cache.load_cache(args.cal_cache)
    # The checkpoint's mu/sigma are already subset to the model's own columns,
    # so subset the raw cached features first, then standardise with them.
    col_idx = np.array([features.GATE_FEATURES.index(n) for n in ckpt["feature_names"]])
    X = ((np.asarray(cal["X"])[:, col_idx] - ckpt["mu"]) / ckpt["sigma"]).astype(np.float32)
    y = np.asarray(cal["y"]).astype(np.uint8)

    aux = np.asarray(cal["aux"])
    chi2, n_window, eta = aux[:, 0], aux[:, 1], aux[:, 2]

    logits = train.predict_logits(model, X, device=args.device)
    raw_pred = 1.0 / (1.0 + np.exp(-logits))

    # --- Platt fits ------------------------------------------------------
    a, b = calibration.fit_platt(logits, y)
    cal_pred = calibration.apply_platt(logits, a, b)
    occ_params = calibration.fit_platt_occupancy(logits, y, n_window)
    occ_pred = calibration.apply_platt_occupancy(logits, n_window, occ_params)

    # --- chi2 baseline ---------------------------------------------------
    lam = np.clip(np.exp(-np.where(np.isfinite(chi2), chi2, np.inf) / 2.0), 0.0, 1.0)

    eta_s = metrics.eta_strata(eta)
    occ_s = metrics.quintile_strata(n_window)

    audit = {
        "platt_2param": {"a": a, "b": b},
        "platt_4param": {
            "a0": occ_params[0], "a1": occ_params[1],
            "b0": occ_params[2], "b1": occ_params[3],
        },
        "prior_logit_shift_from_training": ckpt.get("prior_logit_shift", 0.0),
        "n_cal_rows": int(len(y)),
        "positive_fraction": float(y.mean()),
        "bin_scheme": {
            "type": "logit_uniform",
            "n_bins": len(EDGES) - 1,
            "p_range": [float(EDGES[0]), float(EDGES[-1])],
        },
        "ece": {
            "chi2_lambda": metrics.expected_calibration_error(lam, y, edges=EDGES),
            "gate_raw": metrics.expected_calibration_error(raw_pred, y, edges=EDGES),
            "gate_platt2": metrics.expected_calibration_error(cal_pred, y, edges=EDGES),
            "gate_platt4": metrics.expected_calibration_error(occ_pred, y, edges=EDGES),
        },
        # ECE is count-weighted, so at 1:193 it is dominated by the trivial
        # near-zero region. These two are the numbers that actually gate
        # deployability; read them first.
        "decision_region_ece": {
            "chi2_lambda": metrics.decision_region_ece(lam, y),
            "gate_raw": metrics.decision_region_ece(raw_pred, y),
            "gate_platt2": metrics.decision_region_ece(cal_pred, y),
            "gate_platt4": metrics.decision_region_ece(occ_pred, y),
        },
        "max_calibration_error": {
            "chi2_lambda": metrics.max_calibration_error(lam, y, edges=EDGES),
            "gate_raw": metrics.max_calibration_error(raw_pred, y, edges=EDGES),
            "gate_platt2": metrics.max_calibration_error(cal_pred, y, edges=EDGES),
            "gate_platt4": metrics.max_calibration_error(occ_pred, y, edges=EDGES),
        },
    }

    # Sanity band from spec §4.1.
    if abs(a) > 2.0:
        audit.setdefault("warnings", []).append(
            f"Platt slope a={a:.3f} is far from 1; the raw network is badly "
            f"calibrated — debug rather than patch"
        )

    # --- Plots -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for i, (pred, name) in enumerate(
        [(lam, r"$\Lambda_{\chi^2}$"), (raw_pred, "gate (raw)"), (cal_pred, "gate (Platt)")]
    ):
        ece = metrics.expected_calibration_error(pred, y, edges=EDGES)
        _plot_curve(ax, metrics.reliability_bins(pred, y, edges=EDGES), COLORS[i], f"{name}: ECE={ece:.4f}")
    _style(ax, "Predicted probability", "Gate calibration on the calibration split")
    ax.legend(loc="lower right")
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"reliability_gate.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    audit["eta_strata_platt2"] = _stratified(
        ax, cal_pred, y, eta_s, "Predicted probability (gate, Platt)", "Gate reliability by η"
    )
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"reliability_eta.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    audit["occupancy_strata_platt2"] = _stratified(
        ax, cal_pred, y, occ_s,
        "Predicted probability (gate, Platt)", "Gate reliability by $n_{window}$ quintile"
    )
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"reliability_occupancy.{ext}", bbox_inches="tight")
    plt.close(fig)

    # --- Figure G3: chi2 vs calibrated gate, side by side, stratified ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    audit["G3"] = {
        "chi2_eta": _stratified(
            axes[0, 0], lam, y, eta_s, r"$\Lambda_{\chi^2}$", r"$\chi^2$ by η"
        ),
        "gate_eta": _stratified(
            axes[0, 1], cal_pred, y, eta_s, "gate (Platt)", "Calibrated gate by η"
        ),
        "chi2_occupancy": _stratified(
            axes[1, 0], lam, y, occ_s, r"$\Lambda_{\chi^2}$", r"$\chi^2$ by occupancy"
        ),
        "gate_occupancy": _stratified(
            axes[1, 1], cal_pred, y, occ_s, "gate (Platt)", "Calibrated gate by occupancy"
        ),
    }
    fig.suptitle(
        r"G3: $\chi^2$-implied probability is miscalibrated and stratum-dependent; "
        r"the calibrated gate is not",
        fontsize=14,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"figure_G3.{ext}", bbox_inches="tight")
    plt.close(fig)

    # --- Value function: density + reliability (spec §3.8) ---------------
    # Two panels because they answer different questions. The 2D histogram shows
    # the joint density (where the states actually are, and whether the bimodal
    # structure is reproduced); the reliability curve shows the conditional mean
    # E[V^{π†} | V_φ], which is what soft BCE is minimised by and therefore the
    # actual calibration test for the value head.
    if args.value_predictions:
        vp = np.load(args.value_predictions)
        pred_v, target_v = vp["pred"], vp["target"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        h = axes[0].hist2d(
            target_v, pred_v, bins=50, range=[[0, 1], [0, 1]],
            norm=matplotlib.colors.LogNorm(),
        )
        fig.colorbar(h[3], ax=axes[0], label="states")
        axes[0].plot([0, 1], [0, 1], color="white", linestyle="--", linewidth=1)
        axes[0].set_xlabel(r"$V^{\pi^\dagger}$ (Tier 2 target)")
        axes[0].set_ylabel(r"$V_\varphi$ prediction")
        axes[0].set_title("Joint density")
        axes[0].set_aspect("equal")

        vdata = metrics.soft_reliability_bins(pred_v, target_v, n_bins=20)
        sce = metrics.soft_calibration_error(pred_v, target_v)
        drawn = [b for b in vdata["bins"] if b["count"] and not b["sparse"]]
        axes[1].plot([0, 1], [0, 1], **DIAGONAL)
        axes[1].plot(
            [b["mean_predicted"] for b in drawn],
            [b["mean_target"] for b in drawn],
            "o-", color=COLORS[0], markersize=4,
            label=rf"$V_\varphi$ (soft ECE = {sce:.4f})",
        )
        axes[1].fill_between(
            [b["mean_predicted"] for b in drawn],
            [b["ci_lower"] for b in drawn],
            [b["ci_upper"] for b in drawn],
            alpha=0.2, color=COLORS[0],
        )
        axes[1].axvline(0.5, color="red", linestyle=":", linewidth=1,
                        label=r"prune threshold $V=0.5$")
        axes[1].set_xlabel(r"$V_\varphi$ prediction")
        axes[1].set_ylabel(r"mean $V^{\pi^\dagger}$ in bin")
        axes[1].set_title(r"Reliability: $E[V^{\pi^\dagger} \mid V_\varphi]$")
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].set_aspect("equal")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="lower right")

        fig.suptitle(
            r"Value function $V_\varphi$ against the Tier 2 target "
            r"$V^{\pi^\dagger}$ (validation split)",
            fontsize=14,
        )
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"value_reliability.{ext}", bbox_inches="tight")
        plt.close(fig)

        audit["value"] = {
            "mse": float(np.mean((pred_v - target_v) ** 2)),
            "soft_calibration_error": sce,
            # Fraction of states the 0.5 prune threshold would mis-classify:
            # both a branch killed that was actually matchable and one kept that
            # was not. The first kind is the expensive one under DAgger.
            "wrongly_pruned_frac": float(
                np.mean((pred_v < 0.5) & (target_v >= 0.5))
            ),
            "wrongly_kept_frac": float(
                np.mean((pred_v >= 0.5) & (target_v < 0.5))
            ),
            "n_bins_drawn": len(drawn),
        }

    # --- Pass/fail against §4.2 targets ----------------------------------
    # Skipped strata carry ece=None; exclude them rather than crashing on max().
    stratum_eces = [
        v["ece"]
        for v in (
            list(audit.get("eta_strata_platt2", {}).values())
            + list(audit.get("occupancy_strata_platt2", {}).values())
        )
        if v.get("ece") is not None
    ]
    worst_stratum = max(stratum_eces) if stratum_eces else 0.0
    audit["worst_stratum_ece"] = worst_stratum
    audit["passes_overall_target"] = audit["ece"]["gate_platt2"] < ECE_TARGET_OVERALL
    audit["passes_stratum_target"] = worst_stratum < ECE_TARGET_STRATUM

    # The overall-ECE target is necessary but weak: count-weighting means it can
    # pass on the easy-negative bulk alone. Require the decision region too, or a
    # gate that is useless at every reachable τ would still report "passes".
    dre = audit["decision_region_ece"]["gate_platt2"]["ece"]
    audit["passes_decision_region_target"] = bool(
        dre == dre and dre < ECE_TARGET_STRATUM  # NaN-safe: NaN != NaN
    )
    audit["headline_beats_chi2"] = (
        audit["decision_region_ece"]["gate_platt2"]["ece"]
        < audit["decision_region_ece"]["chi2_lambda"]["ece"]
    )

    # §4.3: only adopt the 4-parameter form if it actually helps the strata.
    audit["recommend_4param_platt"] = (
        not audit["passes_stratum_target"]
        and audit["ece"]["gate_platt4"] < audit["ece"]["gate_platt2"]
    )

    (out_dir / "platt_params.json").write_text(
        json.dumps({"two_param": audit["platt_2param"],
                    "four_param": audit["platt_4param"]}, indent=2)
    )
    (out_dir / "calibration_audit.json").write_text(json.dumps(audit, indent=2, default=str))

    print(json.dumps({k: v for k, v in audit.items() if k != "G3"}, indent=2, default=str))
    print(f"\nfigures written to {out_dir}")


if __name__ == "__main__":
    main()
