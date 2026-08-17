"""Figure set for the gate S1 sampling ablation.

Reads the bundles written by scripts/export_gate_curves.py and writes six
figures. Every panel uses fixed, shared axes so the three arms and the chi2
baseline are directly comparable.

Three of these figures carry a subtlety a reader would otherwise take for a
bug, so each states it in its own caption:

* F1 -- val BCE is not comparable across arms (B and C carry deliberate
  subsampling bias that Platt later removes, and BCE penalises exactly that).
* F2/F6 -- ROC, PR and both AUCs are invariant under 2-param Platt, because
  a > 0 makes it a strictly increasing map and those metrics see only ranking.

Usage
-----
    python scripts/plot_gate_figures.py \\
        --bundle-dir results/curves --out-dir figures/gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cckf import curves, metrics

ARMS = ("A", "B", "C")
ARM_LABEL = {
    "A": "A — no subsampling",
    "B": "B — uniform 1:5",
    "C": "C — hard-neg ∝1/χ²",
}
ARM_COLOR = {"A": "#1b5e9c", "B": "#2e8b57", "C": "#c0392b"}
EST_LABEL = {
    "chi2_lambda": "χ²_λ baseline",
    "gate_raw": "gate, raw",
    "gate_platt2": "gate + Platt-2",
    "gate_platt4": "gate + Platt-4",
}
EXTS = ("png", "pdf")


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in EXTS:
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")
    return out_dir / f"{name}.png"


def _load(bundle_dir: Path) -> dict:
    return {
        arm: {
            "arrays": np.load(bundle_dir / f"gate_{arm}.npz"),
            "scalars": json.loads((bundle_dir / f"gate_{arm}.json").read_text()),
        }
        for arm in ARMS
    }


def figure_loss_curves(data: dict, out_dir: Path) -> Path:
    """F1: per-epoch train/val BCE for each arm."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, arm in zip(axes, ARMS):
        hist = data[arm]["scalars"]["training_history"]
        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        ax.semilogy(epochs, hist["train_loss"], color=ARM_COLOR[arm], label="train BCE")
        ax.semilogy(
            epochs, hist["val_loss"], color=ARM_COLOR[arm], ls="--", label="val BCE"
        )
        best = int(np.argmin(hist["val_loss"]))
        ax.axvline(best + 1, color="0.5", lw=0.8, ls=":")
        ax.set_title(
            f"{ARM_LABEL[arm]}\nbest epoch {best + 1} of {len(epochs)}", fontsize=9
        )
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("BCE")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "F1  Gate training loss. Val BCE is NOT comparable across arms: B and C "
        "carry deliberate subsampling\nbias in their logits that Platt later "
        "removes, and BCE penalises exactly that bias. Compare AUC instead.",
        fontsize=9,
    )
    return _save(fig, out_dir, "F1_loss_curves")


def figure_roc(data: dict, out_dir: Path) -> Path:
    """F2: ROC per arm, full range plus a log-FPR view of the usable region."""
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for arm in ARMS:
        arr, sc = data[arm]["arrays"], data[arm]["scalars"]
        auc = sc["metrics"]["gate_platt2"]["auc_roc"]
        for ax in (ax_full, ax_zoom):
            ax.plot(
                arr["gate_platt2__fpr"],
                arr["gate_platt2__tpr"],
                color=ARM_COLOR[arm],
                lw=1.6,
                label=f"{ARM_LABEL[arm]}  AUC {auc:.4f}",
            )
    arr = data["A"]["arrays"]
    auc_chi2 = data["A"]["scalars"]["metrics"]["chi2_lambda"]["auc_roc"]
    for ax in (ax_full, ax_zoom):
        ax.plot(
            arr["chi2_lambda__fpr"],
            arr["chi2_lambda__tpr"],
            color="0.35",
            lw=1.2,
            ls="-.",
            label=f"χ²_λ baseline  AUC {auc_chi2:.4f}",
        )
        ax.grid(alpha=0.3)
        ax.set_ylabel("true-positive rate (efficiency)")
        ax.legend(fontsize=8, loc="lower right")
    ax_full.plot([0, 1], [0, 1], color="0.7", lw=0.8, ls=":")
    ax_full.set_xlabel("false-positive rate")
    ax_full.set_title("full range", fontsize=9)
    ax_zoom.set_xscale("log")
    ax_zoom.set_xlim(1e-6, 1.0)
    ax_zoom.set_xlabel("false-positive rate (log)")
    ax_zoom.set_title("usable region — FPR ≲ 1e-2", fontsize=9)
    fig.suptitle(
        "F2  ROC on the val split. Curves depend only on score *ranking*, so "
        "2-param Platt cannot move them\n(a > 0 is strictly increasing) — raw "
        "and Platt-2 ROCs are identical by construction, not coincidence.",
        fontsize=9,
    )
    return _save(fig, out_dir, "F2_roc")


def figure_pr(data: dict, out_dir: Path) -> Path:
    """F3: precision-recall per arm, with the base-rate floor drawn."""
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    base = data["A"]["scalars"]["metrics"]["gate_platt2"]["base_rate"]
    for arm in ARMS:
        arr, sc = data[arm]["arrays"], data[arm]["scalars"]
        rec, prec = arr["gate_platt2__tpr"], arr["gate_platt2__precision"]
        ok = np.isfinite(prec) & (prec > 0)
        ax.plot(
            rec[ok],
            prec[ok],
            color=ARM_COLOR[arm],
            lw=1.6,
            label=f"{ARM_LABEL[arm]}  AP {sc['metrics']['gate_platt2']['auc_pr']:.4f}",
        )
    arr = data["A"]["arrays"]
    prec = arr["chi2_lambda__precision"]
    ok = np.isfinite(prec) & (prec > 0)
    ax.plot(
        arr["chi2_lambda__tpr"][ok],
        prec[ok],
        color="0.35",
        lw=1.2,
        ls="-.",
        label=f"χ²_λ  AP {data['A']['scalars']['metrics']['chi2_lambda']['auc_pr']:.4f}",
    )
    ax.axhline(base, color="0.7", lw=1.0, ls=":", label=f"no-skill floor = {base:.4%}")
    ax.set_yscale("log")
    ax.set_xlabel("recall (efficiency)")
    ax.set_ylabel("precision (purity, log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(
        f"F3  Precision-recall on the val split. At a {base:.2%} base rate, "
        "ROC's FPR denominator\n(~24M negatives) hides operational cost; PR's "
        "denominator is TP+FP, which is what a\nbranch budget actually pays.",
        fontsize=9,
    )
    return _save(fig, out_dir, "F3_pr")


def figure_calibration_nll(data: dict, out_dir: Path) -> Path:
    """F4: L-BFGS-B convergence of both Platt fits, per arm."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for arm in ARMS:
        sc = data[arm]["scalars"]
        for key, ls, tag in (
            ("calibration_nll_trace_2param", "-", "Platt-2"),
            ("calibration_nll_trace_4param", "--", "Platt-4"),
        ):
            trace = np.asarray(sc[key], dtype=float)
            ax.plot(
                np.arange(len(trace)),
                trace,
                color=ARM_COLOR[arm],
                ls=ls,
                lw=1.5,
                marker="o",
                ms=3,
                label=f"{ARM_LABEL[arm]} — {tag}  final {trace[-1]:.5f}",
            )
    ax.set_yscale("log")
    ax.set_xlabel("L-BFGS-B iteration (0 = initial guess a=1, b=0)")
    ax.set_ylabel("mean NLL on the calibration split (log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax.set_title(
        "F4  Calibration fit convergence. The NLL is convex in the Platt "
        "parameters, so these traces are\nmonotone and the optimum is global — "
        "the curve shows cost, not risk of a bad local minimum.",
        fontsize=9,
    )
    return _save(fig, out_dir, "F4_calibration_nll")


def figure_reliability(data: dict, out_dir: Path) -> Path:
    """F5: reliability diagrams, one panel per arm, on shared log-log axes.

    Log-log rather than linear, matching scripts/calibrate_and_audit.py: the
    gate spans several orders of magnitude in probability and a linear axis
    compresses everything below p~0.05 into the origin, which is exactly where
    the decision lives. Sparse bins (count < MIN_BIN_COUNT) are drawn as bare
    Wilson intervals with no marker, so a thin region reads as "uncertain"
    rather than silently absent.
    """
    lo, hi = metrics.DECISION_REGION
    t_lo, t_hi = metrics.THRESHOLD_REGION
    axis_lo = 1e-5

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharex=True, sharey=True)
    for ax, arm in zip(axes, ARMS):
        sc = data[arm]["scalars"]
        for est, color, marker in (
            ("chi2_lambda", "0.35", "s"),
            ("gate_raw", "#e08a1e", "^"),
            ("gate_platt2", ARM_COLOR[arm], "o"),
        ):
            bins = sc["reliability"][est]["bins"]
            solid = [b for b in bins if b["count"] > 0 and not b["sparse"]]
            if solid:
                ax.plot(
                    [b["mean_predicted"] for b in solid],
                    [b["observed_fraction"] for b in solid],
                    marker=marker,
                    ms=4,
                    lw=1.3,
                    color=color,
                    label=(
                        f"{EST_LABEL[est]}  ECE {sc['metrics'][est]['ece']:.2e}, "
                        f"MCE {sc['metrics'][est]['mce']:.3f}"
                    ),
                )
                ax.fill_between(
                    [b["mean_predicted"] for b in solid],
                    [max(b["ci_lower"], axis_lo) for b in solid],
                    [b["ci_upper"] for b in solid],
                    alpha=0.18,
                    color=color,
                )
            for b in bins:
                if b["count"] > 0 and b["sparse"]:
                    ax.plot(
                        [b["mean_predicted"]] * 2,
                        [max(b["ci_lower"], axis_lo), b["ci_upper"]],
                        color=color,
                        alpha=0.35,
                        lw=1,
                    )
        ax.plot([axis_lo, 1], [axis_lo, 1], color="0.7", lw=0.8, ls=":")
        ax.axvspan(lo, hi, color="#4a90d9", alpha=0.08, zorder=0)
        ax.axvspan(t_lo, t_hi, color="#4a90d9", alpha=0.08, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(axis_lo, 1.0)
        ax.set_ylim(axis_lo, 1.0)
        ax.set_aspect("equal")
        ax.set_xlabel("mean predicted probability")
        ax.set_title(ARM_LABEL[arm], fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("observed positive fraction")
    fig.suptitle(
        "F5  Reliability on the val split. Bins uniform in log-odds (30 bins, "
        f"p∈[1e-5, 0.99999]); dotted line is\nperfect calibration. Shaded bands "
        f"are Wilson intervals; shaded columns are the audited region [{lo}, "
        f"{hi}]\nand, darker, the threshold view [{t_lo}, {t_hi}]. Bin edges are "
        "fixed constants shared by every curve.",
        fontsize=9,
    )
    return _save(fig, out_dir, "F5_reliability")


def figure_reliability_linear(data: dict, out_dir: Path, n_bins: int = 20) -> Path:
    """F5b: the textbook reliability diagram -- linear axes, equal-width bins.

    Equal-width bins over [0, 1] on linear axes is the conventional form, and
    for chi2_lambda it is the *right* form: being a p-value, chi2_lambda is
    roughly uniform on [0, 1] under the correct-hit hypothesis, so equal-width
    bins are well populated across the whole range.

    For the gate it is the less informative view, and the figure should be read
    knowing why: at a 0.57% base rate the gate concentrates ~99.5% of its mass
    below 0.01, so the first bin swallows nearly every row and the upper bins go
    thin. That is a property of the data, not a defect of the model -- the
    log-odds view (F5) is where the gate's decision region is resolved.

    Bins come from summing the fine 1000-bin sufficient statistics shipped in
    the bundle, so both the counts and the *mean prediction* per bin are exact.
    Bin midpoints are never substituted for the mean prediction: the first bin's
    rows average ~0.001 against a midpoint of 0.025.
    """
    missing = [
        arm
        for arm in ARMS
        if "chi2_lambda__hist_count" not in data[arm]["arrays"].files
    ]
    if missing:
        print(
            f"skipping F5b: bundles for {', '.join(missing)} predate the "
            "prob_histogram statistics; re-run export_curves to add them"
        )
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharex=True, sharey=True)
    for ax, arm in zip(axes, ARMS):
        entry, sc = data[arm], data[arm]["scalars"]
        for est, color, marker in (
            ("chi2_lambda", "0.35", "s"),
            ("gate_raw", "#e08a1e", "^"),
            ("gate_platt2", ARM_COLOR[arm], "o"),
        ):
            hist = {
                "count": entry["arrays"][f"{est}__hist_count"],
                "sum_prob": entry["arrays"][f"{est}__hist_sum_prob"],
                "sum_label": entry["arrays"][f"{est}__hist_sum_label"],
            }
            bins = curves.rebin_equal_width(
                hist, n_bins=n_bins, min_count=metrics.MIN_BIN_COUNT
            )["bins"]
            solid = [b for b in bins if b["count"] > 0 and not b["sparse"]]
            thin = [b for b in bins if 0 < b["count"] < metrics.MIN_BIN_COUNT]
            if solid:
                # Wilson error bars, asymmetric by construction: the interval on
                # a proportion near 0 or 1 is not symmetric about the estimate,
                # and bin occupancy here spans five orders of magnitude, so the
                # bars are what tell a real deviation from a thinly-populated
                # one. Most are smaller than the marker -- that is the honest
                # signal that these deviations are not sampling noise.
                obs = np.array([b["observed_fraction"] for b in solid])
                ax.errorbar(
                    [b["mean_predicted"] for b in solid],
                    obs,
                    yerr=np.vstack(
                        [
                            obs - np.array([b["ci_lower"] for b in solid]),
                            np.array([b["ci_upper"] for b in solid]) - obs,
                        ]
                    ),
                    marker=marker,
                    ms=5,
                    lw=1.4,
                    elinewidth=1.1,
                    capsize=2.5,
                    color=color,
                    label=f"{EST_LABEL[est]}  ECE {sc['metrics'][est]['ece']:.2e}",
                )
            if thin:
                obs_t = np.array([b["observed_fraction"] for b in thin])
                ax.errorbar(
                    [b["mean_predicted"] for b in thin],
                    obs_t,
                    yerr=np.vstack(
                        [
                            obs_t - np.array([b["ci_lower"] for b in thin]),
                            np.array([b["ci_upper"] for b in thin]) - obs_t,
                        ]
                    ),
                    marker=marker,
                    ms=4,
                    lw=0,
                    elinewidth=0.9,
                    capsize=2,
                    mfc="none",
                    color=color,
                    alpha=0.6,
                    label=f"{EST_LABEL[est]} (sparse, <100 rows)",
                )
        ax.plot([0, 1], [0, 1], color="0.7", lw=0.9, ls=":", label="perfect")
        ax.axhline(
            sc["metrics"]["gate_platt2"]["base_rate"],
            color="0.55",
            lw=0.8,
            ls="--",
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("mean predicted probability")
        ax.set_title(ARM_LABEL[arm], fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="upper left")
    axes[0].set_ylabel("observed positive fraction")
    fig.suptitle(
        f"F5b  Reliability on the val split — linear axes, {n_bins} equal-width "
        "bins. Error bars are 95% Wilson intervals\n(asymmetric near 0 and 1); "
        "open markers are sparse bins (<100 rows). Dashed line is the base rate. "
        "χ²_λ = exp(-χ²/2)\nis a *p-value* under the correct-hit hypothesis, not "
        "a posterior, so it carries no prior and sits flat near that rate.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, out_dir, "F5b_reliability_linear")


def figure_before_after(data: dict, out_dir: Path) -> Path:
    """F6: metric deltas from calibration, grouped bars."""
    lo, hi = metrics.DECISION_REGION
    keys = [
        ("auc_roc", "AUC-ROC"),
        ("auc_pr", "AUC-PR"),
        ("ece", "ECE"),
        ("dr_ece", f"DR-ECE [{lo}, {hi}]"),
        ("mce", "MCE"),
    ]
    fig, axes = plt.subplots(1, len(keys), figsize=(18, 5.4))
    width = 0.26
    handles: list = []
    for ax, (key, title) in zip(axes, keys):
        is_log = key in ("ece", "dr_ece", "mce")
        for i, est in enumerate(("gate_raw", "gate_platt2", "gate_platt4")):
            vals = [data[arm]["scalars"]["metrics"][est][key] for arm in ARMS]
            bars = ax.bar(
                np.arange(len(ARMS)) + (i - 1) * width,
                vals,
                width,
                label=EST_LABEL[est],
                color=["#bbbbbb", "#4a90d9", "#1b5e9c"][i],
            )
            if ax is axes[0]:
                handles.append(bars)
            # Value labels, because the axis range cannot serve both the AUC
            # panels (all within 0.96-0.99, differences invisible at 0-1 scale)
            # and the chi2 reference line an order of magnitude away.
            for rect, v in zip(bars, vals):
                ax.annotate(
                    f"{v:.2e}" if is_log else f"{v:.4f}",
                    (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    textcoords="offset points",
                    xytext=(0, 2),
                    ha="center",
                    fontsize=5.5,
                    rotation=90,
                )
        chi2 = data["A"]["scalars"]["metrics"]["chi2_lambda"][key]
        line = ax.axhline(chi2, color="0.35", ls="-.", lw=1.2, label="χ²_λ")
        if ax is axes[0]:
            handles.append(line)
        if is_log:
            ax.set_yscale("log")
        ax.margins(y=0.22)  # headroom for the rotated value labels
        ax.set_xticks(np.arange(len(ARMS)))
        ax.set_xticklabels(ARMS)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y", which="both")
    fig.legend(
        handles=[h for h in handles],
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "F6  Before/after calibration on the val split. AUC-ROC and AUC-PR are "
        "IDENTICAL for raw and Platt-2 by\nconstruction (a > 0 is monotone, and "
        "AUC sees only ranking); they move only under Platt-4, whose slope\n"
        "a(x) = a0 + a1·log n_window is row-dependent and so is not a single "
        "monotone map.",
        fontsize=9,
    )
    # Explicit rect: the 3-line suptitle and the bottom legend both need space
    # the default layout gives to the axes, which collides the panel titles.
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    return _save(fig, out_dir, "F6_before_after")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", default="results/curves")
    parser.add_argument("--out-dir", default="figures/gate")
    args = parser.parse_args()

    data = _load(Path(args.bundle_dir))
    out_dir = Path(args.out_dir)
    for fn in (
        figure_loss_curves,
        figure_roc,
        figure_pr,
        figure_calibration_nll,
        figure_reliability,
        figure_reliability_linear,
        figure_before_after,
    ):
        fn(data, out_dir)

    # AUC invariance under 2-param Platt is arithmetic, not an empirical
    # finding; verify it held on real data rather than trusting the figure.
    #
    # The tolerance is 1e-4, not 0, and the reason is float saturation rather
    # than sloppiness. a*z + b maps a different set of extreme logits to
    # exactly 0.0 or 1.0 in float64 than z does, so recalibration creates and
    # destroys *ties* at the saturated ends even though it preserves the strict
    # ordering everywhere else. Average precision is tie-sensitive, so it
    # shifts in the 6th decimal. A genuine sign error would instead send AUC
    # toward 1 - AUC, which 1e-4 catches with room to spare.
    for arm in ARMS:
        m = data[arm]["scalars"]["metrics"]
        for key in ("auc_roc", "auc_pr"):
            delta = abs(m["gate_raw"][key] - m["gate_platt2"][key])
            if delta > 1e-4:
                print(
                    f"WARNING arm {arm}: {key} moved by {delta:.3e} under 2-param "
                    "Platt, which a strictly increasing map cannot do beyond "
                    "tie effects at float saturation — check the fitted slope's "
                    f"sign (a = {data[arm]['scalars']['platt_2param']['a']:.4f})"
                )

    for arm in ARMS:
        sv = data[arm]["scalars"]["slope_violations_4param"]
        if sv["n_rows_slope_nonpositive"]:
            print(
                f"WARNING arm {arm}: Platt-4 slope <= 0 on "
                f"{sv['frac_rows_slope_nonpositive']:.4%} of val rows "
                f"(n_window >= {sv['n_window_at_slope_zero']:.0f}); the "
                "calibrator inverts the model's ranking there"
            )


if __name__ == "__main__":
    main()
