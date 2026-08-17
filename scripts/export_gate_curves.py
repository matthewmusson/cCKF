"""Export plot-data bundles for the gate figure set.

Reads a frozen ``gate_model.pt``, runs inference on the val and cal caches,
fits both Platt forms on **cal** (never val -- the calibration split is the
only split allowed to see a calibrator fit), applies them to **val**, and
writes a small bundle of grid curves, reliability bins and scalar metrics.

Evaluating before/after on val rather than cal keeps the calibrator's fit and
its evaluation on disjoint data, and makes the "before" AUCs directly
comparable to the numbers already in experiments/LOG.md.

The bundle is deliberately small -- 4,001-point curves, 30 reliability bins per
estimator, and scalars, roughly 1 MB per arm -- so figure iteration happens on
a laptop instead of round-tripping 68M inference scores off the volume.

Usage
-----
    python scripts/export_gate_curves.py \\
        --model-dir /data/models/gate_A \\
        --val-cache /data/cache/gate/val \\
        --cal-cache /data/cache/gate/cal \\
        --out-dir /data/results/curves
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, logit

from cckf import cache, calibration, curves, features, metrics, models, train

#: aux.f32 column order written by cckf.cache.build_gate_cache.
AUX_CHI2, AUX_N_WINDOW, AUX_ETA = 0, 1, 2

#: Keep probabilities off the closed interval's endpoints before taking logits,
#: so an exactly-saturated 0.0 or 1.0 does not become -+inf on a plot axis.
_P_EPS = 1e-12


def _load_model(model_dir: Path) -> dict:
    """Load a frozen gate checkpoint and put it in eval mode."""
    ckpt = torch.load(
        model_dir / "gate_model.pt", map_location="cpu", weights_only=False
    )
    model = models.GateMLP(
        n_features=ckpt["n_features"], width=ckpt["width"], depth=ckpt["depth"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "ckpt": ckpt}


def _logits_for(loaded: dict, cache_dir: str, device: str) -> tuple[np.ndarray, dict]:
    """Raw logits over one whole cache, streamed a batch at a time.

    Uses ``train.StandardizedView`` rather than materialising the standardised
    matrix: the cal cache is 44.4M x 26 float32, so an eager copy is ~4.6 GB and
    the transient standardisation doubles that.
    """
    ckpt = loaded["ckpt"]
    cached = cache.load_cache(cache_dir)
    col_idx = np.array([features.GATE_FEATURES.index(n) for n in ckpt["feature_names"]])

    # The checkpoint's mu/sigma are already subset to the model's own columns,
    # but StandardizedView indexes mu/sigma by *source* column before narrowing
    # with col_idx, so scatter them back to full width first.
    mu = np.zeros(len(features.GATE_FEATURES), dtype=np.float32)
    sigma = np.ones(len(features.GATE_FEATURES), dtype=np.float32)
    mu[col_idx] = ckpt["mu"]
    sigma[col_idx] = ckpt["sigma"]

    view = train.StandardizedView(
        cached["X"], np.arange(len(cached["y"])), mu, sigma, col_idx
    )
    return train.predict_logits(loaded["model"], view, device=device), cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--cal-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    arm = model_dir.name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = _load_model(model_dir)
    training_metrics = json.loads((model_dir / "gate_metrics.json").read_text())

    cal_logits, cal = _logits_for(loaded, args.cal_cache, args.device)
    cal_labels = np.asarray(cal["y"]).astype(np.float64)
    cal_nw = np.asarray(cal["aux"][:, AUX_N_WINDOW], dtype=np.float64)
    print(f"{arm}: cal inference done, {len(cal_labels):,} rows")

    # Platt is fitted on cal only.
    trace2: list[float] = []
    a, b = calibration.fit_platt(cal_logits, cal_labels, trace=trace2)
    trace4: list[float] = []
    p4 = calibration.fit_platt_occupancy(cal_logits, cal_labels, cal_nw, trace=trace4)
    print(f"{arm}: platt2 a={a:.4f} b={b:.4f}  platt4 {p4}")

    val_logits, val = _logits_for(loaded, args.val_cache, args.device)
    val_labels = np.asarray(val["y"]).astype(bool)
    val_nw = np.asarray(val["aux"][:, AUX_N_WINDOW], dtype=np.float64)
    print(f"{arm}: val inference done, {len(val_labels):,} rows")

    estimators = {
        # chi2-implied probability on the same val rows, so the baseline and the
        # gate share one axis. exp(-chi2/2) is the chi2_2 survival function.
        "chi2_lambda": np.clip(
            np.exp(
                -0.5
                * np.nan_to_num(
                    np.asarray(val["aux"][:, AUX_CHI2], dtype=np.float64),
                    nan=np.inf,
                    posinf=np.inf,
                )
            ),
            0.0,
            1.0,
        ),
        "gate_raw": expit(val_logits),
        "gate_platt2": calibration.apply_platt(val_logits, a, b),
        "gate_platt4": calibration.apply_platt_occupancy(val_logits, val_nw, p4),
    }

    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, object] = {
        "arm": arm,
        "split_curves": "val",
        "split_platt_fit": "cal",
        "platt_2param": {"a": a, "b": b},
        "platt_4param": dict(zip(("a0", "a1", "b0", "b1"), p4)),
        "prior_logit_shift": float(loaded["ckpt"].get("prior_logit_shift", 0.0)),
        "calibration_nll_trace_2param": [float(v) for v in trace2],
        "calibration_nll_trace_4param": [float(v) for v in trace4],
        "slope_violations_4param": calibration.platt_occupancy_slope_violations(
            val_nw, p4
        ),
        "training_history": training_metrics["history"],
        "n_val_rows": int(len(val_labels)),
        "n_cal_rows": int(len(cal_labels)),
        "decision_region": list(metrics.DECISION_REGION),
        "threshold_region": list(metrics.THRESHOLD_REGION),
        "metrics": {},
        "reliability": {},
    }

    rel_edges = metrics.logit_bin_edges(30)
    for name, prob in estimators.items():
        p = np.clip(prob, _P_EPS, 1.0 - _P_EPS)
        c = curves.grid_curves(logit(p), val_labels)
        for key in ("threshold_logit", "tpr", "fpr", "precision"):
            arrays[f"{name}__{key}"] = c[key].astype(np.float32)
        arrays[f"{name}__tp"] = c["tp"]
        arrays[f"{name}__fp"] = c["fp"]

        scalars["metrics"][name] = curves.metric_bundle(prob, val_labels)
        # Real reliability bins on full data -- counts and Wilson intervals
        # included -- rather than reconstructing them from the threshold grid.
        # 30 bins per estimator is a few kilobytes.
        scalars["reliability"][name] = metrics.reliability_bins(
            prob, val_labels, edges=rel_edges
        )
        m = scalars["metrics"][name]
        print(
            f"{arm} {name}: AUC-ROC {m['auc_roc']:.6f} AUC-PR {m['auc_pr']:.6f} "
            f"ECE {m['ece']:.3e} DR-ECE {m['dr_ece']:.3e} MCE {m['mce']:.4f}"
        )

    np.savez_compressed(out_dir / f"{arm}.npz", **arrays)
    (out_dir / f"{arm}.json").write_text(json.dumps(scalars, indent=2, default=str))
    print(f"wrote {out_dir / f'{arm}.npz'} and {arm}.json")


if __name__ == "__main__":
    main()
