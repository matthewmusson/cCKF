"""Dump per-stratum reliability curves for BOTH chi2_lambda and the calibrated
gate, so they can be overlaid on shared axes rather than shown side by side.

Reuses calibrate_and_audit.py's exact binning (logit-uniform, 30 bins) and
strata definitions, so the curves are directly comparable to the audit.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/global/cfs/cdirs/atlas/mussonm/cCKF")
import torch
from cckf import cache, calibration, features, metrics, models, train

MODEL = "/pscratch/sd/m/mussonm/cckf/models_v3/gate_maj/gate_model.pt"
CAL   = "/pscratch/sd/m/mussonm/cckf/caches_v3/cal"
OUT   = "/pscratch/sd/m/mussonm/cckf/models_v3/gate_maj/g3_overlay.json"

ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
model = models.GateMLP(n_features=ckpt["n_features"], width=ckpt["width"],
                       depth=ckpt["depth"])
model.load_state_dict(ckpt["state_dict"])

cal = cache.load_cache(CAL)
col = np.array([features.GATE_FEATURES.index(n) for n in ckpt["feature_names"]])
X = ((np.asarray(cal["X"])[:, col] - ckpt["mu"]) / ckpt["sigma"]).astype(np.float32)
y = np.asarray(cal["y"]).astype(np.uint8)
aux = np.asarray(cal["aux"]); chi2, n_window, eta = aux[:, 0], aux[:, 1], aux[:, 2]

logits = train.predict_logits(model, X, device="cpu")
a, b = calibration.fit_platt(logits, y)
gate = calibration.apply_platt(logits, a, b)            # Platt-2, the audit primary
lam  = np.clip(np.exp(-np.where(np.isfinite(chi2), chi2, np.inf) / 2.0), 0.0, 1.0)

EDGES = metrics.logit_bin_edges(n_bins=30)

def curve(p, m):
    idx = np.digitize(p[m], EDGES) - 1
    ok = (idx >= 0) & (idx < len(EDGES) - 1)
    i, yy, pp = idx[ok], y[m][ok], p[m][ok]
    n = np.bincount(i, minlength=len(EDGES) - 1).astype(float)
    sp = np.bincount(i, weights=pp, minlength=len(EDGES) - 1)
    sy = np.bincount(i, weights=yy, minlength=len(EDGES) - 1)
    with np.errstate(invalid="ignore"):
        return {"n": n.tolist(),
                "pred": np.where(n > 0, sp / np.maximum(n, 1), np.nan).tolist(),
                "obs":  np.where(n > 0, sy / np.maximum(n, 1), np.nan).tolist()}

out = {"edges": EDGES.tolist(), "platt2": {"a": float(a), "b": float(b)},
       "n_cal_rows": int(len(y)), "eta": {}, "occ": {}}
for lab, m in metrics.eta_strata(eta).items():
    out["eta"][lab] = {"n": int(m.sum()), "chi2": curve(lam, m), "gate": curve(gate, m)}
    print(f"eta  {lab:20s} n={int(m.sum()):>10,}")
for lab, m in metrics.quintile_strata(n_window).items():
    out["occ"][lab] = {"n": int(m.sum()), "chi2": curve(lam, m), "gate": curve(gate, m)}
    print(f"occ  {lab:20s} n={int(m.sum()):>10,}")

json.dump(out, open(OUT, "w"))
print("wrote", OUT)
