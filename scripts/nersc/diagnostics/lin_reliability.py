"""Dump reliability curves on a LINEAR p grid for the four gate estimators.

Reproduces calibrate_and_audit.py's definitions exactly (chi2_lambda,
gate raw, Platt-2, occupancy-conditional Platt-4), then bins on a uniform
grid so the curve can be read on a normal axis rather than a logit one.
Writes a small JSON; all plotting happens locally.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/global/cfs/cdirs/atlas/mussonm/cCKF")
import torch
from cckf import cache, calibration, features, models, train

MODEL = "/pscratch/sd/m/mussonm/cckf/models_v3/gate_maj/gate_model.pt"
CAL   = "/pscratch/sd/m/mussonm/cckf/caches_v3/cal"
OUT   = "/pscratch/sd/m/mussonm/cckf/models_v3/gate_maj/linear_reliability.json"

ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
model = models.GateMLP(n_features=ckpt["n_features"], width=ckpt["width"],
                       depth=ckpt["depth"])
model.load_state_dict(ckpt["state_dict"])

cal = cache.load_cache(CAL)
col = np.array([features.GATE_FEATURES.index(n) for n in ckpt["feature_names"]])
X = ((np.asarray(cal["X"])[:, col] - ckpt["mu"]) / ckpt["sigma"]).astype(np.float32)
y = np.asarray(cal["y"]).astype(np.uint8)
aux = np.asarray(cal["aux"]); chi2, n_window = aux[:, 0], aux[:, 1]

logits = train.predict_logits(model, X, device="cpu")
raw = 1.0 / (1.0 + np.exp(-logits))
a, b = calibration.fit_platt(logits, y)
p2 = calibration.apply_platt(logits, a, b)
occ = calibration.fit_platt_occupancy(logits, y, n_window)
p4 = calibration.apply_platt_occupancy(logits, n_window, occ)
lam = np.clip(np.exp(-np.where(np.isfinite(chi2), chi2, np.inf) / 2.0), 0.0, 1.0)

EDGES = np.linspace(0.1, 1.0, 19)          # 0.05-wide linear bins
out = {"edges": EDGES.tolist(), "n_cal_rows": int(len(y)),
       "positive_fraction": float(y.mean())}
for name, p in (("chi2_lambda", lam), ("gate_raw", raw),
                ("gate_platt2", p2), ("gate_platt4", p4)):
    idx = np.digitize(p, EDGES) - 1
    ok = (idx >= 0) & (idx < len(EDGES) - 1)
    n = np.bincount(idx[ok], minlength=len(EDGES) - 1)
    sp = np.bincount(idx[ok], weights=p[ok], minlength=len(EDGES) - 1)
    sy = np.bincount(idx[ok], weights=y[ok], minlength=len(EDGES) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[name] = {"n": n.tolist(),
                     "mean_pred": np.where(n > 0, sp / np.maximum(n, 1), np.nan).tolist(),
                     "obs_freq":  np.where(n > 0, sy / np.maximum(n, 1), np.nan).tolist()}
    print(f"{name:14s} rows in [0.1,1]: {int(n.sum()):>10,}")

json.dump(out, open(OUT, "w"))
print("wrote", OUT)
