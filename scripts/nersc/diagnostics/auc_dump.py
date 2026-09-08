"""AUC-ROC and AUC-PR for all four estimators on the calibration split.

Platt-2 is a single monotone map of the logit and therefore cannot change
either AUC. Platt-4 is occupancy-conditional -- a different monotone map per
stratum -- so it CAN reorder candidates across strata. This measures whether
it actually does.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/global/cfs/cdirs/atlas/mussonm/cCKF")
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from cckf import cache, calibration, features, models, train

ckpt = torch.load("/pscratch/sd/m/mussonm/cckf/models_v3/gate_maj/gate_model.pt",
                  map_location="cpu", weights_only=False)
model = models.GateMLP(n_features=ckpt["n_features"], width=ckpt["width"],
                       depth=ckpt["depth"])
model.load_state_dict(ckpt["state_dict"])

cal = cache.load_cache("/pscratch/sd/m/mussonm/cckf/caches_v3/cal")
col = np.array([features.GATE_FEATURES.index(n) for n in ckpt["feature_names"]])
X = ((np.asarray(cal["X"])[:, col] - ckpt["mu"]) / ckpt["sigma"]).astype(np.float32)
y = np.asarray(cal["y"]).astype(np.uint8)
aux = np.asarray(cal["aux"]); chi2, n_window = aux[:, 0], aux[:, 1]

logits = train.predict_logits(model, X, device="cpu")
raw = logits                                   # rank-equivalent to sigmoid(logits)
a, b = calibration.fit_platt(logits, y)
p2 = calibration.apply_platt(logits, a, b)
occ = calibration.fit_platt_occupancy(logits, y, n_window)
p4 = calibration.apply_platt_occupancy(logits, n_window, occ)
lam = np.clip(np.exp(-np.where(np.isfinite(chi2), chi2, np.inf) / 2.0), 0.0, 1.0)

out = {}
for name, sc in (("chi2_lambda", lam), ("gate_raw", raw),
                 ("gate_platt2", p2), ("gate_platt4", p4)):
    s = np.asarray(sc, dtype=np.float64)
    out[name] = {"auc_roc": float(roc_auc_score(y, s)),
                 "auc_pr":  float(average_precision_score(y, s))}
    print(f"{name:14s} AUC-ROC={out[name]['auc_roc']:.5f}  AUC-PR={out[name]['auc_pr']:.5f}")

# does platt4 actually reorder relative to raw?
n = min(2_000_000, len(y))
idx = np.random.default_rng(0).choice(len(y), n, replace=False)
r_raw = np.argsort(np.argsort(raw[idx]))
r_p4  = np.argsort(np.argsort(p4[idx]))
out["spearman_raw_vs_platt4"] = float(np.corrcoef(r_raw, r_p4)[0, 1])
print(f"\nrank correlation raw vs Platt-4 (2M sample): {out['spearman_raw_vs_platt4']:.6f}")
out["n_rows"] = int(len(y)); out["positive_fraction"] = float(y.mean())
json.dump(out, open("/pscratch/sd/m/mussonm/cckf/models_v3/gate_maj/auc_variants.json","w"))
print("wrote auc_variants.json")
