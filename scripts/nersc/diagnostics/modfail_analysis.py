"""Module failure + window failure stratified by occupancy x sensor.

Per event, over the CKF-selected branch of each seed (winfail_branches join),
majority-defined states only:

  module failure : ~majority_true_hit_on_surface (simhit-level truth)
  window failure : among true-hit-in-10-box candidate rows (label_same),
                   d = max(|r0|/sqrt(S00), |r1|/sqrt(S11)) > n  (1D: r0 leg only)

Stratified by eta (40 bins), sensor (strip/pixel), occupancy
(geometric_density fixed edges). No truth pT cut (branch majority particles).
"""
import sys
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd

EV = int(sys.argv[1])
S = sys.argv[2]  # scratch base
OUT = f"{S}/modfail_v1/modfail_event{EV:03d}.npz"

ETA_BINS = np.linspace(-4, 4, 41)
N_ETA = 40
N_VALUES = [3, 5, 7, 10]
OCC_EDGES = [0, 2, 5, 12, np.inf]  # geometric_density strata
N_OCC = 4
N_SENSOR = 2  # 0=strip 1=pixel

sel = pd.read_parquet(f"{S}/winfail_branches/branches_event{EV:09d}.parquet",
                      columns=["seed_id", "branch_id"])
sel_keys = set(zip(sel.seed_id.astype(np.int64), sel.branch_id.astype(np.int64)))

cols = ["seed_id", "branch_id", "step_k", "cand_hit_id", "state_theta",
        "is_pixel", "geometric_density", "majority_undefined",
        "majority_true_hit_on_surface", "branch_majority_pid",
        "contrib_pids", "residual_l0", "residual_l1", "S00", "S11", "is_1d"]

pf = pq.ParquetFile(f"{S}/reexpanded/expanded_event{EV:09d}.parquet")

mod_total = np.zeros((N_ETA, N_SENSOR, N_OCC), dtype=np.int64)
mod_fail = np.zeros((N_ETA, N_SENSOR, N_OCC), dtype=np.int64)
win_total = np.zeros((N_ETA, N_SENSOR, N_OCC), dtype=np.int64)
win_fail = np.zeros((len(N_VALUES), N_ETA, N_SENSOR, N_OCC), dtype=np.int64)

for batch in pf.iter_batches(batch_size=2_000_000, columns=cols):
    df = batch.to_pandas()
    key = list(zip(df.seed_id.astype(np.int64), df.branch_id.astype(np.int64)))
    on_sel = np.fromiter((k in sel_keys for k in key), dtype=bool, count=len(df))
    df = df[on_sel & (~df.majority_undefined.astype(bool))]
    if not len(df):
        continue
    theta = np.clip(df.state_theta.to_numpy() / 2.0, 1e-10, np.pi - 1e-10)
    eta = -np.log(np.tan(theta))
    ok = np.isfinite(eta)
    df = df[ok]; eta = eta[ok]
    ei = np.clip(np.digitize(eta, ETA_BINS) - 1, 0, N_ETA - 1)
    si = np.nan_to_num(df.is_pixel.to_numpy(), nan=0.0).astype(int)
    oi = np.clip(np.digitize(np.nan_to_num(df.geometric_density.to_numpy(),
                 nan=0.0), OCC_EDGES) - 1, 0, N_OCC - 1)

    # ---- module failure: one entry per STATE (dedupe on seed,branch,step)
    stf = pd.DataFrame({"seed": df.seed_id.to_numpy(), "br": df.branch_id.to_numpy(),
                        "st": df.step_k.to_numpy(), "ei": ei, "si": si, "oi": oi,
                        "on_mod": df.majority_true_hit_on_surface.to_numpy()})
    stf = stf.drop_duplicates(subset=["seed", "br", "st"])
    np.add.at(mod_total, (stf.ei, stf.si, stf.oi), 1)
    fail_mask = ~stf.on_mod.astype(bool)
    np.add.at(mod_fail, (stf.ei[fail_mask], stf.si[fail_mask], stf.oi[fail_mask]), 1)

    # ---- window failure: true-hit candidate rows only
    cand = df[df.cand_hit_id.to_numpy() >= 0]
    if not len(cand):
        continue
    ce, cs, co = ei[df.cand_hit_id.to_numpy() >= 0], si[df.cand_hit_id.to_numpy() >= 0], oi[df.cand_hit_id.to_numpy() >= 0]
    maj = cand.branch_majority_pid.to_numpy()
    label = np.fromiter(
        (m in (c if c is not None else []) for m, c in zip(maj, cand.contrib_pids)),
        dtype=bool, count=len(cand))
    t = cand[label]; te, ts, to = ce[label], cs[label], co[label]
    d0 = np.abs(t.residual_l0.to_numpy()) / np.sqrt(t.S00.to_numpy())
    is1d = t.is_1d.to_numpy().astype(bool)
    with np.errstate(invalid="ignore"):
        d1 = np.abs(t.residual_l1.to_numpy()) / np.sqrt(t.S11.to_numpy())
    d = np.where(is1d, d0, np.maximum(d0, np.nan_to_num(d1, nan=np.inf)))
    good = np.isfinite(d)
    t_e, t_s, t_o, d = te[good], ts[good], to[good], d[good]
    np.add.at(win_total, (t_e, t_s, t_o), 1)
    for ni, n in enumerate(N_VALUES):
        f = d > n
        np.add.at(win_fail, (np.full(f.sum(), ni), t_e[f], t_s[f], t_o[f]), 1)

import os
os.makedirs(f"{S}/modfail_v1", exist_ok=True)
np.savez(OUT, mod_total=mod_total, mod_fail=mod_fail,
         win_total=win_total, win_fail=win_fail,
         eta_bins=ETA_BINS, n_values=N_VALUES, occ_edges=OCC_EDGES[:-1])
print("wrote", OUT, "| states:", mod_total.sum(), "| modfail:", mod_fail.sum(),
      "| truehit rows:", win_total.sum())
