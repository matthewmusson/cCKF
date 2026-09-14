#!/usr/bin/env python3
"""Score one branch's candidates with a trained gate, and its states with a
trained value function, straight from an expanded parquet.

Reproduces offline what the C++ does at inference: build the feature
vectors from the parquet columns (cckf.features / build_value_cache), apply
the checkpoint's standardisation, run the MLP, apply the gate's
occupancy-conditional Platt calibration. Used for the worked example in
docs/08 and docs/09.

Usage
-----
    python scripts/diagnostics/score_branch.py <expanded.parquet> <seed_id> \
        --gate <gate_model.pt> --platt <platt_params.json> \
        --value <value_model.pt> [--step-k K]

Runs on a login node: the parquet read is filtered on seed_id. Needs torch,
pyarrow, pandas and this repository on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cckf import features, models  # noqa: E402
from cckf.calibration import apply_platt_occupancy  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def load_mlp(path: str, kind: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    names = list(ckpt["feature_names"])
    cls = models.GateMLP if kind == "gate" else models.ValueMLP
    m = cls(n_features=len(names), width=ckpt.get("width", 128), depth=ckpt.get("depth", 3 if kind == "gate" else 2))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, names, np.asarray(ckpt["mu"], np.float32), np.asarray(ckpt["sigma"], np.float32)


def run(m, X, mu, sigma):
    Z = (X - mu) / np.where(sigma > 1e-30, sigma, 1.0)
    with torch.no_grad():
        return m(torch.from_numpy(Z.astype(np.float32))).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet"); ap.add_argument("seed_id", type=int)
    ap.add_argument("--gate", default=""); ap.add_argument("--platt", default="")
    ap.add_argument("--value", default=""); ap.add_argument("--step-k", type=int, default=None)
    a = ap.parse_args()

    cols = sorted(set(features.GATE_SOURCE_COLUMNS) | set(features.VALUE_SOURCE_COLUMNS)
                  | {"is_ckf_selected", "contrib_pids", "branch_majority_pid", "surface_id", "layer_id", "volume_id"})
    d = pq.read_table(a.parquet, columns=cols, filters=[("seed_id", "==", a.seed_id)]).to_pandas()
    d = d.sort_values(["step_k", "chi2_inc"]).reset_index(drop=True)
    print(f"branch {a.seed_id}: {len(d)} rows, {d.step_k.nunique()} states")

    if a.gate:
        m, names, mu, sigma = load_mlp(a.gate, "gate")
        X = features.build_gate_features(d)
        idx = [features.GATE_FEATURES.index(n) for n in names]
        logit = run(m, X[:, idx], mu, sigma)
        d["gate_logit"] = logit
        if a.platt:
            cal = json.load(open(a.platt))
            cal = cal.get("four_param", cal)
            d["gate_p"] = apply_platt_occupancy(logit, d["n_window"].to_numpy(), (cal["a0"], cal["a1"], cal["b0"], cal["b1"]))
        d["label"] = [int(int(m_) in list(c)) for m_, c in zip(d.branch_majority_pid, d.contrib_pids)]
        sel = d if a.step_k is None else d[d.step_k == a.step_k]
        show = ["step_k", "volume_id", "layer_id", "surface_id", "cand_hit_id", "residual_l0", "residual_l1", "chi2_inc",
                "n_window", "clus_s_u", "clus_s_v", "is_ckf_selected", "label", "gate_logit"] + (["gate_p"] if a.platt else [])
        print("\n=== gate ===")
        print(sel[sel.cand_hit_id >= 0][show].to_string(index=False))
        # the feature vector of the selected candidate at --step-k
        if a.step_k is not None:
            r = np.flatnonzero((d.step_k == a.step_k) & d.is_ckf_selected)
            if len(r):
                print(f"\nfeature vector of the accepted candidate at step {a.step_k} (name: raw -> standardised):")
                z = (X[r[0], idx] - mu) / np.where(sigma > 1e-30, sigma, 1.0)
                for n, raw, zz in zip(names, X[r[0], idx], z):
                    print(f"  {n:18s} {raw:12.5g} -> {zz:+.3f}")

    if a.value:
        from scripts.build_value_cache import _state_features
        m, names, mu, sigma = load_mlp(a.value, "value")
        st = _state_features(d.assign(branch_id=d.seed_id, event_id=0))
        X = st[list(names)].to_numpy(np.float32)
        logit = run(m, X, mu, sigma)
        st["value_p"] = 1.0 / (1.0 + np.exp(-logit))
        print("\n=== value ===")
        print(st[["step_k"] + [n for n in names if n != "step_k"] + ["value_p"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
