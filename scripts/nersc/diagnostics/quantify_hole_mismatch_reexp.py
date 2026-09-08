"""Quantify train/inference mismatch on n_holes / n_seq_holes (NERSC version).

Training (expansion.py compute_branch_history): hole iff n_window == 0,
over ALL trace states including passive-material surfaces.

C++ inference:
  - gate walk: isHole() states only (sensitive); material states skipped,
    do NOT break a consecutive-hole run.
  - value stopper: nHoles() (sensitive); seq walk `if (!isHole()) break`,
    so a material state TRUNCATES the run.

Usage: python quantify_hole_mismatch_nersc.py <event_id> [<event_id> ...]
"""

import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PARQUET_DIR = "/pscratch/sd/m/mussonm/cckf/reexpanded"


def analyze_event(event_id: int) -> dict:
    path = f"{PARQUET_DIR}/expanded_event{event_id:09d}.parquet"
    cols = ["branch_id", "step_k", "volume_id", "n_window",
            "n_holes", "n_seq_holes"]
    df = pq.read_table(path, columns=cols).to_pandas()

    st = df.drop_duplicates(subset=["branch_id", "step_k"]).sort_values(
        ["branch_id", "step_k"]).reset_index(drop=True)
    del df

    sens_vols = set(st.loc[st["n_window"] > 0, "volume_id"].unique())
    all_vols = st["volume_id"].value_counts().to_dict()
    passive_vols = sorted(set(all_vols) - sens_vols)

    b = st["branch_id"].to_numpy()
    new_branch = np.empty(len(st), dtype=bool)
    new_branch[0] = True
    new_branch[1:] = b[1:] != b[:-1]

    hole_train = (st["n_window"] == 0).to_numpy()
    is_passive = st["volume_id"].isin(passive_vols).to_numpy()
    hole_sens = hole_train & ~is_passive

    def running_counts(hole, skip=None, truncate_seq_at_skip=False):
        n = len(hole)
        nh = np.zeros(n, dtype=np.int64)
        seq = np.zeros(n, dtype=np.int64)
        run_h = run_s = 0
        for i in range(n):
            if new_branch[i]:
                run_h = run_s = 0
            nh[i] = run_h
            seq[i] = run_s
            if skip is not None and skip[i]:
                if truncate_seq_at_skip:
                    run_s = 0
                continue
            if hole[i]:
                run_h += 1
                run_s += 1
            else:
                run_s = 0
        return nh, seq

    nh_train, seq_train = running_counts(hole_train)
    nh_gate, seq_gate = running_counts(hole_sens, skip=is_passive)
    _, seq_value = running_counts(hole_sens, skip=is_passive,
                                  truncate_seq_at_skip=True)

    stored_nh = st["n_holes"].to_numpy()
    stored_seq = st["n_seq_holes"].to_numpy()

    d_nh = nh_train - nh_gate
    d_seq_gate = seq_train - seq_gate
    d_seq_value = seq_train - seq_value

    cand = st["n_window"].to_numpy() > 0
    allm = np.ones(len(st), bool)

    def stats(x, mask):
        v = x[mask]
        if len(v) == 0:
            return {}
        qs = np.percentile(v, [50, 75, 90, 99])
        return {"mean": round(float(v.mean()), 3),
                "p50": float(qs[0]), "p75": float(qs[1]),
                "p90": float(qs[2]), "p99": float(qs[3]),
                "max": int(v.max()),
                "frac_nonzero": round(float(np.mean(v > 0)), 4)}

    prof = pd.DataFrame({"step_k": st["step_k"].to_numpy(), "d_nh": d_nh})[cand]
    step_prof = prof.groupby("step_k")["d_nh"].mean()
    step_prof = {int(k): round(float(v), 2)
                 for k, v in step_prof.items() if k <= 40}

    return {
        "event_id": event_id,
        "n_states": int(len(st)),
        "n_branches": int(new_branch.sum()),
        "passive_vols": {int(v): int(all_vols[v]) for v in passive_vols},
        "sens_vols": sorted(int(v) for v in sens_vols),
        "recompute_match_nh": round(float(np.mean(nh_train == stored_nh)), 4),
        "recompute_match_seq": round(float(np.mean(seq_train == stored_seq)), 4),
        "frac_states_passive": round(float(is_passive.mean()), 4),
        "cand_states": int(cand.sum()),
        "nh_train_at_cand": stats(nh_train, cand),
        "nh_gate_at_cand": stats(nh_gate, cand),
        "d_nh_at_cand": stats(d_nh, cand),
        "seq_train_at_cand": stats(seq_train, cand),
        "seq_gate_at_cand": stats(seq_gate, cand),
        "d_seq_gate_at_cand": stats(d_seq_gate, cand),
        "d_nh_at_all": stats(d_nh, allm),
        "d_seq_value_at_all": stats(d_seq_value, allm),
        "seq_train_at_all": stats(seq_train, allm),
        "seq_value_at_all": stats(seq_value, allm),
        "step_profile_d_nh": step_prof,
    }


if __name__ == "__main__":
    events = [int(a) for a in sys.argv[1:]]
    out = []
    for ev in events:
        r = analyze_event(ev)
        out.append(r)
        print(json.dumps(r), flush=True)
    with open("hole_mismatch_results.json", "w") as f:
        json.dump(out, f, indent=1)
