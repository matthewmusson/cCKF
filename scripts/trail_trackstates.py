#!/usr/bin/env python3
"""Follow one particle through trackstates_ckf.root and an expanded parquet.

Companion to scripts/trail_edm4hep.py (doc 01) and scripts/trail_csv.py
(doc 02); this one produced the tables in docs/03, 05 and 06. It lists the
CKF tracks that hold at least three states on the particle, prints the
cleanest one state by state in ROOT order with the propagation-order index
beside it, and then prints that branch's rows in the expanded parquet.

Usage
-----
    python scripts/trail_trackstates.py <run_dir> <expanded.parquet> <event> pv sv part gen sub

`run_dir` must contain trackstates_ckf.root. Reading every track of an
event needs tens of GB of memory: run it on a compute node (debug queue,
--mem=0), not a login node. Needs uproot, awkward, pyarrow, pandas and this
repository on PYTHONPATH.

Tier 3 (diagnostic).
"""

from __future__ import annotations

import sys

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot

from expansion import encode_particle_id

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

_STATE_FIELDS = [
    "event_nr", "volume_id", "layer_id", "module_id", "stateType", "pathLength",
    "l_x_hit", "l_y_hit", "chi2", "eLOC0_prt", "eLOC1_prt", "err_eLOC0_prt",
    "err_eLOC1_prt", "eQOP_prt", "pathInX0_interval", "clus_size_u",
    "clus_size_v", "clus_qtot", "alpha_u", "alpha_v",
    "particle_ids_vertex_primary", "particle_ids_vertex_secondary",
    "particle_ids_particle", "particle_ids_generation",
    "particle_ids_sub_particle",
]

_PARQUET_COLS = [
    "seed_id", "step_k", "volume_id", "layer_id", "surface_id", "cand_hit_id",
    "residual_l0", "residual_l1", "S00", "S11", "chi2_inc", "n_window",
    "is_ckf_selected", "sel_l0", "action_taken", "n_hits", "n_holes",
    "n_seq_holes", "contrib_pids", "branch_majority_pid", "majority_undefined",
    "majority_true_hit_on_surface", "clus_s_u", "clus_q_tot", "pitch_u",
    "is_pixel", "is_1d",
]


def main(argv: list[str]) -> int:
    if len(argv) != 9:
        print(__doc__)
        return 2
    run_dir, parquet, event = argv[1], argv[2], int(argv[3])
    pv, sv, part, gen, sub = (int(x) for x in argv[4:9])
    packed = int(
        encode_particle_id(
            np.array([pv]), np.array([sv]), np.array([part]),
            np.array([gen]), np.array([sub]),
        )[0]
    )
    print("packed pid:", packed)

    tree = uproot.open(f"{run_dir}/trackstates_ckf.root")["trackstates"]
    fields = [f for f in _STATE_FIELDS if f in set(tree.keys())]
    print(f"reading {len(fields)} branches over {tree.num_entries} tracks")
    a = tree.arrays(fields, library="ak")
    a = a[ak.to_numpy(a["event_nr"]) == event]
    print(f"tracks in event {event}: {len(a)}")

    ours = (
        (a["particle_ids_particle"] == part)
        & (a["particle_ids_vertex_primary"] == pv)
        & (a["particle_ids_vertex_secondary"] == sv)
        & (a["particle_ids_generation"] == gen)
        & (a["particle_ids_sub_particle"] == sub)
    )
    state_ours = ak.any(ours, axis=2)
    n_ours = ak.to_numpy(ak.sum(state_ours, axis=1))
    has_hit = ~np.isnan(ak.fill_none(a["l_x_hit"], np.nan))
    n_meas = ak.to_numpy(ak.sum(has_hit, axis=1))
    cand = np.where(n_ours >= 3)[0]
    print(f"tracks with >=3 states on our particle: {len(cand)}")
    if len(cand) == 0:
        return 1
    summary = pd.DataFrame(
        {
            "track_nr": cand,
            "n_states": ak.to_numpy(ak.num(a["volume_id"], axis=1))[cand],
            "n_meas": n_meas[cand],
            "n_ours": n_ours[cand],
        }
    )
    summary["purity"] = summary.n_ours / summary.n_meas
    print(summary.sort_values(["n_ours", "purity"], ascending=False).head(12).to_string(index=False))

    best = int(summary.sort_values(["purity", "n_ours"], ascending=False).iloc[0].track_nr)
    print(f"\n=== track {best} (cleanest): ROOT order (index 0 = outermost); step_k = propagation order ===")
    tr = a[best]
    n = len(tr["volume_id"])
    rows = []
    for j in range(n):
        row = {
            "root_idx": j,
            "step_k": n - 1 - j,
            "vol": tr["volume_id"][j],
            "lay": tr["layer_id"][j],
            "mod": tr["module_id"][j],
            "pathLength": round(float(tr["pathLength"][j]), 1),
            "l_x_hit": tr["l_x_hit"][j],
            "eLOC0_prt": tr["eLOC0_prt"][j],
            "err_eLOC0_prt": tr["err_eLOC0_prt"][j],
            "chi2": tr["chi2"][j],
            "qop_prt": tr["eQOP_prt"][j],
            "X0_int": tr["pathInX0_interval"][j],
            "clus_u": tr["clus_size_u"][j],
            "clus_v": tr["clus_size_v"][j],
            "contrib_part": list(tr["particle_ids_particle"][j]),
            "ours": bool(state_ours[best][j]),
        }
        if "stateType" in fields:
            row["stateType"] = tr["stateType"][j]
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== expanded parquet rows for that branch ===")
    tbl = pq.read_table(parquet, columns=_PARQUET_COLS, filters=[("seed_id", "==", best)])
    d = tbl.to_pandas().sort_values(["step_k", "chi2_inc"])
    print(
        f"rows: {len(d)}, states: {d.step_k.nunique()}, "
        f"branch_majority_pid: {d.branch_majority_pid.iloc[0]} (ours = {packed}), "
        f"undefined: {bool(d.majority_undefined.iloc[0])}"
    )
    print(d.to_string(index=False, max_rows=120))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
