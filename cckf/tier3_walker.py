"""Tier-3 walker: classify branch states for truth-greedy rollout.

Per docs/superpowers/specs/2026-08-25-tier3-rollout-design.md. Walks each
branch tip -> seed and marks every state:

  collapse   : branch's next action == pi-dagger's choice -> V target reuses
               the child's rollout, zero propagation
  divergence : actions differ -> fresh C++ rollout needed from this state
  tip        : last logged state -> fresh rollout (pi-dagger continues to
               detector exit)

Only divergence + tip states enter the C++ worklist. Reuse is licensed by
state equality, certified by hit-IDENTITY equality (not is-truth equality):
same parent state + same hit -> bit-identical filtered child.
"""

from __future__ import annotations

import argparse
import sys

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_COLS = [
    "seed_id", "step_k", "cand_hit_id", "is_ckf_selected", "chi2_inc",
    "contrib_pids", "branch_majority_pid", "majority_undefined",
    "action_taken", "volume_id", "layer_id",
]


def pi_dagger_pick(cands: pd.DataFrame) -> int:
    """pi-dagger's choice among the truth candidates of ONE state. Breaks the rare instance of a 
    tie by choosing the measurement with the lowest chi^2 among those with the majority particle contributing. 

    Parameters
    ----------
    cands : rows of one (seed_id, step_k) with is_truth == True.
    """
    idx = np.lexsort((cands["cand_hit_id"].to_numpy(),
                      cands["chi2_inc"].to_numpy()))
    return int(cands["cand_hit_id"].to_numpy()[idx[0]])


def classify_event(parquet_path: str) -> pd.DataFrame:
    """Classify every (seed_id, step_k) state of one event.

    Returns a per-state frame: seed_id, step_k, state_class in
    {collapse, divergence, tip}, plus sel_hit / truth_pick for audit.
    """
    tbl = pq.read_table(parquet_path, columns=_COLS)
    df = tbl.drop_columns(["contrib_pids"]).to_pandas()

    # is_truth per candidate row: branch majority pid among contributors.
    # awkward broadcast, no python loop over 46M lists. The list column goes
    # through arrow -> awkward directly; a pandas object column of ndarrays
    # is not a valid ak.to_layout input.
    contribs = ak.from_arrow(tbl.column("contrib_pids").combine_chunks())
    majority = df["branch_majority_pid"].to_numpy()
    is_truth = ak.to_numpy(ak.any(contribs == majority[:, None], axis=1))
    is_truth &= ~df["majority_undefined"].to_numpy(dtype=bool)
    is_truth &= df["cand_hit_id"].to_numpy() >= 0
    df["is_truth"] = is_truth

    # Per-state reductions.
    g = df.groupby(["seed_id", "step_k"], sort=False)
    st = g.agg(
        n_truth=("is_truth", "sum"),
        volume_id=("volume_id", "first"),
        layer_id=("layer_id", "first"),
        sel_hit=("cand_hit_id",
                 lambda s: -1),  # placeholder, filled vectorized below
    ).reset_index()

    sel = df.loc[df["is_ckf_selected"],
                 ["seed_id", "step_k", "cand_hit_id"]].rename(
                     columns={"cand_hit_id": "sel_hit_real"})
    st = st.merge(sel, on=["seed_id", "step_k"], how="left")
    st["sel_hit"] = st["sel_hit_real"].fillna(-1).astype(np.int64)
    st = st.drop(columns=["sel_hit_real"])

    # pi-dagger pick per state, vectorized. THIS IS THE SAME RULE AS
    # pi_dagger_pick above (lowest chi2_inc, ties to lowest cand_hit_id);
    # the sort keys ARE the rule. If either site changes, change both --
    # tests/test_tier3_walker.py::test_pick_rule_sites_agree fails if they
    # diverge.
    tdf = df.loc[df["is_truth"],
                 ["seed_id", "step_k", "cand_hit_id", "chi2_inc"]]
    tdf = tdf.sort_values(["seed_id", "step_k", "chi2_inc", "cand_hit_id"])
    picks = tdf.groupby(["seed_id", "step_k"], sort=False).first().reset_index()
    picks = picks.rename(columns={"cand_hit_id": "truth_pick"})
    st = st.merge(picks[["seed_id", "step_k", "truth_pick"]],
                  on=["seed_id", "step_k"], how="left")
    st["truth_pick"] = st["truth_pick"].fillna(-1).astype(np.int64)

    # Branch's action at the NEXT logged state vs pi-dagger's choice there.
    st = st.sort_values(["seed_id", "step_k"]).reset_index(drop=True)
    same_branch_next = st["seed_id"].shift(-1) == st["seed_id"]
    next_sel = st["sel_hit"].shift(-1).fillna(-1).astype(np.int64)
    next_pick = st["truth_pick"].shift(-1).fillna(-1).astype(np.int64)

    # Agreement: same hit taken, or both hole (no selected hit AND no truth
    # hit available). Any other combination is a divergence.
    agree = (next_sel == next_pick) & same_branch_next
    st["state_class"] = np.where(
        ~same_branch_next, "tip", np.where(agree, "collapse", "divergence"))
    return st


_FLT_BRANCHES = [
    "eLOC0_flt", "eLOC1_flt", "ePHI_flt", "eTHETA_flt", "eQOP_flt", "eT_flt",
    "err_eLOC0_flt", "err_eLOC1_flt", "err_ePHI_flt", "err_eTHETA_flt",
    "err_eQOP_flt", "err_eT_flt",
]

#: Predicted fallbacks, same order. A hole state has no measurement update,
#: so its filtered state IS the predicted state -- coalescing flt-else-prt
#: is exact, not an approximation. Without it 60% of divergence/tip states
#: (those whose action was a hole) have NaN parameters and the worklist
#: guard refuses to emit.
_PRT_BRANCHES = [
    "eLOC0_prt", "eLOC1_prt", "ePHI_prt", "eTHETA_prt", "eQOP_prt", "eT_prt",
    "err_eLOC0_prt", "err_eLOC1_prt", "err_ePHI_prt", "err_eTHETA_prt",
    "err_eQOP_prt", "err_eT_prt",
]


def emit_worklist(st: pd.DataFrame, trackstates_root: str, event_id: int,
                  detectors_csv: str, parquet_path: str,
                  out_csv: str) -> int:
    """Write the C++ executor's worklist: divergence + tip states with their
    FILTERED parameters and variance diagonals from the trackstates ROOT.

    Joins on the proven (seed_id, all-state state_idx) key. geometry_id is
    translated to the TRUE surface id via detectors.csv -- the walker's
    (vol, lay, sen) triple has extra=0, but real endcap surface ids carry
    the ring index in the extra byte, and TrackingGeometry::findSurface
    needs the real id (the extra-byte bug in reverse).
    """
    import uproot

    with uproot.open(trackstates_root) as fh:
        tree = fh["trackstates"]
        fields = _FLT_BRANCHES + _PRT_BRANCHES + [
            "volume_id", "layer_id", "module_id"]
        if "event_nr" in set(tree.keys()):
            fields = fields + ["event_nr"]
        arrays = tree.arrays(fields, library="ak")
    if "event_nr" in arrays.fields:
        arrays = arrays[ak.to_numpy(arrays["event_nr"]) == int(event_id)]

    n_states = ak.to_numpy(ak.num(arrays["volume_id"], axis=1))
    seed_id = np.repeat(np.arange(len(arrays), dtype=np.int64), n_states)
    state_idx = ak.to_numpy(
        ak.flatten(ak.local_index(arrays["volume_id"], axis=1), axis=1)
    ).astype(np.int64)
    root = pd.DataFrame({"seed_id": seed_id, "step_k": state_idx})
    for b in _FLT_BRANCHES + _PRT_BRANCHES + [
            "volume_id", "layer_id", "module_id"]:
        root[b] = ak.to_numpy(
            ak.flatten(ak.fill_none(arrays[b], np.nan), axis=1))
    for flt, prt in zip(_FLT_BRANCHES, _PRT_BRANCHES):
        root[flt] = np.where(
            np.isfinite(root[flt].to_numpy()), root[flt], root[prt])

    det = pd.read_csv(detectors_csv)
    gcol = [c for c in det.columns if "geometry" in c][0]
    g = det[gcol].astype(np.uint64).to_numpy()
    trip = pd.DataFrame({
        "volume_id": (g >> 56) & 0xFF,
        "layer_id": (g >> 36) & 0xFFF,
        "module_id": (g >> 8) & 0xFFFFF,
        "true_gid": g.astype(np.int64),
    })

    work = st.loc[st["state_class"].isin(["divergence", "tip"]),
                  ["seed_id", "step_k"]].copy()
    work = work.merge(root, on=["seed_id", "step_k"], how="left")
    work = work.merge(trip, on=["volume_id", "layer_id", "module_id"],
                      how="left")

    # Rollout starts on PASSIVE surfaces (module_id == 0: layer/approach
    # material surfaces, absent from detectors.csv) are launched from the
    # branch's most recent SENSITIVE state instead. Equivalent by
    # construction: between that sensor and the passive surface there are no
    # decision points (a sensitive surface there would itself be a logged
    # state), so the propagator carries the same filtered state through the
    # same material and reaches the first decision surface identically.
    # 60.03% of event-1 starts are such states; without this they were all
    # refused by the guard.
    par_cols = _FLT_BRANCHES + ["true_gid"]
    donor = root.merge(trip, on=["volume_id", "layer_id", "module_id"],
                       how="left")
    donor = donor.sort_values(["seed_id", "step_k"])
    sens = donor["true_gid"].notna()
    for c in par_cols:
        donor[f"_ff_{c}"] = donor[c].where(sens)
        donor[f"_ff_{c}"] = donor.groupby("seed_id")[f"_ff_{c}"].ffill()
    ff = donor[["seed_id", "step_k"] + [f"_ff_{c}" for c in par_cols]]
    work = work.merge(ff, on=["seed_id", "step_k"], how="left")
    passive = work["true_gid"].isna()
    for c in par_cols:
        work.loc[passive, c] = work.loc[passive, f"_ff_{c}"]

    maj = pq.read_table(
        parquet_path, columns=["seed_id", "branch_majority_pid"]
    ).to_pandas().drop_duplicates("seed_id")
    work = work.merge(maj, on="seed_id", how="left")

    n0 = len(work)
    bad_par = ~np.isfinite(work["eLOC0_flt"].to_numpy(dtype=np.float64))
    bad_gid = work["true_gid"].isna().to_numpy()
    bad_join = work["volume_id"].isna().to_numpy()  # ROOT merge missed entirely
    keep = ~(bad_par | bad_gid)
    if keep.sum() < 0.95 * n0:
        sample = work.loc[bad_par | bad_gid,
                          ["seed_id", "step_k", "volume_id", "layer_id",
                           "module_id", "eLOC0_flt", "true_gid"]].head(5)
        raise ValueError(
            f"worklist join lost {n0 - int(keep.sum()):,} of {n0:,} states "
            f"({1 - keep.sum()/n0:.2%}). Breakdown: root-join-miss "
            f"{int(bad_join.sum()):,}, param-NaN {int(bad_par.sum()):,}, "
            f"gid-miss {int(bad_gid.sum()):,}. Sample lost rows:\n"
            f"{sample.to_string()}")
    work = work.loc[keep].reset_index(drop=True)

    out = pd.DataFrame({
        "rollout_id": np.arange(len(work), dtype=np.int64),
        "seed_id": work["seed_id"],
        "step_k": work["step_k"],
        "geometry_id": work["true_gid"].astype(np.int64),
    })
    for i, b in enumerate(
            ["eLOC0_flt", "eLOC1_flt", "ePHI_flt", "eTHETA_flt",
             "eQOP_flt", "eT_flt"]):
        out[f"par{i}"] = work[b]
    for i, b in enumerate(
            ["err_eLOC0_flt", "err_eLOC1_flt", "err_ePHI_flt",
             "err_eTHETA_flt", "err_eQOP_flt", "err_eT_flt"]):
        out[f"var{i}"] = work[b].to_numpy() ** 2
    out["majority_pid"] = work["branch_majority_pid"].fillna(0).astype(
        np.uint64)
    out.to_csv(out_csv, index=False)
    return len(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", required=True)
    p.add_argument("--trackstates-root", default="")
    p.add_argument("--event-id", type=int, default=-1)
    p.add_argument("--detectors-csv", default="")
    p.add_argument("--worklist-out", default="")
    args = p.parse_args()

    st = classify_event(args.parquet)
    n = len(st)
    counts = st["state_class"].value_counts()
    n_rollouts = int(counts.get("divergence", 0) + counts.get("tip", 0))
    multi_truth = int((st["n_truth"] > 1).sum())
    print(f"states={n:,}")
    for k in ("collapse", "divergence", "tip"):
        c = int(counts.get(k, 0))
        print(f"  {k:10s} {c:>12,}  ({c / n:.2%})")
    print(f"rollouts needed: {n_rollouts:,}  "
          f"(vs {n:,} naive; saving {1 - n_rollouts / n:.2%})")
    print(f"multi-truth states (tie-break exercised): {multi_truth:,} "
          f"({multi_truth / n:.4%})")
    if args.worklist_out:
        n_rows = emit_worklist(st, args.trackstates_root, args.event_id,
                               args.detectors_csv, args.parquet,
                               args.worklist_out)
        print(f"worklist: {n_rows:,} rows -> {args.worklist_out}")
    # Where do branches end? A tip on the outermost long-strip layer is at
    # the detector edge: pi-dagger has nowhere to continue and the rollout
    # is length zero, so it should not count toward propagation cost.
    tips = st.loc[st["state_class"] == "tip"]
    print("tip volume distribution:")
    print(tips["volume_id"].value_counts().sort_index().to_string())
    outer = tips.groupby("volume_id")["layer_id"].max()
    at_edge = 0
    for vol in (28, 29, 30):
        if vol in outer.index:
            at_edge += int(((tips["volume_id"] == vol) &
                            (tips["layer_id"] == outer[vol])).sum())
    print(f"tips on outermost long-strip layer (zero-rollout): {at_edge:,} "
          f"({at_edge / max(len(tips),1):.2%} of tips)")


if __name__ == "__main__":
    main()
