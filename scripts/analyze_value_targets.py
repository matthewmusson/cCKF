"""Analyze V^{pi-dagger} target distributions across detector regions.

Produces binned statistics for:
  1. Mean vstar_t1/t2 vs eta, stratified by seed purity
  2. Mean vstar_t1/t2 vs occupancy (n_window), stratified by seed purity
  3. Mean vstar_t1/t2 vs step_k

Seed purity categories:
  - "pure"     : 3/3 seed hits from the majority particle
  - "majority" : 2/3 seed hits from the majority particle

Runs on Modal. Outputs a compressed npz to --out-path on the volume.

Usage
-----
    python scripts/analyze_value_targets.py \\
        --parquet-dir /data/results/train32/selected \\
        --out-path /data/analysis/value_target_distributions.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cckf import features as feat, labels as lab, splits, value_target
from cckf.event_selection import resolve_requested_events
from cckf.seed_purity import classify_seed_purity


def _bin_means(values_t1, values_t2, bin_var, edges):
    """Compute mean vstar_t1/t2 in each bin, plus counts."""
    idx = np.digitize(bin_var, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    n_bins = len(edges) - 1
    sum_t1 = np.zeros(n_bins)
    sum_t2 = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    np.add.at(sum_t1, idx, values_t1)
    np.add.at(sum_t2, idx, values_t2)
    np.add.at(counts, idx, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_t1 = np.where(counts > 0, sum_t1 / counts, np.nan)
        mean_t2 = np.where(counts > 0, sum_t2 / counts, np.nan)
    return mean_t1, mean_t2, counts


def process_event(parquet_path: Path, csv_dir: str, event_id: int) -> pd.DataFrame:
    """Load one event, compute value targets, return per-state frame with metadata."""
    from expansion import load_simhits

    needed = [
        "seed_id", "branch_id", "step_k", "cand_hit_id", "is_ckf_selected",
        "contrib_pids", "branch_majority_pid", "majority_undefined",
        "majority_true_hit_on_surface", "state_theta", "n_window",
        "volume_id", "layer_id", "surface_id",
    ]

    table = pq.read_table(parquet_path, columns=needed)
    derived = lab.derive_labels(table)
    df = table.to_pandas()
    df["label_same_particle"] = derived["label_same_particle"]

    df = df.loc[~df["majority_undefined"].astype(bool)].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    purity = classify_seed_purity(df)

    step = value_target.build_step_table(df)
    counts = value_target.particle_simhit_counts(load_simhits(csv_dir, event_id))
    targets = value_target.compute_value_targets(step, counts)

    valid = targets["vstar_t2"].notna() & ~targets["tier_invariant_violated"]
    targets = targets.loc[valid].reset_index(drop=True)

    state_meta = (
        df.groupby(["seed_id", "branch_id", "step_k"], as_index=False)
        .agg(
            state_theta=("state_theta", "first"),
            n_window=("n_window", "first"),
            volume_id=("volume_id", "first"),
            layer_id=("layer_id", "first"),
        )
    )
    state_meta["eta"] = feat.eta_from_theta(
        state_meta["state_theta"].to_numpy(dtype=np.float64)
    )

    result = targets.merge(state_meta, on=["seed_id", "branch_id", "step_k"], how="left")
    result = result.merge(purity, on=["seed_id", "branch_id"], how="left")
    result["event_id"] = event_id
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--only-events", default="")
    args = parser.parse_args()

    assigned = (*splits.TRAIN_EVENTS, *splits.VAL_EVENTS, *splits.CAL_EVENTS)
    events = resolve_requested_events(args.only_events, assigned)
    splits.assert_not_test(events)

    from cckf import stage1_map

    eta_edges = np.linspace(-4.0, 4.0, 161)
    occ_edges = np.concatenate([np.arange(0, 50, 1), np.arange(50, 110, 5)])
    step_edges = np.arange(-0.5, 40.5, 1.0)

    accumulators = {}
    for cat in ("all", "pure", "majority"):
        for xvar in ("eta", "occ", "step"):
            edges = {"eta": eta_edges, "occ": occ_edges, "step": step_edges}[xvar]
            n = len(edges) - 1
            accumulators[(cat, xvar)] = {
                "sum_t1": np.zeros(n),
                "sum_t2": np.zeros(n),
                "counts": np.zeros(n, dtype=np.int64),
            }

    vol_layer_records = []

    for event_id in sorted(events):
        path = Path(args.parquet_dir) / f"expanded_event{event_id:09d}.parquet"
        csv_dir = stage1_map.csv_dir_for(event_id)
        print(f"processing event {event_id}...")
        result = process_event(path, csv_dir, event_id)
        if result.empty:
            continue

        vt1 = result["vstar_t1"].to_numpy(dtype=np.float64)
        vt2 = result["vstar_t2"].to_numpy(dtype=np.float64)
        eta = result["eta"].to_numpy(dtype=np.float64)
        occ = result["n_window"].to_numpy(dtype=np.float64)
        step = result["step_k"].to_numpy(dtype=np.float64)
        purity = result["seed_purity"].to_numpy()
        pure_mask = purity == "pure"
        maj_mask = purity == "majority"

        for cat, mask in [("all", np.ones(len(vt1), dtype=bool)),
                          ("pure", pure_mask), ("majority", maj_mask)]:
            for xvar, vals, edges in [("eta", eta, eta_edges),
                                      ("occ", occ, occ_edges),
                                      ("step", step, step_edges)]:
                v = vals[mask]
                t1 = vt1[mask]
                t2 = vt2[mask]
                if len(v) == 0:
                    continue
                idx = np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2)
                np.add.at(accumulators[(cat, xvar)]["sum_t1"], idx, t1)
                np.add.at(accumulators[(cat, xvar)]["sum_t2"], idx, t2)
                np.add.at(accumulators[(cat, xvar)]["counts"], idx, 1)

        vl = (
            result.groupby(["volume_id", "layer_id"], as_index=False)
            .agg(
                mean_t1=("vstar_t1", "mean"),
                mean_t2=("vstar_t2", "mean"),
                count=("vstar_t2", "size"),
                mean_eta=("eta", "mean"),
                mean_occ=("n_window", "mean"),
            )
        )
        vl["event_id"] = event_id
        vol_layer_records.append(vl)

        n_states = len(result)
        n_pure = int(pure_mask.sum())
        n_maj = int(maj_mask.sum())
        print(
            f"  event {event_id}: {n_states:,} states "
            f"({n_pure:,} pure, {n_maj:,} majority)"
        )

    save_dict = {}
    for (cat, xvar), acc in accumulators.items():
        edges = {"eta": eta_edges, "occ": occ_edges, "step": step_edges}[xvar]
        counts = acc["counts"]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_t1 = np.where(counts > 0, acc["sum_t1"] / counts, np.nan)
            mean_t2 = np.where(counts > 0, acc["sum_t2"] / counts, np.nan)
        save_dict[f"{cat}_{xvar}_edges"] = edges
        save_dict[f"{cat}_{xvar}_mean_t1"] = mean_t1
        save_dict[f"{cat}_{xvar}_mean_t2"] = mean_t2
        save_dict[f"{cat}_{xvar}_counts"] = counts

    if vol_layer_records:
        vl_all = pd.concat(vol_layer_records, ignore_index=True)
        vl_agg = (
            vl_all.groupby(["volume_id", "layer_id"], as_index=False)
            .apply(lambda g: pd.Series({
                "mean_t1": np.average(g["mean_t1"], weights=g["count"]),
                "mean_t2": np.average(g["mean_t2"], weights=g["count"]),
                "total_count": g["count"].sum(),
                "mean_eta": np.average(g["mean_eta"], weights=g["count"]),
                "mean_occ": np.average(g["mean_occ"], weights=g["count"]),
            }))
        )
        save_dict["vol_layer_volume_id"] = vl_agg["volume_id"].to_numpy()
        save_dict["vol_layer_layer_id"] = vl_agg["layer_id"].to_numpy()
        save_dict["vol_layer_mean_t1"] = vl_agg["mean_t1"].to_numpy()
        save_dict["vol_layer_mean_t2"] = vl_agg["mean_t2"].to_numpy()
        save_dict["vol_layer_count"] = vl_agg["total_count"].to_numpy()
        save_dict["vol_layer_mean_eta"] = vl_agg["mean_eta"].to_numpy()
        save_dict["vol_layer_mean_occ"] = vl_agg["mean_occ"].to_numpy()

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **save_dict)
    print(f"saved to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
