"""Build the value-function cache: one row per (branch, surface) with V^{π†}.

Prerequisites
-------------
1. ``is_ckf_selected`` present in the Parquet (``scripts/patch_is_selected.py``).
2. Per-event ``simhits.csv`` reachable, for ``N_total_true``.

Iteration-0 note (spec §3.2): the two accumulated-gate-log-odds features are
built from the χ²-implied log-odds, log(Λ/(1−Λ)) with Λ = exp(−χ²/2), because no
trained gate exists yet. That keeps them on the same probability scale as the
learned gate log-odds that replace them at DAgger iteration 1.

Interim tier-invariant handling (see ``cckf.value_target`` module docstring,
"Known limitation" section)
----------------------------------------------------------------------------
``compute_value_targets`` flags rows where ``vstar_t1 < vstar_t2`` (a surface
revisit double-counted ``maj_hit_on_surface`` badly enough to invert the
documented ``vstar_t1 >= vstar_t2`` invariant) via ``tier_invariant_violated``.
The principled fix — dedupe ``maj_hit_on_surface`` per ``surface_id`` — is
deferred because it changes ``build_step_table``'s input contract. Until then
this script drops flagged rows, counts how many were dropped, and reports the
count and fraction in ``meta.json`` (``n_tier_invariant_dropped``,
``frac_tier_invariant_dropped``) so the size of the deferred problem is
measured rather than assumed.

χ² log-odds saturation check
-----------------------------
``sum_gate_logodds`` / ``min_gate_logodds`` are built from
``cckf.features.chi2_log_odds``, whose Λ = exp(−χ²/2) is clipped at 1e-6 —
saturating at χ² ≳ 27.6 (−2·ln(1e-6) ≈ 27.63). The collection config caps
*accepted* measurements at χ² = 16.26 (``configs/envelope.yaml``), which
should keep every accepted hit's contribution off the clip. This script
measures that on real data instead of assuming it: it records the maximum
accepted χ² and the count of accepted hits above the saturation point
(``max_accepted_chi2``, ``n_accepted_chi2_above_saturation`` in
``meta.json``), and prints a warning if any are found.

Usage
-----
    python scripts/build_value_cache.py --split train \\
        --parquet-dir /data/results/train32/selected \\
        --csv-dir /data/results/train32/csv \\
        --out-dir /data/cache/value/train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cckf import features as feat
from cckf import labels as lab
from cckf import splits, value_target

_NEEDED = (
    "seed_id", "branch_id", "step_k", "cand_hit_id", "is_ckf_selected",
    "contrib_pids", "branch_majority_pid", "majority_undefined",
    "majority_true_hit_on_surface",
    "state_theta", "state_qop", "cov_00", "cov_06",
    "n_hits", "n_holes", "n_seq_holes", "chi2_inc", "pathInX0_interval",
)

#: chi2_log_odds clips Λ at 1e-6, i.e. -2*ln(1e-6) ≈ 27.63; round down to 27.6
#: so this check does not itself hide a value right at the boundary.
_CHI2_SATURATION_THRESHOLD = 27.6


def _state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce candidate rows to per-state value features.

    ``sum_gate_logodds`` and ``min_gate_logodds`` accumulate over the branch's
    *accepted* hits only — a rejected candidate never entered the state, so its
    score says nothing about the branch's quality.
    """
    df = df.copy()
    df["gate_logodds"] = feat.chi2_log_odds(df["chi2_inc"].to_numpy(dtype=np.float64))
    sel = df["is_ckf_selected"].to_numpy(dtype=bool) & (
        df["cand_hit_id"].to_numpy(dtype=np.int64) != -1
    )
    df["sel_logodds"] = np.where(sel, df["gate_logodds"], np.nan)

    per_state = df.groupby(["seed_id", "branch_id", "step_k"], as_index=False).agg(
        state_theta=("state_theta", "first"),
        state_qop=("state_qop", "first"),
        sigma2_l0=("cov_00", "first"),
        sigma2_l1=("cov_06", "first"),
        n_hits=("n_hits", "first"),
        n_holes=("n_holes", "first"),
        n_seq_holes=("n_seq_holes", "first"),
        pathInX0_interval=("pathInX0_interval", "first"),
        step_logodds=("sel_logodds", "max"),
    )
    per_state = per_state.sort_values(["seed_id", "branch_id", "step_k"]).reset_index(drop=True)

    grp = per_state.groupby(["seed_id", "branch_id"], sort=False)
    filled = per_state["step_logodds"].fillna(0.0)
    per_state["sum_gate_logodds"] = filled.groupby(
        [per_state["seed_id"], per_state["branch_id"]]
    ).cumsum().to_numpy()
    grouped_min = per_state["step_logodds"].groupby(
        [per_state["seed_id"], per_state["branch_id"]]
    ).cummin()
    # Forward-fill WITHIN each branch so a hole step reports the running worst
    # accepted score rather than resetting it. cummin leaves the NaN row's own
    # output NaN; filling before the ffill (as `.fillna(0.0)` alone does) would
    # overwrite it with 0.0 log-odds = p 0.500, i.e. "the worst hit so far was a
    # coin flip" -- optimistically wrong at exactly the hole states V_phi must
    # judge. The trailing fillna(0.0) then covers only leading holes, before any
    # hit has been accepted and a minimum is genuinely undefined.
    per_state["min_gate_logodds"] = (
        grouped_min.groupby(
            [per_state["seed_id"], per_state["branch_id"]]
        ).ffill().fillna(0.0).to_numpy()
    )
    per_state["x0_accumulated"] = grp["pathInX0_interval"].cumsum().to_numpy()
    per_state["eta"] = feat.eta_from_theta(per_state["state_theta"].to_numpy(dtype=np.float64))
    return per_state


def process_event(
    parquet_path: Path, csv_dir: str, event_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return ``(X, y, aux, event_meta)`` for one event.

    ``X`` is ``(n_states, 11)`` float32 in :data:`cckf.features.VALUE_FEATURES`
    order, ``y`` is the Tier-2 soft target, and ``aux`` is
    ``(vstar_t1, step_k, eta)`` — Tier 1 for the tier-gap diagnostic and the
    last two for stratified plots.

    ``event_meta`` carries two interim diagnostics (see module docstring):
    ``n_tier_invariant_dropped`` / ``n_valid_target`` for the tier-invariant
    drop rate, and ``max_accepted_chi2`` / ``n_accepted_chi2_above_saturation``
    for the χ² log-odds saturation check. Both are measured over this event
    only; :func:`main` aggregates across events.
    """
    from expansion import load_simhits

    available = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    missing = set(_NEEDED) - available
    if missing:
        raise ValueError(f"{parquet_path} missing columns: {sorted(missing)}")

    table = pq.read_table(parquet_path, columns=list(_NEEDED))
    derived = lab.derive_labels(table)
    df = table.to_pandas()
    df["label_same_particle"] = derived["label_same_particle"]

    # --- Required addition 2: χ² log-odds saturation check ---------------
    # Measured over selected/accepted hits only, before the majority_undefined
    # filter below (this is a question about the gate's chi2 scale on this
    # event's Parquet, not about which branches have a defined majority).
    accepted_mask = df["is_ckf_selected"].to_numpy(dtype=bool) & (
        df["cand_hit_id"].to_numpy(dtype=np.int64) != -1
    )
    accepted_chi2 = df.loc[accepted_mask, "chi2_inc"].to_numpy(dtype=np.float64)
    accepted_chi2 = accepted_chi2[np.isfinite(accepted_chi2)]
    if accepted_chi2.size:
        max_accepted_chi2 = float(accepted_chi2.max())
        n_accepted_chi2_above_saturation = int(
            np.sum(accepted_chi2 > _CHI2_SATURATION_THRESHOLD)
        )
    else:
        max_accepted_chi2 = float("nan")
        n_accepted_chi2_above_saturation = 0
    if n_accepted_chi2_above_saturation > 0:
        print(
            f"WARNING event {event_id}: {n_accepted_chi2_above_saturation} accepted "
            f"hits have chi2 > {_CHI2_SATURATION_THRESHOLD} — chi2_log_odds is "
            f"saturating (max accepted chi2 = {max_accepted_chi2:.3f})"
        )

    event_meta = {
        "max_accepted_chi2": max_accepted_chi2,
        "n_accepted_chi2_above_saturation": n_accepted_chi2_above_saturation,
        "n_tier_invariant_dropped": 0,
        "n_valid_target": 0,
    }

    # A branch with no defined majority has no defined value target.
    df = df.loc[~df["majority_undefined"].astype(bool)].reset_index(drop=True)
    if df.empty:
        return (
            np.empty((0, len(feat.VALUE_FEATURES)), np.float32),
            np.empty(0, np.float32),
            np.empty((0, 3), np.float32),
            event_meta,
        )

    step = value_target.build_step_table(df)
    counts = value_target.particle_simhit_counts(load_simhits(csv_dir, event_id))
    targets = value_target.compute_value_targets(step, counts)

    state_feats = _state_features(df)
    merged = state_feats.merge(
        targets[
            ["seed_id", "branch_id", "step_k", "vstar_t2", "vstar_t1", "tier_invariant_violated"]
        ],
        on=["seed_id", "branch_id", "step_k"],
        how="inner",
    )

    # --- Required addition 1: drop tier-invariant-violated rows ----------
    # Independent of the NaN-target filter below (the flag is False for
    # NaN-target rows), so both filters are needed. `n_valid_target` is the
    # denominator for the drop fraction: rows with a defined target, whether
    # or not they were also flagged.
    valid_target = merged["vstar_t2"].notna().to_numpy()
    tier_violated = merged["tier_invariant_violated"].to_numpy(dtype=bool)
    event_meta["n_valid_target"] = int(valid_target.sum())
    event_meta["n_tier_invariant_dropped"] = int(tier_violated.sum())
    merged = merged.loc[valid_target & ~tier_violated].reset_index(drop=True)

    X = np.column_stack(
        [merged[name].to_numpy(dtype=np.float64) for name in feat.VALUE_FEATURES]
    )
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y = merged["vstar_t2"].to_numpy(dtype=np.float32)
    aux = np.column_stack([
        merged["vstar_t1"].to_numpy(dtype=np.float64),
        merged["step_k"].to_numpy(dtype=np.float64),
        merged["eta"].to_numpy(dtype=np.float64),
    ]).astype(np.float32)
    return X.astype(np.float32), y, aux, event_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train", "val", "cal"])
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    events = splits.events_for(args.split)
    splits.assert_not_test(events)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xs, ys, auxs = [], [], []
    total_tier_dropped = 0
    total_valid_target = 0
    total_chi2_above_saturation = 0
    max_accepted_chi2_overall = float("-inf")
    for event_id in events:
        path = Path(args.parquet_dir) / f"expanded_event{event_id:09d}.parquet"
        X, y, aux, event_meta = process_event(path, args.csv_dir, event_id)
        print(f"event {event_id}: {len(y):,} states")
        xs.append(X)
        ys.append(y)
        auxs.append(aux)
        total_tier_dropped += event_meta["n_tier_invariant_dropped"]
        total_valid_target += event_meta["n_valid_target"]
        total_chi2_above_saturation += event_meta["n_accepted_chi2_above_saturation"]
        if not np.isnan(event_meta["max_accepted_chi2"]):
            max_accepted_chi2_overall = max(
                max_accepted_chi2_overall, event_meta["max_accepted_chi2"]
            )

    X = np.concatenate(xs)
    y = np.concatenate(ys)
    aux = np.concatenate(auxs)

    X.tofile(out_dir / "X.f32")
    y.tofile(out_dir / "y.f32")
    aux.tofile(out_dir / "aux.f32")

    frac_tier_dropped = (
        total_tier_dropped / total_valid_target if total_valid_target > 0 else 0.0
    )
    max_accepted_chi2_report = (
        None if max_accepted_chi2_overall == float("-inf") else max_accepted_chi2_overall
    )

    # Prominent, not buried in the JSON: this fraction is currently
    # unmeasured on real data and quantifies the deferred surface-revisit
    # over-count problem (cckf/value_target.py, "Known limitation").
    print(
        f"tier-invariant drop: {total_tier_dropped:,} / {total_valid_target:,} "
        f"valid-target rows dropped ({frac_tier_dropped:.4%}) — "
        f"vstar_t1 < vstar_t2 from surface-revisit over-count"
    )
    if total_chi2_above_saturation > 0:
        print(
            f"WARNING: {total_chi2_above_saturation} accepted hits across all "
            f"events exceed chi2={_CHI2_SATURATION_THRESHOLD} "
            f"(chi2_log_odds saturating); max accepted chi2 = "
            f"{max_accepted_chi2_overall:.3f}"
        )
    else:
        shown = f"{max_accepted_chi2_overall:.3f}" if max_accepted_chi2_report is not None else "n/a"
        print(
            f"chi2 log-odds saturation check: no accepted hits exceed "
            f"chi2={_CHI2_SATURATION_THRESHOLD} (max accepted chi2 = {shown})"
        )

    marginal = float(np.mean((y >= 0.2) & (y <= 0.8)))
    meta = {
        "n_rows": int(len(y)),
        "n_features": len(feat.VALUE_FEATURES),
        "feature_names": list(feat.VALUE_FEATURES),
        "aux_columns": ["vstar_t1", "step_k", "eta"],
        "mean_vstar_t2": float(y.mean()),
        "marginal_fraction_0.2_0.8": marginal,
        # Tier-gap diagnostic: how much Tier 2's surface restriction gives up
        # against the optimistic bound. Free, since both tiers are computed
        # together.
        "tier1_minus_tier2_mean": float(np.mean(aux[:, 0] - y)),
        # Interim tier-invariant-violation handling (see module docstring).
        "n_tier_invariant_dropped": int(total_tier_dropped),
        "n_valid_target_before_tier_drop": int(total_valid_target),
        "frac_tier_invariant_dropped": float(frac_tier_dropped),
        # χ² log-odds saturation check (see module docstring).
        "max_accepted_chi2": max_accepted_chi2_report,
        "n_accepted_chi2_above_saturation": int(total_chi2_above_saturation),
        "chi2_saturation_threshold": _CHI2_SATURATION_THRESHOLD,
        "events": list(events),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    if args.split == "train":
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0.0] = 1.0
        for j, name in enumerate(feat.VALUE_FEATURES):
            if name in feat.NO_STANDARDIZE:
                mu[j], sigma[j] = 0.0, 1.0
        np.savez(out_dir / "norm_stats.npz", mu=mu.astype(np.float32),
                 sigma=sigma.astype(np.float32),
                 feature_names=np.array(feat.VALUE_FEATURES))
        print("wrote norm_stats.npz")


if __name__ == "__main__":
    main()
