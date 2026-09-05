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
from cckf import splits, stage1_map, value_target
from cckf.event_selection import resolve_requested_events

_NEEDED = (
    "seed_id",
    "branch_id",
    "step_k",
    "cand_hit_id",
    "is_ckf_selected",
    "contrib_pids",
    "branch_majority_pid",
    "majority_undefined",
    "majority_true_hit_on_surface",
    "state_theta",
    "state_qop",
    "cov_00",
    "cov_06",
    "n_hits",
    "n_holes",
    "n_seq_holes",
    "chi2_inc",
    "pathInX0_interval",
)

#: chi2_log_odds clips Λ at 1e-6, i.e. -2*ln(1e-6) ≈ 27.63; round down to 27.6
#: so this check does not itself hide a value right at the boundary.
_CHI2_SATURATION_THRESHOLD = 27.6

#: Columns read from a tier-3 targets Parquet (scripts/stitch_tier3.py's
#: output contract). window_nsigma is present in that file but is not read
#: here -- the CLI's own --window-nsigma is the source of truth for the
#: constant feature column (see apply_window_targets).
_TARGETS_COLUMNS = ("seed_id", "step_k", "vstar_tier3")


def _fmt_nsigma(nsig: float) -> str:
    """Format a window nsigma for directory/file names.

    Mirrors ``scripts/stitch_tier3.py::_fmt_nsig``: the plan's values (0, 3,
    5, 10) are all integral, so this renders them bare (``"10"``, not
    ``"10.0"``) to match the ``tier3_targets/vstar_nsig{N}_event...``
    filename convention; a genuinely fractional value falls back to
    ``str()``.
    """
    if float(nsig).is_integer():
        return str(int(nsig))
    return str(nsig)


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
    per_state = per_state.sort_values(["seed_id", "branch_id", "step_k"]).reset_index(
        drop=True
    )

    grp = per_state.groupby(["seed_id", "branch_id"], sort=False)
    filled = per_state["step_logodds"].fillna(0.0)
    per_state["sum_gate_logodds"] = (
        filled.groupby([per_state["seed_id"], per_state["branch_id"]])
        .cumsum()
        .to_numpy()
    )
    grouped_min = (
        per_state["step_logodds"]
        .groupby([per_state["seed_id"], per_state["branch_id"]])
        .cummin()
    )
    # Forward-fill WITHIN each branch so a hole step reports the running worst
    # accepted score rather than resetting it. cummin leaves the NaN row's own
    # output NaN; filling before the ffill (as `.fillna(0.0)` alone does) would
    # overwrite it with 0.0 log-odds = p 0.500, i.e. "the worst hit so far was a
    # coin flip" -- optimistically wrong at exactly the hole states V_phi must
    # judge. The trailing fillna(0.0) then covers only leading holes, before any
    # hit has been accepted and a minimum is genuinely undefined.
    per_state["min_gate_logodds"] = (
        grouped_min.groupby([per_state["seed_id"], per_state["branch_id"]])
        .ffill()
        .fillna(0.0)
        .to_numpy()
    )
    per_state["x0_accumulated"] = grp["pathInX0_interval"].cumsum().to_numpy()
    per_state["eta"] = feat.eta_from_theta(
        per_state["state_theta"].to_numpy(dtype=np.float64)
    )
    return per_state


def apply_window_targets(
    frame: pd.DataFrame, targets: pd.DataFrame, nsig: float
) -> tuple[pd.DataFrame, int]:
    """Join tier-3 window-conditioned targets onto a per-state frame.

    Window-conditioned tier-3 value plan, Task 6: replaces the tier-2 soft
    target with the tier-3 ``vstar_tier3`` value (``scripts/stitch_tier3.py``'s
    per-``(event, nsig)`` rollout target) and appends the constant
    ``window_nsigma`` feature column that lets a single value network
    condition on the rollout window.

    Parameters
    ----------
    frame : pandas.DataFrame
        Per-state frame, one row per ``(seed_id, step_k)``, carrying at least
        those two join columns plus every name in
        :data:`cckf.features.VALUE_FEATURES`.
    targets : pandas.DataFrame
        Tier-3 targets for one ``(event, nsig)`` pair —
        ``scripts/stitch_tier3.py``'s output contract: columns ``seed_id``,
        ``step_k``, ``vstar_tier3`` (a ``window_nsigma`` column may also be
        present but is not read here; ``nsig`` is the source of truth for the
        constant feature column so the caller controls it explicitly).
    nsig : float
        Rollout acceptance window this cache build is for. Broadcast as the
        12th feature column, ``window_nsigma``.

    Returns
    -------
    tuple[pandas.DataFrame, int]
        The joined frame — rows with no matching tier-3 target dropped, with
        a ``vstar_tier3`` column (from the join) and a new constant
        ``window_nsigma`` float32 feature column — and the count of dropped
        rows.
    """
    merged = frame.merge(
        targets[["seed_id", "step_k", "vstar_tier3"]],
        on=["seed_id", "step_k"],
        how="left",
    )
    has_target = merged["vstar_tier3"].notna().to_numpy()
    n_dropped = int((~has_target).sum())
    merged = merged.loc[has_target].reset_index(drop=True)
    merged["window_nsigma"] = np.float32(nsig)
    return merged, n_dropped


def process_event(
    parquet_path: Path,
    csv_dir: str,
    event_id: int,
    pure_seeds_only: bool = False,
    targets_df: pd.DataFrame | None = None,
    window_nsigma: float | None = None,
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

    When ``pure_seeds_only`` is True, purity is computed inline from the same
    Parquet read and ``derive_labels`` call — no separate pre-computation pass.
    ``_PURITY_COLUMNS`` is a strict subset of ``_NEEDED``, so this adds zero
    extra I/O.

    When ``targets_df`` is given (window-conditioned tier-3 value plan, Task
    6; ``window_nsigma`` must be given alongside it), ``y`` instead comes from
    ``targets_df``'s ``vstar_tier3`` column joined on ``(seed_id, step_k)``
    via :func:`apply_window_targets`, ``X`` is ``(n_states, 12)`` in
    :data:`cckf.features.VALUE_FEATURES_WINDOWED` order (12th column the
    constant ``window_nsigma``), and ``event_meta`` gains
    ``n_window_target_dropped`` — states that survived the Tier-2 filtering
    above but have no matching tier-3 target, dropped and counted. ``aux`` is
    unchanged (still Tier-2's ``vstar_t1``). When both are omitted, behaviour
    is byte-identical to the un-windowed path.
    """
    from expansion import load_simhits

    windowed = targets_df is not None
    if windowed != (window_nsigma is not None):
        raise ValueError(
            "targets_df and window_nsigma must be given together "
            f"(got targets_df={'set' if windowed else None}, "
            f"window_nsigma={window_nsigma})"
        )
    value_features = feat.VALUE_FEATURES_WINDOWED if windowed else feat.VALUE_FEATURES

    available = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    missing = set(_NEEDED) - available
    if missing:
        raise ValueError(f"{parquet_path} missing columns: {sorted(missing)}")

    table = pq.read_table(parquet_path, columns=list(_NEEDED))
    derived = lab.derive_labels(table)
    df = table.to_pandas()
    del table
    df["label_same_particle"] = derived["label_same_particle"]

    # --- χ² log-odds saturation check (before any filtering) ---------------
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
            f"saturating (max accepted chi2 = {max_accepted_chi2:.3f})",
            flush=True,
        )

    event_meta = {
        "max_accepted_chi2": max_accepted_chi2,
        "n_accepted_chi2_above_saturation": n_accepted_chi2_above_saturation,
        "n_tier_invariant_dropped": 0,
        "n_valid_target": 0,
        "n_pure_branches": None,
    }
    if windowed:
        event_meta["n_window_target_dropped"] = 0

    _empty = (
        np.empty((0, len(value_features)), np.float32),
        np.empty(0, np.float32),
        np.empty((0, 3), np.float32),
        event_meta,
    )

    df = df.loc[~df["majority_undefined"].astype(bool)].reset_index(drop=True)
    if df.empty:
        return _empty

    if pure_seeds_only:
        from cckf.seed_purity import classify_seed_purity

        purity = classify_seed_purity(df)
        pure = purity.loc[purity["seed_purity"] == "pure"]
        pure_set = set(zip(pure["seed_id"].tolist(), pure["branch_id"].tolist()))
        event_meta["n_pure_branches"] = len(pure_set)
        if pure_set:
            pure_index = pd.MultiIndex.from_tuples(
                pure_set, names=["seed_id", "branch_id"]
            )
            row_index = pd.MultiIndex.from_arrays(
                [df["seed_id"].to_numpy(), df["branch_id"].to_numpy()]
            )
            pure_mask = row_index.isin(pure_index)
        else:
            pure_mask = np.zeros(len(df), dtype=bool)
        df = df.loc[pure_mask].reset_index(drop=True)
        if df.empty:
            return _empty

    step = value_target.build_step_table(df)
    counts = value_target.particle_simhit_counts(load_simhits(csv_dir, event_id))
    targets = value_target.compute_value_targets(step, counts)

    state_feats = _state_features(df)
    merged = state_feats.merge(
        targets[
            [
                "seed_id",
                "branch_id",
                "step_k",
                "vstar_t2",
                "vstar_t1",
                "tier_invariant_violated",
            ]
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

    if windowed:
        # apply_window_targets left-merges onto `merged`, so every existing
        # column -- including vstar_t1, used by `aux` below -- survives on
        # the rows that keep a tier-3 target; only `vstar_tier3` and the new
        # `window_nsigma` column are added.
        merged, n_window_dropped = apply_window_targets(
            merged, targets_df, window_nsigma
        )
        event_meta["n_window_target_dropped"] = n_window_dropped
        if n_window_dropped > 0:
            print(
                f"WARNING event {event_id} nsig {window_nsigma}: "
                f"{n_window_dropped:,} states with no tier-3 target dropped",
                flush=True,
            )
        y_column = "vstar_tier3"
    else:
        y_column = "vstar_t2"

    X = np.column_stack(
        [merged[name].to_numpy(dtype=np.float64) for name in value_features]
    )
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y = merged[y_column].to_numpy(dtype=np.float32)
    aux = np.column_stack(
        [
            merged["vstar_t1"].to_numpy(dtype=np.float64),
            merged["step_k"].to_numpy(dtype=np.float64),
            merged["eta"].to_numpy(dtype=np.float64),
        ]
    ).astype(np.float32)
    return X.astype(np.float32), y, aux, event_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train", "val", "cal"])
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument(
        "--csv-dir",
        default="",
        help=(
            "Directory holding event{id:09d}-simhits.csv. Optional: when "
            "omitted, each event's directory is looked up per event in "
            "cckf.stage1_map, which is correct for the 32-event training set "
            "because Stage 1 wrote its CSVs into 16 separate per-batch pilot "
            "directories -- no single directory holds them all. Pass this "
            "only to override the map with one directory for every event "
            "(e.g. a hand-assembled scratch dir)."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--only-events",
        default="",
        help=(
            "Comma-separated event id subset (e.g. '0,1') to build the cache "
            "from, instead of every event assigned to --split. Validated "
            "against --split's own events, so an id from a different split "
            "or the sealed test range raises. Omit for the full split "
            "(default, unchanged behaviour)."
        ),
    )
    parser.add_argument(
        "--pure-seeds-only",
        action="store_true",
        help="Filter to pure seeds (3/3 seed hits from majority particle).",
    )
    parser.add_argument(
        "--targets-dir",
        default="",
        help=(
            "Window-conditioned tier-3 value plan (Task 6). Directory of "
            "per-event tier-3 targets Parquets, "
            "'vstar_nsig{N}_event{id:09d}.parquet' "
            "(scripts/stitch_tier3.py's output contract). Requires "
            "--window-nsigma. When both are omitted, output is "
            "byte-identical to the un-windowed (Tier-2) cache."
        ),
    )
    parser.add_argument(
        "--window-nsigma",
        type=float,
        default=None,
        help=(
            "Rollout acceptance window (N in vstar_nsig{N}_event...) this "
            "cache build is for. Broadcast as the 12th feature column, "
            "window_nsigma. Requires --targets-dir."
        ),
    )
    args = parser.parse_args()

    windowed = bool(args.targets_dir) or args.window_nsigma is not None
    if bool(args.targets_dir) != (args.window_nsigma is not None):
        parser.error("--targets-dir and --window-nsigma must be given together")

    # Validate against this split's own events, not the train+val+cal union:
    # otherwise '--split train --only-events 4' would succeed on a validation
    # event and silently mix splits. Same reasoning as
    # build_gate_cache.resolve_split_events.
    split_events = splits.events_for(args.split)
    events = resolve_requested_events(args.only_events, split_events)
    splits.assert_not_test(events)
    is_staged = bool(args.only_events.strip())

    if is_staged and args.split == "train":
        # norm_stats.npz fit on a subset is not what a full run would produce,
        # and no downstream consumer can tell from the file alone -- hence the
        # meta.json flag below as well.
        print(
            "WARNING: --only-events was given with --split train. "
            f"norm_stats.npz will be fit on {len(events)}/{len(split_events)} "
            f"train events ({list(events)}), NOT the full train split. This is "
            "a STAGED cache -- do not use it as a real training run's output."
        )

    out_dir = Path(args.out_dir)
    if windowed:
        # One cache build per n; the training set is the concatenation across
        # n (a later task's concern). Nesting under nsig{N} keeps three
        # per-n builds from colliding in the same split directory.
        out_dir = out_dir / f"nsig{_fmt_nsigma(args.window_nsigma)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    xs, ys, auxs = [], [], []
    total_tier_dropped = 0
    total_valid_target = 0
    total_chi2_above_saturation = 0
    total_window_dropped = 0
    max_accepted_chi2_overall = float("-inf")
    for event_id in events:
        path = Path(args.parquet_dir) / f"expanded_event{event_id:09d}.parquet"
        csv_dir = args.csv_dir or stage1_map.csv_dir_for(event_id)
        targets_df = None
        if windowed:
            targets_path = (
                Path(args.targets_dir) / f"vstar_nsig{_fmt_nsigma(args.window_nsigma)}_"
                f"event{event_id:09d}.parquet"
            )
            if not targets_path.exists():
                raise FileNotFoundError(
                    f"tier-3 targets not found for event {event_id}, "
                    f"nsig {args.window_nsigma}: {targets_path}"
                )
            targets_df = pq.read_table(
                targets_path, columns=list(_TARGETS_COLUMNS)
            ).to_pandas()
        X, y, aux, event_meta = process_event(
            path,
            csv_dir,
            event_id,
            pure_seeds_only=args.pure_seeds_only,
            targets_df=targets_df,
            window_nsigma=args.window_nsigma,
        )
        print(f"event {event_id}: {len(y):,} states", flush=True)
        xs.append(X)
        ys.append(y)
        auxs.append(aux)
        total_tier_dropped += event_meta["n_tier_invariant_dropped"]
        total_valid_target += event_meta["n_valid_target"]
        total_chi2_above_saturation += event_meta["n_accepted_chi2_above_saturation"]
        if windowed:
            total_window_dropped += event_meta["n_window_target_dropped"]
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
        None
        if max_accepted_chi2_overall == float("-inf")
        else max_accepted_chi2_overall
    )

    # Prominent, not buried in the JSON: this fraction is currently
    # unmeasured on real data and quantifies the deferred surface-revisit
    # over-count problem (cckf/value_target.py, "Known limitation").
    print(
        f"tier-invariant drop: {total_tier_dropped:,} / {total_valid_target:,} "
        f"valid-target rows dropped ({frac_tier_dropped:.4%}) — "
        f"vstar_t1 < vstar_t2 from surface-revisit over-count"
    )
    if windowed:
        print(
            f"window-target drop (nsig {args.window_nsigma}): "
            f"{total_window_dropped:,} states with no tier-3 target dropped"
        )
    if total_chi2_above_saturation > 0:
        print(
            f"WARNING: {total_chi2_above_saturation} accepted hits across all "
            f"events exceed chi2={_CHI2_SATURATION_THRESHOLD} "
            f"(chi2_log_odds saturating); max accepted chi2 = "
            f"{max_accepted_chi2_overall:.3f}"
        )
    else:
        shown = (
            f"{max_accepted_chi2_overall:.3f}"
            if max_accepted_chi2_report is not None
            else "n/a"
        )
        print(
            f"chi2 log-odds saturation check: no accepted hits exceed "
            f"chi2={_CHI2_SATURATION_THRESHOLD} (max accepted chi2 = {shown})"
        )

    feature_names = list(
        feat.VALUE_FEATURES_WINDOWED if windowed else feat.VALUE_FEATURES
    )
    y_key = "mean_vstar_tier3" if windowed else "mean_vstar_t2"
    gap_key = "tier1_minus_tier3_mean" if windowed else "tier1_minus_tier2_mean"

    marginal = float(np.mean((y >= 0.2) & (y <= 0.8)))
    meta = {
        "n_rows": int(len(y)),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "aux_columns": ["vstar_t1", "step_k", "eta"],
        y_key: float(y.mean()),
        "marginal_fraction_0.2_0.8": marginal,
        # Tier-gap diagnostic: how much the restricted tier's target gives up
        # against the Tier-1 optimistic bound. Free, since aux is computed
        # alongside y regardless of tier.
        gap_key: float(np.mean(aux[:, 0] - y)),
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
    if windowed:
        # Window-conditioned tier-3 value plan, Task 6.
        meta["window_nsigma"] = args.window_nsigma
        meta["targets_dir"] = str(args.targets_dir)
        meta["n_window_target_dropped"] = int(total_window_dropped)
    if args.pure_seeds_only:
        meta["pure_seeds_only"] = True
    if is_staged:
        # Mark a subset build as partial so no consumer -- or human reading a
        # directory listing -- can mistake it for a full-split cache. Same
        # reasoning as build_gate_cache.py.
        meta["partial_split"] = True
        meta["events_used"] = list(events)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    if args.split == "train":
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0.0] = 1.0
        for j, name in enumerate(feature_names):
            if name in feat.NO_STANDARDIZE:
                mu[j], sigma[j] = 0.0, 1.0
        np.savez(
            out_dir / "norm_stats.npz",
            mu=mu.astype(np.float32),
            sigma=sigma.astype(np.float32),
            feature_names=np.array(feature_names),
        )
        print("wrote norm_stats.npz")


if __name__ == "__main__":
    main()
