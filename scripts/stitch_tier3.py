"""Tier-3 stitch driver: rollout hits + parquet -> V^{pi-dagger}(n) targets.

Per-event, per-window driver for the window-conditioned tier-3 value plan
(``docs/superpowers/plans/2026-09-03-window-conditioned-tier3-value.md``,
Task 5). For one ``(event, nsig)`` pair it:

1. Classifies every branch state (``cckf.tier3_walker.classify_event``).
2. Reads that window's rollout hits + the (window-independent) worklist and
   reduces them to per-rollout futures (``cckf.tier3_stitch.rollout_futures``).
3. Builds past counts and particle totals (``cckf.tier3_inputs``).
4. Composes V^{pi-dagger} targets (``cckf.tier3_stitch.compose_targets`` --
   Tier 1, called here, never modified).
5. Attaches the constant ``window_nsigma`` column and writes the targets
   Parquet.

For ``nsig == 10`` (or ``--tier2-check``), it additionally recomputes tier-2
targets straight from the parquet and runs the truth-suffix gate
(``cckf.tier3_stitch.truth_suffix_check``): on branches that never diverge
from truth, the n=10 rollout window is wide enough (16.26 < 10^2) that
tier-3 must reproduce tier-2 exactly. A disagreement rate >= 1% fails the
run (exit 1) -- this is the plan's acceptance gate, not a bug to silence.

No classification-cache file format exists from the walker runs (only a
worklist CSV, itself derived from a run of ``classify_event``), so this
driver always recomputes ``classify_event`` fresh; it is a single pass over
one event's parquet and cheap relative to the rollout I/O it joins against.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cckf import labels as lab
from cckf import stage1_map, tier3_inputs, tier3_stitch, tier3_walker, value_target

#: Columns needed to recompute tier-2 targets: build_step_table's inputs
#: (via labels.derive_labels for label_same_particle) plus the
#: majority_undefined filter. Mirrors scripts/build_value_cache.py's
#: process_event, minus the state-feature / flatten-to-array columns that
#: feed only the training cache, not the suffix check.
_TIER2_COLUMNS = (
    "seed_id",
    "branch_id",
    "step_k",
    "cand_hit_id",
    "is_ckf_selected",
    "contrib_pids",
    "branch_majority_pid",
    "majority_undefined",
    "majority_true_hit_on_surface",
)


def _fmt_nsig(nsig: float) -> str:
    """Format a window nsigma for directory/file names.

    Parameters
    ----------
    nsig : float
        Rollout acceptance window. The plan's values (0, 3, 5, 10) are all
        integral, so this renders them bare (``"10"``, not ``"10.0"``) to
        match the existing ``tier3_nsig{N}`` directory convention; a
        genuinely fractional value falls back to its plain ``str()``.

    Returns
    -------
    str
        The formatted tag.
    """
    if float(nsig).is_integer():
        return str(int(nsig))
    return str(nsig)


def paths(event: int, nsig: float, scratch: str, hits_dir: str | None) -> dict:
    """Resolve every filesystem path the driver needs for one (event, n).

    Pure path construction -- no filesystem access, no environment lookups
    -- so it is testable with arbitrary synthetic inputs.

    Parameters
    ----------
    event : int
        Event id.
    nsig : float
        Rollout acceptance window; used only for naming the default hits
        directory and the output parquet (Task 4's rollout generation,
        which actually produces the hits, is a separate script).
    scratch : str
        ``$SCRATCH/cckf``-equivalent base directory.
    hits_dir : str or None
        Override for the per-window rollout hits directory. When ``None``,
        defaults to ``{scratch}/tier3_nsig{nsig}/hits``.

    Returns
    -------
    dict
        ``parquet`` (expanded candidate Parquet), ``worklist`` (rollout
        worklist CSV, window-independent), ``hits_dir`` (resolved hits
        directory), ``hits`` (that directory's hits CSV for this event),
        ``out`` (the output targets Parquet path).
    """
    tag = f"{event:09d}"
    resolved_hits_dir = (
        hits_dir
        if hits_dir is not None
        else f"{scratch}/tier3_nsig{_fmt_nsig(nsig)}/hits"
    )
    return {
        "parquet": f"{scratch}/reexpanded/expanded_event{tag}.parquet",
        "worklist": f"{scratch}/tier3/worklists/event{tag}-rollout-worklist.csv",
        "hits_dir": resolved_hits_dir,
        "hits": f"{resolved_hits_dir}/event{tag}-rollout-hits.csv",
        "out": f"{scratch}/tier3_targets/vstar_nsig{_fmt_nsig(nsig)}_event{tag}.parquet",
    }


def attach_window(targets: pd.DataFrame, nsig: float) -> pd.DataFrame:
    """Add the constant ``window_nsigma`` column to a targets frame.

    Parameters
    ----------
    targets : pandas.DataFrame
        Output of ``cckf.tier3_stitch.compose_targets``
        (``seed_id``, ``step_k``, ``vstar_tier3``).
    nsig : float
        The rollout acceptance window this batch of targets was generated
        under.

    Returns
    -------
    pandas.DataFrame
        A copy of ``targets`` with a new ``window_nsigma`` float64 column
        broadcast to every row. ``targets`` itself is not mutated.
    """
    out = targets.copy()
    out["window_nsigma"] = float(nsig)
    return out


def gate(report: dict, tol: float) -> bool:
    """Whether a ``truth_suffix_check`` report passes the acceptance gate.

    Parameters
    ----------
    report : dict
        Output of ``cckf.tier3_stitch.truth_suffix_check``. A report with
        ``n_states_compared == 0`` carries no ``disagree_rate`` key (see
        that function's early return) and is treated as a pass -- there is
        nothing to disagree on.
    tol : float
        Disagreement-rate ceiling. Fails (returns ``False``) iff
        ``disagree_rate >= tol``.

    Returns
    -------
    bool
        ``True`` if the run should proceed, ``False`` if it should exit
        nonzero.
    """
    if report.get("n_states_compared", 0) == 0:
        return True
    return report["disagree_rate"] < tol


def filter_valid_tier2(targets: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop tier-2 rows ``compute_value_targets`` flags as unusable.

    Mirrors ``scripts/build_value_cache.py::process_event``'s
    target-validity filter (same column names, same mask) applied just
    before ``vstar_t2`` is exposed to any consumer: a NaN ``vstar_t2``
    means a failed PID join (the majority particle had no simhit count),
    and ``tier_invariant_violated`` means a surface revisit double-counted
    ``maj_hit_on_surface`` badly enough to invert the ``vstar_t1 >=
    vstar_t2`` invariant (see ``cckf.value_target`` module docstring,
    "Known limitation"). Skipping this filter would let the truth-suffix
    gate compare tier-3 against known-bad tier-2 rows: a violated row's
    disagreement reflects a tier-2 artifact, not the diagonal-seeding bias
    the gate exists to measure, and a NaN row would silently drop out of
    ``disagree_rate``'s numerator while poisoning the mean/max/percentile
    diagnostics with NaN.

    Parameters
    ----------
    targets : pandas.DataFrame
        Output of ``cckf.value_target.compute_value_targets`` (or any frame
        carrying its ``vstar_t2`` and ``tier_invariant_violated`` columns).

    Returns
    -------
    tuple of (pandas.DataFrame, int)
        The filtered frame (index reset) and the number of dropped rows.
    """
    valid_target = targets["vstar_t2"].notna().to_numpy()
    not_violated = ~targets["tier_invariant_violated"].to_numpy(dtype=bool)
    keep = valid_target & not_violated
    n_dropped = int(len(targets) - int(keep.sum()))
    return targets.loc[keep].reset_index(drop=True), n_dropped


def recompute_tier2_from_frames(
    labeled: pd.DataFrame, simhits: pd.DataFrame
) -> pd.DataFrame:
    """Pure core: labeled candidate rows + simhits -> validated tier-2 targets.

    No parquet/CSV I/O -- takes already-loaded frames so it is testable
    with synthetic data. Mirrors
    ``scripts/build_value_cache.py::process_event``'s call sequence from
    its ``build_step_table`` call through the target-validity filter
    (``process_event``'s "Required addition 1" block), which is exactly
    the part of that function this driver needs to reproduce for the
    truth-suffix gate -- the state-feature construction and final
    flatten-to-array step that follow in ``process_event`` feed only the
    training cache and are not needed here.

    Parameters
    ----------
    labeled : pandas.DataFrame
        Candidate rows with ``seed_id``, ``branch_id``, ``step_k``,
        ``cand_hit_id``, ``is_ckf_selected``, ``majority_true_hit_on_surface``,
        ``branch_majority_pid``, ``majority_undefined``, and
        ``label_same_particle`` already attached (the I/O wrapper attaches
        it via ``cckf.labels.derive_labels``, a ``pyarrow.Table`` operation
        kept out of this pure core).
    simhits : pandas.DataFrame
        Output of ``expansion.load_simhits`` (needs ``particle_id``).

    Returns
    -------
    pandas.DataFrame
        ``seed_id``, ``step_k``, ``vstar_t2`` -- one row per valid state,
        with NaN-target and ``tier_invariant_violated`` rows dropped (see
        :func:`filter_valid_tier2`).
    """
    df = labeled.loc[~labeled["majority_undefined"].astype(bool)].reset_index(drop=True)
    step = value_target.build_step_table(df)
    counts = value_target.particle_simhit_counts(simhits)
    targets = value_target.compute_value_targets(step, counts)
    targets, n_dropped = filter_valid_tier2(targets)
    if n_dropped:
        print(
            f"recompute_tier2_from_frames: dropped {n_dropped:,} tier-2 "
            "states (NaN target from a failed PID join, or a tier-invariant "
            "violation)",
            flush=True,
        )
    return targets[["seed_id", "step_k", "vstar_t2"]]


def _recompute_tier2(parquet_path: str, csv_dir: str, event_id: int) -> pd.DataFrame:
    """I/O wrapper around :func:`recompute_tier2_from_frames`.

    The persisted tier-2 value cache
    (``scripts/build_value_cache.py::process_event``) drops ``seed_id`` at
    its final flatten-to-array step, so it cannot be joined back to
    ``(seed_id, step_k)`` for the truth-suffix check. This reads the same
    columns and attaches the same label column that function does, then
    hands off to the pure core.

    Parameters
    ----------
    parquet_path : str
        Expanded candidate Parquet for the event.
    csv_dir : str
        Stage-1 CSV directory (``cckf.stage1_map.csv_dir_for``), for
        ``expansion.load_simhits``.
    event_id : int
        Event id, forwarded to ``expansion.load_simhits``.

    Returns
    -------
    pandas.DataFrame
        See :func:`recompute_tier2_from_frames`. ``seed_id == branch_id``
        on this data, so no separate branch_id column is needed downstream.
    """
    from expansion import load_simhits

    table = pq.read_table(parquet_path, columns=list(_TIER2_COLUMNS))
    derived = lab.derive_labels(table)
    df = table.to_pandas()
    df["label_same_particle"] = derived["label_same_particle"]
    simhits = load_simhits(csv_dir, event_id)
    return recompute_tier2_from_frames(df, simhits)


def filter_majority_defined(states: pd.DataFrame, defined_seeds) -> tuple:
    """Drop classified states whose branch has an undefined majority particle.

    The rollout worklist (``cckf.tier3_walker.emit_worklist``) deliberately
    excludes majority-undefined branches -- they can never match any hit
    under pi-dagger -- so those branches have no rollout and no hits-file
    entry. ``cckf.tier3_walker.classify_event`` classifies every state
    regardless of majority definedness and does not itself carry a
    ``majority_undefined`` column, so calling ``compose_targets`` on the
    unfiltered frame makes every majority-undefined branch's anchor state
    look like a genuinely missing rollout -- observed on event 4 (unbounded
    hits, nsig=10): 1,534,557 states / 1,534,557 branches dropped as
    "missing rollout", dwarfing the ~187 states behind the real rollout
    losses. Filtering here first keeps ``compose_targets``'s own counter
    meaningful.

    Parameters
    ----------
    states : pandas.DataFrame
        ``cckf.tier3_walker.classify_event`` output; needs ``seed_id``.
    defined_seeds : set of int (or any container supporting ``.isin``)
        seed_ids with a defined branch majority -- e.g.
        ``cckf.tier3_inputs.past_counts``'s output seed_id column, which
        already excludes majority-undefined branches by construction.

    Returns
    -------
    tuple of (pandas.DataFrame, int)
        The filtered frame (index reset) and the number of distinct
        excluded seed_ids (branches, not states).
    """
    keep = states["seed_id"].isin(defined_seeds)
    n_excluded_branches = int(states.loc[~keep, "seed_id"].nunique())
    return states.loc[keep].reset_index(drop=True), n_excluded_branches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("event", type=int)
    ap.add_argument("nsig", type=float)
    ap.add_argument("scratch_base")
    ap.add_argument(
        "--hits-dir",
        default=None,
        help="override the per-window rollout hits directory; default "
        "{scratch}/tier3_nsig{nsig}/hits (used for the unbounded dry run "
        "with {scratch}/tier3/hits)",
    )
    ap.add_argument(
        "--tier2-check",
        action="store_true",
        help="run the truth-suffix gate against recomputed tier-2 targets; "
        "always on when nsig == 10 regardless of this flag",
    )
    args = ap.parse_args()

    do_check = args.tier2_check or args.nsig == 10.0
    p = paths(args.event, args.nsig, args.scratch_base, args.hits_dir)
    nsig_tag = _fmt_nsig(args.nsig)

    t_prev = time.time()

    def _stage(name: str) -> None:
        nonlocal t_prev
        now = time.time()
        print(
            f"event {args.event} nsig {nsig_tag}: [{name}] {now - t_prev:.1f}s",
            flush=True,
        )
        t_prev = now

    states = tier3_walker.classify_event(p["parquet"])
    _stage("classify")

    hits = pd.read_csv(p["hits"])
    worklist = pd.read_csv(p["worklist"])
    futures = tier3_stitch.rollout_futures(hits, worklist)
    _stage("futures")

    past = tier3_inputs.past_counts(p["parquet"])
    _stage("past")

    states, n_excluded = filter_majority_defined(states, past["seed_id"].unique())
    if n_excluded:
        print(
            f"event {args.event} nsig {nsig_tag}: excluded {n_excluded:,} "
            "majority-undefined branches before compose_targets",
            flush=True,
        )

    csv_dir = stage1_map.csv_dir_for(args.event)
    n_total = tier3_inputs.n_total_true(p["parquet"], csv_dir, args.event)
    _stage("n_total")

    targets = tier3_stitch.compose_targets(states, futures, past, n_total)
    targets = attach_window(targets, args.nsig)
    _stage("compose")

    Path(p["out"]).parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(p["out"], index=False)
    _stage("write")

    print(f"event {args.event} nsig {nsig_tag}: rows={len(targets):,}", flush=True)

    if do_check:
        tier2 = _recompute_tier2(p["parquet"], csv_dir, args.event)
        _stage("tier2 recompute")
        report = tier3_stitch.truth_suffix_check(
            states, targets, tier2, tol=0.01, value_col="vstar_t2"
        )
        print("SUFFIX_CHECK " + json.dumps(report), flush=True)
        if not gate(report, 0.01):
            print(
                f"event {args.event} nsig {nsig_tag}: SUFFIX GATE FAILED "
                f"(disagree_rate={report.get('disagree_rate')} >= 0.01)",
                flush=True,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
