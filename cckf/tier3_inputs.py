"""Tier-3 stitch input builders: past counts and particle totals.

Feeds :func:`cckf.tier3_stitch.compose_targets` (DO NOT MODIFY that module).
Two independent inputs, each with a pure core (tested directly on
DataFrames) and a thin I/O wrapper (reads the Parquet / CSVs):

``past_counts`` / ``past_counts_from_rows``
    Per (seed_id, step_k): ``n_correct``, ``n_wrong`` accumulated up to and
    including that state, from the ``is_ckf_selected`` row. Membership test
    is identical to ``scripts/winfail_uncensored.build_state_table``'s:
    ``branch_majority_pid in contrib_pids`` of the selected row, with the
    same null-guard for ``contrib_pids`` (Arrow's null sentinel surfaces as
    either ``None`` or a bare float ``NaN``). Holes (``cand_hit_id == -1``)
    contribute 0 to both counters. Branches with ``majority_undefined ==
    True`` are excluded entirely -- their ``branch_majority_pid`` is a -1
    sentinel, not a real particle.

``n_total_true`` / ``n_total_true_from_frames``
    Per seed_id: the majority particle's total simhit count, ``N_total_true``.
    This is the SAME convention tier-2 uses (``cckf.value_target``): simhits
    from ``expansion.load_simhits``, grouped by the custom-encoded
    ``particle_id`` (``expansion.encode_particle_id``), which is the same id
    space as ``branch_majority_pid``. Not measurements -- simhits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cckf.value_target import particle_simhit_counts


def _as_pid_list(c: object) -> list:
    """Normalize a `contrib_pids` cell to a membership-testable list.

    Parquet nulls in a list<int64> column can surface to pandas either as
    `None` or as a bare `float` NaN (arrow's null sentinel loses the list
    type on the empty case). Both mean "no contributors" here.

    Copied from ``scripts/winfail_uncensored._as_pid_list`` (same semantics)
    rather than imported, since ``scripts/`` is not an importable package.
    """
    if c is None or isinstance(c, float):
        return []
    return c


def past_counts_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Cumulative per-branch (n_correct, n_wrong), including the own hit.

    Parameters
    ----------
    rows : pandas.DataFrame
        Candidate rows with ``seed_id``, ``step_k``, ``cand_hit_id``,
        ``is_ckf_selected``, ``contrib_pids``, ``branch_majority_pid``,
        ``majority_undefined``.

    Returns
    -------
    pandas.DataFrame
        ``seed_id``, ``step_k``, ``n_correct``, ``n_wrong`` (all int64), one
        row per state (including hole states) in ascending ``step_k`` within
        each ``seed_id``. Counts are inclusive of the state's own accepted
        hit. ``majority_undefined`` branches are dropped entirely.
    """
    empty_cols = {
        "seed_id": np.int64,
        "step_k": np.int64,
        "n_correct": np.int64,
        "n_wrong": np.int64,
    }
    work = rows[~rows["majority_undefined"].astype(bool)].copy()
    if work.empty:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in empty_cols.items()})

    sel = work["is_ckf_selected"].to_numpy(dtype=bool)
    is_hit = work["cand_hit_id"].to_numpy(dtype=np.int64) >= 0
    maj = work["branch_majority_pid"].to_numpy()
    is_correct = np.fromiter(
        (m in _as_pid_list(c) for m, c in zip(maj, work["contrib_pids"])),
        dtype=bool,
        count=len(work),
    )
    work["_sel_correct"] = (sel & is_hit & is_correct).astype(np.int64)
    work["_sel_wrong"] = (sel & is_hit & ~is_correct).astype(np.int64)

    # Collapse candidate rows to one row per state: every (seed_id, step_k)
    # that appears at all -- hit or hole -- survives via "max" (exactly one
    # row per state can be the is_ckf_selected==True hit row; every other
    # row at that state contributes 0 to both flags).
    state = (
        work.groupby(["seed_id", "step_k"], as_index=False)
        .agg(n_correct=("_sel_correct", "max"), n_wrong=("_sel_wrong", "max"))
        .sort_values(["seed_id", "step_k"])
        .reset_index(drop=True)
    )
    grp = state.groupby("seed_id", sort=False)
    state["n_correct"] = grp["n_correct"].cumsum()
    state["n_wrong"] = grp["n_wrong"].cumsum()

    return state[["seed_id", "step_k", "n_correct", "n_wrong"]].astype(empty_cols)


def past_counts(parquet_path: str) -> pd.DataFrame:
    """I/O wrapper around :func:`past_counts_from_rows`.

    Parameters
    ----------
    parquet_path : str
        Path to the expanded, ``is_ckf_selected``-patched candidate Parquet
        for one event.

    Returns
    -------
    pandas.DataFrame
        See :func:`past_counts_from_rows`.
    """
    cols = [
        "seed_id",
        "step_k",
        "cand_hit_id",
        "is_ckf_selected",
        "contrib_pids",
        "branch_majority_pid",
        "majority_undefined",
    ]
    rows = pq.read_table(parquet_path, columns=cols).to_pandas()
    return past_counts_from_rows(rows)


def n_total_true_from_frames(
    majority_by_seed: pd.DataFrame, simhits: pd.DataFrame
) -> pd.DataFrame:
    """N_total_true per seed: simhit count of that branch's majority particle.

    Parameters
    ----------
    majority_by_seed : pandas.DataFrame
        One row per ``seed_id``, with ``branch_majority_pid`` in the same
        custom-encoded particle-id space as ``simhits["particle_id"]``
        (``expansion.encode_particle_id``).
    simhits : pandas.DataFrame
        Output of ``expansion.load_simhits``; needs ``particle_id``. Counted
        directly -- one row per simhit, never per measurement.

    Returns
    -------
    pandas.DataFrame
        ``seed_id``, ``N_total_true`` (both int64). A seed whose majority pid
        has no simhits at all (a failed join, since a majority-defined branch
        has >= 2 majority simhits) gets 0, not a dropped row -- callers
        (``cckf.tier3_stitch.compose_targets``) treat ``N_total_true <= 0`` as
        a failed join and drop those branches loudly.
    """
    counts = particle_simhit_counts(simhits)
    n_total = majority_by_seed["branch_majority_pid"].map(counts).fillna(0)
    out = majority_by_seed[["seed_id"]].copy()
    out["N_total_true"] = n_total
    return out.astype({"seed_id": "int64", "N_total_true": "int64"})


def n_total_true(parquet_path: str, csv_dir: str, event_id: int) -> pd.DataFrame:
    """I/O wrapper around :func:`n_total_true_from_frames`.

    Parameters
    ----------
    parquet_path : str
        Path to the expanded candidate Parquet for one event; used only to
        read each seed's ``branch_majority_pid``.
    csv_dir : str
        Directory holding that event's raw CSVs (``simhits.csv`` etc.), as
        consumed by ``expansion.load_simhits``.
    event_id : int
        Event id, forwarded to ``expansion.load_simhits``.

    Returns
    -------
    pandas.DataFrame
        See :func:`n_total_true_from_frames`.
    """
    from expansion import load_simhits

    cols = ["seed_id", "branch_majority_pid", "majority_undefined"]
    df = pq.read_table(parquet_path, columns=cols).to_pandas()
    df = df[~df["majority_undefined"].astype(bool)]
    majority_by_seed = df.groupby("seed_id", as_index=False)[
        "branch_majority_pid"
    ].first()
    simhits = load_simhits(csv_dir, event_id)
    return n_total_true_from_frames(majority_by_seed, simhits)
