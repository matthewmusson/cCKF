"""Seed purity classification.

Classifies each (seed_id, branch_id) as 'pure' (3/3 seed hits from the
majority particle) or 'majority' (2/3). Extracted from
analyze_value_targets.py for reuse by the cache builders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import labels as lab

#: Parquet columns required to compute seed purity.
_PURITY_COLUMNS: tuple[str, ...] = (
    "seed_id",
    "branch_id",
    "step_k",
    "is_ckf_selected",
    "cand_hit_id",
    "contrib_pids",
    "branch_majority_pid",
    "majority_undefined",
)


def classify_seed_purity(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each (seed_id, branch_id) as 'pure' or 'majority'.

    Finds the first 3 CKF-selected measurement hits per branch (by step_k
    order), counts how many have label_same_particle == 1. step_k is
    state_idx (including holes), so seed hits are NOT necessarily at 0,1,2.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: seed_id, branch_id, step_k, is_ckf_selected,
        cand_hit_id, label_same_particle.

    Returns
    -------
    pd.DataFrame
        Columns: seed_id, branch_id, seed_purity ('pure' or 'majority').
    """
    sel = df["is_ckf_selected"].to_numpy(dtype=bool)
    hit = df["cand_hit_id"].to_numpy(dtype=np.int64) != -1
    meas = df.loc[
        sel & hit, ["seed_id", "branch_id", "step_k", "label_same_particle"]
    ].copy()
    meas = meas.sort_values(["seed_id", "branch_id", "step_k"])
    meas["rank"] = meas.groupby(["seed_id", "branch_id"]).cumcount()
    seed_hits = meas.loc[meas["rank"] < 3].copy()
    seed_hits["is_same"] = (
        seed_hits["label_same_particle"].to_numpy(dtype=np.int64) == 1
    ).astype(np.int64)

    per_branch = seed_hits.groupby(
        ["seed_id", "branch_id"], as_index=False
    ).agg(n_seed_same=("is_same", "sum"))
    per_branch["seed_purity"] = np.where(
        per_branch["n_seed_same"] >= 3, "pure", "majority"
    )
    return per_branch[["seed_id", "branch_id", "seed_purity"]]


def compute_pure_seed_set(parquet_path: str) -> set[tuple[int, int]]:
    """Return the set of (seed_id, branch_id) that are pure (3/3).

    Reads only the columns needed for purity computation, derives
    label_same_particle via labels.derive_labels, then classifies.

    Parameters
    ----------
    parquet_path : str or Path
        Path to an expanded Parquet file with is_ckf_selected column.

    Returns
    -------
    set of (int, int)
        Pure-seed (seed_id, branch_id) pairs.
    """
    table = pq.read_table(parquet_path, columns=list(_PURITY_COLUMNS))
    derived = lab.derive_labels(table)
    df = table.to_pandas()
    df["label_same_particle"] = derived["label_same_particle"]

    purity = classify_seed_purity(df)
    pure = purity.loc[purity["seed_purity"] == "pure"]
    return set(zip(pure["seed_id"].tolist(), pure["branch_id"].tolist()))
