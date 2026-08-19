import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from cckf.seed_purity import classify_seed_purity, compute_pure_seed_set


def _make_parquet(tmp_path, rows):
    """Write a minimal Parquet with the columns needed for seed purity."""
    df = pd.DataFrame(rows)
    # contrib_pids must be a list column
    table = pa.table({
        "seed_id": pa.array(df["seed_id"], pa.int64()),
        "branch_id": pa.array(df["branch_id"], pa.int64()),
        "step_k": pa.array(df["step_k"], pa.int64()),
        "is_ckf_selected": pa.array(df["is_ckf_selected"], pa.bool_()),
        "cand_hit_id": pa.array(df["cand_hit_id"], pa.int64()),
        "contrib_pids": pa.array(df["contrib_pids"], pa.list_(pa.int64())),
        "branch_majority_pid": pa.array(df["branch_majority_pid"], pa.int64()),
        "majority_undefined": pa.array([False] * len(df), pa.bool_()),
    })
    path = tmp_path / "test.parquet"
    pq.write_table(table, path)
    return path


def test_pure_seed_3_of_3(tmp_path):
    """All 3 seed hits from majority particle → pure."""
    rows = [
        # seed 0, branch 0: 3 selected measurement hits, all from pid=100
        {"seed_id": 0, "branch_id": 0, "step_k": 0, "is_ckf_selected": True,
         "cand_hit_id": 10, "contrib_pids": [100], "branch_majority_pid": 100},
        {"seed_id": 0, "branch_id": 0, "step_k": 1, "is_ckf_selected": True,
         "cand_hit_id": 11, "contrib_pids": [100], "branch_majority_pid": 100},
        {"seed_id": 0, "branch_id": 0, "step_k": 2, "is_ckf_selected": True,
         "cand_hit_id": 12, "contrib_pids": [100], "branch_majority_pid": 100},
    ]
    path = _make_parquet(tmp_path, rows)
    pure_set = compute_pure_seed_set(path)
    assert (0, 0) in pure_set


def test_majority_seed_2_of_3(tmp_path):
    """2 of 3 seed hits from majority → majority (not pure)."""
    rows = [
        {"seed_id": 0, "branch_id": 0, "step_k": 0, "is_ckf_selected": True,
         "cand_hit_id": 10, "contrib_pids": [100], "branch_majority_pid": 100},
        {"seed_id": 0, "branch_id": 0, "step_k": 1, "is_ckf_selected": True,
         "cand_hit_id": 11, "contrib_pids": [200], "branch_majority_pid": 100},
        {"seed_id": 0, "branch_id": 0, "step_k": 2, "is_ckf_selected": True,
         "cand_hit_id": 12, "contrib_pids": [100], "branch_majority_pid": 100},
    ]
    path = _make_parquet(tmp_path, rows)
    pure_set = compute_pure_seed_set(path)
    assert (0, 0) not in pure_set


def test_holes_skipped(tmp_path):
    """Hole rows (cand_hit_id == -1) should be skipped; purity computed from measurements only."""
    rows = [
        {"seed_id": 0, "branch_id": 0, "step_k": 0, "is_ckf_selected": True,
         "cand_hit_id": -1, "contrib_pids": [], "branch_majority_pid": 100},
        {"seed_id": 0, "branch_id": 0, "step_k": 1, "is_ckf_selected": True,
         "cand_hit_id": 11, "contrib_pids": [100], "branch_majority_pid": 100},
        {"seed_id": 0, "branch_id": 0, "step_k": 2, "is_ckf_selected": True,
         "cand_hit_id": 12, "contrib_pids": [100], "branch_majority_pid": 100},
        {"seed_id": 0, "branch_id": 0, "step_k": 3, "is_ckf_selected": True,
         "cand_hit_id": 13, "contrib_pids": [100], "branch_majority_pid": 100},
    ]
    path = _make_parquet(tmp_path, rows)
    pure_set = compute_pure_seed_set(path)
    assert (0, 0) in pure_set
