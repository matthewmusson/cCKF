"""Tests for pure-seed filtering in the gate cache builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cckf.cache import build_gate_cache
from cckf.seed_purity import compute_pure_seed_set


def _make_parquet_with_selected(tmp_path, n_pure=10, n_majority=30):
    """Create a minimal Parquet with both pure and majority seeds."""
    rows = []
    # Pure seed: seed_id=0, branch_id=0
    for step in range(n_pure):
        rows.append({
            "seed_id": 0, "branch_id": 0, "step_k": step,
            "is_ckf_selected": True, "cand_hit_id": step + 100,
            "contrib_pids": [1], "branch_majority_pid": 1,
            "majority_undefined": False, "n_window": 5,
            "residual_l0": 0.01, "residual_l1": 0.02,
            "S00": 0.1, "S01": 0.0, "S11": 0.1,
            "chi2_inc": 2.0, "state_theta": 1.0, "state_qop": 0.5,
            "clus_s_u": 0.01, "clus_s_v": 0.02, "clus_q_tot": 100.0,
            "clus_sigma_uu": 0.001, "clus_sigma_uv": 0.0, "clus_sigma_vv": 0.001,
            "alpha_u": 0.1, "alpha_v": 0.2,
            "pitch_u": 0.05, "pitch_v": 0.05, "thickness": 0.15,
            "is_pixel": True, "is_barrel": True,
            "n_hits": step + 1, "n_holes": 0, "n_seq_holes": 0,
            "pathInX0_interval": 0.01,
        })
    # Majority seed: seed_id=1, branch_id=0 (2/3 from pid=2, 1/3 from pid=3)
    for step in range(n_majority):
        pid = 2 if step != 1 else 3  # step 1 is from wrong particle
        rows.append({
            "seed_id": 1, "branch_id": 0, "step_k": step,
            "is_ckf_selected": True, "cand_hit_id": step + 200,
            "contrib_pids": [pid], "branch_majority_pid": 2,
            "majority_undefined": False, "n_window": 5,
            "residual_l0": 0.01, "residual_l1": 0.02,
            "S00": 0.1, "S01": 0.0, "S11": 0.1,
            "chi2_inc": 2.0, "state_theta": 1.0, "state_qop": 0.5,
            "clus_s_u": 0.01, "clus_s_v": 0.02, "clus_q_tot": 100.0,
            "clus_sigma_uu": 0.001, "clus_sigma_uv": 0.0, "clus_sigma_vv": 0.001,
            "alpha_u": 0.1, "alpha_v": 0.2,
            "pitch_u": 0.05, "pitch_v": 0.05, "thickness": 0.15,
            "is_pixel": True, "is_barrel": True,
            "n_hits": step + 1, "n_holes": 0, "n_seq_holes": 0,
            "pathInX0_interval": 0.01,
        })
    # Build Parquet with list columns for contrib_pids
    df = pd.DataFrame(rows)
    arrays = {}
    for col in df.columns:
        if col == "contrib_pids":
            arrays[col] = pa.array(df[col].tolist(), pa.list_(pa.int64()))
        elif col in ("is_ckf_selected", "majority_undefined", "is_pixel", "is_barrel"):
            arrays[col] = pa.array(df[col], pa.bool_())
        elif col in ("seed_id", "branch_id", "step_k", "cand_hit_id",
                      "branch_majority_pid", "n_window", "n_hits", "n_holes", "n_seq_holes"):
            arrays[col] = pa.array(df[col], pa.int64())
        else:
            arrays[col] = pa.array(df[col], pa.float64())
    table = pa.table(arrays)
    path = tmp_path / "test.parquet"
    pq.write_table(table, path)
    return path


def test_pure_seed_filter_reduces_rows(tmp_path):
    """With pure_seeds_only, cache should contain only pure-seed rows."""
    path = _make_parquet_with_selected(tmp_path, n_pure=10, n_majority=30)
    out_all = tmp_path / "cache_all"
    out_pure = tmp_path / "cache_pure"

    meta_all = build_gate_cache([path], out_all)
    pure_set = {path: compute_pure_seed_set(str(path))}
    meta_pure = build_gate_cache([path], out_pure, pure_seed_sets=pure_set)

    assert meta_pure["n_rows"] < meta_all["n_rows"]
    assert meta_pure["n_rows"] == 10  # only pure seed's 10 rows
    assert meta_pure.get("pure_seeds_only") is True


def test_no_pure_seed_sets_records_false(tmp_path):
    """Without pure_seed_sets, meta.json records pure_seeds_only=False."""
    path = _make_parquet_with_selected(tmp_path, n_pure=10, n_majority=30)
    out = tmp_path / "cache_all"
    meta = build_gate_cache([path], out)
    assert meta.get("pure_seeds_only") is False


def test_pure_seed_filter_empty_set_yields_no_rows(tmp_path):
    """A file present in the dict with an empty pure set contributes zero rows."""
    path = _make_parquet_with_selected(tmp_path, n_pure=10, n_majority=30)
    out = tmp_path / "cache_empty"
    meta = build_gate_cache([path], out, pure_seed_sets={path: set()})
    assert meta["n_rows"] == 0
    assert meta.get("pure_seeds_only") is True
