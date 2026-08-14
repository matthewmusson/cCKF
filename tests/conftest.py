"""Shared test fixtures.

``synthetic_parquet`` builds a small Parquet with the exact 76-column expanded
schema and hand-chosen values whose correct labels, features and value targets
are known by construction. Every downstream test asserts against those known
answers rather than against a recomputation of the same formula.

Layout: one event (id 0), two seeds. Seed 0 / branch 0 walks 3 steps; seed 1 /
branch 0 walks 2 steps and has an undefined majority. Step 1 of seed 0 has two
in-window candidates (one correct, one wrong); step 2 is a hole.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cckf.splits import SCHEMA_76

# Particle barcodes (same encoding as simhits.csv particle_id).
PID_MAJ = 1001
PID_OTHER = 2002


def _base_row() -> dict:
    """A row with every column present and benign defaults."""
    row = {
        "event_id": 0,
        "seed_id": 0,
        "branch_id": 0,
        "parent_branch_id": -1,
        "step_k": 0,
        "layer_id": 2,
        "surface_id": 1152922604118474752,
        "volume_id": 16,
        "state_l0": 1.0,
        "state_l1": 2.0,
        "state_phi": 0.5,
        "state_theta": np.pi / 2,  # η = 0 exactly
        "state_qop": -0.5,
        "state_t": 0.0,
        "pred_l0": 1.0,
        "pred_l1": 2.0,
        "cand_hit_id": 10,
        "residual_l0": 0.0,
        "residual_l1": 0.0,
        "S00": 4.0,  # chol → L00 = 2
        "S01": 2.0,  # chol → L10 = 1
        "S11": 5.0,  # chol → L11 = sqrt(5 - 1) = 2
        "chi2_inc": 0.0,
        "clus_s_u": 2.0,
        "clus_s_v": 3.0,
        "clus_q_tot": 0.4,
        "clus_sigma_uu": 0.01,
        "clus_sigma_uv": 0.0,
        "clus_sigma_vv": 0.02,
        "alpha_u": 0.0,  # tan = 0 → expected_size_u = 1 → kappa_u = clus_s_u
        "alpha_v": 0.0,
        "pitch_u": 0.05,
        "pitch_v": 0.05,
        "thickness": 0.2,
        "is_pixel": True,
        "is_barrel": np.nan,
        "n_window": 2,
        "geometric_density": 5,
        "pathInX0_interval": 0.01,
        "dead_module_flag": 0,
        "layer_embed_idx": 0,
        "n_hits": 1,
        "n_holes": 0,
        "n_seq_holes": 0,
        "action_taken": 0,
        "prune_reason": None,
        "contrib_pids": [PID_MAJ],
        "contrib_charge_frac": [1.0],
        "branch_majority_pid": PID_MAJ,
        "majority_undefined": False,
        "majority_true_hit_on_surface": True,
        "truth_residual_l0": np.nan,
        "truth_residual_l1": np.nan,
        "vstar_soft": np.nan,
        "env_config_hash": "deadbeef",
    }
    for i in range(21):
        row[f"cov_{i:02d}"] = np.nan
    row["cov_00"] = 0.25  # σ²_l0
    row["cov_06"] = 0.36  # σ²_l1
    return row


def _rows() -> list[dict]:
    rows: list[dict] = []

    # --- seed 0, branch 0 -------------------------------------------------
    # step 0: single correct candidate (positive, unambiguous)
    r = _base_row()
    rows.append(r)

    # step 1, candidate A: correct AND ambiguous (merged cluster, 2 contributors)
    r = _base_row()
    r.update(
        step_k=1, cand_hit_id=11, chi2_inc=1.0, residual_l0=0.5,
        contrib_pids=[PID_MAJ, PID_OTHER], contrib_charge_frac=[0.6, 0.4],
        n_hits=2, majority_true_hit_on_surface=True,
    )
    rows.append(r)

    # step 1, candidate B: wrong particle (negative)
    r = _base_row()
    r.update(
        step_k=1, cand_hit_id=12, chi2_inc=9.0, residual_l0=3.0,
        contrib_pids=[PID_OTHER], contrib_charge_frac=[1.0],
        n_hits=2, majority_true_hit_on_surface=True,
    )
    rows.append(r)

    # step 2: hole row (cand_hit_id == -1) — excluded from gate training
    r = _base_row()
    r.update(
        step_k=2, cand_hit_id=-1, residual_l0=np.nan, residual_l1=np.nan,
        chi2_inc=np.nan, contrib_pids=[], contrib_charge_frac=[],
        clus_s_u=np.nan, clus_s_v=np.nan, clus_q_tot=np.nan,
        action_taken=1, n_hits=2, n_holes=1, n_seq_holes=1,
        majority_true_hit_on_surface=False,
    )
    rows.append(r)

    # --- seed 1, branch 0: majority undefined → excluded entirely ---------
    for step in (0, 1):
        r = _base_row()
        r.update(
            seed_id=1, step_k=step, cand_hit_id=20 + step,
            branch_majority_pid=-1, majority_undefined=True,
            contrib_pids=[PID_OTHER], contrib_charge_frac=[1.0],
        )
        rows.append(r)

    return rows


@pytest.fixture
def synthetic_rows() -> list[dict]:
    """The raw row dicts, for tests that want a DataFrame directly."""
    return _rows()


@pytest.fixture
def synthetic_df(synthetic_rows: list[dict]) -> pd.DataFrame:
    """Synthetic expanded data as a DataFrame with columns in schema order."""
    return pd.DataFrame(synthetic_rows)[list(SCHEMA_76)]


@pytest.fixture
def synthetic_parquet(tmp_path: Path, synthetic_df: pd.DataFrame) -> Path:
    """Write the synthetic data to a Parquet file and return its path."""
    path = tmp_path / "expanded_event000000000.parquet"
    pq.write_table(pa.Table.from_pandas(synthetic_df, preserve_index=False), path)
    return path
