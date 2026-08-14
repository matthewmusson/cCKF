"""Tests for gate label derivation and row inclusion."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cckf import labels


def test_label_columns_are_all_in_schema():
    from cckf.splits import SCHEMA_76

    for col in labels.LABEL_COLUMNS:
        assert col in SCHEMA_76


def test_derive_labels_on_synthetic(synthetic_parquet):
    out = labels.derive_labels(pq.read_table(synthetic_parquet))

    # Row order matches the fixture: [s0k0, s0k1candA, s0k1candB, s0k2hole,
    #                                 s1k0, s1k1]
    # Positive iff branch_majority_pid is among contrib_pids.
    assert out["label_same_particle"].tolist() == [1, 1, 0, 0, 0, 0]

    # Ambiguous iff the cluster has more than one contributing particle.
    assert out["label_ambiguous"].tolist() == [False, True, False, False, False, False]

    # Included iff majority is defined AND the row is not a hole.
    # s0k2 is a hole; s1k0/s1k1 have majority_undefined=True.
    assert out["gate_row_mask"].tolist() == [True, True, True, False, False, False]


def test_derive_labels_dtypes_are_compact(synthetic_parquet):
    out = labels.derive_labels(pq.read_table(synthetic_parquet))
    assert out["label_same_particle"].dtype == np.uint8
    assert out["label_ambiguous"].dtype == np.bool_
    assert out["gate_row_mask"].dtype == np.bool_


def test_empty_contrib_pids_is_negative_not_error():
    """A hole row has contrib_pids = []. Membership must be False, not raise."""
    table = pa.table({
        "cand_hit_id": pa.array([-1], pa.int64()),
        "contrib_pids": pa.array([[]], pa.list_(pa.int64())),
        "branch_majority_pid": pa.array([1001], pa.int64()),
        "majority_undefined": pa.array([False]),
    })
    out = labels.derive_labels(table)
    assert out["label_same_particle"].tolist() == [0]
    assert out["gate_row_mask"].tolist() == [False]


def test_undefined_majority_is_excluded_even_when_pid_would_match():
    """majority_undefined dominates: the label is undefined, so drop the row."""
    table = pa.table({
        "cand_hit_id": pa.array([10], pa.int64()),
        "contrib_pids": pa.array([[-1]], pa.list_(pa.int64())),
        "branch_majority_pid": pa.array([-1], pa.int64()),
        "majority_undefined": pa.array([True]),
    })
    out = labels.derive_labels(table)
    assert out["gate_row_mask"].tolist() == [False]


def test_multi_row_group_table_is_handled():
    """Arrow tables from Parquet can be chunked; membership must still align."""
    n = 5
    t1 = pa.table({
        "cand_hit_id": pa.array([10] * n, pa.int64()),
        "contrib_pids": pa.array([[7], [8], [7, 8], [], [8, 7]], pa.list_(pa.int64())),
        "branch_majority_pid": pa.array([7] * n, pa.int64()),
        "majority_undefined": pa.array([False] * n),
    })
    combined = pa.concat_tables([t1, t1])
    out = labels.derive_labels(combined)
    expected = [1, 0, 1, 0, 1] * 2
    assert out["label_same_particle"].tolist() == expected
