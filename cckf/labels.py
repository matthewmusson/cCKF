"""Gate label derivation and row inclusion (spec §1.3, §2.4).

The expanded Parquet has **no** ``label_same_particle`` or ``label_ambiguous``
column, despite the training spec naming them. Both are derived here from the
truth columns that do exist.

Definitions
-----------
``label_same_particle`` (the gate's BCE target)
    1 iff the branch's majority particle is among the particles that
    contributed charge to this candidate's cluster::

        y = 1[ branch_majority_pid ∈ contrib_pids ]

    This is the right target for the question the gate answers — "is this
    candidate from the branch's particle?" — and it treats a merged cluster
    containing the majority particle as a positive, because the Kalman update
    from that cluster does carry the majority particle's position information.

``label_ambiguous``
    True iff the cluster has more than one contributing particle (a merged
    cluster). Per spec §1.3 these rows are kept at **full weight**: merged
    clusters are in-distribution at μ=200 and the gate must learn to score
    them, not be shielded from them. The flag is retained so ablation A8b can
    exclude or down-weight them and measure the effect.

``gate_row_mask`` (row inclusion)
    Keep a row iff the label is defined and the row is a real candidate::

        keep = (not majority_undefined) and (cand_hit_id != -1)

    ``majority_undefined`` marks branches where no particle owns ≥2/3 of the
    seed hits — for those, "the branch's particle" is not a well-posed notion,
    so y is undefined rather than 0. This drops ~90.3% of all rows. Hole rows
    (``cand_hit_id == -1``) have no candidate to score at all; the value
    function, not the gate, is what handles holes.

Implementation note
-------------------
Membership is computed with Arrow list kernels rather than a Python loop over
rows. ``list_flatten`` + ``list_parent_indices`` turn the ragged
``contrib_pids`` column into two flat arrays, the comparison is a single
vectorised ``equal``, and ``np.bincount`` folds the per-element matches back to
per-row booleans. This runs over 1.44B rows in streaming row-group batches;
a per-row Python check would not.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

#: Parquet columns required to derive labels and the inclusion mask.
LABEL_COLUMNS: tuple[str, ...] = (
    "cand_hit_id",
    "contrib_pids",
    "branch_majority_pid",
    "majority_undefined",
)


def derive_labels(table: pa.Table) -> dict[str, np.ndarray]:
    """Derive the gate target, the ambiguity flag, and the inclusion mask.

    Parameters
    ----------
    table : pyarrow.Table
        Any table carrying at least :data:`LABEL_COLUMNS`. May be chunked.

    Returns
    -------
    dict of str to numpy.ndarray
        ``label_same_particle`` : uint8, shape (n_rows,)
            1 if the branch majority particle contributed to this cluster.
        ``label_ambiguous`` : bool, shape (n_rows,)
            True if the cluster has >1 contributing particle.
        ``gate_row_mask`` : bool, shape (n_rows,)
            True if the row is trainable for the gate.
    """
    n_rows = table.num_rows

    # Combine chunks so list offsets are contiguous and parent indices are
    # global row numbers rather than per-chunk ones.
    pids = table.column("contrib_pids").combine_chunks()
    if isinstance(pids, pa.ChunkedArray):
        pids = pids.chunk(0) if pids.num_chunks == 1 else pa.concat_arrays(pids.chunks)

    majority = np.asarray(
        table.column("branch_majority_pid")
        .combine_chunks()
        .to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )

    # Ragged → flat, with a parent row index per element.
    flat = pc.list_flatten(pids)
    parent = np.asarray(
        pc.list_parent_indices(pids).to_numpy(zero_copy_only=False), dtype=np.int64
    )

    same = np.zeros(n_rows, dtype=np.uint8)
    if len(parent) > 0:
        flat_np = np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.int64)
        matches = flat_np == majority[parent]
        if matches.any():
            hit_counts = np.bincount(parent[matches], minlength=n_rows)
            same = (hit_counts > 0).astype(np.uint8)

    n_contrib = np.asarray(pc.list_value_length(pids).to_numpy(zero_copy_only=False))
    n_contrib = np.nan_to_num(n_contrib, nan=0.0).astype(np.int64)
    ambiguous = n_contrib > 1

    undefined = np.asarray(
        table.column("majority_undefined")
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
    ).astype(bool)
    cand_hit_id = np.asarray(
        table.column("cand_hit_id").combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    keep = (~undefined) & (cand_hit_id != -1)

    return {
        "label_same_particle": same,
        "label_ambiguous": ambiguous,
        "gate_row_mask": keep,
    }
