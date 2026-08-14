"""Frozen event→split assignment.

The split is fixed by event ID and must never change. Splitting by row would
leak: hits within an event share the same pileup realisation and detector
state, so two rows from one event are not independent samples.

Events [32, 64) are the sealed shared CKF/cCKF test set, opened exactly once
for headline numbers (cCKF_Specification §6.1). Events [0, 32) are partitioned
24/4/4 into train / validation / calibration.

All 32 events in [0, 32) are fully patched and verified (76 columns, 100% fill
on S00/pitch/volume_id) as of 2026-08-13 — see experiments/LOG.md. Freezing the
assignment here is what closes that file's outstanding "split: TBD" item.

Why val and cal are scattered rather than contiguous: Stage 1 produced the 32
events as 16 sequential batches of 2, i.e. (0,1), (2,3), ..., (30,31). Taking a
contiguous block for validation would sample only a few batches, so any
batch-correlated artefact (a container-level configuration difference, a
partially-corrupted re-expansion) would land entirely inside one split and be
invisible in the others. The picks below take one val and one cal event from
each quarter of the range, and no val event shares a generation batch with a
cal event.

VAL_EVENTS and CAL_EVENTS must never change. Early-stopping decisions and
calibration statistics are only comparable across experiments if the splits
they are measured on are identical.
"""
from __future__ import annotations

from typing import Iterable

# --- Frozen assignment (never modify) -------------------------------------
TRAIN_EVENTS: tuple[int, ...] = (
    0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 13, 14,
    16, 17, 18, 19, 21, 22, 24, 25, 26, 27, 29, 30,
)
VAL_EVENTS: tuple[int, ...] = (4, 12, 20, 28)
CAL_EVENTS: tuple[int, ...] = (7, 15, 23, 31)
TEST_EVENTS: tuple[int, ...] = tuple(range(32, 64))

_LOWER_ASSIGNMENT: dict[int, str] = {
    **{e: "train" for e in TRAIN_EVENTS},
    **{e: "val" for e in VAL_EVENTS},
    **{e: "cal" for e in CAL_EVENTS},
}

# --- Expanded Parquet schema (expansion.py:SCHEMA_COLUMNS, 76 columns) -----
SCHEMA_76: tuple[str, ...] = (
    "event_id",
    "seed_id",
    "branch_id",
    "parent_branch_id",
    "step_k",
    "layer_id",
    "surface_id",
    "volume_id",
    "state_l0",
    "state_l1",
    "state_phi",
    "state_theta",
    "state_qop",
    "state_t",
    *(f"cov_{i:02d}" for i in range(21)),
    "pred_l0",
    "pred_l1",
    "cand_hit_id",
    "residual_l0",
    "residual_l1",
    "S00",
    "S01",
    "S11",
    "chi2_inc",
    "clus_s_u",
    "clus_s_v",
    "clus_q_tot",
    "clus_sigma_uu",
    "clus_sigma_uv",
    "clus_sigma_vv",
    "alpha_u",
    "alpha_v",
    "pitch_u",
    "pitch_v",
    "thickness",
    "is_pixel",
    "is_barrel",
    "n_window",
    "geometric_density",
    "pathInX0_interval",
    "dead_module_flag",
    "layer_embed_idx",
    "n_hits",
    "n_holes",
    "n_seq_holes",
    "action_taken",
    "prune_reason",
    "contrib_pids",
    "contrib_charge_frac",
    "branch_majority_pid",
    "majority_undefined",
    "majority_true_hit_on_surface",
    "truth_residual_l0",
    "truth_residual_l1",
    "vstar_soft",
    "env_config_hash",
)


def split_of(event_id: int) -> str:
    """Return the split name for one event.

    Parameters
    ----------
    event_id : int
        Event index.

    Returns
    -------
    str
        One of ``"train"``, ``"val"``, ``"cal"``, ``"test"``.

    Raises
    ------
    KeyError
        If ``event_id`` has no assignment. All 32 events in [0, 32) are
        assigned, so this fires only for an out-of-range event or if someone
        edits the three tuples inconsistently and drops one — both of which
        should be loud rather than silently defaulting to "train".
    """
    if event_id in TEST_EVENTS:
        return "test"
    try:
        return _LOWER_ASSIGNMENT[event_id]
    except KeyError:
        raise KeyError(
            f"event {event_id} has no split assignment. Events [0, 32) are "
            f"partitioned 24/4/4 across TRAIN/VAL/CAL and [32, 64) is the "
            f"sealed test set; {event_id} is outside both, or the tuples no "
            f"longer partition [0, 32)."
        ) from None


def events_for(split: str) -> tuple[int, ...]:
    """Return the event tuple for a split name."""
    table = {
        "train": TRAIN_EVENTS,
        "val": VAL_EVENTS,
        "cal": CAL_EVENTS,
        "test": TEST_EVENTS,
    }
    if split not in table:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(table)}")
    return table[split]


def assert_not_test(event_ids: Iterable[int]) -> None:
    """Raise if any event is in the sealed test range.

    Call this at the top of every data-collection and training entrypoint.
    The test set is opened exactly once, deliberately, by the final evaluation
    script — never by a training or tuning run.
    """
    sealed = sorted(set(event_ids) & set(TEST_EVENTS))
    if sealed:
        raise ValueError(
            f"refusing to touch sealed test events {sealed}; "
            f"events [32, 64) are opened once for headline numbers only"
        )
