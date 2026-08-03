"""Decision-log Parquet schema and writer skeleton (cCKF Spec §6.5).

Tier 2 scaffolding only. Label functions, losses, and calibrators are Tier 1
and must not live here.

Replay precedent
----------------
Max Zhao's KalmanML Python hit-following path is the named precedent for the
Python replay environment (spec §15.6):

* ``KalmanML/scripts/policy_tracker.py`` — recursive track extension from a
  seed under a policy threshold (``PolicyTracker.track``).
* ``KalmanML/utils/hit_locator.py`` — per-layer hit lookup near a predicted
  intersection (``get_near_hits`` / ``get_hits_around``).
* ``KalmanML/mcts.py`` — ``SingleTracker`` / ``EventTracker`` tree search with
  helix propagation via ``utils/helix.py``.

That stack is a *simplified* helix + KD-tree replay, not the ACTS propagator.
cCKF's label-generating MDP still requires ACTS state/covariance transport;
KalmanML is the structural template (seed → candidates → accept/hole → log),
not a drop-in physics substitute.

IRREVERSIBLE fields (cannot be reconstructed after the run) are always present
in the schema even when a row leaves them empty — the writer refuses to omit
columns.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - Modal image may need pip install
    pa = None  # type: ignore
    pq = None  # type: ignore


# ─── Enums ───────────────────────────────────────────────────────────────


class ActionTaken(IntEnum):
    """Accepted / rejected / hole / branch-pruned (spec §6.5)."""

    ACCEPTED = 0
    REJECTED = 1
    HOLE = 2
    BRANCH_PRUNED = 3


class PruneReason(IntEnum):
    """Which heuristic or tractability cap fired. Logged whenever a prune occurs.

    Values are stable integers for Parquet; add new reasons at the end only.
    """

    NONE = 0
    BRANCH_CAP = 1  # numMeasurementsCutOff / beam width
    CHI2_MEASUREMENT = 2
    CHI2_OUTLIER = 3
    GLOBAL_BRANCH_LIMIT = 4
    OTHER = 255


# ─── Schema ──────────────────────────────────────────────────────────────


# Canonical column order for §6.5. List/array fields use Arrow list types.
DECISION_LOG_COLUMNS: tuple[str, ...] = (
    # lineage [IRREVERSIBLE]
    "event_id",
    "seed_id",
    "branch_id",
    "parent_branch_id",
    "step_k",
    # geometry
    "layer_id",
    "surface_id",
    # predicted state / cov
    "state",  # float[6]
    "cov_packed",  # float[21] lower-triangular
    "pred_local",  # float[2]
    # candidate
    "cand_hit_id",  # -1 = hole
    "residual",  # float[2]
    "chi2_inc",
    "cluster_feats",  # float[C]: s_u, s_v, Q_tot, σ_uu, σ_uv, σ_vv
    # [IRREVERSIBLE] incidence from predicted direction
    "incidence",  # float[2] (α_u, α_v)
    "sensor_props",  # float[3–5]
    "occupancy_feats",
    "context_feats",
    "branch_feats",
    "action_taken",
    # [IRREVERSIBLE]
    "prune_reason",
    "contrib_pids",  # int64[]
    "contrib_charge_frac",  # float[]
    "branch_majority_pid",
    "majority_true_hit_on_surface",
    "truth_residual",  # float[2]
    "vstar_soft",
    "env_config_hash",
)

IRREVERSIBLE_FIELDS: frozenset[str] = frozenset(
    {
        "event_id",
        "seed_id",
        "branch_id",
        "parent_branch_id",
        "step_k",
        "incidence",
        "prune_reason",
        "contrib_pids",
        "contrib_charge_frac",
        "majority_true_hit_on_surface",
        "truth_residual",
    }
)


def _arrow_schema() -> "pa.Schema":
    if pa is None:
        raise ImportError("pyarrow is required to write decision logs")
    return pa.schema(
        [
            ("event_id", pa.int64()),
            ("seed_id", pa.int64()),
            ("branch_id", pa.int64()),
            ("parent_branch_id", pa.int64()),
            ("step_k", pa.int32()),
            ("layer_id", pa.int32()),
            ("surface_id", pa.int64()),
            ("state", pa.list_(pa.float32(), 6)),
            ("cov_packed", pa.list_(pa.float32(), 21)),
            ("pred_local", pa.list_(pa.float32(), 2)),
            ("cand_hit_id", pa.int64()),
            ("residual", pa.list_(pa.float32(), 2)),
            ("chi2_inc", pa.float32()),
            ("cluster_feats", pa.list_(pa.float32())),
            ("incidence", pa.list_(pa.float32(), 2)),
            ("sensor_props", pa.list_(pa.float32())),
            ("occupancy_feats", pa.list_(pa.float32())),
            ("context_feats", pa.list_(pa.float32())),
            ("branch_feats", pa.list_(pa.float32())),
            ("action_taken", pa.int8()),
            ("prune_reason", pa.int8()),
            ("contrib_pids", pa.list_(pa.int64())),
            ("contrib_charge_frac", pa.list_(pa.float32())),
            ("branch_majority_pid", pa.int64()),
            ("majority_true_hit_on_surface", pa.int8()),
            ("truth_residual", pa.list_(pa.float32(), 2)),
            ("vstar_soft", pa.float32()),
            ("env_config_hash", pa.string()),
        ]
    )


def env_config_hash(config: Mapping[str, Any]) -> str:
    """Stable hash of label-generating environment constants.

    Includes window_n, seeding+CKF following knobs, digi_config, and the
    disabled-terminal-cut flags. Rows with mismatched hashes must not be pooled.
    """
    keys = [
        "window_n",
        "num_seeds_per_spm",
        "seed_minPt",
        "seed_impactMax",
        "seed_sigmaScattering",
        "ckf_chi2CutOffMeasurement",
        "ckf_chi2CutOffOutlier",
        "ckf_numMeasurementsCutOff",
        "ckf_nMeasurementsMin",
        "ckf_maxHolesAndOutliers",
        "ckf_ptMin",
        "digi_config",
        "forbid_loc0_cut",
    ]
    payload = {k: config.get(k) for k in keys}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class DecisionRow:
    """One (branch, layer, candidate) row. Defaults leave irreversible fields present."""

    event_id: int
    seed_id: int
    branch_id: int
    parent_branch_id: int
    step_k: int
    layer_id: int = -1
    surface_id: int = -1
    state: Sequence[float] = field(default_factory=lambda: [0.0] * 6)
    cov_packed: Sequence[float] = field(default_factory=lambda: [0.0] * 21)
    pred_local: Sequence[float] = field(default_factory=lambda: [0.0, 0.0])
    cand_hit_id: int = -1
    residual: Sequence[float] = field(default_factory=lambda: [0.0, 0.0])
    chi2_inc: float = float("nan")
    cluster_feats: Sequence[float] = field(default_factory=list)
    incidence: Sequence[float] = field(default_factory=lambda: [float("nan")] * 2)
    sensor_props: Sequence[float] = field(default_factory=list)
    occupancy_feats: Sequence[float] = field(default_factory=list)
    context_feats: Sequence[float] = field(default_factory=list)
    branch_feats: Sequence[float] = field(default_factory=list)
    action_taken: int = int(ActionTaken.HOLE)
    prune_reason: int = int(PruneReason.NONE)
    contrib_pids: Sequence[int] = field(default_factory=list)
    contrib_charge_frac: Sequence[float] = field(default_factory=list)
    branch_majority_pid: int = -1
    majority_true_hit_on_surface: int = -1
    truth_residual: Sequence[float] = field(
        default_factory=lambda: [float("nan")] * 2
    )
    vstar_soft: float = float("nan")
    env_config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        missing = IRREVERSIBLE_FIELDS - d.keys()
        if missing:
            raise RuntimeError(f"IRREVERSIBLE fields missing from row: {missing}")
        return d


class DecisionLogWriter:
    """Buffered Parquet writer for decision rows.

    Does not implement CKF/replay hooks — callers append ``DecisionRow``s.
    Prune events must set ``action_taken=BRANCH_PRUNED`` and a non-NONE
    ``prune_reason`` whenever ``log_prune_reasons`` is enabled in the config.
    """

    def __init__(self, output_path: Path | str, config: Mapping[str, Any]):
        self.output_path = Path(output_path)
        self.config = dict(config)
        self.hash = env_config_hash(self.config)
        self._rows: list[dict[str, Any]] = []
        if self.config.get("forbid_loc0_cut", True) and self.config.get(
            "ckf_loc0_max"
        ) is not None:
            raise ValueError(
                "ckf_loc0_max is set but forbid_loc0_cut=True — loc0 ±4 mm cut "
                "must be absent from envelope collection (spec §4.1)."
            )

    def append(self, row: DecisionRow) -> None:
        if not row.env_config_hash:
            row.env_config_hash = self.hash
        elif row.env_config_hash != self.hash:
            raise ValueError(
                f"row env_config_hash {row.env_config_hash} != writer {self.hash}"
            )
        if (
            self.config.get("log_prune_reasons", True)
            and row.action_taken == int(ActionTaken.BRANCH_PRUNED)
            and row.prune_reason == int(PruneReason.NONE)
        ):
            raise ValueError("BRANCH_PRUNED row missing prune_reason")
        self._rows.append(row.to_dict())

    def __len__(self) -> int:
        return len(self._rows)

    def flush(self) -> Path:
        if pa is None or pq is None:
            raise ImportError("pyarrow is required to flush decision logs")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._rows:
            # Write empty table with full schema so downstream code can open it.
            table = pa.Table.from_pylist([], schema=_arrow_schema())
        else:
            table = pa.Table.from_pylist(self._rows, schema=_arrow_schema())
        # Ensure every schema column exists (from_pylist already does).
        missing = set(DECISION_LOG_COLUMNS) - set(table.column_names)
        if missing:
            raise RuntimeError(f"Parquet missing columns: {sorted(missing)}")
        pq.write_table(table, self.output_path)
        return self.output_path


def write_skeleton_example(
    output_path: Path | str, config: Mapping[str, Any]
) -> Path:
    """Write a one-row skeleton Parquet proving the §6.5 schema is complete."""
    writer = DecisionLogWriter(output_path, config)
    writer.append(
        DecisionRow(
            event_id=0,
            seed_id=0,
            branch_id=0,
            parent_branch_id=-1,
            step_k=0,
            cand_hit_id=-1,
            action_taken=int(ActionTaken.HOLE),
            prune_reason=int(PruneReason.NONE),
            majority_true_hit_on_surface=0,
            cluster_feats=[float("nan")] * 6,
            contrib_pids=[],
            contrib_charge_frac=[],
        )
    )
    return writer.flush()


def assert_schema(path: Path | str) -> list[str]:
    """Return column names; raise if any §6.5 / IRREVERSIBLE column is missing."""
    if pq is None:
        raise ImportError("pyarrow is required")
    schema = pq.read_schema(path)
    names = list(schema.names)
    missing = [c for c in DECISION_LOG_COLUMNS if c not in names]
    if missing:
        raise AssertionError(f"decision log missing columns: {missing}")
    irr_missing = [c for c in IRREVERSIBLE_FIELDS if c not in names]
    if irr_missing:
        raise AssertionError(f"IRREVERSIBLE columns missing: {irr_missing}")
    return names
