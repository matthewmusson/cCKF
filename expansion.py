"""Offline expansion pipeline for cCKF pilot data collection (spec Sec 6.5).

For each CKF branch-state written to ``trackstates_ckf.root``, find every
measurement on the same sensitive surface that falls inside the axis-aligned
Mahalanobis window W_k(n), and emit one row per (branch, surface, candidate).
Surfaces with zero in-window candidates emit a single "hole" row
(``cand_hit_id == -1``).

This module is pure: every function takes data in and returns data out. No
Modal, no remote execution, no ACTS python bindings. It is driven entirely by
``uproot``/``awkward`` (ROOT) and ``pandas``/``numpy`` (CSV).

Data sources per output field
------------------------------
ROOT trackstates
    predicted state, predicted-error diagonal, innovation covariance
    (S00_prt/S01_prt/S11_prt), incidence angles, pathInX0, cluster features
    of the CKF-*selected* hit, per-state contributing particle IDs.
measurements.csv
    candidate local position + measurement variance, for ALL candidates on a
    surface (not just the selected hit).
cells.csv
    channel-level cluster features for ALL candidates.
measurement-simhit-map.csv + simhits.csv
    truth chain: measurement -> hit_id (row index into simhits.csv) ->
    contributing particle(s).

Known gaps (see task-2-report.md for the full list)
-----------------------------------------------------
- ``cov_packed`` (full 6x6 predicted covariance, 21 lower-triangular values)
  is only partially populated: the ROOT tree exposes the 6 diagonal errors
  and the 2x2 local innovation covariance (S00/S01/S11), but not the
  loc-phi/loc-theta/... predicted cross-covariances. Off-diagonal slots are
  NaN.
- ``truth_residual`` (h - h^true_m from the SimTrackerHit) requires a
  global -> local transform through the ACTS tracking geometry, which is not
  available from CSV/ROOT alone. Left as NaN, matching the precedent set in
  ``scripts/expand_pilot.py``.
- ``dead_module_flag`` has no data source yet; always 0.
- ``majority_true_hit_on_surface`` is computed from simhit presence, which
  conflates "genuine hole" and "window failure" (spec Sec 7.4). Splitting
  them requires per-measurement geometry_id inside compute_truth_labels,
  which is not part of the function's given signature.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import awkward as ak
except ImportError:  # pragma: no cover
    ak = None

try:
    import uproot
except ImportError:  # pragma: no cover
    uproot = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None
    pq = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_N_DEFAULT = 10.0
R_GEOM_MM_DEFAULT = 5.0

# ACTS GeometryIdentifier bit layout (Core/Geometry/GeometryIdentifier.hpp).
# Kept as plain ints (not np.uint64) since real ODD volume IDs are small
# (<64), so shifted values stay well under the int64 range and ordinary
# int64 arithmetic (numpy or python) is safe -- no need for unsigned masks.
_VOLUME_SHIFT = 56
_BOUNDARY_SHIFT = 48
_LAYER_SHIFT = 36
_SENSITIVE_SHIFT = 8

_VOLUME_BITS = 0xFF
_LAYER_BITS = 0xFFF
_SENSITIVE_BITS = 0xFFFFF

# Best-effort volume classification (matches the convention already used in
# scripts/expand_pilot.py). NOT independently re-verified against the ACTS
# tracking geometry for this task -- treat as a documented assumption.
PIXEL_VOLUMES_DEFAULT = frozenset({16, 17, 18})
BARREL_VOLUMES_DEFAULT = frozenset({16, 23, 28})

# ROOT trackstate branches this module reads. Missing branches are dropped
# with a warning rather than raising, so the pipeline degrades gracefully on
# older/newer ACTS builds.
#: Absolute tolerance (mm) for matching a candidate's measurement position to
#: the CKF-accepted hit position (l_x_hit) when marking is_ckf_selected.
#: Measured agreement on real data is exact to 1e-6.
SEL_MATCH_TOL = 1e-4

_TRACKSTATE_SCALAR_BRANCHES = [
    "volume_id",
    "layer_id",
    "module_id",
    "predicted",
    "eLOC0_prt",
    "eLOC1_prt",
    "ePHI_prt",
    "eTHETA_prt",
    "eQOP_prt",
    "eT_prt",
    "err_eLOC0_prt",
    "err_eLOC1_prt",
    "err_ePHI_prt",
    "err_eTHETA_prt",
    "err_eQOP_prt",
    "err_eT_prt",
    "eta_prt",
    "S00_prt",
    "S01_prt",
    "S11_prt",
    "pathInX0_interval",
    "l_x_hit",
    "l_y_hit",
    "clus_size_u",
    "clus_size_v",
    "clus_qtot",
    "clus_sigma_uu",
    "clus_sigma_uv",
    "clus_sigma_vv",
    "alpha_u",
    "alpha_v",
]

# Doubly-jagged (per-track, per-state, per-contributor) particle ID
# components for the CKF-*selected* hit at each state.
_TRACKSTATE_PARTICLE_BRANCHES = [
    "particle_ids_particle",
    "particle_ids_vertex_primary",
    "particle_ids_vertex_secondary",
    "particle_ids_generation",
    "particle_ids_sub_particle",
]

# §6.5 schema columns for the final Parquet output.
SCHEMA_COLUMNS = [
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
    *[f"cov_{i:02d}" for i in range(21)],
    "pred_l0",
    "pred_l1",
    "cand_hit_id",
    "residual_l0",
    "residual_l1",
    "S00",
    "S01",
    "S11",
    "chi2_inc",
    "var_local0",
    "var_local1",
    "is_1d",
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
]

# action_taken codes
ACTION_CANDIDATE = 0  # in-window candidate, accept/reject decided offline
ACTION_HOLE_WINDOW_FAILURE = 1  # measurements existed on surface, none in window
ACTION_HOLE_NO_MEASUREMENTS = 2  # zero measurements on this surface at all


# ---------------------------------------------------------------------------
# Geometry ID helpers
# ---------------------------------------------------------------------------


def decode_geometry_id(geometry_id: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a packed ACTS ``geometry_id`` into (volume, layer, sensitive).

    Parameters
    ----------
    geometry_id : array-like of int
        Packed geometry identifiers, as read from a CSV ``geometry_id``
        column.

    Returns
    -------
    volume, layer, sensitive : np.ndarray
        Decoded components. ``sensitive`` corresponds to ``module_id`` in the
        trackstates ROOT tree.
    """
    gid = np.asarray(geometry_id, dtype=np.int64)
    volume = (gid >> _VOLUME_SHIFT) & _VOLUME_BITS
    layer = (gid >> _LAYER_SHIFT) & _LAYER_BITS
    sensitive = (gid >> _SENSITIVE_SHIFT) & _SENSITIVE_BITS
    return volume, layer, sensitive


def normalize_geometry_id(geometry_id: np.ndarray | pd.Series) -> np.ndarray:
    """Zero the ``extra`` byte (bits 0-7) of a packed ACTS ``GeometryIdentifier``.

    ODD endcap volumes (16, 18, 23, 25, 28, 30) carry a nonzero ``extra``
    field -- the ring index, 1-3 -- in the geometry_id written to the
    digitization CSVs. Barrel volumes are all ``extra = 0``. The trackstates
    ROOT tree stores only ``volume_id / layer_id / module_id``, so any
    geometry_id reconstructed from it (:func:`encode_geometry_id`) has
    ``extra = 0`` and can never equal a raw endcap CSV gid.

    This is exactly why the 2026-08 re-expansion produced zero endcap
    candidates: the states-to-measurements merge on geometry_id matched only
    barrels (volumes 17/24/29), and every endcap state -- 39.46% of ROOT
    measurement states on event 1, 1.82M states -- degenerated to a hole row.

    Dropping ``extra`` loses nothing: ``(volume, layer, sensitive)`` is unique
    over all 18,918 surfaces in detectors.csv. Every CSV loader applies this
    at read time so the entire pipeline operates in ``extra = 0`` space.
    """
    gid = np.asarray(geometry_id, dtype=np.int64)
    return gid & ~np.int64(0xFF)


def encode_geometry_id(
    volume_id: np.ndarray | pd.Series,
    layer_id: np.ndarray | pd.Series,
    module_id: np.ndarray | pd.Series,
) -> np.ndarray:
    """Build the composite ``geometry_id`` from trackstates' separate columns.

    ``module_id`` in the trackstates ROOT tree is the ``sensitive()``
    component of the ACTS ``GeometryIdentifier``; the boundary field is
    always 0 for sensitive surfaces, matching the raw ``geometry_id`` values
    written to the digitization CSVs.
    """
    vol = np.asarray(volume_id, dtype=np.int64)
    lay = np.asarray(layer_id, dtype=np.int64)
    mod = np.asarray(module_id, dtype=np.int64)
    return (vol << _VOLUME_SHIFT) | (lay << _LAYER_SHIFT) | (mod << _SENSITIVE_SHIFT)


def encode_particle_id(
    pv: np.ndarray | pd.Series,
    sv: np.ndarray | pd.Series,
    part: np.ndarray | pd.Series,
    gen: np.ndarray | pd.Series,
    subpart: np.ndarray | pd.Series,
) -> np.ndarray:
    """Encode the five simhit particle-barcode fields into a single int64.

    ``pid = (pv << 48) | (sv << 32) | (part << 16) | (gen << 8) | subpart``

    This is a self-consistent bit-packing used only within this pipeline
    (for grouping/matching); it is not guaranteed to match ACTS's internal
    ``SimBarcode`` layout, but that is not required -- both truth labels and
    the seed-majority computation use this same encoding, so consistency
    within the pipeline is all that matters.
    """
    a = np.asarray(pv, dtype=np.int64)
    b = np.asarray(sv, dtype=np.int64)
    c = np.asarray(part, dtype=np.int64)
    d = np.asarray(gen, dtype=np.int64)
    e = np.asarray(subpart, dtype=np.int64)
    return (a << 48) | (b << 32) | (c << 16) | (d << 8) | e


# ---------------------------------------------------------------------------
# Step 1: data loading
# ---------------------------------------------------------------------------


def _find_tree(root_file: Any, name_hints: tuple[str, ...]) -> Any:
    keys = [k.split(";")[0] for k in root_file.keys()]
    for hint in name_hints:
        if hint in keys:
            return root_file[hint]
    # fall back to first key containing any hint substring
    for k in keys:
        if any(h in k.lower() for h in name_hints):
            return root_file[k]
    if keys:
        return root_file[keys[0]]
    raise FileNotFoundError(f"No trees found (hints={name_hints})")


def _to_python_list(x: Any) -> list:
    """Convert one awkward/np entry to a plain python list."""
    if x is None:
        return []
    try:
        return ak.to_list(x)
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        return list(x)
    except TypeError:
        return [x]


def _mode_or_first(values: list[int]) -> int | None:
    """Mode of a list of ints, ties broken by first-listed value."""
    if not values:
        return None
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = values[0]
    best_n = counts[best]
    for v in values:
        if counts[v] > best_n:
            best = v
            best_n = counts[v]
    return best


def load_trackstates(root_path: str, event_id: int) -> pd.DataFrame:
    """Load trackstates for one event from ``trackstates_ckf.root``.

    Filters ROOT entries by the standard ACTS ``event_nr`` branch (a single
    trackstates file can hold multiple events, e.g. Stage 1's pilot run
    writes 2 events into one file). Returns a flat DataFrame with one row
    per (track_nr, state_idx).

    Parameters
    ----------
    root_path : str
        Path to ``trackstates_ckf.root``.
    event_id : int
        Event to select via the ``event_nr`` branch. If the branch is
        absent, all entries are treated as belonging to ``event_id`` (with a
        printed warning) -- this matches single-event files produced by
        older pipelines.

    Returns
    -------
    pd.DataFrame
        Columns: ``event_id, track_nr, state_idx, geometry_id, volume_id,
        layer_id, module_id, is_predicted, pred_l0, pred_l1, pred_phi,
        pred_theta, pred_qop, pred_t, err_l0, err_l1, err_phi, err_theta,
        err_qop, err_t, S00, S01, S11, eta, pathInX0_interval, clus_s_u,
        clus_s_v, clus_q_tot, clus_sigma_uu, clus_sigma_uv, clus_sigma_vv,
        alpha_u, alpha_v, state_primary_pid, state_n_contribs``.

        ``state_primary_pid`` is the mode of that state's (possibly merged)
        contributor list, encoded via :func:`encode_particle_id`; used for
        seed-majority computation. ``state_n_contribs`` is the number of
        contributing particles (int; >0 means a real hit, 0 means a hole).
    """
    if uproot is None or ak is None:
        raise ImportError("uproot and awkward are required to read trackstates ROOT files")

    empty_cols = [
        "event_id", "track_nr", "state_idx", "geometry_id", "volume_id",
        "layer_id", "module_id", "is_predicted", "pred_l0", "pred_l1",
        "pred_phi", "pred_theta", "pred_qop", "pred_t", "err_l0", "err_l1",
        "err_phi", "err_theta", "err_qop", "err_t", "S00", "S01", "S11",
        "eta", "pathInX0_interval", "sel_l0", "sel_l1", "clus_s_u",
        "clus_s_v", "clus_q_tot", "clus_sigma_uu", "clus_sigma_uv",
        "clus_sigma_vv", "alpha_u", "alpha_v", "state_primary_pid",
        "state_n_contribs",
    ]

    with uproot.open(root_path) as f:
        tree = _find_tree(f, ("trackstates",))
        available = set(tree.keys())

        scalar_fields = [b for b in _TRACKSTATE_SCALAR_BRANCHES if b in available]
        missing = [b for b in _TRACKSTATE_SCALAR_BRANCHES if b not in available]
        if missing:
            print(f"[expansion] WARNING: missing trackstate branches: {missing}")

        particle_fields = [b for b in _TRACKSTATE_PARTICLE_BRANCHES if b in available]
        has_particle_ids = len(particle_fields) == len(_TRACKSTATE_PARTICLE_BRANCHES)
        if not has_particle_ids:
            print(
                "[expansion] WARNING: incomplete particle_ids_* branches -- "
                "seed-majority computation will be unavailable"
            )

        fields = scalar_fields + particle_fields
        has_event_nr = "event_nr" in available
        read_fields = fields + (["event_nr"] if has_event_nr else [])
        if not has_event_nr:
            print(
                "[expansion] WARNING: no 'event_nr' branch -- treating entire "
                f"file as event {event_id}"
            )

        # NOTE: uproot's `cut=` expression filtering is not used here -- it
        # was observed to silently no-op (return all entries) against an
        # RNTuple-backed test file in this environment (uproot 5.7.5), and
        # to raise outright for library="np". Boolean-mask indexing on the
        # loaded awkward Array is applied instead, which works identically
        # for both TTree- and RNTuple-backed files and was verified against
        # a synthetic multi-event file.
        arrays = tree.arrays(read_fields, library="ak")
        if has_event_nr:
            event_mask = ak.to_numpy(arrays["event_nr"]) == int(event_id)
            arrays = arrays[event_mask]

    n_tracks = len(arrays)
    if n_tracks == 0 or "volume_id" not in scalar_fields:
        return pd.DataFrame(columns=empty_cols)

    n_states = ak.to_numpy(ak.num(arrays["volume_id"], axis=1))
    track_nr = np.repeat(np.arange(n_tracks, dtype=np.int64), n_states)
    state_idx = ak.to_numpy(ak.flatten(ak.local_index(arrays["volume_id"], axis=1))).astype(np.int64)

    flat: dict[str, np.ndarray] = {"track_nr": track_nr, "state_idx": state_idx}
    for name in scalar_fields:
        vals = ak.flatten(arrays[name], axis=1)
        flat[name] = ak.to_numpy(vals)

    vol = flat.get("volume_id", np.zeros(len(track_nr), dtype=np.int64)).astype(np.int64)
    lay = flat.get("layer_id", np.zeros(len(track_nr), dtype=np.int64)).astype(np.int64)
    mod = flat.get("module_id", np.zeros(len(track_nr), dtype=np.int64)).astype(np.int64)

    out = pd.DataFrame({
        "event_id": event_id,
        "track_nr": track_nr,
        "state_idx": state_idx,
        "geometry_id": encode_geometry_id(vol, lay, mod),
        "volume_id": vol,
        "layer_id": lay,
        "module_id": mod,
        "is_predicted": flat.get("predicted", np.ones(len(track_nr), dtype=bool)).astype(bool),
        "pred_l0": flat.get("eLOC0_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "pred_l1": flat.get("eLOC1_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "pred_phi": flat.get("ePHI_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "pred_theta": flat.get("eTHETA_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "pred_qop": flat.get("eQOP_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "pred_t": flat.get("eT_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "err_l0": flat.get("err_eLOC0_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "err_l1": flat.get("err_eLOC1_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "err_phi": flat.get("err_ePHI_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "err_theta": flat.get("err_eTHETA_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "err_qop": flat.get("err_eQOP_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "err_t": flat.get("err_eT_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "S00": flat.get("S00_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "S01": flat.get("S01_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "S11": flat.get("S11_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "eta": flat.get("eta_prt", np.full(len(track_nr), np.nan)).astype(np.float64),
        "pathInX0_interval": flat.get("pathInX0_interval", np.full(len(track_nr), np.nan)).astype(np.float64),
        # The CKF-selected measurement's local position at this state
        # (all-state branches; NaN at holes / material states). This is what
        # marks is_ckf_selected inline in expand_trackstates -- see the
        # SEL_MATCH_TOL comment there.
        "sel_l0": flat.get("l_x_hit", np.full(len(track_nr), np.nan)).astype(np.float64),
        "sel_l1": flat.get("l_y_hit", np.full(len(track_nr), np.nan)).astype(np.float64),
        "clus_s_u": flat.get("clus_size_u", np.full(len(track_nr), np.nan)).astype(np.float64),
        "clus_s_v": flat.get("clus_size_v", np.full(len(track_nr), np.nan)).astype(np.float64),
        "clus_q_tot": flat.get("clus_qtot", np.full(len(track_nr), np.nan)).astype(np.float64),
        "clus_sigma_uu": flat.get("clus_sigma_uu", np.full(len(track_nr), np.nan)).astype(np.float64),
        "clus_sigma_uv": flat.get("clus_sigma_uv", np.full(len(track_nr), np.nan)).astype(np.float64),
        "clus_sigma_vv": flat.get("clus_sigma_vv", np.full(len(track_nr), np.nan)).astype(np.float64),
        "alpha_u": flat.get("alpha_u", np.full(len(track_nr), np.nan)).astype(np.float64),
        "alpha_v": flat.get("alpha_v", np.full(len(track_nr), np.nan)).astype(np.float64),
    })

    # Per-state primary contributor particle ID (mode of the, possibly
    # merged, contributor list at the CKF-selected hit). Needed for
    # seed-majority computation in compute_branch_majority_pid.
    if has_particle_ids:
        part_flat = ak.flatten(arrays["particle_ids_particle"], axis=1)
        pv_flat = ak.flatten(arrays["particle_ids_vertex_primary"], axis=1)
        sv_flat = ak.flatten(arrays["particle_ids_vertex_secondary"], axis=1)
        gen_flat = ak.flatten(arrays["particle_ids_generation"], axis=1)
        sub_flat = ak.flatten(arrays["particle_ids_sub_particle"], axis=1)

        n_rows = len(out)
        counts = ak.to_numpy(ak.num(part_flat)).astype(np.int64)

        flat_p = ak.to_numpy(ak.flatten(part_flat))
        flat_pv = ak.to_numpy(ak.flatten(pv_flat))
        flat_sv = ak.to_numpy(ak.flatten(sv_flat))
        flat_gen = ak.to_numpy(ak.flatten(gen_flat))
        flat_sub = ak.to_numpy(ak.flatten(sub_flat))

        min_total = min(len(flat_p), len(flat_pv), len(flat_sv),
                        len(flat_gen), len(flat_sub))
        all_pids = encode_particle_id(
            flat_pv[:min_total], flat_sv[:min_total],
            flat_p[:min_total], flat_gen[:min_total], flat_sub[:min_total],
        )

        offsets = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        offsets = np.minimum(offsets, min_total)

        primary_pid = np.full(n_rows, -1, dtype=np.int64)
        has_any = counts > 0
        single = counts == 1
        single_idx = np.where(single)[0]
        if len(single_idx):
            primary_pid[single_idx] = all_pids[offsets[single_idx]]

        multi_idx = np.where(has_any & ~single)[0]
        for idx in multi_idx:
            pids = all_pids[offsets[idx]:offsets[idx + 1]]
            vals, cnts = np.unique(pids, return_counts=True)
            primary_pid[idx] = vals[np.argmax(cnts)]

        out["state_primary_pid"] = primary_pid
        out["state_n_contribs"] = counts
    else:
        out["state_primary_pid"] = -1
        out["state_n_contribs"] = 0

    return out


def load_measurements(csv_dir: str, event_id: int) -> pd.DataFrame:
    """Load ``event{N:09d}-measurements.csv`` for one event.

    Returns
    -------
    pd.DataFrame
        Columns: ``measurement_id, geometry_id, local0, local1, var_local0,
        var_local1``.
    """
    path = Path(csv_dir) / f"event{event_id:09d}-measurements.csv"
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]

    measurement_id = (
        df["measurement_id"].to_numpy(dtype=np.int64)
        if "measurement_id" in df.columns
        else np.arange(len(df), dtype=np.int64)
    )
    out = pd.DataFrame({
        "measurement_id": measurement_id,
        "geometry_id": normalize_geometry_id(df["geometry_id"]),
        "local0": df["local0"].to_numpy(dtype=np.float64),
        "local1": df["local1"].to_numpy(dtype=np.float64),
        "var_local0": df["var_local0"].to_numpy(dtype=np.float64),
        "var_local1": df["var_local1"].to_numpy(dtype=np.float64),
    })
    return out


def load_predicted_cov(csv_dir: str, event_id: int) -> pd.DataFrame:
    """Load ``event{N:09d}-predicted-cov.csv`` for one event.

    This is the predicted covariance projected into local surface coordinates,
    i.e. the ``H C H^T`` term of the innovation covariance, written per track
    state by ``utils/predicted_cov_writer.PredictedCovWriter``.

    It is what makes a *per-candidate* ``S`` possible. The full innovation
    covariance is ``S = V + H C H^T``, where ``H C H^T`` is shared by every
    candidate on a surface (they share the predicted state) and ``V`` is the
    candidate's own measurement variance. Reading ``P`` here and ``V`` from
    :func:`load_measurements` lets :func:`expand_trackstates` build ``S`` from
    first principles rather than reading the ``S11_prt`` column, which is
    corrupted for 34-35% of rows in events 0-3.

    Returns
    -------
    pd.DataFrame
        Columns: ``track_nr, state_idx, P00, P01, P11``. ``state_idx`` is the
        CSV's ``step_k``, renamed to match the trackstates join key.
    """
    path = Path(csv_dir) / f"event{event_id:09d}-predicted-cov.csv"
    # This file is ~2.5 GB per event and pandas' parser dominates the
    # expansion wall time. pyarrow's threaded CSV reader is several times
    # faster; fall back to pandas if it is unavailable.
    try:
        from pyarrow import csv as _pacsv

        tbl = _pacsv.read_csv(
            str(path),
            convert_options=_pacsv.ConvertOptions(
                include_columns=["track_nr", "step_k", "P00", "P01", "P11"]
            ),
        )
        df = tbl.to_pandas()
    except Exception:
        df = pd.read_csv(path, comment="#",
                         usecols=["track_nr", "step_k", "P00", "P01", "P11"])
    df.columns = [c.strip() for c in df.columns]
    return pd.DataFrame({
        "track_nr": df["track_nr"].to_numpy(dtype=np.int64),
        "state_idx": df["step_k"].to_numpy(dtype=np.int64),
        "P00": df["P00"].to_numpy(dtype=np.float64),
        "P01": df["P01"].to_numpy(dtype=np.float64),
        "P11": df["P11"].to_numpy(dtype=np.float64),
    })


def load_cells(csv_dir: str, event_id: int) -> pd.DataFrame:
    """Load ``event{N:09d}-cells.csv`` for one event.

    Returns
    -------
    pd.DataFrame
        Columns: ``geometry_id, measurement_id, channel0, channel1, value``.
    """
    path = Path(csv_dir) / f"event{event_id:09d}-cells.csv"
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({
        "geometry_id": normalize_geometry_id(df["geometry_id"]),
        "measurement_id": df["measurement_id"].to_numpy(dtype=np.int64),
        "channel0": df["channel0"].to_numpy(dtype=np.float64),
        "channel1": df["channel1"].to_numpy(dtype=np.float64),
        "value": df["value"].to_numpy(dtype=np.float64),
    })
    return out


def load_simhits(csv_dir: str, event_id: int) -> pd.DataFrame:
    """Load ``event{N:09d}-simhits.csv`` for one event.

    The row index (0-based, file order) is the ``hit_id`` referenced by
    ``measurement-simhit-map.csv``, so it is preserved explicitly as a
    column.

    Returns
    -------
    pd.DataFrame
        Columns: ``hit_id, geometry_id, particle_id, tx, ty, tz``.
        ``particle_id`` is the int64 encoding from
        :func:`encode_particle_id`.
    """
    path = Path(csv_dir) / f"event{event_id:09d}-simhits.csv"
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]

    def _col(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.zeros(len(df), dtype=np.int64)
        return df[name].to_numpy(dtype=np.int64)

    particle_id = encode_particle_id(
        _col("particle_id_pv"),
        _col("particle_id_sv"),
        _col("particle_id_part"),
        _col("particle_id_gen"),
        _col("particle_id_subpart"),
    )
    out = pd.DataFrame({
        "hit_id": np.arange(len(df), dtype=np.int64),
        "geometry_id": normalize_geometry_id(df["geometry_id"]) if "geometry_id" in df.columns else np.zeros(len(df), dtype=np.int64),
        "particle_id": particle_id,
        "tx": df["tx"].to_numpy(dtype=np.float64) if "tx" in df.columns else np.full(len(df), np.nan),
        "ty": df["ty"].to_numpy(dtype=np.float64) if "ty" in df.columns else np.full(len(df), np.nan),
        "tz": df["tz"].to_numpy(dtype=np.float64) if "tz" in df.columns else np.full(len(df), np.nan),
    })
    return out


def load_measurement_simhit_map(csv_dir: str, event_id: int) -> pd.DataFrame:
    """Load ``event{N:09d}-measurement-simhit-map.csv`` for one event.

    Returns
    -------
    pd.DataFrame
        Columns: ``measurement_id, hit_id`` (hit_id indexes into
        :func:`load_simhits`'s output row order).
    """
    path = Path(csv_dir) / f"event{event_id:09d}-measurement-simhit-map.csv"
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    mcol = "measurement_id" if "measurement_id" in df.columns else df.columns[0]
    hcol = "hit_id" if "hit_id" in df.columns else df.columns[1]
    return pd.DataFrame({
        "measurement_id": df[mcol].to_numpy(dtype=np.int64),
        "hit_id": df[hcol].to_numpy(dtype=np.int64),
    })


# ---------------------------------------------------------------------------
# Step 2: window computation and candidate matching
# ---------------------------------------------------------------------------


def compute_window_bounds(
    S00: np.ndarray, S11: np.ndarray, n: float = WINDOW_N_DEFAULT
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the axis-aligned Mahalanobis window half-widths.

    ``delta_l0 = n * sqrt(S00)``, ``delta_l1 = n * sqrt(S11)``.

    Parameters
    ----------
    S00, S11 : array-like of float
        Diagonal innovation-covariance entries at the predicted state
        (``S00_prt``, ``S11_prt`` from the trackstates ROOT tree).
    n : float
        Window multiplier (environment constant, spec §5.2/§6.3).

    Returns
    -------
    delta_l0, delta_l1 : np.ndarray
    """
    s00 = np.asarray(S00, dtype=np.float64)
    s11 = np.asarray(S11, dtype=np.float64)
    delta_l0 = n * np.sqrt(np.clip(s00, a_min=0.0, a_max=None))
    delta_l1 = n * np.sqrt(np.clip(s11, a_min=0.0, a_max=None))
    return delta_l0, delta_l1


def expand_trackstates(
    states: pd.DataFrame,
    measurements: pd.DataFrame,
    n: float = WINDOW_N_DEFAULT,
    r_geom_mm: float = R_GEOM_MM_DEFAULT,
    predicted_cov: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Match candidate measurements to trackstates within the n-sigma window.

    For each trackstate, finds all measurements on the same surface
    (``geometry_id``) with ``|meas.local0 - pred_l0| <= delta_l0`` and
    ``|meas.local1 - pred_l1| <= delta_l1``, where the window is computed
    once per state from that state's own S00/S11 (the ROOT-provided
    innovation covariance for the CKF-selected hit) via
    :func:`compute_window_bounds`. This is a documented approximation: the
    true per-candidate S varies slightly with each candidate's own
    measurement variance, but a single per-state window keeps the join a
    plain equi-join + vectorized filter (no per-row python loop), matching
    the task's numpy-vectorization requirement. chi2_inc (below) uses this
    same shared S00/S01/S11 for every candidate on the surface.

    Also emits hole rows for states with zero in-window candidates,
    distinguishing (via ``action_taken``) states with no measurements on the
    surface at all from states where measurements existed but none fell in
    the window (spec §7.4's "window failure").

    Implementation note: matching is done via a single vectorized
    ``pandas.merge`` on ``geometry_id`` followed by a boolean-mask filter,
    rather than an explicit python loop over trackstates.

    Parameters
    ----------
    states : pd.DataFrame
        Output of :func:`load_trackstates`.
    measurements : pd.DataFrame
        Output of :func:`load_measurements`.
    n : float
        Window multiplier.
    r_geom_mm : float
        Fixed-radius geometric occupancy window (independent of S), for the
        ``geometric_density`` occupancy feature (spec §6.5's second
        occupancy feature).

    Returns
    -------
    pd.DataFrame
        One row per (track_nr, state_idx, candidate), plus one hole row per
        state with zero in-window candidates. Carries all ``states`` columns
        plus: ``cand_hit_id, residual_l0, residual_l1, S00, S01, S11,
        n_window, geometric_density, action_taken``.
    """
    # NOTE: `S11.notna()` is deliberately NOT part of this mask.
    #
    # Volumes 28/29/30 are genuinely 1D, so instrumentation.patch writes
    # S11_prt = NaN for them by design. Filtering on it dropped every
    # long-strip state, leaving them in only 4 of 32 events -- and the gate
    # trained on that set accepts 0.000% of long-strip candidates, on the
    # surfaces with the HIGHEST base positive rate in the detector.
    # See experiments/LOG.md 2026-08-24.
    #
    # `S00.notna()` is ALSO dropped when predicted_cov is supplied.
    #
    # S00_prt is only filled where the CKF had a calibrated measurement, i.e.
    # for the selected hit. It is finite on just 23.78% of states, so keeping
    # the filter discarded three quarters of them and skewed the survivors
    # hard toward barrel volumes -- the re-expansion produced volumes 17/24/29
    # only, with every endcap missing.
    #
    # It is redundant now: S is rebuilt from var_local0 + P00, and the
    # predicted-cov join covers 100% of states (19,445,455 / 19,445,455).
    _mask = states["is_predicted"] & states["pred_l0"].notna()
    if predicted_cov is None:
        _mask = _mask & states["S00"].notna()
    valid = states[_mask].copy()
    valid = valid.reset_index(drop=True)
    valid["_state_row"] = valid.index.to_numpy()

    # Predicted covariance in local coords (the H C H^T term), per state.
    # Joined here so S can be rebuilt per candidate below.
    if predicted_cov is not None:
        valid = valid.merge(
            predicted_cov, on=["track_nr", "state_idx"], how="left"
        )
        valid["_state_row"] = valid.index.to_numpy()

    cand_cols = ["measurement_id", "geometry_id", "local0", "local1", "var_local0", "var_local1"]
    merged = valid.merge(
        measurements[cand_cols], on="geometry_id", how="left", suffixes=("", "_cand")
    )

    # Per-candidate innovation covariance, from first principles:
    #     S = V + H C H^T
    # with V = diag(var_local0, var_local1) belonging to THIS candidate and
    # H C H^T = (P00, P01, P11) shared across the surface.
    #
    # Two consequences beyond correctness:
    #   - the corrupted S11_prt column is never read, so the events 0-3
    #     corruption drops out of the pipeline entirely;
    #   - `is_1d` becomes exact. A 1D sensor has no var_local1. Deriving it
    #     this way does NOT conflate genuine-1D with corrupted-2D, which
    #     S11.isna() does.
    has_cand = merged["measurement_id"].notna()

    # ODD writes var_local1 == 0.0 for a 1D measurement -- NOT NaN. Verified
    # against event*-measurements.csv: n_nan = 0, n_zero = 36,184 / 188,144
    # (19.2%), matching the long-strip share.
    #
    # `.isna()` was wrong twice over: it never fired on a real 1D candidate,
    # and it DID fire on hole rows, where var_local1 is NaN simply because the
    # left join found no candidate. Hence `has_cand` -- is_1d is a property of
    # a measurement, and a hole has none.
    merged["is_1d"] = has_cand & (
        merged["var_local1"].isna() | (merged["var_local1"] <= 0.0)
    )

    if predicted_cov is not None:
        merged["S00"] = merged["var_local0"] + merged["P00"]
        merged["S01"] = merged["P01"]
        # S11 is left NaN on 1D rows rather than set to P11. A long strip has
        # no l1 measurement, so V has no (1,1) entry and S is genuinely 1x1.
        # Using P11 alone would claim the predicted covariance IS the l1
        # uncertainty, giving a window far tighter than the strip length and
        # rejecting nearly every long-strip candidate.
        merged["S11"] = np.where(
            merged["is_1d"], np.nan, merged["var_local1"] + merged["P11"])
        merged["S01"] = np.where(merged["is_1d"], np.nan, merged["S01"])

    # Dimension-aware n-sigma box. The l1 leg is only testable when l1 was
    # measured; on a 1D strip the box reduces to the l0 leg, where it
    # coincides exactly with the chi2 ellipse.
    delta_l0, delta_l1 = compute_window_bounds(
        merged["S00"].to_numpy(), merged["S11"].to_numpy(), n
    )
    merged["_delta_l0"] = delta_l0
    merged["_delta_l1"] = delta_l1

    r0 = merged["local0"] - merged["pred_l0"]
    r1 = merged["local1"] - merged["pred_l1"]
    in_window = (
        has_cand
        & (r0.abs() <= merged["_delta_l0"])
        & (merged["is_1d"] | (r1.abs() <= merged["_delta_l1"]))
    )
    geom_ok = has_cand & (np.hypot(r0, np.where(merged["is_1d"], 0.0, r1)) <= r_geom_mm)

    merged["_in_window"] = in_window
    merged["_geom_ok"] = geom_ok
    n_window = merged.groupby("_state_row")["_in_window"].transform("sum").astype(np.int64)
    n_geom = merged.groupby("_state_row")["_geom_ok"].transform("sum").astype(np.int64)
    merged["n_window"] = n_window
    merged["geometric_density"] = n_geom

    candidates = merged.loc[in_window].copy()
    candidates["cand_hit_id"] = candidates["measurement_id"].astype(np.int64)
    candidates["residual_l0"] = candidates["local0"] - candidates["pred_l0"]
    # A 1D measurement has no l1, so the residual is undefined rather than
    # zero. Keeping it NaN stops build_gate_features' zero-fill from making
    # every long-strip row look identical.
    candidates["residual_l1"] = np.where(
        candidates["is_1d"], np.nan,
        candidates["local1"] - candidates["pred_l1"])
    candidates["action_taken"] = ACTION_CANDIDATE

    # is_ckf_selected: the candidate whose measurement position equals the
    # accepted hit's position recorded by ROOT at this state (sel_l0/sel_l1
    # from the all-state l_x_hit/l_y_hit branches). Done inline because every
    # offline reconstruction of this flag failed (LOG 2026-08-17/-18/-25):
    # there is no second join here, so there is no key to get wrong.
    # Position agreement measured on real event 1: 100.0000% of states at
    # 1e-6 mm, so SEL_MATCH_TOL = 1e-4 carries three orders of slack.
    _d0 = (candidates["local0"] - candidates["sel_l0"]).abs()
    _d1 = (candidates["local1"] - candidates["sel_l1"]).abs()
    # l1 constrains the match only where the accepted hit has one; a 1D long
    # strip does not, and requiring it would unflag every long-strip state.
    _pos_ok = (
        (_d0 <= SEL_MATCH_TOL)
        & (candidates["is_1d"] | ~np.isfinite(candidates["sel_l1"]) | (_d1 <= SEL_MATCH_TOL))
        & np.isfinite(candidates["sel_l0"])
    ).to_numpy(dtype=bool)
    candidates["is_ckf_selected"] = False
    if _pos_ok.any():
        _close = candidates.loc[_pos_ok, ["_state_row", "cand_hit_id"]].copy()
        _close["_d0"] = _d0.to_numpy()[_pos_ok]
        _close["_orig"] = np.flatnonzero(_pos_ok)
        # closest wins; exact ties break to the lowest cand_hit_id, so at
        # most one candidate per state is ever flagged, deterministically.
        _winners = (
            _close.sort_values(["_state_row", "_d0", "cand_hit_id"])
            .groupby("_state_row", as_index=False)
            .first()
        )
        _sel = candidates["is_ckf_selected"].to_numpy(dtype=bool)
        _idx = candidates.index.to_numpy()[_winners["_orig"].to_numpy()]
        candidates.loc[_idx, "is_ckf_selected"] = True

    # Guard: among states where ROOT says the CKF accepted a hit AND at least
    # one candidate landed in the window, the accepted hit must be found
    # almost always. An all-False column is schema-identical to a valid one
    # and would silently train V_phi on "the CKF never accepted anything" --
    # the 2026-08-17 incident. Fail loudly instead.
    _states_with_sel = candidates.loc[
        np.isfinite(candidates["sel_l0"]), "_state_row"
    ].drop_duplicates()
    if len(_states_with_sel) > 0:
        _matched = candidates.loc[candidates["is_ckf_selected"], "_state_row"]
        _frac = len(_matched) / len(_states_with_sel)
        if _frac < 0.95:
            raise ValueError(
                f"is_ckf_selected: only {_frac:.2%} of "
                f"{len(_states_with_sel):,} states with an accepted hit and "
                f">=1 in-window candidate matched it by position "
                f"(SEL_MATCH_TOL={SEL_MATCH_TOL}); refusing to emit an "
                f"almost-all-False column."
            )

    # Hole rows: one per state with zero in-window candidates. Distinguish
    # "no measurements on surface" (has_cand all False for that state, i.e.
    # the left-merge produced exactly one all-NaN candidate row) from
    # "window failure" (candidates existed, none matched).
    rows_with_candidate_state = set(candidates["_state_row"].unique())
    hole_mask = ~merged["_state_row"].isin(rows_with_candidate_state)
    hole_rows = merged.loc[hole_mask].copy()
    # a state may appear multiple times in `merged` (multiple candidates on
    # its surface, all outside the window) -- collapse to one hole row.
    had_any_measurement_on_surface = hole_rows.groupby("_state_row")["measurement_id"].transform(
        lambda s: s.notna().any()
    )
    hole_rows["_had_meas"] = had_any_measurement_on_surface
    hole_rows = hole_rows.drop_duplicates(subset="_state_row").copy()
    hole_rows["cand_hit_id"] = -1
    hole_rows["residual_l0"] = np.nan
    hole_rows["residual_l1"] = np.nan
    hole_rows["action_taken"] = np.where(
        hole_rows["_had_meas"], ACTION_HOLE_WINDOW_FAILURE, ACTION_HOLE_NO_MEASUREMENTS
    )
    hole_rows["is_ckf_selected"] = False

    _stale = {"S00", "S01", "S11", "P00", "P01", "P11"}
    keep_cols = [c for c in valid.columns
                 if not c.startswith("_") and c not in _stale] + [
        "cand_hit_id", "residual_l0", "residual_l1", "n_window",
        "geometric_density", "action_taken", "is_ckf_selected",
        # per-candidate S and its provenance
        "S00", "S01", "S11", "var_local0", "var_local1", "is_1d",
    ]
    out = pd.concat(
        [candidates[keep_cols], hole_rows[keep_cols]], ignore_index=True, sort=False
    )
    # S00/S01/S11 for chi2_inc: the shared per-state innovation covariance
    # (see docstring note on the window-vs-chi2 approximation).
    out = out.rename(columns={"S00": "S00", "S01": "S01", "S11": "S11"})
    return out.sort_values(["track_nr", "state_idx", "cand_hit_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3: per-candidate feature computation
# ---------------------------------------------------------------------------


def compute_candidate_features(
    state_row: dict[str, float], cand_row: dict[str, float], S00: float, S01: float, S11: float
) -> dict[str, float]:
    """Compute residual and chi2_inc for one (state, candidate) pair.

    Scalar reference implementation matching the task-2-brief formula
    exactly; the vectorized batch version used inside
    :func:`expand_trackstates` (via :func:`chi2_increment_batch`) is what
    actually runs on the full join for performance.

    Parameters
    ----------
    state_row : dict
        Must contain ``pred_l0``, ``pred_l1``.
    cand_row : dict
        Must contain ``local0``, ``local1``.
    S00, S01, S11 : float
        Innovation covariance entries (full 2x2, including the off-diagonal
        S01) for this state.

    Returns
    -------
    dict
        ``{"residual_l0": r0, "residual_l1": r1, "chi2_inc": chi2}``.
    """
    r0 = cand_row["local0"] - state_row["pred_l0"]
    r1 = cand_row["local1"] - state_row["pred_l1"]
    det = S00 * S11 - S01 * S01
    if not np.isfinite(det) or det <= 0:
        s00 = S00 if S00 > 0 else 1e-30
        s11 = S11 if S11 > 0 else 1e-30
        chi2 = (r0 * r0) / s00 + (r1 * r1) / s11
    else:
        chi2 = (S11 * r0 * r0 - 2.0 * S01 * r0 * r1 + S00 * r1 * r1) / det
    return {"residual_l0": r0, "residual_l1": r1, "chi2_inc": float(chi2)}


def chi2_increment_batch(
    r0: np.ndarray, r1: np.ndarray, S00: np.ndarray, S01: np.ndarray,
    S11: np.ndarray, is_1d: np.ndarray | None = None
) -> np.ndarray:
    """Vectorized ``chi2_inc = r^T S^-1 r``, dimension-aware.

    A 1D measurement has no l1 coordinate at all: ``S`` is 1x1 and ``S(1,1)``
    does not exist. The correct statistic is then ``r0^2 / S00``, which is a
    chi2 with one degree of freedom, not a degenerate 2x2 inverse.

    Getting this wrong is why the gate rejects 100% of long strips today:
    every long-strip row received the same degenerate value, so the network
    could not tell them apart and -- at 1.27% of the training set under
    unweighted BCE -- learned to reject them all.

    Parameters
    ----------
    is_1d : array of bool, optional
        True where the measurement has no l1 coordinate. When omitted, falls
        back to treating non-finite ``S11`` as 1D, which is the historical
        behaviour and is ambiguous where ``S11`` is merely corrupt.
    """
    r0 = np.asarray(r0, dtype=np.float64)
    r1 = np.asarray(r1, dtype=np.float64)
    s00 = np.asarray(S00, dtype=np.float64)
    s01 = np.nan_to_num(np.asarray(S01, dtype=np.float64), nan=0.0)
    s11 = np.asarray(S11, dtype=np.float64)

    if is_1d is None:
        is_1d = ~np.isfinite(s11)
    is_1d = np.asarray(is_1d, dtype=bool)

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_1d = np.where(s00 > 0, (r0 * r0) / np.where(s00 > 0, s00, 1.0), np.nan)
        det = s00 * s11 - s01 * s01
        chi2_2d = (s11 * r0 * r0 - 2.0 * s01 * r0 * r1 + s00 * r1 * r1) / det
        # Degenerate 2D S (non-positive determinant): fall back to the
        # independent-coordinate form rather than emitting a negative chi2.
        chi2_2d_fallback = chi2_1d + np.where(
            s11 > 0, (r1 * r1) / np.where(s11 > 0, s11, 1.0), np.nan
        )

    valid_det = np.isfinite(det) & (det > 0)
    chi2_2d = np.where(valid_det, chi2_2d, chi2_2d_fallback)
    return np.where(is_1d, chi2_1d, chi2_2d)


# ---------------------------------------------------------------------------
# Step 4: cluster feature computation from cells
# ---------------------------------------------------------------------------


def compute_cluster_features(cells_for_measurement: pd.DataFrame) -> dict[str, float]:
    """Compute cluster size, charge, and moments for ONE measurement's cells.

    All quantities are in channel (bin) units, not physical units -- the
    conversion to physical units requires pitch (sensor_props), applied
    downstream at gate-training time, not here (spec §6.5 note; task-2-brief
    Step 4 decision).

    Parameters
    ----------
    cells_for_measurement : pd.DataFrame
        Rows of ``cells.csv`` (one measurement's cells), with columns
        ``channel0, channel1, value``.

    Returns
    -------
    dict
        ``s_u, s_v, q_tot, sigma_uu, sigma_uv, sigma_vv``.
    """
    if len(cells_for_measurement) == 0:
        return {
            "s_u": np.nan, "s_v": np.nan, "q_tot": np.nan,
            "sigma_uu": np.nan, "sigma_uv": np.nan, "sigma_vv": np.nan,
        }
    ch0 = cells_for_measurement["channel0"].to_numpy(dtype=np.float64)
    ch1 = cells_for_measurement["channel1"].to_numpy(dtype=np.float64)
    q = cells_for_measurement["value"].to_numpy(dtype=np.float64)

    s_u = float(ch0.max() - ch0.min() + 1.0)
    s_v = float(ch1.max() - ch1.min() + 1.0)
    q_tot = float(q.sum())

    if q_tot > 0:
        w = q / q_tot
        mean_u = float((w * ch0).sum())
        mean_v = float((w * ch1).sum())
        sigma_uu = float((w * (ch0 - mean_u) ** 2).sum())
        sigma_vv = float((w * (ch1 - mean_v) ** 2).sum())
        sigma_uv = float((w * (ch0 - mean_u) * (ch1 - mean_v)).sum())
    else:
        sigma_uu = sigma_vv = sigma_uv = 0.0

    return {
        "s_u": s_u, "s_v": s_v, "q_tot": q_tot,
        "sigma_uu": sigma_uu, "sigma_uv": sigma_uv, "sigma_vv": sigma_vv,
    }


def compute_cluster_features_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Vectorized cluster-feature computation for ALL measurements at once.

    Equivalent to calling :func:`compute_cluster_features` once per
    ``measurement_id`` group, but implemented entirely with
    ``groupby``/``transform`` (no explicit per-group python loop), per the
    task's numpy-vectorization requirement.

    Parameters
    ----------
    cells : pd.DataFrame
        Output of :func:`load_cells`.

    Returns
    -------
    pd.DataFrame
        One row per ``measurement_id``, columns:
        ``measurement_id, s_u, s_v, q_tot, sigma_uu, sigma_uv, sigma_vv``.
    """
    if len(cells) == 0:
        return pd.DataFrame(columns=["measurement_id", "s_u", "s_v", "q_tot", "sigma_uu", "sigma_uv", "sigma_vv"])

    g = cells.groupby("measurement_id", sort=False)
    s_u = (g["channel0"].max() - g["channel0"].min() + 1.0).rename("s_u")
    s_v = (g["channel1"].max() - g["channel1"].min() + 1.0).rename("s_v")
    q_tot = g["value"].sum().rename("q_tot")

    c = cells.copy()
    qtot_bcast = c.groupby("measurement_id")["value"].transform("sum")
    w = np.where(qtot_bcast > 0, c["value"] / qtot_bcast.replace(0, np.nan), 0.0)
    c["_wu"] = w * c["channel0"]
    c["_wv"] = w * c["channel1"]
    mu0 = c.groupby("measurement_id")["_wu"].sum()
    mu1 = c.groupby("measurement_id")["_wv"].sum()

    c = c.join(mu0.rename("_mu0"), on="measurement_id")
    c = c.join(mu1.rename("_mu1"), on="measurement_id")
    du = c["channel0"] - c["_mu0"]
    dv = c["channel1"] - c["_mu1"]
    c["_wuu"] = w * du * du
    c["_wvv"] = w * dv * dv
    c["_wuv"] = w * du * dv

    sigma_uu = c.groupby("measurement_id")["_wuu"].sum().rename("sigma_uu")
    sigma_vv = c.groupby("measurement_id")["_wvv"].sum().rename("sigma_vv")
    sigma_uv = c.groupby("measurement_id")["_wuv"].sum().rename("sigma_uv")

    out = pd.concat([s_u, s_v, q_tot, sigma_uu, sigma_uv, sigma_vv], axis=1)
    out.loc[out["q_tot"] <= 0, ["sigma_uu", "sigma_uv", "sigma_vv"]] = 0.0
    out = out.reset_index().rename(columns={"measurement_id": "measurement_id"})
    return out


# ---------------------------------------------------------------------------
# Sensor properties (best-effort geometry lookup)
# ---------------------------------------------------------------------------


def load_sensor_pitch_table(digi_config_path: str) -> dict[int, dict[str, float]]:
    """Load per-volume pitch/thickness from an ACTS geometric-digitization
    config JSON (e.g. ``configs/odd-digi-geometric-config.json``).

    The config is keyed only by ``volume`` (not by layer/module), so pitch
    and thickness are volume-level constants, not per-module. Pitch is
    derived from the segmentation ``binningdata`` as ``(max - min) / bins``
    for the first two binning axes (loc0, loc1).

    Returns
    -------
    dict
        ``{volume_id: {"pitch_u": float, "pitch_v": float, "thickness":
        float}}``.
    """
    with open(digi_config_path) as f:
        cfg = json.load(f)
    out: dict[int, dict[str, float]] = {}
    for entry in cfg.get("entries", []):
        volume = entry.get("volume")
        if volume is None:
            continue
        geo = entry.get("value", {}).get("geometric")
        if geo is None:
            continue
        bins = geo.get("segmentation", {}).get("binningdata", [])
        pitch_u = pitch_v = np.nan
        if len(bins) > 0:
            b0 = bins[0]
            if b0.get("bins", 0) > 0:
                pitch_u = (b0["max"] - b0["min"]) / b0["bins"]
        if len(bins) > 1:
            b1 = bins[1]
            if b1.get("bins", 0) > 0:
                pitch_v = (b1["max"] - b1["min"]) / b1["bins"]
        out[int(volume)] = {
            "pitch_u": pitch_u,
            "pitch_v": pitch_v,
            "thickness": float(geo.get("thickness", np.nan)),
        }
    return out


def compute_sensor_props(
    volume_id: np.ndarray | pd.Series,
    pitch_table: dict[int, dict[str, float]] | None = None,
    pixel_volumes: frozenset[int] = PIXEL_VOLUMES_DEFAULT,
    barrel_volumes: frozenset[int] | None = None,
) -> pd.DataFrame:
    """Look up sensor properties per trackstate row from its ``volume_id``.

    ``is_barrel`` is left NaN unless ``barrel_volumes`` is explicitly
    supplied: the digitization config is keyed only by volume and does not
    distinguish barrel vs endcap, and this was not independently verified
    against the ACTS tracking geometry for this task (see module docstring
    "Known gaps").

    Parameters
    ----------
    volume_id : array-like of int
    pitch_table : dict, optional
        Output of :func:`load_sensor_pitch_table`. If None, pitch/thickness
        are NaN for all rows.
    pixel_volumes : frozenset[int]
        Volumes classified as pixel (default matches the convention already
        used in ``scripts/expand_pilot.py`` -- NOT independently
        re-verified here).
    barrel_volumes : frozenset[int], optional
        Volumes classified as barrel. If None, ``is_barrel`` is NaN for all
        rows.

    Returns
    -------
    pd.DataFrame
        Columns: ``pitch_u, pitch_v, thickness, is_pixel, is_barrel``,
        aligned to the input index.
    """
    vol = np.asarray(volume_id, dtype=np.int64)
    n = len(vol)
    pitch_u = np.full(n, np.nan)
    pitch_v = np.full(n, np.nan)
    thickness = np.full(n, np.nan)
    if pitch_table:
        for v_int, props in pitch_table.items():
            mask = vol == v_int
            pitch_u[mask] = props.get("pitch_u", np.nan)
            pitch_v[mask] = props.get("pitch_v", np.nan)
            thickness[mask] = props.get("thickness", np.nan)

    is_pixel = np.isin(vol, list(pixel_volumes)).astype(np.int8)
    if barrel_volumes is not None:
        known = vol > 0
        is_barrel = np.full(n, np.nan)
        is_barrel[known] = 0.0
        is_barrel[np.isin(vol, list(barrel_volumes))] = 1.0
    else:
        is_barrel = np.full(n, np.nan)

    return pd.DataFrame({
        "pitch_u": pitch_u, "pitch_v": pitch_v, "thickness": thickness,
        "is_pixel": is_pixel, "is_barrel": is_barrel,
    })


# ---------------------------------------------------------------------------
# Branch bookkeeping (n_hits, n_holes, n_seq_holes) and seed majority
# ---------------------------------------------------------------------------


def compute_branch_history(expanded: pd.DataFrame) -> pd.DataFrame:
    """Compute per-state branch bookkeeping columns: n_hits, n_holes,
    n_seq_holes.

    These count states *before* the current one (i.e. the branch history at
    the moment this state was reached), grouped by (event_id, track_nr) and
    ordered by ``step_k``. A state counts as a "hit" if it has >= 1
    in-window candidate (``n_window > 0``), else as a "hole"
    (``action_taken != ACTION_CANDIDATE``).

    This necessarily runs after :func:`expand_trackstates`, since branch
    history depends on the window-matching outcome -- it cannot be computed
    purely from ``load_trackstates``'s raw per-state fields.

    Parameters
    ----------
    expanded : pd.DataFrame
        Output of :func:`expand_trackstates` (candidate + hole rows).

    Returns
    -------
    pd.DataFrame
        ``expanded`` with three new columns added: ``n_hits, n_holes,
        n_seq_holes``.
    """
    state_level = expanded.drop_duplicates(subset=["event_id", "track_nr", "state_idx"]).copy()
    state_level = state_level.sort_values(["event_id", "track_nr", "state_idx"])
    is_hole = (state_level["n_window"] == 0).to_numpy()

    n_hits = np.zeros(len(state_level), dtype=np.int64)
    n_holes = np.zeros(len(state_level), dtype=np.int64)
    n_seq_holes = np.zeros(len(state_level), dtype=np.int64)

    branch_key = list(zip(state_level["event_id"], state_level["track_nr"]))
    prev_key = None
    hits_run = holes_run = seq_run = 0
    for i, key in enumerate(branch_key):
        if key != prev_key:
            hits_run = holes_run = seq_run = 0
            prev_key = key
        n_hits[i] = hits_run
        n_holes[i] = holes_run
        n_seq_holes[i] = seq_run
        if is_hole[i]:
            holes_run += 1
            seq_run += 1
        else:
            hits_run += 1
            seq_run = 0

    state_level["n_hits"] = n_hits
    state_level["n_holes"] = n_holes
    state_level["n_seq_holes"] = n_seq_holes

    hist = state_level[["event_id", "track_nr", "state_idx", "n_hits", "n_holes", "n_seq_holes"]]
    out = expanded.merge(hist, on=["event_id", "track_nr", "state_idx"], how="left")
    return out


def compute_branch_majority_pid(states: pd.DataFrame) -> pd.DataFrame:
    """Compute the seed/branch-majority particle ID per branch.

    Per task-2-brief: determined from the SEED (first 3 measurement
    states of the branch, ordered by ``state_idx``). A "measurement state"
    here is one with ``state_n_contribs > 0`` (i.e. the CKF
    selected a real hit there, not a hole). Branches with fewer than 3 such
    states, or where all 3 seed hits have different particles, have no
    majority.

    Parameters
    ----------
    states : pd.DataFrame
        Output of :func:`load_trackstates` (must include
        ``state_primary_pid`` and ``state_n_contribs``).

    Returns
    -------
    pd.DataFrame
        One row per (event_id, track_nr), columns: ``branch_majority_pid``
        (int64, -1 if undefined), ``majority_undefined`` (bool).
    """
    meas_states = states[states["state_n_contribs"] > 0].copy()
    meas_states = meas_states.sort_values(["event_id", "track_nr", "state_idx"])

    def _majority(group: pd.DataFrame) -> pd.Series:
        seed = group["state_primary_pid"].to_numpy()[:3]
        if len(seed) < 3:
            return pd.Series({"branch_majority_pid": -1, "majority_undefined": True})
        vals, counts = np.unique(seed, return_counts=True)
        best = vals[np.argmax(counts)]
        best_n = counts.max()
        if best_n < 2:
            return pd.Series({"branch_majority_pid": -1, "majority_undefined": True})
        return pd.Series({"branch_majority_pid": int(best), "majority_undefined": False})

    result = meas_states.groupby(["event_id", "track_nr"], as_index=False).apply(
        _majority, include_groups=False
    )
    result = result.reset_index(drop=True)

    # Branches with zero measurement states at all also get "undefined".
    all_branches = states[["event_id", "track_nr"]].drop_duplicates()
    out = all_branches.merge(result, on=["event_id", "track_nr"], how="left")
    out["branch_majority_pid"] = out["branch_majority_pid"].fillna(-1).astype(np.int64)
    out["majority_undefined"] = out["majority_undefined"].fillna(True).astype(bool)
    return out


# ---------------------------------------------------------------------------
# Step 5: truth label computation
# ---------------------------------------------------------------------------


def compute_truth_labels(
    expanded: pd.DataFrame,
    simhits: pd.DataFrame,
    measurement_particle_map: pd.DataFrame,
    branch_majority_pid: pd.DataFrame,
) -> pd.DataFrame:
    """Add truth columns to the expanded DataFrame.

    Parameters
    ----------
    expanded : pd.DataFrame
        Output of :func:`expand_trackstates` (optionally already merged with
        :func:`compute_branch_history`).
    simhits : pd.DataFrame
        Output of :func:`load_simhits`.
    measurement_particle_map : pd.DataFrame
        Output of :func:`load_measurement_simhit_map` -- the RAW
        (measurement_id, hit_id) map; this function joins it against
        ``simhits`` itself to resolve particle IDs.
    branch_majority_pid : pd.DataFrame
        Output of :func:`compute_branch_majority_pid`.

    Returns
    -------
    pd.DataFrame
        ``expanded`` with added columns: ``contrib_pids`` (list[int64] per
        row), ``contrib_charge_frac`` (list[float] per row, equal-weight
        approximation -- see note below), ``branch_majority_pid``,
        ``majority_undefined``, ``majority_true_hit_on_surface``,
        ``truth_residual_l0``, ``truth_residual_l1`` (NaN placeholder, not
        computed -- see module docstring "Known gaps").

    Notes
    -----
    **Charge-fraction approximation.** The digitization CSV chain does not
    expose per-particle charge contributions to a merged cluster (that would
    require per-cell truth links not present in ``cells.csv``). Each unique
    contributor to a measurement is given equal weight
    ``1 / n_contributors``, matching the approximation already used in
    ``scripts/expand_pilot.py``. This is a known simplification, not an
    exact charge fraction.

    **majority_true_hit_on_surface** is derived from simhit presence
    (regardless of whether that simhit produced a recorded measurement), so
    it conflates spec §7.4's "genuine hole" and "window failure" cases.
    Splitting them precisely requires per-measurement ``geometry_id`` inside
    this function, which is not part of the given signature; deferred.
    """
    out = expanded.copy()

    # measurement_id -> [(particle_id, charge_frac), ...]
    joined = measurement_particle_map.merge(
        simhits[["hit_id", "particle_id", "geometry_id"]], on="hit_id", how="left"
    )
    joined = joined.dropna(subset=["particle_id"])
    contrib_lists = (
        joined.groupby("measurement_id")["particle_id"]
        .apply(lambda s: sorted(set(int(x) for x in s)))
        .rename("contrib_pids")
    )
    contrib_frac = contrib_lists.apply(lambda pids: [1.0 / len(pids)] * len(pids) if pids else [])
    contrib_frac.name = "contrib_charge_frac"

    contrib_table = pd.concat([contrib_lists, contrib_frac], axis=1).reset_index()
    out = out.merge(contrib_table, left_on="cand_hit_id", right_on="measurement_id", how="left")
    out = out.drop(columns=["measurement_id"], errors="ignore")

    empty_list_mask = out["contrib_pids"].isna()
    out.loc[empty_list_mask, "contrib_pids"] = pd.Series(
        [[] for _ in range(empty_list_mask.sum())], index=out.index[empty_list_mask]
    )
    out.loc[empty_list_mask, "contrib_charge_frac"] = pd.Series(
        [[] for _ in range(empty_list_mask.sum())], index=out.index[empty_list_mask]
    )

    out = out.merge(branch_majority_pid, on=["event_id", "track_nr"], how="left")
    out["branch_majority_pid"] = out["branch_majority_pid"].fillna(-1).astype(np.int64)
    out["majority_undefined"] = out["majority_undefined"].fillna(True).astype(bool)

    # majority_true_hit_on_surface: does m have ANY simhit on this state's
    # surface? (conflates genuine-hole vs window-failure -- see Notes above)
    simhit_particle_surface = set(
        zip(simhits["particle_id"].to_numpy(), simhits["geometry_id"].to_numpy())
    )
    maj_hit = np.array([
        (int(m), int(g)) in simhit_particle_surface if not undef else False
        for m, g, undef in zip(
            out["branch_majority_pid"], out["geometry_id"], out["majority_undefined"]
        )
    ])
    out["majority_true_hit_on_surface"] = maj_hit.astype(np.int8)

    out["truth_residual_l0"] = np.nan
    out["truth_residual_l1"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Step 6: Parquet writer
# ---------------------------------------------------------------------------


def compute_env_config_hash(env_config: dict | None) -> str:
    """Deterministic short hash of the collection-envelope config."""
    canon = json.dumps(env_config or {}, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def write_expanded_parquet(
    expanded: pd.DataFrame,
    output_path: str,
    env_config_hash: str,
) -> str:
    """Write the expanded DataFrame to Parquet with the §6.5 schema.

    Columns not present in ``expanded`` are added and filled with a
    schema-appropriate default (NaN for floats, -1 for int placeholders,
    empty list for list columns) so the output always has the full
    :data:`SCHEMA_COLUMNS` layout, even for fields this pipeline does not
    yet populate (``cov_*`` off-diagonals, ``truth_residual_*``,
    ``vstar_soft``, ``prune_reason``, ``parent_branch_id``).

    Parameters
    ----------
    expanded : pd.DataFrame
        Assembled rows (see :func:`run_expansion` for how these are built
        from the individual steps).
    output_path : str
        Destination ``.parquet`` path.
    env_config_hash : str
        Written into every row's ``env_config_hash`` column.

    Returns
    -------
    str
        ``output_path``, for convenience chaining.
    """
    if pa is None or pq is None:
        raise ImportError("pyarrow is required to write parquet output")

    df = expanded.copy()
    df["env_config_hash"] = env_config_hash

    # Convenience aliases so callers assembling `expanded` from this module's
    # intermediate outputs (which use *_l0/_l1 suffixes and internal names
    # like `track_nr`/`state_idx`/`module_id`) map cleanly onto the schema's
    # `state`/`pred_local`/`seed_id`/`branch_id`/`step_k`/`surface_id`
    # fields. Applied BEFORE the default-fill pass below, so a schema column
    # backed by an alias is populated from real data rather than masked by
    # its int/bool placeholder default.
    alias_map = {
        "state_l0": "pred_l0", "state_l1": "pred_l1", "state_phi": "pred_phi",
        "state_theta": "pred_theta", "state_qop": "pred_qop", "state_t": "pred_t",
        "surface_id": "module_id", "seed_id": "track_nr", "branch_id": "track_nr",
        "step_k": "state_idx",
    }
    for schema_col, source_col in alias_map.items():
        if schema_col not in df.columns and source_col in df.columns:
            df[schema_col] = df[source_col]

    list_cols = {"contrib_pids", "contrib_charge_frac"}
    int_defaults = {
        "seed_id", "branch_id", "parent_branch_id", "cand_hit_id",
        "action_taken", "prune_reason", "branch_majority_pid",
        "majority_true_hit_on_surface", "is_pixel", "n_window",
        "geometric_density", "n_hits", "n_holes", "n_seq_holes",
        "dead_module_flag", "layer_embed_idx", "volume_id",
    }
    bool_defaults = {"majority_undefined"}

    for col in SCHEMA_COLUMNS:
        if col in df.columns:
            continue
        if col in list_cols:
            df[col] = [[] for _ in range(len(df))]
        elif col in int_defaults:
            df[col] = -1
        elif col in bool_defaults:
            df[col] = True
        else:
            df[col] = np.nan

    out_df = df[SCHEMA_COLUMNS].copy()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    schema_overrides = {
        "contrib_pids": pa.list_(pa.int64()),
        "contrib_charge_frac": pa.list_(pa.float32()),
    }
    table = pa.Table.from_pandas(out_df, preserve_index=False)
    for col_name, pa_type in schema_overrides.items():
        if col_name in table.column_names:
            idx = table.column_names.index(col_name)
            try:
                col = table.column(col_name).cast(pa_type)
                table = table.set_column(idx, col_name, col)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                pass

    pq.write_table(table, output_path, compression="zstd")
    return output_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_expansion(
    trackstates_root: str,
    csv_dir: str,
    event_id: int,
    output_path: str,
    window_n: float = WINDOW_N_DEFAULT,
    r_geom_mm: float = R_GEOM_MM_DEFAULT,
    digi_config_path: str | None = None,
    env_config: dict | None = None,
    n_chunks: int = 1,
    chunk_idx: int = 0,
) -> pd.DataFrame:
    """Run the full expansion pipeline for one event and write Parquet.

    ``n_chunks``/``chunk_idx`` partition the work by ``track_nr`` so a single
    event can be expanded in bounded memory. The largest events (7, 5, 14)
    OOM a 503 GB node outright once the valid-mask fix quadrupled the state
    count, and Perlmutter exposes no largemem partition here. Each chunk is a
    self-contained Parquet; the gate cache builder already accepts a list of
    files per split, so parts need no merging.

    Ties together Steps 1-6. Intended as the single entry point for a
    caller (e.g. a future Modal function in ``modal_build_acts.py``) that
    just wants "trackstates + CSVs in, expanded Parquet out" for one event.

    Parameters
    ----------
    trackstates_root : str
        Path to ``trackstates_ckf.root``.
    csv_dir : str
        Directory containing the per-event digitization CSVs.
    event_id : int
        Event to expand.
    output_path : str
        Destination Parquet path.
    window_n : float
        Window multiplier n (spec §6.3).
    r_geom_mm : float
        Fixed-radius geometric occupancy window.
    digi_config_path : str, optional
        Path to the geometric-digitization config JSON, for sensor pitch
        lookup. If None, pitch/thickness columns are NaN.
    env_config : dict, optional
        Collection-envelope config, hashed into ``env_config_hash``.

    Returns
    -------
    pd.DataFrame
        The final expanded DataFrame, as written to ``output_path``.
    """
    states = load_trackstates(trackstates_root, event_id)
    if n_chunks > 1:
        # Partition by track_nr, not by row: a branch's states must stay
        # together or the backward history walk in compute_branch_history
        # would see a truncated branch and emit wrong n_hits/n_holes.
        _tn = states["track_nr"].to_numpy()
        states = states[(_tn % n_chunks) == chunk_idx].copy()
    measurements = load_measurements(csv_dir, event_id)
    cells = load_cells(csv_dir, event_id)
    simhits = load_simhits(csv_dir, event_id)
    meas_map = load_measurement_simhit_map(csv_dir, event_id)

    predicted_cov = load_predicted_cov(csv_dir, event_id)
    expanded = expand_trackstates(states, measurements, n=window_n,
                                  r_geom_mm=r_geom_mm,
                                  predicted_cov=predicted_cov)
    expanded = compute_branch_history(expanded)

    cluster_table = compute_cluster_features_table(cells)
    expanded = expanded.merge(
        cluster_table.rename(columns={
            "s_u": "clus_s_u", "s_v": "clus_s_v", "q_tot": "clus_q_tot",
            "sigma_uu": "clus_sigma_uu", "sigma_uv": "clus_sigma_uv", "sigma_vv": "clus_sigma_vv",
        }),
        left_on="cand_hit_id", right_on="measurement_id", how="left",
        suffixes=("_selected", ""),
    )
    expanded = expanded.drop(columns=["measurement_id"], errors="ignore")
    # For hole rows and states whose selected-hit cluster features are the
    # only ones available (candidate lookup missed), fall back to the
    # ROOT-provided selected-hit cluster features.
    for feat in ("clus_s_u", "clus_s_v", "clus_q_tot", "clus_sigma_uu", "clus_sigma_uv", "clus_sigma_vv"):
        sel_col = f"{feat}_selected"
        if sel_col in expanded.columns:
            expanded[feat] = expanded[feat].where(expanded[feat].notna(), expanded[sel_col])
            expanded = expanded.drop(columns=[sel_col])

    pitch_table = load_sensor_pitch_table(digi_config_path) if digi_config_path else None
    sensor_df = compute_sensor_props(
        expanded["volume_id"].to_numpy(),
        pitch_table=pitch_table,
        barrel_volumes=BARREL_VOLUMES_DEFAULT,
    )
    expanded = pd.concat([expanded.reset_index(drop=True), sensor_df.reset_index(drop=True)], axis=1)

    expanded["chi2_inc"] = chi2_increment_batch(
        expanded["residual_l0"].to_numpy(),
        expanded["residual_l1"].to_numpy(),
        expanded["S00"].to_numpy(),
        expanded["S01"].to_numpy(),
        expanded["S11"].to_numpy(),
        expanded["is_1d"].to_numpy(),
    )

    majority = compute_branch_majority_pid(states)
    expanded = compute_truth_labels(expanded, simhits, meas_map, majority)

    env_hash = compute_env_config_hash(env_config)
    write_expanded_parquet(expanded, output_path, env_hash)
    return expanded
