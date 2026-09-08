"""Expand ACTS track-states to full pilot-protocol Parquet (spec §6.5).

For each CKF branch-state with a predicted bound state, find every measurement
on the same sensitive surface that falls inside the axis-aligned Mahalanobis
window W_k(n), and emit one row per (branch, surface, candidate).  Additionally
emits a 'hole' row (cand_hit_id = -1) for surfaces with zero candidates.

This replaces the slim expand_trackstates_to_chi2_rows.py for the full data
collection pipeline.  The innovation covariance S = HPH' + V is now read
directly from ROOT (S00_prt, S01_prt, S11_prt from our ACTS patch) rather
than reconstructed from a separate predicted-cov CSV.

Cluster features for all candidates (not just the CKF-selected hit) come from
cells.csv, pre-aggregated into a per-measurement lookup table.

Data sources per row field:
  ROOT trackstates: predicted state, predicted errors, S, incidence angles,
    pathInX0, selected-hit cluster features, branch counters, particle IDs
  measurements.csv: candidate local position, measurement variance
  cells.csv: cluster features for ALL candidates
  measurement-simhit-map.csv + simhits.csv: contributor PIDs, charge fracs,
    majority_true_hit_on_surface, truth_residual
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import uproot
except ImportError:
    uproot = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

WINDOW_N = 10.0
R_GEOM_MM = 5.0

# ACTS GeometryIdentifier masks
_VOL_MASK = np.uint64(0xFF00000000000000)
_LAY_MASK = np.uint64(0x0000FFF000000000)
_SEN_MASK = np.uint64(0x000000000FFFFF00)

PidKey = tuple[int, int, int, int, int]

# ODD v5 volume IDs for pixel vs strip classification
PIXEL_VOLUMES = {16, 17, 18}
BARREL_VOLUMES = {16, 23}


def decode_geo_id(gid: int) -> tuple[int, int, int]:
    g = np.uint64(gid)
    volume = int((g & _VOL_MASK) >> np.uint64(56))
    layer = int((g & _LAY_MASK) >> np.uint64(36))
    sensitive = int((g & _SEN_MASK) >> np.uint64(8))
    return volume, layer, sensitive


def _pid_key(pv: int, sv: int, part: int, gen: int, sub: int) -> PidKey:
    return (int(pv), int(sv), int(part), int(gen), int(sub))


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_measurements(path: Path) -> dict[tuple[int, int, int], dict[str, np.ndarray]]:
    """Measurements indexed by (volume, layer, module)."""
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    gids = df["geometry_id"].to_numpy(dtype=np.uint64)
    vol = ((gids & _VOL_MASK) >> np.uint64(56)).astype(np.int32)
    lay = ((gids & _LAY_MASK) >> np.uint64(36)).astype(np.int32)
    sen = ((gids & _SEN_MASK) >> np.uint64(8)).astype(np.int32)
    mid = (
        df["measurement_id"].to_numpy(dtype=np.int64)
        if "measurement_id" in df.columns
        else np.arange(len(df), dtype=np.int64)
    )
    l0 = df["local0"].to_numpy(dtype=np.float64)
    l1 = df["local1"].to_numpy(dtype=np.float64)
    v0 = df["var_local0"].to_numpy(dtype=np.float64)
    v1 = df["var_local1"].to_numpy(dtype=np.float64)
    v0 = np.where(v0 > 0, v0, 0.015**2)
    v1 = np.where(v1 > 0, v1, 0.015**2)

    by_surface: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}
    order = np.lexsort((sen, lay, vol))
    vol_s, lay_s, sen_s = vol[order], lay[order], sen[order]
    starts = np.flatnonzero(
        np.r_[True, (vol_s[1:] != vol_s[:-1]) | (lay_s[1:] != lay_s[:-1]) | (sen_s[1:] != sen_s[:-1])]
    )
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        key = (int(vol_s[s]), int(lay_s[s]), int(sen_s[s]))
        by_surface[key] = {
            "l0": l0[idx], "l1": l1[idx],
            "v0": v0[idx], "v1": v1[idx],
            "mid": mid[idx],
        }
    return by_surface


def load_cluster_features(cells_path: Path) -> dict[int, dict[str, float]]:
    """measurement_id → {s_u, s_v, q_tot, sigma_uu, sigma_uv, sigma_vv}.

    Computed from channel-level cells.csv (geometric digi output).
    """
    if not cells_path.exists():
        return {}
    df = pd.read_csv(cells_path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    required = {"measurement_id", "channel0", "channel1", "value"}
    if not required.issubset(df.columns):
        return {}

    result: dict[int, dict[str, float]] = {}
    for mid, g in df.groupby("measurement_id"):
        ch0 = g["channel0"].to_numpy(dtype=np.float64)
        ch1 = g["channel1"].to_numpy(dtype=np.float64)
        q = g["value"].to_numpy(dtype=np.float64)
        s_u = 1.0 + float(ch0.max() - ch0.min()) if len(ch0) else 0.0
        s_v = 1.0 + float(ch1.max() - ch1.min()) if len(ch1) else 0.0
        q_tot = float(q.sum())
        if q_tot > 0:
            w = q / q_tot
            mu0 = float((w * ch0).sum())
            mu1 = float((w * ch1).sum())
            sig_uu = float((w * (ch0 - mu0) ** 2).sum())
            sig_vv = float((w * (ch1 - mu1) ** 2).sum())
            sig_uv = float((w * (ch0 - mu0) * (ch1 - mu1)).sum())
        else:
            sig_uu = sig_vv = sig_uv = 0.0
        result[int(mid)] = {
            "s_u": s_u, "s_v": s_v, "q_tot": q_tot,
            "sigma_uu": sig_uu, "sigma_uv": sig_uv, "sigma_vv": sig_vv,
        }
    return result


def load_meas_contributors(
    map_path: Path, simhits_path: Path,
) -> tuple[
    dict[int, list[tuple[PidKey, float]]],
    set[tuple[PidKey, int, int, int]],
]:
    """Return (meas_contribs, simhit_positions).

    meas_contribs: measurement_id → [(pid_key, charge_fraction), ...]
    simhit_on_surface: set of (pid_key, vol, lay, mod) where a simhit exists
    """
    mmap = pd.read_csv(map_path, comment="#")
    mmap.columns = [c.strip() for c in mmap.columns]
    mcol = "measurement_id" if "measurement_id" in mmap.columns else mmap.columns[0]
    hcol = "hit_id" if "hit_id" in mmap.columns else mmap.columns[1]

    sim = pd.read_csv(simhits_path, comment="#")
    sim.columns = [c.strip() for c in sim.columns]
    n_sim = len(sim)
    zeros = np.zeros(n_sim, dtype=np.int64)

    def _col(name: str) -> np.ndarray:
        return sim[name].to_numpy(dtype=np.int64, copy=False) if name in sim.columns else zeros

    sim_pids = [
        _pid_key(int(a), int(b), int(c), int(d), int(e))
        for a, b, c, d, e in zip(
            _col("particle_id_pv"), _col("particle_id_sv"),
            _col("particle_id_part"), _col("particle_id_gen"),
            _col("particle_id_subpart"),
        )
    ]

    # SimHit presence indexed by (pid, vol, lay, mod).
    # The simhits CSV has GLOBAL coordinates (tx, ty, tz), not local.
    # truth_residual requires a global→local transform (needs DD4hep geometry),
    # so we defer it. We do record majority_true_hit_on_surface (boolean).
    simhit_on_surface: set[tuple[PidKey, int, int, int]] = set()
    if "geometry_id" in sim.columns:
        sim_gids = sim["geometry_id"].to_numpy(dtype=np.uint64)
        for i in range(n_sim):
            gid = sim_gids[i]
            vol = int((gid & _VOL_MASK) >> np.uint64(56))
            lay = int((gid & _LAY_MASK) >> np.uint64(36))
            mod = int((gid & _SEN_MASK) >> np.uint64(8))
            simhit_on_surface.add((sim_pids[i], vol, lay, mod))

    # Measurement → contributors with charge fractions
    # DECISION: charge fraction = cell_value contributed by particle / total cluster charge.
    # The meas-simhit map gives (measurement_id, hit_id). The simhit has the particle.
    # For charge fractions, we'd need per-particle cell contributions which aren't in the
    # standard CSV. We approximate: each unique contributor gets equal weight.
    mids = mmap[mcol].to_numpy(dtype=np.int64)
    hids = mmap[hcol].to_numpy(dtype=np.int64)

    contribs_raw: dict[int, list[PidKey]] = defaultdict(list)
    valid = (hids >= 0) & (hids < len(sim_pids))
    for mid_val, hid in zip(mids[valid], hids[valid]):
        pid = sim_pids[int(hid)]
        bucket = contribs_raw[int(mid_val)]
        if pid not in bucket:
            bucket.append(pid)

    meas_contribs: dict[int, list[tuple[PidKey, float]]] = {}
    for mid_val, pids in contribs_raw.items():
        frac = 1.0 / len(pids) if pids else 0.0
        meas_contribs[mid_val] = [(p, frac) for p in pids]

    return meas_contribs, simhit_on_surface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_python(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        return x.tolist()
    except Exception:
        try:
            return list(x)
        except TypeError:
            return [x]


def _int_list(x: Any) -> list[int]:
    out: list[int] = []
    for item in _to_python(x):
        if item is None:
            continue
        if isinstance(item, (list, tuple, np.ndarray)):
            for v in item:
                try:
                    out.append(int(v))
                except (TypeError, ValueError):
                    pass
            continue
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def majority_from_barcode_fields(
    part_nested, pv_nested, sv_nested, gen_nested, sub_nested,
) -> tuple[PidKey | None, bool]:
    """Mode particle across states; undefined if mode owns < 2/3."""
    parts_per = _to_python(part_nested)
    pvs_per = _to_python(pv_nested)
    svs_per = _to_python(sv_nested)
    gens_per = _to_python(gen_nested)
    subs_per = _to_python(sub_nested)

    keys: list[PidKey] = []
    for i, state_parts in enumerate(parts_per):
        parts = _int_list(state_parts)
        if not parts or all(p == 0 for p in parts):
            continue
        mode_part = Counter(parts).most_common(1)[0][0]
        try:
            j = parts.index(mode_part)
        except ValueError:
            j = 0

        def _field(nested: list, default: int = 0) -> int:
            if i >= len(nested):
                return default
            vals = _int_list(nested[i])
            if not vals:
                return default
            return vals[j] if j < len(vals) else vals[0]

        keys.append(_pid_key(
            _field(pvs_per), _field(svs_per), mode_part,
            _field(gens_per), _field(subs_per),
        ))
    if not keys:
        return None, True
    maj, n = Counter(keys).most_common(1)[0]
    return maj, bool(n * 3 < 2 * len(keys))


def chi2_full(r0: float, r1: float, s00: float, s01: float, s11: float) -> float:
    det = s00 * s11 - s01 * s01
    if not np.isfinite(det) or det <= 0:
        return (r0 * r0) / max(s00, 1e-30) + (r1 * r1) / max(s11, 1e-30)
    return (s11 * r0 * r0 - 2.0 * s01 * r0 * r1 + s00 * r1 * r1) / det


def compute_env_config_hash(config: dict) -> str:
    """Deterministic hash of the envelope config for reproducibility."""
    canon = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# ROOT branch lists
# ---------------------------------------------------------------------------

# Fields we read from the trackstates ROOT tree per track
TRACK_FIELDS = [
    # geometry
    "volume_id", "layer_id", "module_id",
    # state type and flags
    "stateType", "predicted", "chi2",
    # predicted state
    "eLOC0_prt", "eLOC1_prt", "ePHI_prt", "eTHETA_prt", "eQOP_prt", "eT_prt",
    # predicted errors (diagonal of predicted cov)
    "err_eLOC0_prt", "err_eLOC1_prt", "err_ePHI_prt",
    "err_eTHETA_prt", "err_eQOP_prt", "err_eT_prt",
    # eta for context
    "eta_prt",
    # patched branches: innovation covariance
    "S00_prt", "S01_prt", "S11_prt",
    # patched branches: accumulated material
    "pathInX0_interval",
    # patched branches: cluster features (selected hit only)
    "clus_size_u", "clus_size_v", "clus_qtot",
    "clus_sigma_uu", "clus_sigma_uv", "clus_sigma_vv",
    # patched branches: incidence angles
    "alpha_u", "alpha_v",
    # measurement hit position
    "l_x_hit", "l_y_hit",
    # particle IDs for majority
    "particle_ids_particle", "particle_ids_vertex_primary",
    "particle_ids_vertex_secondary", "particle_ids_generation",
    "particle_ids_sub_particle",
    # branch-level counters
    "nMeasurements", "nStates",
]


# ---------------------------------------------------------------------------
# Main expansion
# ---------------------------------------------------------------------------

def expand_event(
    *,
    event_id: int,
    trackstates_path: Path,
    measurements_path: Path,
    cells_path: Path,
    map_path: Path,
    simhits_path: Path,
    window_n: float = WINDOW_N,
    r_geom_mm: float = R_GEOM_MM,
    batch_size: int = 512,
    max_tracks: int = 25_000,
    sample_seed: int = 42,
    env_config: dict | None = None,
) -> pd.DataFrame:
    """Expand track-states to full pilot rows (spec §6.5).

    Returns a DataFrame with the full schema. Variable-length fields
    (contrib_pids, contrib_charge_frac) are stored as Python lists.
    """
    if uproot is None:
        raise ImportError("uproot required")

    t0 = time.time()
    env_hash = compute_env_config_hash(env_config or {})

    print(f"[expand] event {event_id}: loading data...", flush=True)
    by_surface = load_measurements(measurements_path)
    cluster_feats = load_cluster_features(cells_path)
    meas_contribs, simhit_on_surface = load_meas_contributors(map_path, simhits_path)
    print(
        f"[expand] surfaces={len(by_surface)} clusters={len(cluster_feats)} "
        f"contribs={len(meas_contribs)} simhit_surfaces={len(simhit_on_surface)} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    # Accumulate rows as lists of dicts for clarity; convert to DataFrame at end
    rows: list[dict] = []

    # Read available branches (some may be missing in older ACTS builds)
    with uproot.open(trackstates_path) as f:
        keys = list(f.keys())
        tree_key = next(
            (k for k in keys if "trackstate" in k.lower() or "states" in k.lower()),
            keys[0],
        )
        tree = f[tree_key]
        available = set(tree.keys())
        fields = [fld for fld in TRACK_FIELDS if fld in available]
        missing = [fld for fld in TRACK_FIELDS if fld not in available]
        if missing:
            print(f"[expand] WARNING: missing branches: {missing}", flush=True)

        n_tracks_total = int(tree.num_entries)
        if n_tracks_total <= 0:
            return pd.DataFrame()

        rng = np.random.default_rng(sample_seed + int(event_id))
        if max_tracks and n_tracks_total > max_tracks:
            start = int(rng.integers(0, n_tracks_total - max_tracks + 1))
            entry_start, entry_stop = start, start + max_tracks
            print(
                f"[expand] sample tracks[{entry_start}:{entry_stop}] "
                f"of {n_tracks_total}",
                flush=True,
            )
        else:
            entry_start, entry_stop = 0, n_tracks_total

        n_kept = 0
        for batch_start in range(entry_start, entry_stop, batch_size):
            batch_stop = min(batch_start + batch_size, entry_stop)
            batch = tree.arrays(
                fields, entry_start=batch_start, entry_stop=batch_stop, library="np"
            )
            n_batch = batch_stop - batch_start

            for local_i in range(n_batch):
                track_nr = batch_start + local_i
                maj_pid, maj_undef = majority_from_barcode_fields(
                    batch["particle_ids_particle"][local_i],
                    batch["particle_ids_vertex_primary"][local_i],
                    batch["particle_ids_vertex_secondary"][local_i],
                    batch["particle_ids_generation"][local_i],
                    batch["particle_ids_sub_particle"][local_i],
                )

                def _get(name: str) -> list:
                    return _to_python(batch[name][local_i]) if name in batch else []

                vols = _get("volume_id")
                lays = _get("layer_id")
                mods = _get("module_id")
                e0 = _get("eLOC0_prt")
                e1 = _get("eLOC1_prt")
                e_phi = _get("ePHI_prt")
                e_theta = _get("eTHETA_prt")
                e_qop = _get("eQOP_prt")
                e_t = _get("eT_prt")
                err0 = _get("err_eLOC0_prt")
                err1 = _get("err_eLOC1_prt")
                has_prt = _get("predicted")
                eta_arr = _get("eta_prt")
                chi2_acts = _get("chi2")

                # Patched branches
                s00_arr = _get("S00_prt")
                s01_arr = _get("S01_prt")
                s11_arr = _get("S11_prt")
                pathx0 = _get("pathInX0_interval")
                clus_su = _get("clus_size_u")
                clus_sv = _get("clus_size_v")
                clus_qt = _get("clus_qtot")
                clus_suu = _get("clus_sigma_uu")
                clus_suv = _get("clus_sigma_uv")
                clus_svv = _get("clus_sigma_vv")
                au = _get("alpha_u")
                av = _get("alpha_v")

                n_meas = int(_get("nMeasurements")[0]) if _get("nMeasurements") else 0
                n_states = len(vols)

                # Branch-level counters (accumulated over states)
                n_hits_so_far = 0
                n_holes_so_far = 0
                n_seq_holes = 0

                for step in range(n_states):
                    if step >= len(has_prt) or not bool(has_prt[step]):
                        continue
                    try:
                        pred0 = float(e0[step])
                        pred1 = float(e1[step])
                    except (TypeError, ValueError, IndexError):
                        continue
                    if not (np.isfinite(pred0) and np.isfinite(pred1)):
                        continue

                    vol_id = int(vols[step])
                    lay_id = int(lays[step])
                    mod_id = int(mods[step])
                    surf_key = (vol_id, lay_id, mod_id)
                    surf = by_surface.get(surf_key)

                    # Predicted state vector
                    state_vec = [
                        pred0, pred1,
                        float(e_phi[step]) if step < len(e_phi) else float("nan"),
                        float(e_theta[step]) if step < len(e_theta) else float("nan"),
                        float(e_qop[step]) if step < len(e_qop) else float("nan"),
                        float(e_t[step]) if step < len(e_t) else float("nan"),
                    ]
                    eta = float(eta_arr[step]) if step < len(eta_arr) else float("nan")

                    # Innovation covariance from patched branches
                    p00 = float(err0[step]) ** 2 if step < len(err0) else 0.05**2
                    p11 = float(err1[step]) ** 2 if step < len(err1) else 0.05**2
                    # S00/S01/S11 from ROOT are the full innovation cov for the
                    # selected hit. For other candidates, S varies with their V_i.
                    s00_root = float(s00_arr[step]) if step < len(s00_arr) else float("nan")
                    s01_root = float(s01_arr[step]) if step < len(s01_arr) else float("nan")
                    s11_root = float(s11_arr[step]) if step < len(s11_arr) else float("nan")

                    # Context features
                    path_x0 = float(pathx0[step]) if step < len(pathx0) else 0.0
                    alpha_u = float(au[step]) if step < len(au) else float("nan")
                    alpha_v = float(av[step]) if step < len(av) else float("nan")

                    # Sensor properties (from volume ID)
                    is_pixel = int(vol_id in PIXEL_VOLUMES)
                    is_barrel = int(vol_id in BARREL_VOLUMES)

                    # Majority particle has a simhit on this surface?
                    maj_has_hit = 0
                    if maj_pid is not None and not maj_undef:
                        truth_key = (maj_pid, vol_id, lay_id, mod_id)
                        if truth_key in simhit_on_surface:
                            maj_has_hit = 1

                    # Base row fields shared across all candidates on this surface
                    base = {
                        "event_id": event_id,
                        "seed_id": track_nr,
                        "branch_id": track_nr,
                        "step_k": step,
                        "layer_id": lay_id,
                        "surface_id": mod_id,
                        "volume_id": vol_id,
                        # Predicted state (6D)
                        "pred_l0": state_vec[0],
                        "pred_l1": state_vec[1],
                        "pred_phi": state_vec[2],
                        "pred_theta": state_vec[3],
                        "pred_qop": state_vec[4],
                        "pred_t": state_vec[5],
                        # Predicted errors (diagonal of 6x6 cov)
                        "err_l0": float(err0[step]) if step < len(err0) else float("nan"),
                        "err_l1": float(err1[step]) if step < len(err1) else float("nan"),
                        # Innovation covariance (from ROOT, for the selected hit)
                        "S00_selected": s00_root,
                        "S01_selected": s01_root,
                        "S11_selected": s11_root,
                        # Context
                        "eta": eta,
                        "qop": state_vec[4],
                        "pathInX0_interval": path_x0,
                        "alpha_u": alpha_u,
                        "alpha_v": alpha_v,
                        # Sensor
                        "is_pixel": is_pixel,
                        "is_barrel": is_barrel,
                        # Branch counters
                        "n_hits": n_hits_so_far,
                        "n_holes": n_holes_so_far,
                        "n_seq_holes": n_seq_holes,
                        # Truth
                        "branch_majority_pid": hash(maj_pid) if maj_pid else 0,
                        "majority_undefined": int(maj_undef),
                        "majority_true_hit_on_surface": maj_has_hit,
                        # Config
                        "env_config_hash": env_hash,
                    }

                    if surf is None:
                        # Hole: no measurements on this surface at all
                        row = {**base}
                        row["cand_hit_id"] = -1
                        row["residual_l0"] = float("nan")
                        row["residual_l1"] = float("nan")
                        row["chi2_inc"] = float("nan")
                        row["S00"] = float("nan")
                        row["S01"] = float("nan")
                        row["S11"] = float("nan")
                        row["window_count"] = 0
                        row["geometric_density"] = 0
                        row["action_taken"] = 2  # hole
                        for feat in ("clus_s_u", "clus_s_v", "clus_q_tot",
                                     "clus_sigma_uu", "clus_sigma_uv", "clus_sigma_vv"):
                            row[feat] = float("nan")
                        row["contrib_pids"] = []
                        row["contrib_charge_frac"] = []
                        row["cluster_merged"] = 0
                        row["label"] = 0
                        row["truth_residual_l0"] = float("nan")
                        row["truth_residual_l1"] = float("nan")
                        rows.append(row)
                        n_holes_so_far += 1
                        n_seq_holes += 1
                        continue

                    # Surface has measurements — do window expansion
                    l0 = surf["l0"]
                    l1 = surf["l1"]
                    v0 = surf["v0"]
                    v1 = surf["v1"]
                    mid_arr = surf["mid"]

                    # P is the predicted covariance (local block)
                    # S_i = P + V_i for each candidate (V diagonal from digi)
                    # p01 = S01_root - 0 (V has no off-diagonal in geometric digi)
                    # Actually: S = HPH' + V. For loc0/loc1 H = I_{2x2}.
                    # So P00 = S00_root - V_selected, P01 = S01_root, P11 = S11_root - V_selected
                    # But we don't know V_selected here. Use err0^2, err1^2 as P diagonal.
                    p01 = s01_root if np.isfinite(s01_root) else 0.0

                    # Window: axis-aligned Mahalanobis pre-filter
                    s00_win = p00 + v0
                    s11_win = p11 + v1
                    r0_all = l0 - pred0
                    r1_all = l1 - pred1
                    in_window = (
                        (np.abs(r0_all) <= window_n * np.sqrt(s00_win))
                        & (np.abs(r1_all) <= window_n * np.sqrt(s11_win))
                    )
                    n_window = int(in_window.sum())
                    n_geom = int((np.hypot(r0_all, r1_all) <= r_geom_mm).sum())

                    if n_window == 0:
                        # Window found no candidates (window failure)
                        row = {**base}
                        row["cand_hit_id"] = -1
                        row["residual_l0"] = float("nan")
                        row["residual_l1"] = float("nan")
                        row["chi2_inc"] = float("nan")
                        row["S00"] = float("nan")
                        row["S01"] = float("nan")
                        row["S11"] = float("nan")
                        row["window_count"] = 0
                        row["geometric_density"] = n_geom
                        row["action_taken"] = 2  # hole (window failure)
                        for feat in ("clus_s_u", "clus_s_v", "clus_q_tot",
                                     "clus_sigma_uu", "clus_sigma_uv", "clus_sigma_vv"):
                            row[feat] = float("nan")
                        row["contrib_pids"] = []
                        row["contrib_charge_frac"] = []
                        row["cluster_merged"] = 0
                        row["label"] = 0
                        row["truth_residual_l0"] = float("nan")
                        row["truth_residual_l1"] = float("nan")
                        rows.append(row)
                        n_holes_so_far += 1
                        n_seq_holes += 1
                        continue

                    n_seq_holes = 0
                    n_hits_so_far += 1

                    idxs = np.nonzero(in_window)[0]
                    for j in idxs:
                        mid = int(mid_arr[j])
                        r0 = float(r0_all[j])
                        r1 = float(r1_all[j])
                        s00_cand = float(p00 + v0[j])
                        s01_cand = float(p01)
                        s11_cand = float(p11 + v1[j])
                        chi2_val = chi2_full(r0, r1, s00_cand, s01_cand, s11_cand)

                        # Cluster features from cells.csv lookup
                        cf = cluster_feats.get(mid, {})

                        # Contributors
                        contribs = meas_contribs.get(mid, [])
                        c_pids = [hash(p) for p, _ in contribs]
                        c_fracs = [f for _, f in contribs]
                        is_merged = int(len(contribs) >= 2)

                        # Label: majority particle in contributor list
                        label = 0
                        if not maj_undef and maj_pid is not None:
                            label = int(any(p == maj_pid for p, _ in contribs))

                        row = {**base}
                        row["cand_hit_id"] = mid
                        row["residual_l0"] = r0
                        row["residual_l1"] = r1
                        row["chi2_inc"] = chi2_val
                        row["S00"] = s00_cand
                        row["S01"] = s01_cand
                        row["S11"] = s11_cand
                        row["window_count"] = n_window
                        row["geometric_density"] = n_geom
                        row["action_taken"] = 0  # candidate (accept/reject TBD offline)
                        row["clus_s_u"] = cf.get("s_u", float("nan"))
                        row["clus_s_v"] = cf.get("s_v", float("nan"))
                        row["clus_q_tot"] = cf.get("q_tot", float("nan"))
                        row["clus_sigma_uu"] = cf.get("sigma_uu", float("nan"))
                        row["clus_sigma_uv"] = cf.get("sigma_uv", float("nan"))
                        row["clus_sigma_vv"] = cf.get("sigma_vv", float("nan"))
                        row["contrib_pids"] = c_pids
                        row["contrib_charge_frac"] = c_fracs
                        row["cluster_merged"] = is_merged
                        row["label"] = label
                        row["truth_residual_l0"] = float("nan")
                        row["truth_residual_l1"] = float("nan")
                        rows.append(row)

                n_kept += 1
                if n_kept % 2000 < batch_size or batch_stop >= entry_stop:
                    print(
                        f"[expand] kept={n_kept}/{entry_stop - entry_start} "
                        f"rows={len(rows)} ({time.time() - t0:.1f}s)",
                        flush=True,
                    )

    print(
        f"[expand] done event {event_id}: {len(rows)} rows from "
        f"{n_kept} tracks in {time.time() - t0:.1f}s",
        flush=True,
    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write DataFrame to Parquet with proper types for list columns."""
    if pa is None or pq is None:
        raise ImportError("pyarrow required")
    path.parent.mkdir(parents=True, exist_ok=True)

    schema_overrides = {}
    if "contrib_pids" in df.columns:
        schema_overrides["contrib_pids"] = pa.list_(pa.int64())
    if "contrib_charge_frac" in df.columns:
        schema_overrides["contrib_charge_frac"] = pa.list_(pa.float32())

    table = pa.Table.from_pandas(df, preserve_index=False)

    for col_name, pa_type in schema_overrides.items():
        if col_name in table.column_names:
            idx = table.column_names.index(col_name)
            col = table.column(col_name)
            try:
                col = col.cast(pa_type)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                pass
            table = table.set_column(idx, col_name, col)

    pq.write_table(table, path, compression="zstd")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event-id", type=int, required=True)
    p.add_argument("--trackstates", type=Path, required=True)
    p.add_argument("--measurements", type=Path, required=True)
    p.add_argument("--cells", type=Path, required=True)
    p.add_argument("--meas-simhit-map", type=Path, required=True)
    p.add_argument("--simhits", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--window-n", type=float, default=WINDOW_N)
    p.add_argument("--max-tracks", type=int, default=25_000)
    p.add_argument("--env-config", type=Path, default=None)
    args = p.parse_args()

    env_config = {}
    if args.env_config and args.env_config.exists():
        import yaml
        with open(args.env_config) as f:
            env_config = yaml.safe_load(f)

    df = expand_event(
        event_id=args.event_id,
        trackstates_path=args.trackstates,
        measurements_path=args.measurements,
        cells_path=args.cells,
        map_path=args.meas_simhit_map,
        simhits_path=args.simhits,
        window_n=args.window_n,
        max_tracks=args.max_tracks,
        env_config=env_config,
    )
    write_parquet(df, args.out)
    n_cand = len(df[df["cand_hit_id"] >= 0]) if "cand_hit_id" in df.columns else len(df)
    n_hole = len(df[df["cand_hit_id"] < 0]) if "cand_hit_id" in df.columns else 0
    print(
        f"event {args.event_id}: {len(df)} rows ({n_cand} candidates, {n_hole} holes) "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()
