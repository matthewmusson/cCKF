"""Expand ACTS track-states + digi CSVs into slim chi²-calibration rows.

For each CKF-visited measurement state with a predicted bound state, evaluate
every measurement on the same sensitive surface that falls inside the
axis-aligned Mahalanobis window (spec §5.2, window_n), and emit one slim row.

χ² (chi2_inc) is recomputed offline to match MeasurementSelector::calculateChi2:
    S = H P Hᵀ + V,  χ² = rᵀ S⁻¹ r
with the full 2×2 S (not ACTS's per-selected-hit state.chi2(), which is only
available for the accepted measurement on each track state). Predicted
off-diagonal P₀₁ comes from ``*-predicted-cov.csv`` (PredictedCovWriter);
measurement V is taken from digi CSV (typically diagonal).

Logged innovation covariance columns: S00, S01, S11.

geometric_density: count of measurements on the same surface with
    hypot(Δl₀, Δl₁) ≤ R_GEOM_MM
independent of S. R_GEOM_MM = 5.0 mm (fixed for every row).

Labels (Tier-3 operational definitions for this analysis only — not the
locked training-label module):
  * branch majority = most frequent contributing particle on the track's
    measurement states; majority_undefined if no particle owns ≥ 2/3 of
    those states.
  * label = 1 iff majority_pid appears in the candidate's contributor list
    (y^any).
  * cluster_merged = 1 iff the candidate has ≥ 2 contributing particles.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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

WINDOW_N = 10.0
R_GEOM_MM = 5.0  # fixed geometric density radius

SLIM_COLUMNS = [
    "event_id",
    "seed_id",
    "branch_id",
    "step_k",
    "layer_id",
    "chi2_inc",
    "chi2_diag",
    "S00",
    "S01",
    "S11",
    "window_count",
    "geometric_density",
    "eta",
    "label",
    "majority_undefined",
    "cluster_merged",
]

# ACTS GeometryIdentifier masks (Core/Geometry/GeometryIdentifier.hpp)
# Use uint64 so high-bit masks don't overflow signed int64.
_VOL_MASK = np.uint64(0xFF00000000000000)
_LAY_MASK = np.uint64(0x0000FFF000000000)
_SEN_MASK = np.uint64(0x000000000FFFFF00)

PidKey = tuple[int, int, int, int, int]


def decode_geo_id(gid: int) -> tuple[int, int, int]:
    g = np.uint64(gid)
    volume = int((g & _VOL_MASK) >> np.uint64(56))
    layer = int((g & _LAY_MASK) >> np.uint64(36))
    sensitive = int((g & _SEN_MASK) >> np.uint64(8))
    return volume, layer, sensitive


def _pid_key(pv: int, sv: int, part: int, gen: int, sub: int) -> PidKey:
    """Stable particle barcode key (tuple — avoids int64 overflow)."""
    return (int(pv), int(sv), int(part), int(gen), int(sub))


def load_measurements(path: Path) -> dict[tuple[int, int, int], dict[str, np.ndarray]]:
    """Load measurements indexed by (volume, layer, module)."""
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    gids = df["geometry_id"].to_numpy(dtype=np.uint64)
    vol = ((gids & _VOL_MASK) >> np.uint64(56)).astype(np.int32)
    lay = ((gids & _LAY_MASK) >> np.uint64(36)).astype(np.int32)
    sen = ((gids & _SEN_MASK) >> np.uint64(8)).astype(np.int32)
    if "measurement_id" in df.columns:
        mid = df["measurement_id"].to_numpy(dtype=np.int64)
    else:
        mid = np.arange(len(df), dtype=np.int64)
    l0 = df["local0"].to_numpy(dtype=np.float64)
    l1 = df["local1"].to_numpy(dtype=np.float64)
    v0 = df["var_local0"].to_numpy(dtype=np.float64)
    v1 = df["var_local1"].to_numpy(dtype=np.float64)
    v0 = np.where(v0 > 0, v0, 0.015**2)
    v1 = np.where(v1 > 0, v1, 0.015**2)

    by_surface: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}
    # Group via sorting
    order = np.lexsort((sen, lay, vol))
    vol_s, lay_s, sen_s = vol[order], lay[order], sen[order]
    starts = np.flatnonzero(
        np.r_[
            True,
            (vol_s[1:] != vol_s[:-1])
            | (lay_s[1:] != lay_s[:-1])
            | (sen_s[1:] != sen_s[:-1]),
        ]
    )
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        key = (int(vol_s[s]), int(lay_s[s]), int(sen_s[s]))
        by_surface[key] = {
            "l0": l0[idx],
            "l1": l1[idx],
            "v0": v0[idx],
            "v1": v1[idx],
            "mid": mid[idx],
        }
    return by_surface


def load_meas_to_pids(map_path: Path, simhits_path: Path) -> dict[int, list[PidKey]]:
    """measurement_id → list of contributing particle keys."""
    mmap = pd.read_csv(map_path, comment="#")
    mmap.columns = [c.strip() for c in mmap.columns]
    mcol = "measurement_id" if "measurement_id" in mmap.columns else mmap.columns[0]
    hcol = "hit_id" if "hit_id" in mmap.columns else mmap.columns[1]

    sim = pd.read_csv(simhits_path, comment="#")
    sim.columns = [c.strip() for c in sim.columns]
    n_sim = len(sim)
    zeros = np.zeros(n_sim, dtype=np.int64)

    def _col(name: str) -> np.ndarray:
        if name not in sim.columns:
            return zeros
        return sim[name].to_numpy(dtype=np.int64, copy=False)

    pids = [
        _pid_key(int(a), int(b), int(c), int(d), int(e))
        for a, b, c, d, e in zip(
            _col("particle_id_pv"),
            _col("particle_id_sv"),
            _col("particle_id_part"),
            _col("particle_id_gen"),
            _col("particle_id_subpart"),
        )
    ]
    mids = mmap[mcol].to_numpy(dtype=np.int64)
    hids = mmap[hcol].to_numpy(dtype=np.int64)
    out: dict[int, list[PidKey]] = defaultdict(list)
    valid = (hids >= 0) & (hids < len(pids))
    for mid, hid in zip(mids[valid], hids[valid]):
        pid = pids[int(hid)]
        bucket = out[int(mid)]
        if pid not in bucket:
            bucket.append(pid)
    return dict(out)


def _to_python(x: Any) -> Any:
    """Convert uproot/awkward jagged entry to nested Python lists."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    # awkward Array / STLVector
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
    part_nested: Any,
    pv_nested: Any,
    sv_nested: Any,
    gen_nested: Any,
    sub_nested: Any,
) -> tuple[PidKey | None, bool]:
    """Mode particle across states; undefined if mode owns < 2/3 of states."""
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

        keys.append(
            _pid_key(
                _field(pvs_per),
                _field(svs_per),
                mode_part,
                _field(gens_per),
                _field(subs_per),
            )
        )
    if not keys:
        return None, True
    maj, n = Counter(keys).most_common(1)[0]
    return maj, bool(n * 3 < 2 * len(keys))


def load_predicted_cov(
    path: Path | None,
) -> dict[tuple[int, int], tuple[float, float, float]]:
    """(track_nr, step_k) → (P00, P01, P11)."""
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    out: dict[tuple[int, int], tuple[float, float, float]] = {}
    for row in df.itertuples(index=False):
        out[(int(row.track_nr), int(row.step_k))] = (
            float(row.P00),
            float(row.P01),
            float(row.P11),
        )
    return out


def chi2_full_and_diag(
    r0: np.ndarray,
    r1: np.ndarray,
    s00: float,
    s01: float,
    s11: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (chi2_true, chi2_diag) for residuals against one 2×2 S."""
    det = s00 * s11 - s01 * s01
    if not np.isfinite(det) or det <= 0:
        # Degenerate — fall back to diagonal
        chi2_diag = (r0 * r0) / max(s00, 1e-30) + (r1 * r1) / max(s11, 1e-30)
        return chi2_diag, chi2_diag
    # S^{-1} = (1/det) [[S11, -S01], [-S01, S00]]
    chi2_true = (s11 * r0 * r0 - 2.0 * s01 * r0 * r1 + s00 * r1 * r1) / det
    chi2_diag = (r0 * r0) / s00 + (r1 * r1) / s11
    return chi2_true, chi2_diag


def _empty_columns() -> dict[str, list]:
    return {c: [] for c in SLIM_COLUMNS}


def _append_rows(
    cols: dict[str, list],
    *,
    event_id: int,
    track_nr: int,
    step: int,
    layer_id: int,
    chi2: np.ndarray,
    idxs: np.ndarray,
    mid_arr: np.ndarray,
    n_window: int,
    n_geom: int,
    eta: float,
    s00: float,
    s01: float,
    s11: float,
    maj_pid: PidKey | None,
    maj_undef: bool,
    meas_to_pids: dict[int, list[PidKey]],
) -> None:
    for j in idxs:
        mid = int(mid_arr[j])
        contrib = meas_to_pids.get(mid, [])
        label = int((not maj_undef) and (maj_pid is not None) and (maj_pid in contrib))
        cols["event_id"].append(event_id)
        cols["seed_id"].append(track_nr)
        cols["branch_id"].append(track_nr)
        cols["step_k"].append(step)
        cols["layer_id"].append(layer_id)
        cols["chi2_inc"].append(float(chi2[j]))
        cols["S00"].append(float(s00))
        cols["S01"].append(float(s01))
        cols["S11"].append(float(s11))
        cols["window_count"].append(n_window)
        cols["geometric_density"].append(n_geom)
        cols["eta"].append(eta)
        cols["label"].append(label)
        cols["majority_undefined"].append(int(maj_undef))
        cols["cluster_merged"].append(int(len(contrib) >= 2))


def expand_event(
    *,
    event_id: int,
    trackstates_path: Path,
    measurements_path: Path,
    map_path: Path,
    simhits_path: Path,
    predicted_cov_path: Path | None = None,
    window_n: float = WINDOW_N,
    r_geom_mm: float = R_GEOM_MM,
    batch_size: int = 512,
    max_tracks: int = 25_000,
    sample_seed: int = 42,
) -> pd.DataFrame:
    """Expand track-states to slim chi² rows.

    Envelope CKF with ambi off writes O(10^6) track candidates per event.
    Expanding all of them exceeds the ~40 GB slim budget (~100M+ rows/event).
    We therefore draw a fixed-size uniform sample of track indices
    (``max_tracks``, seeded) and expand only those. Documented in the
    analysis SUMMARY. Set ``max_tracks=0`` to expand all (not recommended).

    ``predicted_cov_path`` supplies P00,P01,P11. Without it, P01=0 and
    chi2_inc falls back to the diagonal approximation (flagged in logs).
    """
    if uproot is None:
        raise ImportError("uproot required")

    t0 = time.time()
    print(f"[expand] event {event_id}: loading measurements/simhit map ...", flush=True)
    by_surface = load_measurements(measurements_path)
    meas_to_pids = load_meas_to_pids(map_path, simhits_path)
    pred_cov = load_predicted_cov(predicted_cov_path)
    if pred_cov:
        print(f"[expand] predicted-cov entries: {len(pred_cov)}", flush=True)
    else:
        print(
            "[expand] WARNING: no predicted-cov CSV — using diagonal S "
            "(P01=0); chi2_inc is NOT the MeasurementSelector statistic",
            flush=True,
        )
    print(
        f"[expand] surfaces={len(by_surface)} mapped_meas={len(meas_to_pids)} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    cols = _empty_columns()
    fields = [
        "volume_id",
        "layer_id",
        "module_id",
        "eLOC0_prt",
        "eLOC1_prt",
        "err_eLOC0_prt",
        "err_eLOC1_prt",
        "predicted",
        "eta_prt",
        "particle_ids_particle",
        "particle_ids_vertex_primary",
        "particle_ids_vertex_secondary",
        "particle_ids_generation",
        "particle_ids_sub_particle",
    ]

    with uproot.open(trackstates_path) as f:
        keys = list(f.keys())
        if not keys:
            raise FileNotFoundError(
                f"No trees in {trackstates_path} (empty/corrupt trackstates)"
            )
        tree_key = next(
            (k for k in keys if "trackstate" in k.lower() or "states" in k.lower()),
            keys[0],
        )
        tree = f[tree_key]
        n_tracks_total = int(tree.num_entries)
        if n_tracks_total <= 0:
            raise FileNotFoundError(
                f"Zero tracks in {trackstates_path} (empty trackstates)"
            )
        rng = np.random.default_rng(sample_seed + int(event_id))
        # Contiguous random window: sparse uniform sampling forces near-full
        # ROOT reads (avg gap ~ n/max_tracks) and is ~100× slower. A seeded
        # contiguous window keeps I/O sequential while staying in budget.
        if max_tracks and n_tracks_total > max_tracks:
            start = int(rng.integers(0, n_tracks_total - max_tracks + 1))
            entry_start, entry_stop = start, start + max_tracks
            print(
                f"[expand] contiguous sample tracks[{entry_start}:{entry_stop}] "
                f"of {n_tracks_total} (seed={sample_seed + event_id})",
                flush=True,
            )
        else:
            entry_start, entry_stop = 0, n_tracks_total
            print(
                f"[expand] trackstates={trackstates_path} n_tracks={n_tracks_total} "
                f"(no subsample)",
                flush=True,
            )

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
                vols = _to_python(batch["volume_id"][local_i])
                lays = _to_python(batch["layer_id"][local_i])
                mods = _to_python(batch["module_id"][local_i])
                e0 = _to_python(batch["eLOC0_prt"][local_i])
                e1 = _to_python(batch["eLOC1_prt"][local_i])
                err0 = _to_python(batch["err_eLOC0_prt"][local_i])
                err1 = _to_python(batch["err_eLOC1_prt"][local_i])
                has_prt = _to_python(batch["predicted"][local_i])
                eta_prt = _to_python(batch["eta_prt"][local_i])
                n_states = len(vols)

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
                    key = (int(vols[step]), int(lays[step]), int(mods[step]))
                    surf = by_surface.get(key)
                    if surf is None:
                        continue

                    cov_key = (int(track_nr), int(step))
                    if cov_key in pred_cov:
                        p00, p01, p11 = pred_cov[cov_key]
                    else:
                        p00 = (
                            float(err0[step]) ** 2
                            if step < len(err0) and np.isfinite(float(err0[step]))
                            else 0.05**2
                        )
                        p11 = (
                            float(err1[step]) ** 2
                            if step < len(err1) and np.isfinite(float(err1[step]))
                            else 0.05**2
                        )
                        p01 = 0.0
                    eta = (
                        float(eta_prt[step])
                        if step < len(eta_prt) and np.isfinite(float(eta_prt[step]))
                        else float("nan")
                    )

                    l0 = surf["l0"]
                    l1 = surf["l1"]
                    # V is diagonal in geometric digi CSV (no cov_local0_local1)
                    # S is per-candidate only through V; P is shared across the
                    # window on this surface, so S00/S01/S11 vary with V_i.
                    # For the gate cut ACTS uses per-candidate S = HPHᵀ+V_i.
                    # We store the state-level P+median(V) in columns as the
                    # common part; per-row chi2_inc uses that candidate's V.
                    v0 = surf["v0"]
                    v1 = surf["v1"]
                    # Window prefilter uses diagonal S (same as ACTS AABB shortcut)
                    s00_win = p00 + v0
                    s11_win = p11 + v1
                    r0 = l0 - pred0
                    r1 = l1 - pred1
                    in_window = (np.abs(r0) <= window_n * np.sqrt(s00_win)) & (
                        np.abs(r1) <= window_n * np.sqrt(s11_win)
                    )
                    n_window = int(in_window.sum())
                    if n_window == 0:
                        continue
                    n_geom = int((np.hypot(r0, r1) <= r_geom_mm).sum())
                    idxs = np.nonzero(in_window)[0]
                    # Per-candidate full S and χ²
                    for j in idxs:
                        s00 = float(p00 + v0[j])
                        s01 = float(p01)  # V01 = 0
                        s11 = float(p11 + v1[j])
                        chi2_t, chi2_d = chi2_full_and_diag(
                            np.array([r0[j]]), np.array([r1[j]]), s00, s01, s11
                        )
                        mid = int(surf["mid"][j])
                        contrib = meas_to_pids.get(mid, [])
                        label = int(
                            (not maj_undef)
                            and (maj_pid is not None)
                            and (maj_pid in contrib)
                        )
                        cols["event_id"].append(event_id)
                        cols["seed_id"].append(track_nr)
                        cols["branch_id"].append(track_nr)
                        cols["step_k"].append(step)
                        cols["layer_id"].append(int(lays[step]))
                        cols["chi2_inc"].append(float(chi2_t[0]))
                        cols["chi2_diag"].append(float(chi2_d[0]))
                        cols["S00"].append(s00)
                        cols["S01"].append(s01)
                        cols["S11"].append(s11)
                        cols["window_count"].append(n_window)
                        cols["geometric_density"].append(n_geom)
                        cols["eta"].append(eta)
                        cols["label"].append(label)
                        cols["majority_undefined"].append(int(maj_undef))
                        cols["cluster_merged"].append(int(len(contrib) >= 2))
                n_kept += 1

            if n_kept % 2000 < batch_size or batch_stop >= entry_stop:
                print(
                    f"[expand] kept={n_kept}/{entry_stop - entry_start} "
                    f"rows={len(cols['event_id'])} ({time.time() - t0:.1f}s)",
                    flush=True,
                )

    print(
        f"[expand] done event {event_id}: {len(cols['event_id'])} rows from "
        f"{n_kept} tracks in {time.time() - t0:.1f}s",
        flush=True,
    )
    if not cols["event_id"]:
        return pd.DataFrame(columns=SLIM_COLUMNS)
    return pd.DataFrame(cols, columns=SLIM_COLUMNS)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    if pa is None or pq is None:
        raise ImportError("pyarrow required")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df[SLIM_COLUMNS], preserve_index=False)
    pq.write_table(table, path, compression="zstd")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event-id", type=int, required=True)
    p.add_argument("--trackstates", type=Path, required=True)
    p.add_argument("--measurements", type=Path, required=True)
    p.add_argument("--meas-simhit-map", type=Path, required=True)
    p.add_argument("--simhits", type=Path, required=True)
    p.add_argument("--predicted-cov", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--window-n", type=float, default=WINDOW_N)
    p.add_argument("--r-geom-mm", type=float, default=R_GEOM_MM)
    args = p.parse_args()
    df = expand_event(
        event_id=args.event_id,
        trackstates_path=args.trackstates,
        measurements_path=args.measurements,
        map_path=args.meas_simhit_map,
        simhits_path=args.simhits,
        predicted_cov_path=args.predicted_cov,
        window_n=args.window_n,
        r_geom_mm=args.r_geom_mm,
    )
    write_parquet(df, args.out)
    print(
        f"event {args.event_id}: {len(df)} rows -> {args.out} "
        f"(R_geom={args.r_geom_mm} mm, window_n={args.window_n})"
    )


if __name__ == "__main__":
    main()
