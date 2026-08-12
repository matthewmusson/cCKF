"""Patch expanded parquet files to fill NaN columns.

Fills three groups of columns that were dropped or never populated during
the original expansion:

1. Innovation covariance (S00, S01, S11) — read from the ROOT trackstates
   file and joined on (seed_id, step_k) = (track_nr, state_idx).
2. Volume ID — read from the same ROOT file, needed for sensor property
   lookup.  Added as a new column.
3. Sensor properties (pitch_u, pitch_v, thickness, is_barrel) — looked up
   from the geometric-digitization config JSON by volume_id.

The patched file replaces the original in-place.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

try:
    import uproot
    import awkward as ak
except ImportError:
    uproot = ak = None

import json
from pathlib import Path


PIXEL_VOLUMES = frozenset({16, 17, 18})
BARREL_VOLUMES = frozenset({16, 23, 28})

_TRACKSTATE_BRANCHES = [
    "volume_id",
    "S00_prt",
    "S01_prt",
    "S11_prt",
]


def load_digi_config(path: str) -> dict[int, dict[str, float]]:
    """Parse the geometric-digitization config into a volume -> props map."""
    with open(path) as f:
        cfg = json.load(f)
    out = {}
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


def load_root_columns(root_path: str, event_id: int) -> pd.DataFrame:
    """Read volume_id and S00/S01/S11 from the ROOT trackstates tree.

    Returns a DataFrame with columns (track_nr, state_idx, volume_id,
    S00, S01, S11) — one row per trackstate.
    """
    with uproot.open(root_path) as f:
        tree_names = [k for k in f.keys() if "trackstate" in k.lower()]
        if not tree_names:
            tree_names = list(f.keys())
        tree = f[tree_names[0]]

        available = set(tree.keys())
        read_fields = [b for b in _TRACKSTATE_BRANCHES if b in available]
        has_event_nr = "event_nr" in available
        if has_event_nr:
            read_fields.append("event_nr")

        arrays = tree.arrays(read_fields, library="ak")
        if has_event_nr:
            mask = ak.to_numpy(arrays["event_nr"]) == int(event_id)
            arrays = arrays[mask]

    n_tracks = len(arrays)
    if n_tracks == 0:
        return pd.DataFrame(columns=["track_nr", "state_idx", "volume_id", "S00", "S01", "S11"])

    n_states = ak.to_numpy(ak.num(arrays["volume_id"], axis=1))
    track_nr = np.repeat(np.arange(n_tracks, dtype=np.int64), n_states)
    state_idx = ak.to_numpy(
        ak.flatten(ak.local_index(arrays["volume_id"], axis=1))
    ).astype(np.int64)

    vol = ak.to_numpy(ak.flatten(arrays["volume_id"])).astype(np.int64)
    s00 = ak.to_numpy(ak.flatten(arrays["S00_prt"])).astype(np.float64) if "S00_prt" in available else np.full(len(track_nr), np.nan)
    s01 = ak.to_numpy(ak.flatten(arrays["S01_prt"])).astype(np.float64) if "S01_prt" in available else np.full(len(track_nr), np.nan)
    s11 = ak.to_numpy(ak.flatten(arrays["S11_prt"])).astype(np.float64) if "S11_prt" in available else np.full(len(track_nr), np.nan)

    return pd.DataFrame({
        "track_nr": track_nr,
        "state_idx": state_idx,
        "volume_id": vol,
        "S00": s00,
        "S01": s01,
        "S11": s11,
    })


def patch_single_event(
    parquet_path: str,
    root_path: str,
    event_id: int,
    digi_table: dict[int, dict[str, float]],
) -> dict:
    """Patch one expanded parquet file in-place.

    Returns a summary dict with row count and fill rates.
    """
    df = pd.read_parquet(parquet_path)
    n_rows = len(df)
    original_cols = list(df.columns)

    root_df = load_root_columns(root_path, event_id)

    # The parquet uses seed_id (= track_nr) and step_k (= state_idx)
    # Each (seed_id, step_k) pair may appear multiple times in the parquet
    # (one per candidate hit on that surface), but S00/S01/S11 and volume_id
    # are the same for all candidates at a given state — they're properties
    # of the predicted state, not the candidate.
    df = df.merge(
        root_df,
        left_on=["seed_id", "step_k"],
        right_on=["track_nr", "state_idx"],
        how="left",
        suffixes=("", "_root"),
    )
    df.drop(columns=["track_nr", "state_idx"], inplace=True, errors="ignore")

    # If any S00/S01/S11 columns got _root suffixes from the merge
    # (shouldn't happen since originals don't have them, but defensive)
    for col in ("S00", "S01", "S11", "volume_id"):
        root_col = f"{col}_root"
        if root_col in df.columns:
            df[col] = df[root_col]
            df.drop(columns=[root_col], inplace=True)

    # Fill sensor properties from volume_id (vectorized)
    vol = df["volume_id"].to_numpy().astype(np.int64) if "volume_id" in df.columns else np.full(n_rows, -1, dtype=np.int64)
    pitch_u = np.full(n_rows, np.nan)
    pitch_v = np.full(n_rows, np.nan)
    thickness = np.full(n_rows, np.nan)
    is_barrel = np.full(n_rows, np.nan)

    for v_int, props in digi_table.items():
        mask = vol == v_int
        pitch_u[mask] = props["pitch_u"]
        pitch_v[mask] = props["pitch_v"]
        thickness[mask] = props["thickness"]

    known_vol_mask = vol > 0
    barrel_mask = np.isin(vol, list(BARREL_VOLUMES))
    is_barrel[known_vol_mask] = 0.0
    is_barrel[barrel_mask] = 1.0

    df["pitch_u"] = pitch_u
    df["pitch_v"] = pitch_v
    df["thickness"] = thickness
    df["is_barrel"] = is_barrel

    # Build final column list: original schema + new columns (S00, S01, S11, volume_id)
    insert_after_chi2 = ["S00", "S01", "S11"]
    final_cols = []
    for c in original_cols:
        final_cols.append(c)
        if c == "chi2_inc":
            final_cols.extend(insert_after_chi2)
        elif c == "surface_id":
            final_cols.append("volume_id")

    # Add any remaining columns not yet included
    for c in df.columns:
        if c not in final_cols:
            final_cols.append(c)
    # Only keep columns that exist in df
    final_cols = [c for c in final_cols if c in df.columns]

    df = df[final_cols]

    # Write back
    schema_overrides = {
        "contrib_pids": pa.list_(pa.int64()),
        "contrib_charge_frac": pa.list_(pa.float32()),
    }
    table = pa.Table.from_pandas(df, preserve_index=False)
    for col_name, pa_type in schema_overrides.items():
        if col_name in table.column_names:
            idx = table.column_names.index(col_name)
            try:
                col = table.column(col_name).cast(pa_type)
                table = table.set_column(idx, col_name, col)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                pass
    pq.write_table(table, parquet_path, compression="zstd")

    # Summary
    s00_fill = float((~np.isnan(df["S00"].to_numpy())).mean()) if "S00" in df.columns else 0
    pitch_fill = float((~np.isnan(df["pitch_u"].to_numpy())).mean()) if "pitch_u" in df.columns else 0
    vol_fill = float((df["volume_id"] > 0).mean()) if "volume_id" in df.columns else 0

    return {
        "event_id": event_id,
        "rows": n_rows,
        "S00_fill_pct": round(100 * s00_fill, 1),
        "pitch_fill_pct": round(100 * pitch_fill, 1),
        "volume_id_fill_pct": round(100 * vol_fill, 1),
    }
