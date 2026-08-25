"""Tests for geometry_id packing and the extra-byte normalization.

ODD endcap volumes write a nonzero ``extra`` field (ring index 1-3) into the
geometry_id of the digitization CSVs; the trackstates ROOT tree stores only
(volume, layer, module), so gids reconstructed from it always have extra = 0.
Joining the two spaces raw matched zero endcap surfaces and silently turned
39.46% of event 1's measurement states into hole rows (2026-08-25).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from expansion import (
    encode_geometry_id,
    load_cells,
    load_measurements,
    load_simhits,
    normalize_geometry_id,
)


def _pack(vol: int, lay: int, sen: int, extra: int) -> int:
    return (vol << 56) | (lay << 36) | (sen << 8) | extra


def test_normalize_strips_only_the_extra_byte():
    raw = np.array([_pack(28, 4, 17, 2), _pack(17, 2, 100, 0)], dtype=np.int64)
    out = normalize_geometry_id(raw)
    assert out.tolist() == [_pack(28, 4, 17, 0), _pack(17, 2, 100, 0)]


def test_normalized_endcap_gid_equals_root_reconstruction():
    """The whole point: an endcap CSV gid must equal encode_geometry_id of the
    (vol, lay, mod) triple the ROOT tree carries for the same surface."""
    csv_gid = np.array([_pack(16, 4, 60, 1)], dtype=np.int64)
    root_gid = encode_geometry_id(np.array([16]), np.array([4]), np.array([60]))
    np.testing.assert_array_equal(normalize_geometry_id(csv_gid), root_gid)


def test_normalize_is_idempotent():
    raw = np.array([_pack(23, 6, 5, 3)], dtype=np.int64)
    np.testing.assert_array_equal(
        normalize_geometry_id(normalize_geometry_id(raw)), normalize_geometry_id(raw)
    )


def _write_csv(path, df):
    df.to_csv(path, index=False)


def test_loaders_normalize_at_read_time(tmp_path):
    """Every CSV loader must emit extra = 0 so the pipeline never sees a raw
    endcap gid. A single un-normalized loader reintroduces the bug through
    whichever join it feeds (candidates, clusters, or truth labels)."""
    gid_raw = _pack(30, 8, 12, 2)
    gid_norm = _pack(30, 8, 12, 0)

    _write_csv(
        tmp_path / "event000000000-measurements.csv",
        pd.DataFrame(
            {
                "measurement_id": [0],
                "geometry_id": [gid_raw],
                "local0": [1.0],
                "local1": [2.0],
                "var_local0": [0.1],
                "var_local1": [0.1],
            }
        ),
    )
    _write_csv(
        tmp_path / "event000000000-cells.csv",
        pd.DataFrame(
            {
                "geometry_id": [gid_raw],
                "measurement_id": [0],
                "channel0": [1.0],
                "channel1": [1.0],
                "value": [1.0],
            }
        ),
    )
    _write_csv(
        tmp_path / "event000000000-simhits.csv",
        pd.DataFrame(
            {
                "geometry_id": [gid_raw],
                "particle_id_pv": [0],
                "particle_id_sv": [0],
                "particle_id_part": [1],
                "particle_id_gen": [0],
                "particle_id_subpart": [0],
                "tx": [0.0],
                "ty": [0.0],
                "tz": [0.0],
            }
        ),
    )

    assert load_measurements(str(tmp_path), 0)["geometry_id"].tolist() == [gid_norm]
    assert load_cells(str(tmp_path), 0)["geometry_id"].tolist() == [gid_norm]
    assert load_simhits(str(tmp_path), 0)["geometry_id"].tolist() == [gid_norm]
