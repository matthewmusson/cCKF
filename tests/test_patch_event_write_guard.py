"""Tests for ``patch_event``'s write-or-refuse behaviour.

``patch_event`` writes the Parquet that carries ``is_ckf_selected``, the
column the whole value target is derived from -- and until now it had no
tests at all. The match-fraction floor lived in ``main()``, so every caller
that imports ``patch_event`` directly (``modal_train.patch_selected_all``,
the only path that runs at scale) skipped the check. The first real 2-event
run wrote ``is_ckf_selected`` all-False with ``frac_states_matched == 0.0``
and exited 0.

The ROOT parsing is covered by the ``_root_residuals_from_arrays`` /
``_select_contributors_from_arrays`` tests in test_patch_is_selected.py, so
these stub ``load_root_residuals`` and test only the guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.patch_is_selected import patch_event


def _tiny_expanded_parquet(tmp_path, residuals, surfaces=None):
    """One state with ``len(residuals)`` candidate rows.

    Carries the columns ``patch_event`` reads (including the geometry
    triplet it composes into ``geometry_id``) plus a filler column, so the
    round-trip also proves untouched columns survive. ``volume_id=16,
    layer_id=2, surface_id=100`` for every row -- all candidates and the
    stubbed ROOT state are on the same single state.
    """
    n = len(residuals)
    surfaces = list(surfaces) if surfaces is not None else [100] * n
    table = pa.table(
        {
            "seed_id": pa.array([0] * n, pa.int64()),
            "volume_id": pa.array([16] * n, pa.int64()),
            "layer_id": pa.array([2] * n, pa.int64()),
            "surface_id": pa.array(surfaces, pa.int64()),
            # One state, so step_k is constant. patch_event reads step_k and
            # pred_* unconditionally; root_res here carries neither state_idx
            # nor sel_l0, so match_selected still takes the legacy residual
            # route, which is what these guard tests are about.
            "step_k": pa.array([0] * n, pa.int64()),
            "cand_hit_id": pa.array(list(range(n)), pa.int64()),
            "residual_l0": pa.array(list(residuals), pa.float64()),
            "residual_l1": pa.array([0.0] * n, pa.float64()),
            "pred_l0": pa.array([0.0] * n, pa.float64()),
            "pred_l1": pa.array([0.0] * n, pa.float64()),
            "filler": pa.array(["keep"] * n),
        }
    )
    path = tmp_path / "expanded_event000000000.parquet"
    pq.write_table(table, path)
    return path


def _geometry_id(volume_id: int = 16, layer_id: int = 2, surface_id: int = 100) -> int:
    from expansion import encode_geometry_id

    return int(
        encode_geometry_id(
            np.array([volume_id]), np.array([layer_id]), np.array([surface_id])
        )[0]
    )


def _stub_root(monkeypatch, root_res):
    from scripts import patch_is_selected

    monkeypatch.setattr(
        patch_is_selected, "load_root_residuals", lambda path, event_id: root_res
    )


def test_writes_when_match_fraction_is_high(tmp_path, monkeypatch):
    src = _tiny_expanded_parquet(tmp_path, [0.5, 1.5])
    _stub_root(
        monkeypatch,
        pd.DataFrame(
            {
                "seed_id": [0],
                "geometry_id": [_geometry_id()],
                "res_l0": [0.5],
                "res_l1": [0.0],
            }
        ),
    )

    out = tmp_path / "out.parquet"
    report = patch_event(str(src), "unused.root", 0, str(out))

    assert report["frac_states_matched"] == 1.0
    table = pq.read_table(out)
    assert table.column("is_ckf_selected").to_pylist() == [True, False]
    assert table.column("filler").to_pylist() == ["keep", "keep"]


def test_refuses_and_writes_nothing_on_zero_match(tmp_path, monkeypatch):
    """The observed real-data failure: the residual join matches nothing.

    An all-False ``is_ckf_selected`` is a column of the right name and dtype
    that silently asserts "the CKF never accepted a hit". Nothing downstream
    can tell it from a valid one, so the write must not happen at all.
    """
    src = _tiny_expanded_parquet(tmp_path, [0.5, 1.5])
    _stub_root(
        monkeypatch,
        pd.DataFrame(
            {
                "seed_id": [0],
                "geometry_id": [_geometry_id()],
                "res_l0": [999.0],
                "res_l1": [999.0],
            }
        ),
    )

    out = tmp_path / "out.parquet"
    with pytest.raises(ValueError, match="0.00% of"):
        patch_event(str(src), "unused.root", 0, str(out))
    assert not out.exists(), "a refused patch must leave no output behind"


def test_refuses_on_partial_match_below_floor(tmp_path, monkeypatch):
    """Half-matched is still refused -- the floor is 0.95, not 'any match'.

    Both states must exist in the Parquet. The floor is measured against the
    states the Parquet actually carries, not every state in the ROOT file:
    expansion legitimately omits states with no predicted parameters and
    states with no candidate inside the n-sigma box (2.80M of 4.62M on
    event 1), so a raw ROOT denominator caps the ratio near 0.6 and the floor
    could never pass however correct the join is.
    """
    src = _tiny_expanded_parquet(tmp_path, [0.5, 0.5], surfaces=[100, 200])
    _stub_root(
        monkeypatch,
        pd.DataFrame(
            {
                "seed_id": [0, 0],
                "geometry_id": [_geometry_id(), _geometry_id(surface_id=200)],
                "res_l0": [0.5, 999.0],
                "res_l1": [0.0, 999.0],
            }
        ),
    )

    out = tmp_path / "out.parquet"
    with pytest.raises(ValueError, match="50.00% of"):
        patch_event(str(src), "unused.root", 0, str(out))
    assert not out.exists()


def test_report_separates_coverage_from_match_rate(tmp_path, monkeypatch):
    """A ROOT state the Parquet never had is excluded from the floor but stays
    visible as coverage, so a collapse in expansion output is still detectable.
    """
    src = _tiny_expanded_parquet(tmp_path, [0.5])
    _stub_root(
        monkeypatch,
        pd.DataFrame(
            {
                "seed_id": [0, 0],
                "geometry_id": [_geometry_id(), _geometry_id(surface_id=999)],
                "res_l0": [0.5, 7.0],
                "res_l1": [0.0, 0.0],
            }
        ),
    )

    report = patch_event(str(src), "unused.root", 0, str(tmp_path / "out.parquet"))
    assert report["n_states"] == 2
    assert report["n_joinable_states"] == 1
    assert report["frac_joinable_matched"] == 1.0
    assert report["frac_states_matched"] == 0.5


def test_floor_is_overridable_for_diagnostics(tmp_path, monkeypatch):
    src = _tiny_expanded_parquet(tmp_path, [0.5])
    _stub_root(
        monkeypatch,
        pd.DataFrame(
            {
                "seed_id": [0],
                "geometry_id": [_geometry_id()],
                "res_l0": [999.0],
                "res_l1": [999.0],
            }
        ),
    )

    out = tmp_path / "out.parquet"
    report = patch_event(str(src), "unused.root", 0, str(out), min_frac_matched=0.0)
    assert report["frac_states_matched"] == 0.0
    assert out.exists()
