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

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.patch_is_selected import patch_event


def _tiny_expanded_parquet(tmp_path, residuals):
    """One state with ``len(residuals)`` candidate rows.

    Carries the five columns ``patch_event`` reads plus a filler column, so
    the round-trip also proves untouched columns survive.
    """
    n = len(residuals)
    table = pa.table(
        {
            "seed_id": pa.array([0] * n, pa.int64()),
            "step_k": pa.array([0] * n, pa.int64()),
            "cand_hit_id": pa.array(list(range(n)), pa.int64()),
            "residual_l0": pa.array(list(residuals), pa.float64()),
            "residual_l1": pa.array([0.0] * n, pa.float64()),
            "filler": pa.array(["keep"] * n),
        }
    )
    path = tmp_path / "expanded_event000000000.parquet"
    pq.write_table(table, path)
    return path


def _stub_root(monkeypatch, root_res):
    from scripts import patch_is_selected

    monkeypatch.setattr(
        patch_is_selected, "load_root_residuals", lambda path, event_id: root_res
    )


def test_writes_when_match_fraction_is_high(tmp_path, monkeypatch):
    src = _tiny_expanded_parquet(tmp_path, [0.5, 1.5])
    _stub_root(
        monkeypatch,
        pd.DataFrame({"seed_id": [0], "step_k": [0], "res_l0": [0.5], "res_l1": [0.0]}),
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
            {"seed_id": [0], "step_k": [0], "res_l0": [999.0], "res_l1": [999.0]}
        ),
    )

    out = tmp_path / "out.parquet"
    with pytest.raises(ValueError, match="0.00% of"):
        patch_event(str(src), "unused.root", 0, str(out))
    assert not out.exists(), "a refused patch must leave no output behind"


def test_refuses_on_partial_match_below_floor(tmp_path, monkeypatch):
    """Half-matched is still refused -- the floor is 0.95, not 'any match'."""
    src = _tiny_expanded_parquet(tmp_path, [0.5])
    _stub_root(
        monkeypatch,
        pd.DataFrame(
            {
                "seed_id": [0, 0],
                "step_k": [0, 1],
                "res_l0": [0.5, 999.0],
                "res_l1": [0.0, 999.0],
            }
        ),
    )

    out = tmp_path / "out.parquet"
    with pytest.raises(ValueError, match="50.00% of"):
        patch_event(str(src), "unused.root", 0, str(out))
    assert not out.exists()


def test_floor_is_overridable_for_diagnostics(tmp_path, monkeypatch):
    src = _tiny_expanded_parquet(tmp_path, [0.5])
    _stub_root(
        monkeypatch,
        pd.DataFrame(
            {"seed_id": [0], "step_k": [0], "res_l0": [999.0], "res_l1": [999.0]}
        ),
    )

    out = tmp_path / "out.parquet"
    report = patch_event(str(src), "unused.root", 0, str(out), min_frac_matched=0.0)
    assert report["frac_states_matched"] == 0.0
    assert out.exists()
