"""Tests for inline is_ckf_selected in expand_trackstates.

The ROOT trackstates tree records, at every state, the local position of the
measurement the CKF accepted (l_x_hit / l_y_hit). Expansion is already at
that state with every measurement on the surface joined, so the accepted
candidate is identified by a direct position comparison -- no second pass, no
key reconstruction. This replaces scripts/patch_is_selected.py, whose offline
join failed three times (LOG 2026-08-17 / -18 / -25).

Position agreement was measured on real event 1 at 100.0000% of states to
1e-6 mm, so the 1e-4 mm tolerance here is three orders of magnitude of slack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from expansion import expand_trackstates, encode_geometry_id


def _gid(vol: int, lay: int, mod: int) -> int:
    return int(encode_geometry_id(np.array([vol]), np.array([lay]), np.array([mod]))[0])


_STATE_DEFAULTS = {
    "event_id": 0,
    "is_predicted": True,
    "pred_phi": 0.0, "pred_theta": 1.0, "pred_qop": 0.5, "pred_t": 0.0,
    "err_l0": 0.1, "err_l1": 0.1, "err_phi": 0.01, "err_theta": 0.01,
    "err_qop": 0.01, "err_t": 0.0,
    "S00": np.nan, "S01": np.nan, "S11": np.nan,
    "eta": 0.5, "pathInX0_interval": 0.01,
    "clus_s_u": 0.01, "clus_s_v": 0.01, "clus_q_tot": 100.0,
    "clus_sigma_uu": 1e-3, "clus_sigma_uv": 0.0, "clus_sigma_vv": 1e-3,
    "alpha_u": 0.1, "alpha_v": 0.1,
    "state_primary_pid": 1, "state_n_contribs": 1,
}


def _states(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        d = dict(_STATE_DEFAULTS)
        d.update(r)
        gid = _gid(d["volume_id"], d["layer_id"], d["module_id"])
        d["geometry_id"] = gid
        out.append(d)
    return pd.DataFrame(out)


def _measurements(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _pcov(states: pd.DataFrame) -> pd.DataFrame:
    """Predicted covariance for every state: small isotropic H C H^T."""
    return pd.DataFrame(
        {
            "track_nr": states["track_nr"],
            "state_idx": states["state_idx"],
            "P00": 0.01,
            "P01": 0.0,
            "P11": 0.01,
        }
    )


def _base_case():
    """One 2D state with three in-window candidates; CKF accepted hit 1."""
    states = _states(
        [
            {
                "track_nr": 0, "state_idx": 3,
                "volume_id": 17, "layer_id": 2, "module_id": 10,
                "pred_l0": 1.0, "pred_l1": 2.0,
                "sel_l0": 1.25, "sel_l1": 2.10,
            }
        ]
    )
    gid = states["geometry_id"].iloc[0]
    meas = _measurements(
        [
            {"measurement_id": 0, "geometry_id": gid, "local0": 0.90,
             "local1": 1.95, "var_local0": 0.01, "var_local1": 0.01},
            {"measurement_id": 1, "geometry_id": gid, "local0": 1.25,
             "local1": 2.10, "var_local0": 0.01, "var_local1": 0.01},
            {"measurement_id": 2, "geometry_id": gid, "local0": 1.40,
             "local1": 2.30, "var_local0": 0.01, "var_local1": 0.01},
        ]
    )
    return states, meas


def test_selected_candidate_is_flagged_by_position():
    states, meas = _base_case()
    out = expand_trackstates(states, meas, predicted_cov=_pcov(states))
    cand = out[out["cand_hit_id"] >= 0].sort_values("cand_hit_id")
    assert "is_ckf_selected" in out.columns
    assert cand["is_ckf_selected"].tolist() == [False, True, False]


def test_exactly_one_selected_per_state():
    states, meas = _base_case()
    out = expand_trackstates(states, meas, predicted_cov=_pcov(states))
    per_state = out.groupby(["track_nr", "state_idx"])["is_ckf_selected"].sum()
    assert (per_state == 1).all()


def test_one_dimensional_state_matches_on_l0_alone():
    """Long strips have no l1: sel_l1 is NaN and var_local1 == 0 (the ODD 1D
    sentinel). The match must not require l1 or every long-strip state loses
    its selected flag -- the defect that removed long strips from training."""
    states = _states(
        [
            {
                "track_nr": 0, "state_idx": 0,
                "volume_id": 28, "layer_id": 4, "module_id": 7,
                "pred_l0": 5.0, "pred_l1": np.nan,
                "sel_l0": 5.50, "sel_l1": np.nan,
            }
        ]
    )
    gid = states["geometry_id"].iloc[0]
    meas = _measurements(
        [
            {"measurement_id": 0, "geometry_id": gid, "local0": 4.60,
             "local1": 0.0, "var_local0": 0.04, "var_local1": 0.0},
            {"measurement_id": 1, "geometry_id": gid, "local0": 5.50,
             "local1": 0.0, "var_local0": 0.04, "var_local1": 0.0},
        ]
    )
    out = expand_trackstates(states, meas, predicted_cov=_pcov(states))
    cand = out[out["cand_hit_id"] >= 0].sort_values("cand_hit_id")
    assert cand["is_ckf_selected"].tolist() == [False, True]


def test_state_without_accepted_hit_selects_nothing():
    """A ROOT hole state (sel_l0 NaN) has no accepted hit; its candidates all
    stay False."""
    states, meas = _base_case()
    states["sel_l0"] = np.nan
    states["sel_l1"] = np.nan
    out = expand_trackstates(states, meas, predicted_cov=_pcov(states))
    assert not out["is_ckf_selected"].any()


def test_hole_rows_are_never_selected():
    """A state with no in-window candidate emits a hole row; the flag is
    False there by construction."""
    states, meas = _base_case()
    meas = meas.iloc[:0]  # no measurements at all -> hole row
    out = expand_trackstates(states, meas, predicted_cov=_pcov(states))
    assert (out["cand_hit_id"] == -1).all()
    assert not out["is_ckf_selected"].any()


def test_ties_break_to_the_closest_then_lowest_hit_id():
    """Two candidates inside tolerance: the closer one wins; exact ties go to
    the lower cand_hit_id so the output is deterministic."""
    states, meas = _base_case()
    extra = meas.iloc[[1]].copy()
    extra["measurement_id"] = 3
    extra["local0"] = 1.25 + 5e-5  # inside 1e-4 tol, but farther than hit 1
    meas = pd.concat([meas, extra], ignore_index=True)
    out = expand_trackstates(states, meas, predicted_cov=_pcov(states))
    sel = out.loc[out["is_ckf_selected"], "cand_hit_id"]
    assert sel.tolist() == [1]


def test_guard_raises_when_accepted_hits_never_match():
    """If ROOT says the CKF accepted hits but no candidate ever matches, the
    expansion must fail loudly rather than write an all-False column -- the
    silent-failure mode that produced the 2026-08-17 incident."""
    states, meas = _base_case()
    states["sel_l0"] = 999.0  # accepted-hit position matches nothing
    with pytest.raises(ValueError, match="is_ckf_selected"):
        expand_trackstates(states, meas, predicted_cov=_pcov(states))


def test_guard_ignores_states_without_candidates():
    """The guard denominator is states whose accepted hit could have been
    found (>= 1 in-window candidate). A state whose surface has no
    measurements must not count against it."""
    states, meas = _base_case()
    lonely = _states(
        [
            {
                "track_nr": 1, "state_idx": 0,
                "volume_id": 24, "layer_id": 2, "module_id": 99,
                "pred_l0": 0.0, "pred_l1": 0.0,
                "sel_l0": 0.5, "sel_l1": 0.5,
            }
        ]
    )
    states = pd.concat([states, lonely], ignore_index=True)
    out = expand_trackstates(states, meas, predicted_cov=_pcov(states))
    sel_rows = out.loc[out["is_ckf_selected"]]
    assert len(sel_rows) == 1 and sel_rows["cand_hit_id"].iloc[0] == 1
