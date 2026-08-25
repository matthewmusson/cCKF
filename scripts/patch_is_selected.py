"""Recover the CKF-accepted hit per state, for the value target's backward half.

Why this exists
---------------
The expansion emits one row per (branch, surface, **candidate**) and stamps every
candidate with ``action_taken = 0`` — "in-window candidate, accept/reject decided
offline". Nothing in the 76-column schema records which candidate the CKF
accepted, nor that candidate's contributor list. So spec §3.3's "iterate over
accepted hits on steps 0 through k" is not computable from the Parquet alone.

Scope: this affects only ``n_correct`` and ``n_wrong``, and it affects **all
three tiers equally** — every tier needs those counts. The forward half needs
nothing from here — both tiers read ``majority_true_hit_on_surface``.

Two routes, deliberately both
-----------------------------
**Primary — contributor join, exact.** ROOT's per-state ``particle_ids_*``
vectors are the contributor list at the CKF-selected hit
(``expansion.py:472``). Decode them with the same barcode packing that
``simhits.csv`` and ``branch_majority_pid`` use, then test membership. No float
tolerance, and it reproduces the spec's own criterion.

  Do **not** substitute ROOT's ``state_primary_pid``: that is the *mode* of the
  contributor list, so it disagrees with membership on exactly the merged
  clusters that §1.3 keeps at full weight. A branch whose majority particle is a
  minority contributor to an accepted cluster would be scored as having taken a
  wrong hit, deflating purity and biasing V^{π†} downward precisely on the
  ambiguous states where the value function's decision matters most.

**Cross-check — residual match.** ACTS stores the selected measurement's
residual against the predicted state, ``res_eLOC0_prt = l_x_hit − eLOC0_prt``,
which is exactly the expanded row's ``residual_l0 = local0 − pred_l0`` for the
accepted candidate. Matching that pair within tolerance flags
``is_ckf_selected``. Matching on the *residual* rather than the hit position
avoids depending on whether the expansion's ``pred_l0`` and ROOT's
``eLOC0_prt`` rounded identically — both sides are differences from the same
predicted state.

Keep both: the flag is independently useful for reconstructing branch
trajectories, and the two routes must agree on ``sel_correct``. A disagreement
rate above ~0.1% means one of them is broken — much cheaper to catch here than
after V_φ has trained on corrupt targets.

Usage
-----
    python scripts/patch_is_selected.py \\
        --parquet /data/results/train32/expanded/expanded_event000000000.parquet \\
        --root /data/results/<pilot_dir>/trackstates_ckf.root \\
        --event-id 0 \\
        --out /data/results/train32/selected/expanded_event000000000.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cckf.splits import assert_not_test

# Reuse the expansion's own barcode packing rather than reimplementing it.
# expansion.encode_particle_id's docstring is explicit that the layout is a
# pipeline-internal convention, not ACTS's SimBarcode -- so a local copy that
# drifted would break the majority-membership test silently.
from expansion import encode_particle_id, encode_geometry_id

#: Residual match tolerance in mm. ROOT stores float32, the expansion computes
#: in float64, so agreement is limited by float32 epsilon on values of order
#: 1-100 mm — roughly 1e-5. 1e-4 is comfortably above that and far below the
#: spacing between distinct candidates on a surface (>= one pitch, ~0.05 mm).
DEFAULT_TOL = 1e-4


#: particle_ids_* branches, doubly-jagged (track, state, contributor) -- see
#: expansion.py's ``_TRACKSTATE_PARTICLE_BRANCHES`` docstring.
_PARTICLE_ID_FIELDS = (
    "particle_ids_vertex_primary",
    "particle_ids_vertex_secondary",
    "particle_ids_particle",
    "particle_ids_generation",
    "particle_ids_sub_particle",
)


def _select_contributors_from_arrays(arrays, event_id: int) -> pd.DataFrame:
    """Pure parsing step of :func:`load_selected_contributors`.

    Takes an already-loaded awkward Array (one entry per track, as read from
    the ``trackstates`` tree with ``library="ak"``) so it can be exercised
    directly against synthetic doubly-jagged fixtures without a ROOT file.

    The tree is one entry per **track**, not per state (expansion.py:418-425).
    ``particle_ids_particle`` (and its four sibling fields) is doubly-jagged:
    (track, state, contributor) -- expansion.py:135-136. This mirrors
    ``expansion.py``'s ``load_trackstates`` exactly: ``ak.num(..., axis=1)``
    on the per-state layer builds ``seed_id``, then a single ``axis=1``
    flatten drops the per-track nesting (leaving one, possibly empty, jagged
    contributor list per state) before a full flatten and re-grouping by
    per-state counts recovers the contributor lists.

    ``volume_id``/``layer_id``/``module_id`` are singly-jagged (track,
    state), the same shape as the residual branches in
    :func:`_root_residuals_from_arrays` -- a single ``axis=1`` flatten (not
    the doubly-jagged pattern above) recovers one value per state, which is
    packed into ``geometry_id`` via :func:`expansion.encode_geometry_id`.
    ``step_k`` (a per-track state ordinal) does not mean the same thing in
    the Parquet as in the ROOT tree, so it cannot be used to join the two.
    """
    import awkward as ak

    if "event_nr" in arrays.fields:
        event_mask = ak.to_numpy(arrays["event_nr"]) == int(event_id)
        arrays = arrays[event_mask]

    empty = pd.DataFrame(
        columns=[
            "seed_id", "state_idx", "geometry_id", "sel_contrib_pids",
            "sel_has_hit",
        ]
    )
    n_tracks = len(arrays)
    if n_tracks == 0:
        return empty

    # particle_ids_particle is doubly-jagged (track, state, contributor):
    # axis=1 num operates on the per-state layer directly.
    n_states = ak.to_numpy(ak.num(arrays["particle_ids_particle"], axis=1)).astype(
        np.int64
    )
    if n_states.sum() == 0:
        return empty
    track_nr = np.repeat(np.arange(n_tracks, dtype=np.int64), n_states)

    # volume_id/layer_id/module_id are singly-jagged (track, state): a plain
    # axis=1 flatten recovers one value per state, same as the residuals.
    vol = ak.to_numpy(ak.flatten(arrays["volume_id"], axis=1)).astype(np.int64)
    lay = ak.to_numpy(ak.flatten(arrays["layer_id"], axis=1)).astype(np.int64)
    mod = ak.to_numpy(ak.flatten(arrays["module_id"], axis=1)).astype(np.int64)
    gid = encode_geometry_id(vol, lay, mod)

    # Per-track state index, matching the Parquet's step_k. Every branch read
    # here is all-state (contributors exist for holes too, as an empty list),
    # so this is a plain local index with no measurement mask -- unlike
    # _root_residuals_from_arrays, which must mask because res_*_prt is
    # measurement-only.
    sidx = ak.to_numpy(
        ak.flatten(ak.local_index(arrays["volume_id"], axis=1), axis=1)
    ).astype(np.int64)

    # Drop the per-track nesting only -- each element is now one (possibly
    # empty) jagged contributor list per state, in (track, state) order.
    per_state = {name: ak.flatten(arrays[name], axis=1) for name in _PARTICLE_ID_FIELDS}
    counts = ak.to_numpy(ak.num(per_state["particle_ids_particle"])).astype(np.int64)

    flat_pv = ak.to_numpy(ak.flatten(per_state["particle_ids_vertex_primary"]))
    flat_sv = ak.to_numpy(ak.flatten(per_state["particle_ids_vertex_secondary"]))
    flat_p = ak.to_numpy(ak.flatten(per_state["particle_ids_particle"]))
    flat_gen = ak.to_numpy(ak.flatten(per_state["particle_ids_generation"]))
    flat_sub = ak.to_numpy(ak.flatten(per_state["particle_ids_sub_particle"]))

    codes = encode_particle_id(flat_pv, flat_sv, flat_p, flat_gen, flat_sub)
    nested = ak.unflatten(codes, counts)

    df = pd.DataFrame(
        {
            "seed_id": track_nr,
            "state_idx": sidx,
            "geometry_id": gid,
            "sel_contrib_pids": ak.to_list(nested),
        }
    )
    df["sel_has_hit"] = df["sel_contrib_pids"].map(len) > 0
    return df


def load_selected_contributors(root_path: str, event_id: int) -> pd.DataFrame:
    """Read the contributor list of each state's CKF-selected hit.

    Returns
    -------
    pandas.DataFrame
        ``seed_id``, ``geometry_id``, ``sel_contrib_pids`` (list of int64
        barcodes), ``sel_has_hit`` (False for holes, where the contributor
        list is empty).
    """
    import uproot

    with uproot.open(root_path) as fh:
        tree = fh["trackstates"]
        available = set(tree.keys())
        fields = list(_PARTICLE_ID_FIELDS) + list(_GEOMETRY_FIELDS)
        if "event_nr" in available:
            fields = fields + ["event_nr"]
        # uproot's `cut=` filtering silently no-ops in this environment (see
        # expansion.py's load_trackstates); boolean-mask the loaded awkward
        # array instead.
        arrays = tree.arrays(fields, library="ak")

    return _select_contributors_from_arrays(arrays, event_id)


def selected_correctness(sel: pd.DataFrame, majority: pd.DataFrame) -> pd.DataFrame:
    """Split each state's accepted hit into correct / wrong / neither.

    Parameters
    ----------
    sel : pandas.DataFrame
        Output of :func:`load_selected_contributors`.
    majority : pandas.DataFrame
        ``seed_id`` → ``branch_majority_pid``, from the Parquet.

    Returns
    -------
    pandas.DataFrame
        ``seed_id``, ``geometry_id``, ``sel_correct``, ``sel_wrong`` in
        {0, 1}, with at most one set per state. A hole is neither: it costs
        completeness but does not pollute purity.
    """
    out = sel.merge(majority, on="seed_id", how="left")
    maj = out["branch_majority_pid"].to_numpy()
    member = np.array(
        [
            (m in pids) if isinstance(pids, (list, tuple)) else False
            for m, pids in zip(maj, out["sel_contrib_pids"])
        ]
    )
    has_hit = out["sel_has_hit"].to_numpy(dtype=bool)
    out["sel_correct"] = (has_hit & member).astype(np.int64)
    out["sel_wrong"] = (has_hit & ~member).astype(np.int64)
    return out[["seed_id", "geometry_id", "sel_correct", "sel_wrong"]]


#: Measurement-only residual branches.  These are shorter than the all-state
#: branches (``volume_id``, ``eLOC0_prt``, etc.) -- they contain one entry
#: per measurement state, not per state overall.  ``l_x_hit`` (all-state) is
#: read alongside the geometry fields so we can mask geometry down to the
#: same measurement-state subset before flattening.
_RESIDUAL_FIELDS = ("res_eLOC0_prt", "res_eLOC1_prt")

#: All-state geometry branches.  These have one entry per state (measurements
#: + holes + material interactions), so they are *longer* than the
#: measurement-only ``res_*_prt`` branches.  A ``l_x_hit``-based mask selects
#: the measurement subset so the two can be placed in the same DataFrame.
_GEOMETRY_FIELDS = ("volume_id", "layer_id", "module_id")


def _root_residuals_from_arrays(arrays, event_id: int) -> pd.DataFrame:
    """Pure parsing step of :func:`load_root_residuals`.

    Takes an already-loaded awkward Array (one entry per track) so it can be
    exercised directly against synthetic fixtures without a ROOT file.

    The ROOT tree has two jagged length groups per track:

    * **All-state** (one entry per CKF state -- measurements, holes, material
      interactions): ``volume_id``, ``layer_id``, ``module_id``, ``l_x_hit``,
      ``eLOC0_prt``, etc.
    * **Measurement-only** (one entry per state that has a hit):
      ``res_eLOC0_prt``, ``res_eLOC1_prt``, ``pull_*_prt``, ``dim_hit``, etc.

    The geometry fields are all-state while the residuals are measurement-only,
    so the geometry arrays must be masked to the measurement subset before
    flattening.  ``l_x_hit`` (all-state, finite only at measurement states)
    provides that mask.
    """
    import awkward as ak

    if "event_nr" in arrays.fields:
        event_mask = ak.to_numpy(arrays["event_nr"]) == int(event_id)
        arrays = arrays[event_mask]

    empty = pd.DataFrame(
        columns=["seed_id", "state_idx", "geometry_id", "res_l0", "res_l1"])
    n_tracks = len(arrays)
    if n_tracks == 0:
        return empty

    n_meas = ak.to_numpy(ak.num(arrays["res_eLOC0_prt"], axis=1)).astype(np.int64)
    if n_meas.sum() == 0:
        return empty
    track_nr = np.repeat(np.arange(n_tracks, dtype=np.int64), n_meas)

    res0 = ak.to_numpy(ak.flatten(arrays["res_eLOC0_prt"], axis=1)).astype(np.float64)
    res1 = ak.to_numpy(ak.flatten(arrays["res_eLOC1_prt"], axis=1)).astype(np.float64)

    # Mask all-state geometry down to measurement states only.
    hit_mask = ~ak.is_none(arrays["l_x_hit"]) & ak.is_valid(arrays["l_x_hit"])
    lx = ak.fill_none(arrays["l_x_hit"], np.nan)
    hit_mask = np.isfinite(ak.to_numpy(ak.flatten(lx, axis=1)))
    # Rebuild as jagged mask matching the all-state shape.
    all_n = ak.to_numpy(ak.num(arrays["volume_id"], axis=1))
    hit_mask_jagged = ak.unflatten(hit_mask, all_n)

    vol = ak.to_numpy(ak.flatten(arrays["volume_id"][hit_mask_jagged], axis=1)).astype(np.int64)
    lay = ak.to_numpy(ak.flatten(arrays["layer_id"][hit_mask_jagged], axis=1)).astype(np.int64)
    mod = ak.to_numpy(ak.flatten(arrays["module_id"][hit_mask_jagged], axis=1)).astype(np.int64)
    gid = encode_geometry_id(vol, lay, mod)

    # Global state index for each measurement state. The res_*_prt branches are
    # measurement-only, so they carry no state index of their own -- but the
    # same hit_mask_jagged that selects the measurement subset from the
    # all-state geometry arrays also selects it from a local index. This gives
    # an EXACT (track_nr, state_idx) key, which is what the Parquet stores as
    # (seed_id, step_k). See expansion.py's rename map.
    #
    # Why this matters: matching on (seed_id, geometry_id) is one-to-many --
    # one selected state against every candidate on that surface -- so the
    # selected hit had to be identified by a 1e-4 mm residual coincidence.
    # That is fragile by construction and has failed twice on this project
    # (LOG 2026-08-17/18, and again after re-expansion widened the candidate
    # set, where it matched 0.03% of states).
    sidx = ak.to_numpy(
        ak.flatten(ak.local_index(arrays["volume_id"], axis=1)[hit_mask_jagged], axis=1)
    ).astype(np.int64)

    # The SELECTED measurement's local coordinates at each state.
    #
    # This, not res_eLOC0_prt, is what identifies which candidate the CKF
    # accepted. res_*_prt is a measurement-only branch (4.62M entries on
    # event 1 against 19.45M all-state) and is NOT l_x_hit - eLOC0_prt:
    # measured against the real files, |residual| agreement between Parquet
    # candidates and res_eLOC0_prt peaks at 0.36% of states even on an exact
    # (seed_id, state_idx) join, while l_x_hit matches a Parquet candidate at
    # 100.0000% of states to 1e-6. l_x_hit/l_y_hit are all-state branches, so
    # they need no cross-length reindexing at all.
    lx_flat = ak.to_numpy(ak.flatten(lx, axis=1))
    sel_l0 = lx_flat[hit_mask].astype(np.float64)
    if "l_y_hit" in arrays.fields:
        ly = ak.fill_none(arrays["l_y_hit"], np.nan)
        sel_l1 = ak.to_numpy(ak.flatten(ly, axis=1))[hit_mask].astype(np.float64)
    else:
        sel_l1 = np.full(sel_l0.shape, np.nan, dtype=np.float64)

    df = pd.DataFrame(
        {
            "seed_id": track_nr,
            "state_idx": sidx,
            "geometry_id": gid,
            "res_l0": res0,
            "res_l1": res1,
            "sel_l0": sel_l0,
            "sel_l1": sel_l1,
        }
    )
    # Keep any state that carries a selected measurement. sel_l0 is finite by
    # construction here (hit_mask is exactly isfinite(l_x_hit)); res_l0 is
    # retained only for the legacy residual path and must not gate the rows,
    # or states usable by the measurement join would be dropped before it runs.
    return df.loc[np.isfinite(df["sel_l0"])].reset_index(drop=True)


def load_root_residuals(root_path: str, event_id: int) -> pd.DataFrame:
    """Read per-state selected-hit residuals from the trackstates ROOT tree.

    Returns
    -------
    pandas.DataFrame
        Columns ``seed_id``, ``geometry_id``, ``res_l0``, ``res_l1``; one row
        per state that has a selected measurement (rows where either residual
        is NaN, i.e. holes, are dropped).
    """
    import uproot

    with uproot.open(root_path) as fh:
        tree = fh["trackstates"]
        available = set(tree.keys())
        fields = list(_RESIDUAL_FIELDS) + list(_GEOMETRY_FIELDS) + ["l_x_hit"]
        if "l_y_hit" in available:
            fields = fields + ["l_y_hit"]
        if "event_nr" in available:
            fields = fields + ["event_nr"]
        # uproot's `cut=` filtering silently no-ops in this environment (see
        # expansion.py's load_trackstates); boolean-mask the loaded awkward
        # array instead. Read as awkward, not "np" -- these are jagged
        # per-state branches, not one flat value per row.
        arrays = tree.arrays(fields, library="ak")

    return _root_residuals_from_arrays(arrays, event_id)


def match_selected(
    cand: pd.DataFrame, root_res: pd.DataFrame, tol: float = DEFAULT_TOL
) -> np.ndarray:
    """Flag the candidate the CKF accepted at each state.

    Two routes, chosen by which columns are present:

    * **exact** -- ``(seed_id, state_idx)`` join, selected candidate found by
      matching the measurement position ``residual_l0 + pred_l0`` against the
      ROOT state's ``l_x_hit``. Requires ``state_idx``/``step_k`` and
      ``pred_l0`` on ``cand`` and ``sel_l0`` on ``root_res``. Grouping is per
      state.
    * **legacy** -- ``(seed_id, geometry_id)`` join with a residual tolerance.
      Grouping is per surface. Kept for older ROOT files and the fixtures.

    Parameters
    ----------
    cand : pandas.DataFrame
        Candidate rows with ``seed_id``, ``geometry_id``, ``cand_hit_id``,
        ``residual_l0``, ``residual_l1``; plus ``step_k`` (or ``state_idx``)
        and ``pred_l0``/``pred_l1`` to enable the exact route.
    root_res : pandas.DataFrame
        Per-state selected hit from :func:`load_root_residuals`.
    tol : float
        Absolute match tolerance in mm on each residual component.

    Returns
    -------
    numpy.ndarray
        Bool array aligned with ``cand``'s rows. At most one True per state;
        exactly zero for states absent from ``root_res`` or where no candidate
        matches within ``tol``. Ties (identical residuals) are broken by
        smallest ``cand_hit_id`` so the result is deterministic.
    """
    work = cand.reset_index(drop=True)
    if "state_idx" not in work.columns and "step_k" in work.columns:
        work = work.rename(columns={"step_k": "state_idx"})

    exact = (
        "state_idx" in work.columns
        and {"state_idx", "sel_l0"} <= set(root_res.columns)
        and {"pred_l0", "residual_l0"} <= set(work.columns)
    )

    if exact:
        # PRIMARY ROUTE. Join on (seed_id, state_idx) -- one ROOT state to its
        # own candidates, never one state against a whole surface -- then
        # identify the accepted candidate by its MEASUREMENT position.
        #
        # Both halves of this were wrong before and each failed on its own:
        #   - (seed_id, geometry_id) is one-to-many, so the selected hit had to
        #     be recovered by a 1e-4 mm residual coincidence (LOG 2026-08-17).
        #   - the exact key alone still failed, because the residual it was
        #     compared against, res_eLOC0_prt, is a measurement-only branch
        #     that is not l_x_hit - eLOC0_prt.
        # Verified on event 1: pred_l0 == eLOC0_prt at 100.0000%, and the
        # measurement match below fires on 100.0000% of joinable states.
        work = work.copy()
        work["_loc0"] = work["residual_l0"] + work["pred_l0"]
        if {"residual_l1", "pred_l1"} <= set(work.columns):
            work["_loc1"] = work["residual_l1"] + work["pred_l1"]
        else:
            work["_loc1"] = np.nan
        merged = work.merge(
            root_res[["seed_id", "state_idx", "sel_l0", "sel_l1"]],
            on=["seed_id", "state_idx"],
            how="left",
        )
        d0 = (merged["_loc0"] - merged["sel_l0"]).abs().to_numpy(dtype=np.float64)
        d1 = (merged["_loc1"] - merged["sel_l1"]).abs().to_numpy(dtype=np.float64)
        # l1 is absent on 1D sensors (volumes 28/29/30 digitise (0,) only), so
        # it constrains the match where it exists and is ignored where it does
        # not. Requiring it outright would reject every long strip.
        close = (d0 <= tol) & (~np.isfinite(d1) | (d1 <= tol))
        close &= np.isfinite(merged["sel_l0"].to_numpy(dtype=np.float64))
        group_keys = ["seed_id", "state_idx"]
        rank = d0
    else:
        # LEGACY ROUTE, retained for ROOT files written before l_x_hit was
        # recovered and for the unit fixtures. Fragile by construction: it
        # compares one state's residual against every candidate on the surface.
        merged = work.merge(root_res, on=["seed_id", "geometry_id"], how="left")
        close = (
            (merged["residual_l0"] - merged["res_l0"]).abs().le(tol)
            & (merged["residual_l1"] - merged["res_l1"]).abs().le(tol)
        ).to_numpy(dtype=bool)
        close = close & np.isfinite(merged["res_l0"].to_numpy(dtype=np.float64))
        group_keys = ["seed_id", "geometry_id"]
        rank = np.zeros(len(merged), dtype=np.float64)

    selected = np.zeros(len(work), dtype=bool)
    if not close.any():
        return selected

    cands_close = merged.loc[close, group_keys + ["cand_hit_id"]].copy()
    cands_close["_row"] = np.flatnonzero(close)
    cands_close["_rank"] = rank[close]
    # Closest measurement wins; cand_hit_id breaks exact ties so the result is
    # deterministic and at most one candidate per state is ever flagged.
    winners = (
        cands_close.sort_values(group_keys + ["_rank", "cand_hit_id"])
        .groupby(group_keys, as_index=False)
        .first()
    )
    selected[winners["_row"].to_numpy()] = True
    return selected


def patch_event(
    parquet_path: str,
    root_path: str,
    event_id: int,
    out_path: str,
    min_frac_matched: float = 0.95,
) -> dict:
    """Add ``is_ckf_selected`` to one event's Parquet and write it out.

    Parameters
    ----------
    min_frac_matched : float
        Refuse to write if fewer than this fraction of the ROOT file's states
        matched a candidate. The check lives *here*, before the write, not in
        :func:`main` -- it used to sit in ``main`` only, which meant every
        caller that imports ``patch_event`` directly (``modal_train.
        patch_selected_all``, i.e. the only path that actually runs at scale)
        silently skipped it. That is not a hypothetical: the first real
        2-event run wrote ``is_ckf_selected`` all-False with
        ``frac_states_matched == 0.0`` and exited 0, because a residual-join
        failure is invisible in the output schema -- the column exists and is
        the right dtype, it is just uniformly wrong, which would train the
        value function on a target that says "the CKF never accepted any
        hit". Set to ``0.0`` to disable (diagnostics only).

    Returns
    -------
    dict
        ``event_id``, ``n_rows``, ``n_selected``, ``n_states``,
        ``frac_states_matched`` — the last is the key health metric and should
        be close to 1.0 for states that have a selected hit.

    Raises
    ------
    ValueError
        If ``frac_states_matched < min_frac_matched``. Raised *before* the
        Parquet is written, so a failed run leaves no output to mistake for
        a good one.
    """
    assert_not_test([event_id])

    df = pq.read_table(
        parquet_path,
        columns=[
            "seed_id",
            "step_k",
            "volume_id",
            "layer_id",
            "surface_id",
            "cand_hit_id",
            "residual_l0",
            "residual_l1",
            # pred_* reconstructs the measurement position
            # (local = residual + prediction), which is what match_selected's
            # exact route compares against ROOT's l_x_hit/l_y_hit.
            "pred_l0",
            "pred_l1",
        ],
    ).to_pandas()
    df["geometry_id"] = encode_geometry_id(
        df["volume_id"], df["layer_id"], df["surface_id"]
    )
    root_res = load_root_residuals(root_path, event_id)
    selected = match_selected(df, root_res)

    n_states = len(root_res)
    n_selected = int(selected.sum())

    # Denominator. Expansion does not emit a row for every ROOT state: a state
    # is absent when it has no predicted parameters or no candidate inside the
    # n-sigma box. On event 1 that is 2.80M of 4.62M states, so n_selected /
    # n_states caps at 60.5% and a 0.95 floor on it can never pass however
    # correct the join is. The health metric is therefore conditional on the
    # states the Parquet actually contains; raw coverage is reported alongside
    # so a collapse in expansion output is still visible.
    if "state_idx" in root_res.columns and "step_k" in df.columns:
        pq_states = set(map(tuple, df[["seed_id", "step_k"]].drop_duplicates().to_numpy()))
        root_states = set(map(tuple, root_res[["seed_id", "state_idx"]].to_numpy()))
    else:
        pq_states = set(map(tuple, df[["seed_id", "geometry_id"]].drop_duplicates().to_numpy()))
        root_states = set(map(tuple, root_res[["seed_id", "geometry_id"]].to_numpy()))
    n_joinable = len(pq_states & root_states)

    frac = (n_selected / n_states) if n_states else 0.0
    frac_joinable = (n_selected / n_joinable) if n_joinable else 0.0
    report = {
        "event_id": event_id,
        "n_rows": len(df),
        "n_selected": n_selected,
        "n_states": n_states,
        "n_joinable_states": n_joinable,
        "frac_states_matched": frac,
        "frac_joinable_matched": frac_joinable,
    }
    if frac_joinable < min_frac_matched:
        raise ValueError(
            f"event {event_id}: only {frac_joinable:.2%} of {n_joinable:,} "
            f"joinable ROOT states "
            f"matched a candidate within the residual tolerance "
            f"(need >= {min_frac_matched:.0%}); refusing to write "
            f"{out_path}. is_ckf_selected would be almost entirely False, "
            "which is indistinguishable from a valid column downstream. "
            f"Diagnose with modal_train.py::diagnose_join --event-id "
            f"{event_id} (checks key overlap, residual sign convention and "
            f"tolerance). Report: {report}"
        )

    table = pq.read_table(parquet_path)
    table = table.append_column("is_ckf_selected", pa.array(selected, pa.bool_()))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--event-id", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # patch_event itself enforces the match-fraction floor before writing, so
    # this wrapper only has to translate the failure into an exit status.
    try:
        report = patch_event(args.parquet, args.root, args.event_id, args.out)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(report)


if __name__ == "__main__":
    main()
