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
from expansion import encode_particle_id

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
    and ``ak.local_index(..., axis=1)`` peel off the per-state layer to
    build ``seed_id``/``step_k``, then a single ``axis=1`` flatten drops the
    per-track nesting (leaving one, possibly empty, jagged contributor list
    per state) before a full flatten and re-grouping by per-state counts
    recovers the contributor lists.
    """
    import awkward as ak

    if "event_nr" in arrays.fields:
        event_mask = ak.to_numpy(arrays["event_nr"]) == int(event_id)
        arrays = arrays[event_mask]

    empty = pd.DataFrame(
        columns=["seed_id", "step_k", "sel_contrib_pids", "sel_has_hit"]
    )
    n_tracks = len(arrays)
    if n_tracks == 0:
        return empty

    # particle_ids_particle is doubly-jagged (track, state, contributor):
    # axis=1 num/local_index operate on the per-state layer directly.
    n_states = ak.to_numpy(ak.num(arrays["particle_ids_particle"], axis=1)).astype(
        np.int64
    )
    if n_states.sum() == 0:
        return empty
    track_nr = np.repeat(np.arange(n_tracks, dtype=np.int64), n_states)
    state_idx = ak.to_numpy(
        ak.flatten(ak.local_index(arrays["particle_ids_particle"], axis=1))
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
            "step_k": state_idx,
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
        ``seed_id``, ``step_k``, ``sel_contrib_pids`` (list of int64 barcodes),
        ``sel_has_hit`` (False for holes, where the contributor list is empty).
    """
    import uproot

    with uproot.open(root_path) as fh:
        tree = fh["trackstates"]
        available = set(tree.keys())
        fields = list(_PARTICLE_ID_FIELDS)
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
        ``seed_id``, ``step_k``, ``sel_correct``, ``sel_wrong`` in {0, 1}, with
        at most one set per state. A hole is neither: it costs completeness but
        does not pollute purity.
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
    return out[["seed_id", "step_k", "sel_correct", "sel_wrong"]]


#: Per-state residual branches. Same ``_prt`` convention as ``eLOC0_prt`` etc.
#: in expansion.py's ``_TRACKSTATE_SCALAR_BRANCHES``: singly-jagged
#: (track, state), not one entry per state.
_RESIDUAL_FIELDS = ("res_eLOC0_prt", "res_eLOC1_prt")


def _root_residuals_from_arrays(arrays, event_id: int) -> pd.DataFrame:
    """Pure parsing step of :func:`load_root_residuals`.

    Takes an already-loaded awkward Array (one entry per track) so it can be
    exercised directly against synthetic fixtures without a ROOT file.
    ``res_eLOC0_prt``/``res_eLOC1_prt`` are singly-jagged, per-state scalar
    branches -- the tree is one entry per track (expansion.py:418-429), not
    one entry per state as originally assumed; ``ak.num``/``ak.local_index``
    at ``axis=1`` peel off the per-state layer to build ``seed_id``/``step_k``
    exactly as expansion.py's ``load_trackstates`` does for its own
    ``_prt``-suffixed scalar branches.
    """
    import awkward as ak

    if "event_nr" in arrays.fields:
        event_mask = ak.to_numpy(arrays["event_nr"]) == int(event_id)
        arrays = arrays[event_mask]

    empty = pd.DataFrame(columns=["seed_id", "step_k", "res_l0", "res_l1"])
    n_tracks = len(arrays)
    if n_tracks == 0:
        return empty

    n_states = ak.to_numpy(ak.num(arrays["res_eLOC0_prt"], axis=1)).astype(np.int64)
    if n_states.sum() == 0:
        return empty
    track_nr = np.repeat(np.arange(n_tracks, dtype=np.int64), n_states)
    state_idx = ak.to_numpy(
        ak.flatten(ak.local_index(arrays["res_eLOC0_prt"], axis=1))
    ).astype(np.int64)

    res0 = ak.to_numpy(ak.flatten(arrays["res_eLOC0_prt"], axis=1)).astype(np.float64)
    res1 = ak.to_numpy(ak.flatten(arrays["res_eLOC1_prt"], axis=1)).astype(np.float64)

    df = pd.DataFrame(
        {
            "seed_id": track_nr,
            "step_k": state_idx,
            "res_l0": res0,
            "res_l1": res1,
        }
    )
    return df.loc[np.isfinite(df["res_l0"]) & np.isfinite(df["res_l1"])].reset_index(
        drop=True
    )


def load_root_residuals(root_path: str, event_id: int) -> pd.DataFrame:
    """Read per-state selected-hit residuals from the trackstates ROOT tree.

    Returns
    -------
    pandas.DataFrame
        Columns ``seed_id``, ``step_k``, ``res_l0``, ``res_l1``; one row per
        state that has a selected measurement (rows where either residual is
        NaN, i.e. holes, are dropped).
    """
    import uproot

    with uproot.open(root_path) as fh:
        tree = fh["trackstates"]
        available = set(tree.keys())
        fields = list(_RESIDUAL_FIELDS)
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
    """Flag the CKF-selected candidate within each ``(seed_id, step_k)`` state.

    Parameters
    ----------
    cand : pandas.DataFrame
        Candidate rows with ``seed_id``, ``step_k``, ``cand_hit_id``,
        ``residual_l0``, ``residual_l1``.
    root_res : pandas.DataFrame
        Per-state selected residuals with ``seed_id``, ``step_k``, ``res_l0``,
        ``res_l1``.
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
    merged = work.merge(root_res, on=["seed_id", "step_k"], how="left")

    close = (
        (merged["residual_l0"] - merged["res_l0"]).abs().le(tol)
        & (merged["residual_l1"] - merged["res_l1"]).abs().le(tol)
    ).to_numpy(dtype=bool)
    close = close & np.isfinite(merged["res_l0"].to_numpy(dtype=np.float64))

    selected = np.zeros(len(work), dtype=bool)
    if not close.any():
        return selected

    cands_close = merged.loc[close, ["seed_id", "step_k", "cand_hit_id"]].copy()
    cands_close["_row"] = np.flatnonzero(close)
    winners = (
        cands_close.sort_values(["seed_id", "step_k", "cand_hit_id"])
        .groupby(["seed_id", "step_k"], as_index=False)
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
        columns=["seed_id", "step_k", "cand_hit_id", "residual_l0", "residual_l1"],
    ).to_pandas()
    root_res = load_root_residuals(root_path, event_id)
    selected = match_selected(df, root_res)

    n_states = len(root_res)
    n_selected = int(selected.sum())
    frac = (n_selected / n_states) if n_states else 0.0
    report = {
        "event_id": event_id,
        "n_rows": len(df),
        "n_selected": n_selected,
        "n_states": n_states,
        "frac_states_matched": frac,
    }
    if frac < min_frac_matched:
        raise ValueError(
            f"event {event_id}: only {frac:.2%} of {n_states:,} ROOT states "
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
