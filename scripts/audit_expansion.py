"""Audit one expanded Parquet against its ROOT track states. Fails loudly.

Run this on ONE event before launching a re-expansion wave, and again on a
few events after. Every check prints PASS/FAIL with the number behind it;
any FAIL exits 1. Nothing here modifies data.

Why these checks (each one caught, or would have caught, a real defect):

  root_order          ROOT stores states outermost-first; step_k must be
                      propagation order (LOG 2026-09-08: inverted for a month).
  parquet_vs_root     the parquet's (seed_id, step_k) -> (volume, layer) must
                      equal ROOT's propagation-order sequence.
  volumes             candidates in all nine sensitive ODD volumes and the
                      passive volume 20 present (LOG 2026-08-25: barrel-only
                      parquets from the geometry_id extra byte).
  hole_fraction       states with no in-window candidate, vs a reference
                      expansion (LOG 2026-08-25 validation gate).
  candidate_shares    pixel / short-strip / long-strip share of candidate
                      rows, vs reference (LOG 2026-08-24: long strips missing).
  history             n_hits + n_holes equals the state's position within the
                      branch, n_hits non-decreasing (counts states BEFORE k).
  majority_label      branch_majority_pid re-derived from the three INNERMOST
                      selected hits agrees (the label was taken from the
                      outermost three before the ordering fix).
  selected_flag       at most one is_ckf_selected row per state; joinable
                      fraction reported (LOG 2026-08-17/25 join failures).
  vstar_range         vstar_soft within [0, 1] where finite.

Usage (NERSC, one event, against the previous expansion as reference):

    python scripts/audit_expansion.py \\
        --parquet $SCRATCH/cckf/reexpanded_v2/expanded_event000000004.parquet \\
        --trackstates $SCRATCH/cckf/modal_backup/results/pilot_1786525888/trackstates_ckf.root \\
        --event 4 \\
        --reference $SCRATCH/cckf/reexpanded/expanded_event000000004.parquet \\
        --json audit_event4.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expansion import check_propagation_order, propagation_order_index  # noqa: E402

PIXEL_VOLUMES = (16, 17, 18)
SSTRIP_VOLUMES = (23, 24, 25)
LSTRIP_VOLUMES = (28, 29, 30)
SENSOR_VOLUMES = PIXEL_VOLUMES + SSTRIP_VOLUMES + LSTRIP_VOLUMES
PASSIVE_VOLUMES = (20,)

PARQUET_COLUMNS = [
    "seed_id", "step_k", "volume_id", "layer_id", "cand_hit_id",
    "is_ckf_selected", "contrib_pids", "branch_majority_pid",
    "majority_undefined", "n_hits", "n_holes", "vstar_soft",
]


@dataclass
class Check:
    name: str
    passed: bool
    value: float | None
    detail: str

    def line(self) -> str:
        v = "" if self.value is None else f" value={self.value:.4g}"
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}{v}  {self.detail}"


# ---------------------------------------------------------------- loaders --

def load_parquet_states(path: str) -> pd.DataFrame:
    cols = [c for c in PARQUET_COLUMNS if c in pq.read_schema(path).names]
    return pq.read_table(path, columns=cols).to_pandas()


def load_root_sequence(trackstates: str, event_id: int) -> pd.DataFrame:
    """(track_nr, step_k, volume_id, layer_id, pathLength) in propagation order."""
    import awkward as ak
    import uproot

    with uproot.open(trackstates) as fh:
        tree = fh["trackstates"]
        fields = ["volume_id", "layer_id"]
        has_pl = "pathLength" in set(tree.keys())
        if has_pl:
            fields.append("pathLength")
        has_ev = "event_nr" in set(tree.keys())
        arrays = tree.arrays(fields + (["event_nr"] if has_ev else []), library="ak")
    if has_ev:
        arrays = arrays[ak.to_numpy(arrays["event_nr"]) == int(event_id)]
    n = ak.to_numpy(ak.num(arrays["volume_id"], axis=1))
    out = pd.DataFrame({
        "track_nr": np.repeat(np.arange(len(arrays), dtype=np.int64), n),
        "step_k": propagation_order_index(arrays["volume_id"]),
        "volume_id": ak.to_numpy(ak.flatten(arrays["volume_id"], axis=1)).astype(np.int64),
        "layer_id": ak.to_numpy(ak.flatten(arrays["layer_id"], axis=1)).astype(np.int64),
    })
    out["pathLength"] = (
        ak.to_numpy(ak.flatten(arrays["pathLength"], axis=1)).astype(np.float64)
        if has_pl else np.nan
    )
    return out


# ----------------------------------------------------------------- checks --

def check_root_order(root: pd.DataFrame) -> Check:
    if root["pathLength"].isna().all():
        return Check("root_order", False, None,
                     "no pathLength branch: cannot verify propagation order")
    try:
        check_propagation_order(root["track_nr"].to_numpy(),
                                root["step_k"].to_numpy(),
                                root["pathLength"].to_numpy())
    except ValueError as exc:
        return Check("root_order", False, None, str(exc))
    seed = root[root.step_k == 0]
    return Check("root_order", True, float(seed["pathLength"].abs().max()),
                 f"pathLength non-decreasing along step_k on {root.track_nr.nunique():,} "
                 f"tracks; max |pathLength| at step 0 = {seed['pathLength'].abs().max():.3g} mm")


def check_parquet_vs_root(states: pd.DataFrame, root: pd.DataFrame) -> Check:
    """Every parquet state must sit on the ROOT surface at the same
    propagation index. Compared on the intersection (the expansion emits no
    row for a state with no prediction)."""
    pq_states = states.drop_duplicates(["seed_id", "step_k"])[
        ["seed_id", "step_k", "volume_id", "layer_id"]]
    j = pq_states.merge(
        root[["track_nr", "step_k", "volume_id", "layer_id"]].rename(
            columns={"track_nr": "seed_id"}),
        on=["seed_id", "step_k"], how="left", suffixes=("", "_root"))
    unjoined = int(j["volume_id_root"].isna().sum())
    joined = j[j["volume_id_root"].notna()]
    mismatch = int(((joined["volume_id"] != joined["volume_id_root"]) |
                    (joined["layer_id"] != joined["layer_id_root"])).sum())
    frac = mismatch / max(len(joined), 1)
    return Check("parquet_vs_root", mismatch == 0 and unjoined == 0, frac,
                 f"{len(joined):,} parquet states joined to ROOT, {mismatch:,} on a different "
                 f"(volume, layer), {unjoined:,} with no ROOT state at that step_k")


def check_volumes(states: pd.DataFrame) -> Check:
    cand_vols = set(states.loc[states.cand_hit_id >= 0, "volume_id"].unique().tolist())
    all_vols = set(states["volume_id"].unique().tolist())
    missing = sorted(set(SENSOR_VOLUMES) - cand_vols)
    unexpected = sorted(all_vols - set(SENSOR_VOLUMES) - set(PASSIVE_VOLUMES))
    passive_missing = sorted(set(PASSIVE_VOLUMES) - all_vols)
    ok = not missing and not unexpected and not passive_missing
    return Check("volumes", ok, float(len(cand_vols)),
                 f"candidate volumes {sorted(cand_vols)}; missing sensitive {missing}; "
                 f"unexpected {unexpected}; passive missing {passive_missing}")


def hole_fraction(states: pd.DataFrame) -> float:
    per_state = states.groupby(["seed_id", "step_k"])["cand_hit_id"].max()
    return float((per_state < 0).mean())


def check_hole_fraction(states: pd.DataFrame, reference: pd.DataFrame | None,
                        tol: float) -> Check:
    f = hole_fraction(states)
    if reference is None:
        return Check("hole_fraction", True, f, "no reference given; reported only")
    r = hole_fraction(reference)
    return Check("hole_fraction", abs(f - r) <= tol, f,
                 f"states with no candidate: {f:.4f} vs reference {r:.4f} (tol {tol})")


def candidate_shares(states: pd.DataFrame) -> dict[str, float]:
    cand = states.loc[states.cand_hit_id >= 0, "volume_id"]
    n = max(len(cand), 1)
    return {
        "pixel": float(cand.isin(PIXEL_VOLUMES).sum() / n),
        "sstrip": float(cand.isin(SSTRIP_VOLUMES).sum() / n),
        "lstrip": float(cand.isin(LSTRIP_VOLUMES).sum() / n),
    }


def check_candidate_shares(states: pd.DataFrame, reference: pd.DataFrame | None,
                           tol: float) -> Check:
    s = candidate_shares(states)
    if reference is None:
        return Check("candidate_shares", all(v > 0 for v in s.values()),
                     s["pixel"], f"{s}; no reference given (all three must be > 0)")
    r = candidate_shares(reference)
    worst = max(abs(s[k] - r[k]) for k in s)
    return Check("candidate_shares", worst <= tol, worst,
                 f"{ {k: round(v, 4) for k, v in s.items()} } vs reference "
                 f"{ {k: round(v, 4) for k, v in r.items()} } (tol {tol})")


def check_history(states: pd.DataFrame) -> Check:
    st = states.drop_duplicates(["seed_id", "step_k"]).sort_values(["seed_id", "step_k"])
    pos = st.groupby("seed_id").cumcount().to_numpy()
    total = (st["n_hits"] + st["n_holes"]).to_numpy()
    bad_total = int((total != pos).sum())
    d = st.groupby("seed_id")["n_hits"].diff().fillna(0).to_numpy()
    bad_mono = int((d < 0).sum())
    ok = bad_total == 0 and bad_mono == 0
    return Check("history", ok, bad_total / max(len(st), 1),
                 f"{bad_total:,} states where n_hits+n_holes != position in branch; "
                 f"{bad_mono:,} where n_hits decreases")


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def recompute_majority(states: pd.DataFrame) -> pd.DataFrame:
    """Majority particle from the three innermost CKF-selected hits.

    Mirrors expansion.compute_branch_majority_pid: per selected state the
    primary particle is the mode of its contributor list; the branch majority
    is the mode over the first three measurement states in propagation order
    and needs >= 2 agreeing.
    """
    sel = states[states["is_ckf_selected"] & (states["cand_hit_id"] >= 0)].copy()
    sel = sel[sel["contrib_pids"].map(lambda c: c is not None and len(c) > 0)]
    sel["primary"] = sel["contrib_pids"].map(lambda c: _mode(np.asarray(c, dtype=np.int64)))
    sel = sel.sort_values(["seed_id", "step_k"])
    first3 = sel.groupby("seed_id").head(3)

    def _maj(g: pd.Series) -> int:
        if len(g) < 3:
            return -1
        vals, counts = np.unique(g.to_numpy(), return_counts=True)
        return int(vals[np.argmax(counts)]) if counts.max() >= 2 else -1

    out = first3.groupby("seed_id")["primary"].apply(_maj).rename("recomputed_pid")
    return out.reset_index()


def check_majority_label(states: pd.DataFrame, min_agreement: float) -> Check:
    stored = states.drop_duplicates("seed_id")[
        ["seed_id", "branch_majority_pid", "majority_undefined"]]
    stored = stored[~stored["majority_undefined"]]
    rec = recompute_majority(states)
    j = stored.merge(rec, on="seed_id", how="left")
    j["recomputed_pid"] = j["recomputed_pid"].fillna(-1).astype(np.int64)
    agree = float((j["branch_majority_pid"] == j["recomputed_pid"]).mean()) if len(j) else 1.0
    return Check("majority_label", agree >= min_agreement, agree,
                 f"stored majority pid agrees with the three innermost selected hits on "
                 f"{agree:.4f} of {len(j):,} majority-defined branches (floor {min_agreement})")


def check_selected_flag(states: pd.DataFrame, min_joinable: float) -> Check:
    per_state = states.groupby(["seed_id", "step_k"]).agg(
        n_sel=("is_ckf_selected", "sum"), has_cand=("cand_hit_id", lambda s: (s >= 0).any()))
    multi = int((per_state["n_sel"] > 1).sum())
    with_cand = per_state[per_state["has_cand"]]
    joinable = float((with_cand["n_sel"] == 1).mean()) if len(with_cand) else 1.0
    ok = multi == 0 and joinable >= min_joinable
    return Check("selected_flag", ok, joinable,
                 f"{multi:,} states with >1 selected row; {joinable:.4f} of states with "
                 f"candidates carry exactly one selected row (floor {min_joinable})")


def check_vstar_range(states: pd.DataFrame) -> Check:
    if "vstar_soft" not in states:
        return Check("vstar_range", True, None, "no vstar_soft column; skipped")
    v = states["vstar_soft"].to_numpy(dtype=np.float64)
    finite = np.isfinite(v)
    bad = int(((v < 0) | (v > 1))[finite].sum())
    return Check("vstar_range", bad == 0, float(finite.mean()),
                 f"{bad:,} finite vstar_soft outside [0, 1]; {finite.mean():.4f} finite")


def run_all(states: pd.DataFrame, root: pd.DataFrame,
            reference: pd.DataFrame | None, tol: float,
            min_agreement: float = 0.98, min_joinable: float = 0.95) -> list[Check]:
    return [
        check_root_order(root),
        check_parquet_vs_root(states, root),
        check_volumes(states),
        check_hole_fraction(states, reference, tol),
        check_candidate_shares(states, reference, tol),
        check_history(states),
        check_majority_label(states, min_agreement),
        check_selected_flag(states, min_joinable),
        check_vstar_range(states),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--trackstates", required=True, help="trackstates_ckf.root")
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--reference", default=None,
                    help="previous expansion of the same event for hole/share tolerances")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="absolute tolerance on hole fraction and candidate shares")
    ap.add_argument("--min-majority-agreement", type=float, default=0.98)
    ap.add_argument("--min-joinable", type=float, default=0.95)
    ap.add_argument("--json", default=None, help="write the check table here")
    args = ap.parse_args(argv)

    states = load_parquet_states(args.parquet)
    root = load_root_sequence(args.trackstates, args.event)
    reference = load_parquet_states(args.reference) if args.reference else None
    print(f"parquet rows {len(states):,}, states {states.drop_duplicates(['seed_id', 'step_k']).shape[0]:,}, "
          f"branches {states.seed_id.nunique():,}; ROOT states {len(root):,}")

    checks = run_all(states, root, reference, args.tol,
                     args.min_majority_agreement, args.min_joinable)
    for c in checks:
        print(c.line())
    n_fail = sum(not c.passed for c in checks)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"parquet": args.parquet, "trackstates": args.trackstates, "event": args.event,
             "reference": args.reference, "checks": [asdict(c) for c in checks]}, indent=2))
    print(f"AUDIT {'PASS' if n_fail == 0 else 'FAIL'}: {n_fail} of {len(checks)} checks failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
