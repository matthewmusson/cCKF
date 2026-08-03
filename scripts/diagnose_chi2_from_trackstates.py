"""Diagonal-vs-full χ² diagnostic from existing RootTrackStatesWriter dumps.

Events 0–3 lack S01 in the slim Parquet. This script uses the ACTS-stored
per-state ``chi2`` (MeasurementSelector full-S value for the *selected* hit)
together with ``pull_x_hit`` / ``pull_y_hit`` (which are r_i/√S_ii using the
diagonal of the full innovation covariance) to recover:

  chi2_true = state.chi2
  chi2_diag = pull_x² + pull_y²
  rho from the quadratic relating chi2_true, chi2_diag, and pulls

geometric_density is recomputed from measurements CSV (fixed R=5 mm circle).

Scope note: this is only for the selected hit on each CKF-visited state, not
every window candidate. That is enough for the confound check (does Δχ²
vary with local occupancy?).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import uproot
except ImportError as e:  # pragma: no cover
    raise SystemExit("uproot required") from e

R_GEOM_MM = 5.0
_VOL_MASK = np.uint64(0xFF00000000000000)
_LAY_MASK = np.uint64(0x0000FFF000000000)
_SEN_MASK = np.uint64(0x000000000FFFFF00)


def decode_geo(gid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = gid.astype(np.uint64)
    return (
        ((g & _VOL_MASK) >> np.uint64(56)).astype(np.int32),
        ((g & _LAY_MASK) >> np.uint64(36)).astype(np.int32),
        ((g & _SEN_MASK) >> np.uint64(8)).astype(np.int32),
    )


def load_measurements(path: Path) -> dict[tuple[int, int, int], pd.DataFrame]:
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    vol, lay, sen = decode_geo(df["geometry_id"].to_numpy())
    df["volume_id"], df["layer_id"], df["module_id"] = vol, lay, sen
    return {k: g for k, g in df.groupby(["volume_id", "layer_id", "module_id"])}


def rho_from_pulls(chi2_true: float, a: float, b: float) -> float:
    """Solve chi2_true = (a²+b² - 2ρab)/(1-ρ²) for ρ ∈ [-1,1]."""
    chi2_diag = a * a + b * b
    if not np.isfinite(chi2_true) or not np.isfinite(chi2_diag):
        return float("nan")
    if abs(chi2_true - chi2_diag) < 1e-12:
        return 0.0
    # chi2_true ρ² - 2ab ρ + (chi2_diag - chi2_true) = 0
    A = chi2_true
    B = -2.0 * a * b
    C = chi2_diag - chi2_true
    if abs(A) < 1e-30:
        return float("nan")
    disc = B * B - 4 * A * C
    if disc < 0:
        disc = 0.0
    sqrt_d = np.sqrt(disc)
    cands = [( -B + sqrt_d) / (2 * A), (-B - sqrt_d) / (2 * A)]
    valid = [r for r in cands if -1.0 <= r <= 1.0]
    if not valid:
        return float(np.clip(cands[0], -1, 1))
    # Prefer the root whose sign matches ab (positive corr → residuals same sign)
    if a * b != 0:
        prefer = [r for r in valid if r * a * b >= 0]
        if prefer:
            return float(prefer[0])
    return float(valid[0])


def summarize(x: np.ndarray) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "median": float("nan"), "p05": float("nan"), "p95": float("nan")}
    return {
        "n": int(len(x)),
        "median": float(np.median(x)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def collect_event(
    event_id: int,
    trackstates: Path,
    measurements: Path,
    max_tracks: int = 50_000,
    sample_seed: int = 42,
) -> pd.DataFrame:
    by_surf = load_measurements(measurements)
    rows: list[dict[str, Any]] = []
    with uproot.open(trackstates) as f:
        keys = list(f.keys())
        tree_key = next(
            (k for k in keys if "trackstate" in k.lower()), keys[0]
        )
        tree = f[tree_key]
        n_tracks = int(tree.num_entries)
        rng = np.random.default_rng(sample_seed + event_id)
        if max_tracks and n_tracks > max_tracks:
            start = int(rng.integers(0, n_tracks - max_tracks + 1))
            entry_start, entry_stop = start, start + max_tracks
        else:
            entry_start, entry_stop = 0, n_tracks

        fields = [
            "volume_id",
            "layer_id",
            "module_id",
            "chi2",
            "predicted",
            "eLOC0_prt",
            "eLOC1_prt",
            "pull_x_hit",
            "pull_y_hit",
            "eta_prt",
        ]
        batch = tree.arrays(
            fields, entry_start=entry_start, entry_stop=entry_stop, library="np"
        )
        for local_i in range(entry_stop - entry_start):
            vols = list(batch["volume_id"][local_i])
            lays = list(batch["layer_id"][local_i])
            mods = list(batch["module_id"][local_i])
            chi2s = list(batch["chi2"][local_i])
            has_prt = list(batch["predicted"][local_i])
            e0 = list(batch["eLOC0_prt"][local_i])
            e1 = list(batch["eLOC1_prt"][local_i])
            px = list(batch["pull_x_hit"][local_i])
            py = list(batch["pull_y_hit"][local_i])
            etas = list(batch["eta_prt"][local_i])
            n = len(vols)
            for step in range(n):
                if step >= len(has_prt) or not bool(has_prt[step]):
                    continue
                if step >= len(chi2s) or not np.isfinite(float(chi2s[step])):
                    continue
                if step >= len(px) or step >= len(py):
                    continue
                a = float(px[step])
                b = float(py[step])
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue
                chi2_true = float(chi2s[step])
                chi2_diag = a * a + b * b
                rho = rho_from_pulls(chi2_true, a, b)
                key = (int(vols[step]), int(lays[step]), int(mods[step]))
                surf = by_surf.get(key)
                n_geom = 0
                if surf is not None and step < len(e0) and np.isfinite(float(e0[step])):
                    r0 = surf["local0"].to_numpy() - float(e0[step])
                    r1 = surf["local1"].to_numpy() - float(e1[step])
                    n_geom = int((np.hypot(r0, r1) <= R_GEOM_MM).sum())
                eta = (
                    float(etas[step])
                    if step < len(etas) and np.isfinite(float(etas[step]))
                    else float("nan")
                )
                rows.append(
                    {
                        "event_id": event_id,
                        "layer_id": int(lays[step]),
                        "eta": eta,
                        "chi2_true": chi2_true,
                        "chi2_diag": chi2_diag,
                        "delta": chi2_true - chi2_diag,
                        "ratio": chi2_true / chi2_diag if abs(chi2_diag) > 1e-12 else float("nan"),
                        "rho": rho,
                        "geometric_density": n_geom,
                    }
                )
    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame) -> dict[str, Any]:
    rho = df["rho"].to_numpy()
    delta = df["delta"].to_numpy()
    dens = df["geometric_density"].to_numpy(dtype=np.float64)
    eta = df["eta"].to_numpy()

    report: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_events": int(df["event_id"].nunique()),
        "scope": "selected hit per CKF state (ACTS state.chi2), not full window",
        "geometric_density_radius_mm": R_GEOM_MM,
        "rho_global": summarize(rho),
        "rho_by_barrel_endcap": {},
        "rho_by_layer": {},
        "rho_by_eta_bin": {},
        "chi2_delta": summarize(delta),
        "chi2_ratio": summarize(df["ratio"].to_numpy()),
        "delta_by_geom_quintile": {},
    }

    ae = np.abs(eta)
    barrel = np.isfinite(eta) & (ae < 1.5)
    endcap = np.isfinite(eta) & (ae >= 1.5)
    report["rho_by_barrel_endcap"]["barrel"] = summarize(rho[barrel])
    report["rho_by_barrel_endcap"]["endcap"] = summarize(rho[endcap])

    eb0 = np.isfinite(eta) & (ae < 1.0)
    eb1 = np.isfinite(eta) & (ae >= 1.0) & (ae < 2.0)
    eb2 = np.isfinite(eta) & (ae >= 2.0)
    report["rho_by_eta_bin"]["eta_lt1"] = summarize(rho[eb0])
    report["rho_by_eta_bin"]["eta_1_to_2"] = summarize(rho[eb1])
    report["rho_by_eta_bin"]["eta_gt2"] = summarize(rho[eb2])

    for layer, g in df.groupby("layer_id"):
        if len(g) < 50:
            continue
        report["rho_by_layer"][str(int(layer))] = summarize(g["rho"].to_numpy())

    edges = np.quantile(dens[np.isfinite(dens)], np.linspace(0, 1, 6))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    q = np.clip(np.digitize(dens, edges[1:-1], right=False), 0, 4)
    means = []
    for j in range(5):
        m = q == j
        md = float(np.mean(delta[m])) if m.any() else float("nan")
        means.append(md)
        report["delta_by_geom_quintile"][f"Q{j+1}"] = {
            "geom_lo": float(edges[j]),
            "geom_hi": float(edges[j + 1]),
            "n": int(m.sum()),
            "mean_delta": md,
            "median_delta": float(np.median(delta[m])) if m.any() else float("nan"),
        }

    means_f = [m for m in means if np.isfinite(m)]
    spread = float(max(means_f) - min(means_f)) if means_f else float("nan")
    rho_med_abs = float(np.median(np.abs(rho[np.isfinite(rho)])))
    # Spearman via rank correlation (no scipy dependency)
    ok = np.isfinite(dens) & np.isfinite(delta)
    if ok.sum() > 10:
        rd = dens[ok].argsort().argsort().astype(float)
        rr = delta[ok].argsort().argsort().astype(float)
        corr = float(np.corrcoef(rd, rr)[0, 1])
    else:
        corr = float("nan")
    corr_d = {"spearman_delta_vs_geom": corr}

    safe = bool(
        np.isfinite(spread)
        and spread < 0.1
        and rho_med_abs < 0.2
        and (not np.isfinite(corr_d["spearman_delta_vs_geom"]) or abs(corr_d["spearman_delta_vs_geom"]) < 0.05)
    )
    report["verdict"] = {
        "mean_delta_spread_across_geom_quintiles": spread,
        "rho_median_abs": rho_med_abs,
        **corr_d,
        "diagonal_safe_for_main_analysis_on_existing_rows": safe,
        "main_analysis_on_events_0_3": (
            "PROCEED with caution"
            if safe
            else "RE-RUN REQUIRED with full S00/S01/S11 logged"
        ),
        "criterion": (
            "safe if |median rho|<0.2 AND quintile mean-Δ spread<0.1 AND "
            "|spearman(delta, geom_density)|<0.05"
        ),
        "note_log_det": (
            "log det(2πS) = log(2πS00)+log(2πS11)+log(1-ρ²); "
            "diagonal drops log(1-ρ²)"
        ),
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--event-dir",
        type=Path,
        action="append",
        required=True,
        help="Directory containing trackstates_ckf.root and measurements CSV",
    )
    p.add_argument("--event-id", type=int, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tracks", type=int, default=50_000)
    args = p.parse_args()
    if len(args.event_dir) != len(args.event_id):
        raise SystemExit("Need matching --event-dir and --event-id counts")

    frames = []
    for eid, ed in zip(args.event_id, args.event_dir):
        ts = ed / "trackstates_ckf.root"
        meas = next(ed.glob("event*-measurements.csv"))
        print(f"event {eid}: {ts}", flush=True)
        frames.append(
            collect_event(eid, ts, meas, max_tracks=args.max_tracks)
        )
    df = pd.concat(frames, ignore_index=True)
    report = analyze(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
