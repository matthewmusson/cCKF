"""Diagnostic: diagonal χ² approximation vs full innovation covariance.

Requires slim rows with S00, S01, S11 (from PredictedCovWriter + expander).
Computes on each row:
  rho = S01 / sqrt(S00 S11)
  chi2_true = rᵀ S⁻¹ r   (already stored as chi2_inc when S is full)
  chi2_diag = r0²/S00 + r1²/S11

For (b)/(c) we need residuals. If residual columns are absent we reconstruct
chi2_diag from S and the identity
  chi2_true = (chi2_diag - 2 rho a b) / (1 - rho²)
which is underdetermined for the difference alone — instead we recompute both
from S and stored chi2_inc by using the closed form:

  det = S00 S11 - S01²
  chi2_true = chi2_inc  (by construction in the expander)
  chi2_diag cannot be recovered from chi2_inc alone without (r0, r1).

Therefore this script expects optional columns r0, r1; if missing, it
re-derives chi2_diag only when |rho| is small enough that the difference is
bounded, and reports rho + the confound check using
  delta_proxy = chi2_inc * rho² / (1 - rho²)
which equals chi2_true - chi2_diag when the residual is aligned with the
leading eigenvector of the correlation (upper bound on |Δχ²| scale).

Preferred path: rows written by the updated expander store S00/S01/S11 and
chi2_inc = chi2_true; we also accept a companion parquet with r0/r1 if present.
For the short diagnostic rollout, the expander is patched to also emit
chi2_diag and residual components.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None


def load_df(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.suffix == ".parquet":
            frames.append(
                pq.read_table(p).to_pandas() if pq else pd.read_parquet(p)
            )
        else:
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    need = {"S00", "S01", "S11", "chi2_inc", "geometric_density"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(
            f"Rows lack {missing}. Events 0–3 as collected cannot support "
            "this diagnostic — re-run with PredictedCovWriter + S columns."
        )
    return df


def eta_region(eta: np.ndarray) -> np.ndarray:
    """0=barrel |η|<1, 1=transition, 2=endcap |η|≥2, -1=unknown."""
    b = np.zeros(len(eta), dtype=np.int32)
    ae = np.abs(eta)
    b[ae >= 1.0] = 1
    b[ae >= 2.0] = 2
    b[~np.isfinite(eta)] = -1
    return b


def barrel_endcap(eta: np.ndarray) -> np.ndarray:
    """0=barrel |η|<1.5, 1=endcap |η|≥1.5, -1=unknown."""
    b = np.full(len(eta), -1, dtype=np.int32)
    ae = np.abs(eta)
    ok = np.isfinite(eta)
    b[ok & (ae < 1.5)] = 0
    b[ok & (ae >= 1.5)] = 1
    return b


def summarize_rho(rho: np.ndarray) -> dict[str, float]:
    rho = rho[np.isfinite(rho)]
    if len(rho) == 0:
        return {"n": 0, "median": float("nan"), "p05": float("nan"), "p95": float("nan")}
    return {
        "n": int(len(rho)),
        "median": float(np.median(rho)),
        "p05": float(np.percentile(rho, 5)),
        "p95": float(np.percentile(rho, 95)),
        "mean": float(np.mean(rho)),
        "std": float(np.std(rho)),
    }


def run(df: pd.DataFrame) -> dict[str, Any]:
    s00 = df["S00"].to_numpy(dtype=np.float64)
    s01 = df["S01"].to_numpy(dtype=np.float64)
    s11 = df["S11"].to_numpy(dtype=np.float64)
    chi2_true = df["chi2_inc"].to_numpy(dtype=np.float64)
    dens = df["geometric_density"].to_numpy(dtype=np.float64)

    denom = np.sqrt(np.maximum(s00 * s11, 0.0))
    rho = np.divide(s01, denom, out=np.full_like(s01, np.nan), where=denom > 0)
    rho = np.clip(rho, -1.0, 1.0)

    # Prefer explicit chi2_diag / residuals if present
    if "chi2_diag" in df.columns:
        chi2_diag = df["chi2_diag"].to_numpy(dtype=np.float64)
    elif {"r0", "r1"}.issubset(df.columns):
        r0 = df["r0"].to_numpy(dtype=np.float64)
        r1 = df["r1"].to_numpy(dtype=np.float64)
        chi2_diag = (r0 * r0) / np.maximum(s00, 1e-30) + (r1 * r1) / np.maximum(
            s11, 1e-30
        )
    else:
        # Without residuals we cannot form chi2_diag exactly. Report rho-only
        # for (a) and mark (b)/(c) as requiring chi2_diag column.
        chi2_diag = np.full_like(chi2_true, np.nan)

    delta = chi2_true - chi2_diag
    ratio = np.divide(
        chi2_true,
        chi2_diag,
        out=np.full_like(chi2_true, np.nan),
        where=np.isfinite(chi2_diag) & (np.abs(chi2_diag) > 1e-12),
    )

    report: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_events": int(df["event_id"].nunique()) if "event_id" in df.columns else None,
        "rho_global": summarize_rho(rho),
        "rho_by_barrel_endcap": {},
        "rho_by_layer": {},
        "rho_by_eta_bin": {},
        "chi2_delta": {},
        "chi2_ratio": {},
        "delta_by_geom_quintile": {},
        "verdict": {},
    }

    if "eta" in df.columns:
        be = barrel_endcap(df["eta"].to_numpy())
        for code, name in [(0, "barrel"), (1, "endcap")]:
            report["rho_by_barrel_endcap"][name] = summarize_rho(rho[be == code])
        eb = eta_region(df["eta"].to_numpy())
        for code, name in [(0, "eta_lt1"), (1, "eta_1_to_2"), (2, "eta_gt2")]:
            report["rho_by_eta_bin"][name] = summarize_rho(rho[eb == code])

    if "layer_id" in df.columns:
        for layer, gidx in df.groupby("layer_id").groups.items():
            idx = np.asarray(list(gidx))
            if len(idx) < 50:
                continue
            report["rho_by_layer"][str(int(layer))] = summarize_rho(rho[idx])

    if np.isfinite(delta).any():
        d = delta[np.isfinite(delta)]
        r = ratio[np.isfinite(ratio)]
        report["chi2_delta"] = {
            "median": float(np.median(d)),
            "p05": float(np.percentile(d, 5)),
            "p95": float(np.percentile(d, 95)),
            "mean": float(np.mean(d)),
            "std": float(np.std(d)),
        }
        report["chi2_ratio"] = {
            "median": float(np.median(r)),
            "p05": float(np.percentile(r, 5)),
            "p95": float(np.percentile(r, 95)),
            "mean": float(np.mean(r)),
        }

        # (c) mean Δχ² within geometric-density quintiles
        edges = np.quantile(dens[np.isfinite(dens)], np.linspace(0, 1, 6))
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9
        q = np.digitize(dens, edges[1:-1], right=False)
        q = np.clip(q, 0, 4)
        for j in range(5):
            m = (q == j) & np.isfinite(delta)
            report["delta_by_geom_quintile"][f"Q{j+1}"] = {
                "geom_lo": float(edges[j]),
                "geom_hi": float(edges[j + 1]),
                "n": int(m.sum()),
                "mean_delta": float(np.mean(delta[m])) if m.any() else float("nan"),
                "median_delta": float(np.median(delta[m])) if m.any() else float("nan"),
            }

        means = [
            report["delta_by_geom_quintile"][f"Q{j+1}"]["mean_delta"] for j in range(5)
        ]
        means_f = [m for m in means if np.isfinite(m)]
        spread = float(max(means_f) - min(means_f)) if means_f else float("nan")
        # Decisive: if mean Δ varies across quintiles by ≳ 0.1 in χ² units,
        # diagonal approx can fake an occupancy separation.
        report["verdict"] = {
            "mean_delta_spread_across_quintiles": spread,
            "rho_median_abs": float(np.median(np.abs(rho[np.isfinite(rho)]))),
            "diagonal_safe_for_main_analysis": bool(
                np.isfinite(spread) and spread < 0.1
                and float(np.median(np.abs(rho[np.isfinite(rho)]))) < 0.2
            ),
            "criterion": (
                "safe if |median rho|<0.2 AND spread of mean(chi2_true-chi2_diag) "
                "across geom quintiles < 0.1"
            ),
        }
    else:
        report["verdict"] = {
            "diagonal_safe_for_main_analysis": False,
            "reason": "chi2_diag/residuals missing — cannot complete confound check",
        }

    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    df = load_df(args.inputs)
    # Keep defined-majority rows for consistency with main analysis
    if "majority_undefined" in df.columns:
        df = df[df["majority_undefined"].astype(int) == 0].copy()
    report = run(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
