"""Model-free calibration test of the constant χ² gate (Tier 3).

Reads slim decision rows, excludes majority_undefined, keeps cluster_merged.
Occupancy quintile edges are computed ONCE globally, then applied inside every
χ² quantile bin. Bootstrap over EVENTS (1000 resamples).

Primary occupancy: geometric_density. Secondary: window_count.
Third dimension: layer_id (and η bin as a companion).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None

N_BOOT = 1000
N_CHI2_BINS = 10
N_OCC_Q = 5
RNG_SEED = 42


def load_rows(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.suffix == ".parquet":
            if pq is None:
                frames.append(pd.read_parquet(p))
            else:
                frames.append(pq.read_table(p).to_pandas())
        else:
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    # Exclude majority_undefined; keep cluster_merged
    df = df[df["majority_undefined"].astype(int) == 0].copy()
    df = df[np.isfinite(df["chi2_inc"].to_numpy())].copy()
    df = df[df["chi2_inc"] >= 0].copy()
    return df


def eta_bin(eta: np.ndarray) -> np.ndarray:
    # 3 bins: barrel-ish |η|<1, transition, forward
    b = np.zeros(len(eta), dtype=np.int32)
    ae = np.abs(eta)
    b[ae >= 1.0] = 1
    b[ae >= 2.0] = 2
    b[~np.isfinite(eta)] = -1
    return b


def fixed_edges_from_quantiles(x: np.ndarray, n: int) -> np.ndarray:
    qs = np.linspace(0, 1, n + 1)
    edges = np.quantile(x, qs)
    # Ensure strictly increasing edges for digitize
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # bins in 0..n-1; values == max edge go to last bin
    b = np.digitize(x, edges[1:-1], right=False)
    return np.clip(b, 0, len(edges) - 2)


def cell_stats(
    df: pd.DataFrame,
    chi2_edges: np.ndarray,
    occ_edges: np.ndarray,
    occ_col: str,
) -> dict[str, Any]:
    chi2_bin = assign_bins(df["chi2_inc"].to_numpy(), chi2_edges)
    occ_bin = assign_bins(df[occ_col].to_numpy(dtype=np.float64), occ_edges)
    n_c, n_o = len(chi2_edges) - 1, len(occ_edges) - 1
    n = np.zeros((n_c, n_o), dtype=np.int64)
    k = np.zeros((n_c, n_o), dtype=np.int64)
    n_events = np.zeros((n_c, n_o), dtype=np.int64)
    labels = df["label"].to_numpy(dtype=np.int64)
    events = df["event_id"].to_numpy(dtype=np.int64)
    for i in range(n_c):
        for j in range(n_o):
            m = (chi2_bin == i) & (occ_bin == j)
            n[i, j] = int(m.sum())
            k[i, j] = int(labels[m].sum())
            n_events[i, j] = int(len(np.unique(events[m]))) if n[i, j] else 0
    frac = np.full_like(n, np.nan, dtype=np.float64)
    np.divide(k, n, out=frac, where=n > 0)
    return {
        "n": n,
        "k": k,
        "frac": frac,
        "n_events": n_events,
        "chi2_edges": chi2_edges,
        "occ_edges": occ_edges,
        "chi2_centers": 0.5 * (chi2_edges[:-1] + chi2_edges[1:]),
    }


def log_odds(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x[m], y[m], 1)
    return float(slope), float(intercept)


def bootstrap_event(
    df: pd.DataFrame,
    chi2_edges: np.ndarray,
    occ_edges: np.ndarray,
    occ_col: str,
    n_boot: int = N_BOOT,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    rng = rng or np.random.default_rng(RNG_SEED)
    event_ids = np.array(sorted(df["event_id"].unique()))
    n_c, n_o = len(chi2_edges) - 1, len(occ_edges) - 1
    fracs = np.full((n_boot, n_c, n_o), np.nan)
    # Pre-split by event for speed
    by_evt = {e: g for e, g in df.groupby("event_id")}
    for b in range(n_boot):
        sample = rng.choice(event_ids, size=len(event_ids), replace=True)
        parts = [by_evt[e] for e in sample if e in by_evt]
        if not parts:
            continue
        boot_df = pd.concat(parts, ignore_index=True)
        st = cell_stats(boot_df, chi2_edges, occ_edges, occ_col)
        fracs[b] = st["frac"]
    std = np.nanstd(fracs, axis=0, ddof=1)
    lo = np.nanpercentile(fracs, 2.5, axis=0)
    hi = np.nanpercentile(fracs, 97.5, axis=0)
    # Paired sparse-vs-dense log-odds separation per chi2 bin
    lo_odds = log_odds(fracs)
    # occupancy quintile 0 = lowest density, 4 = highest
    sep = lo_odds[:, :, 0] - lo_odds[:, :, -1]  # (boot, chi2)
    # Headline: mean separation across chi2 bins (finite cells only), per boot
    headline = np.full(n_boot, np.nan)
    for b in range(n_boot):
        s = sep[b]
        m = np.isfinite(s)
        if m.any():
            headline[b] = float(np.nanmean(s[m]))
    return {
        "frac_std": std,
        "frac_lo": lo,
        "frac_hi": hi,
        "sep_per_chi2_mean": np.nanmean(sep, axis=0),
        "sep_per_chi2_std": np.nanstd(sep, axis=0, ddof=1),
        "sep_per_chi2_lo": np.nanpercentile(sep, 2.5, axis=0),
        "sep_per_chi2_hi": np.nanpercentile(sep, 97.5, axis=0),
        "headline_sep_mean": float(np.nanmean(headline)),
        "headline_sep_std": float(np.nanstd(headline, ddof=1)),
        "headline_sep_lo": float(np.nanpercentile(headline, 2.5)),
        "headline_sep_hi": float(np.nanpercentile(headline, 97.5)),
        "n_boot": n_boot,
        "n_events_total": int(len(event_ids)),
    }


def analyze(
    df: pd.DataFrame,
    occ_col: str,
    out_dir: Path,
    tag: str,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chi2_edges = fixed_edges_from_quantiles(df["chi2_inc"].to_numpy(), N_CHI2_BINS)
    occ_edges = fixed_edges_from_quantiles(
        df[occ_col].to_numpy(dtype=np.float64), N_OCC_Q
    )
    base = cell_stats(df, chi2_edges, occ_edges, occ_col)
    boot = bootstrap_event(df, chi2_edges, occ_edges, occ_col, n_boot=n_boot)

    # Contingency table
    contingency = []
    for i in range(N_CHI2_BINS):
        for j in range(N_OCC_Q):
            contingency.append(
                {
                    "chi2_bin": i,
                    "chi2_lo": float(chi2_edges[i]),
                    "chi2_hi": float(chi2_edges[i + 1]),
                    "occ_quintile": j,
                    "occ_lo": float(occ_edges[j]),
                    "occ_hi": float(occ_edges[j + 1]),
                    "n_rows": int(base["n"][i, j]),
                    "n_correct": int(base["k"][i, j]),
                    "frac": float(base["frac"][i, j]) if base["n"][i, j] else None,
                    "frac_std_boot": float(boot["frac_std"][i, j]),
                    "frac_ci95": [
                        float(boot["frac_lo"][i, j]),
                        float(boot["frac_hi"][i, j]),
                    ],
                    "n_events": int(base["n_events"][i, j]),
                }
            )
    pd.DataFrame(contingency).to_csv(
        out_dir / f"contingency_{tag}.csv", index=False
    )

    # Fits per occupancy quintile: log-odds vs chi2
    centers = base["chi2_centers"]
    fits = []
    for j in range(N_OCC_Q):
        y = log_odds(base["frac"][:, j])
        slope, intercept = fit_line(centers, y)
        fits.append(
            {
                "occ_quintile": j,
                "slope": slope,
                "intercept": intercept,
                "gaussian_model_slope": -0.5,
            }
        )

    # Plots
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for j in range(N_OCC_Q):
            y = base["frac"][:, j]
            yerr = boot["frac_std"][:, j]
            m = np.isfinite(y)
            ax.errorbar(
                centers[m],
                y[m],
                yerr=yerr[m],
                marker="o",
                ms=4,
                lw=1.2,
                label=f"Q{j+1} ({occ_edges[j]:.2g}–{occ_edges[j+1]:.2g})",
            )
        ax.set_xlabel(r"$\chi^2$ (quantile-bin center)")
        ax.set_ylabel("Observed fraction correct")
        ax.set_title(f"P(correct | χ²) by {occ_col} quintile")
        ax.legend(fontsize=8, frameon=False)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(out_dir / f"prob_vs_chi2_{tag}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        x_line = np.linspace(max(centers.min(), 1e-3), centers.max(), 50)
        for j in range(N_OCC_Q):
            y = log_odds(base["frac"][:, j])
            m = np.isfinite(y)
            ax.plot(centers[m], y[m], "o-", ms=4, lw=1.2, label=f"Q{j+1} data")
            slope, intercept = fits[j]["slope"], fits[j]["intercept"]
            if np.isfinite(slope):
                ax.plot(
                    x_line,
                    slope * x_line + intercept,
                    "--",
                    lw=1.0,
                    alpha=0.7,
                    label=f"Q{j+1} fit slope={slope:.3f}",
                )
            # Gaussian-uniform-background: parallel lines slope -1/2
            # anchor intercept so line passes through middle point of this quintile
            if m.any():
                mid = len(centers) // 2
                if np.isfinite(y[mid]):
                    b0 = y[mid] + 0.5 * centers[mid]
                    ax.plot(
                        x_line,
                        -0.5 * x_line + b0,
                        ":",
                        lw=1.0,
                        alpha=0.5,
                        color="gray",
                    )
        ax.set_xlabel(r"$\chi^2$")
        ax.set_ylabel(r"$\log[p/(1-p)]$")
        ax.set_title(
            f"Log-odds vs χ² by {occ_col} (dotted: slope −1/2 anchors)"
        )
        ax.legend(fontsize=7, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"logodds_vs_chi2_{tag}.png", dpi=150)
        plt.close(fig)

    report = {
        "tag": tag,
        "occ_col": occ_col,
        "n_rows": int(len(df)),
        "n_events": int(df["event_id"].nunique()),
        "chi2_edges": chi2_edges.tolist(),
        "occ_edges": occ_edges.tolist(),
        "fits": fits,
        "headline_logodds_sep_Q1_minus_Q5": {
            "mean": boot["headline_sep_mean"],
            "std": boot["headline_sep_std"],
            "ci95": [boot["headline_sep_lo"], boot["headline_sep_hi"]],
            "definition": (
                "mean over χ² bins of [log-odds(lowest occ quintile) − "
                "log-odds(highest occ quintile)]; positive ⇒ sparse windows "
                "more correct at fixed χ²"
            ),
        },
        "sep_per_chi2": {
            "mean": boot["sep_per_chi2_mean"].tolist(),
            "std": boot["sep_per_chi2_std"].tolist(),
            "ci95_lo": boot["sep_per_chi2_lo"].tolist(),
            "ci95_hi": boot["sep_per_chi2_hi"].tolist(),
        },
        "n_boot": n_boot,
        "r_geom_mm": R_GEOM_MM if occ_col == "geometric_density" else None,
    }
    (out_dir / f"report_{tag}.json").write_text(json.dumps(report, indent=2))
    return report


R_GEOM_MM = 5.0  # must match expand_trackstates_to_chi2_rows.R_GEOM_MM


def analyze_strata(df: pd.DataFrame, out_dir: Path, n_boot: int) -> dict[str, Any]:
    results: dict[str, Any] = {"global": {}, "by_layer": {}, "by_eta": {}}
    for occ_col, tag in [
        ("geometric_density", "geom"),
        ("window_count", "window"),
    ]:
        results["global"][tag] = analyze(
            df, occ_col, out_dir / "global", tag, n_boot=n_boot
        )

    # Layer strata (primary third dimension)
    for layer, g in df.groupby("layer_id"):
        if len(g) < 500 or g["event_id"].nunique() < 5:
            continue
        results["by_layer"][str(int(layer))] = analyze(
            g,
            "geometric_density",
            out_dir / "by_layer" / f"layer_{int(layer)}",
            f"geom_layer{int(layer)}",
            n_boot=n_boot,
        )

    df = df.copy()
    df["eta_bin"] = eta_bin(df["eta"].to_numpy())
    for eb, g in df.groupby("eta_bin"):
        if int(eb) < 0 or len(g) < 500 or g["event_id"].nunique() < 5:
            continue
        results["by_eta"][str(int(eb))] = analyze(
            g,
            "geometric_density",
            out_dir / "by_eta" / f"etabin_{int(eb)}",
            f"geom_eta{int(eb)}",
            n_boot=n_boot,
        )
    return results


def summarize(results: dict[str, Any]) -> str:
    g = results["global"]["geom"]
    fits = g["fits"]
    slopes = [f["slope"] for f in fits if np.isfinite(f["slope"])]
    sep = g["headline_logodds_sep_Q1_minus_Q5"]
    lines = []
    lines.append("=== χ² gate calibration (model-free) ===")
    lines.append(f"Rows (after majority_undefined cut): {g['n_rows']}")
    lines.append(f"Events: {g['n_events']}")
    lines.append(
        f"Headline log-odds sep (sparse−dense): "
        f"{sep['mean']:.3f} ± {sep['std']:.3f} "
        f"[{sep['ci95'][0]:.3f}, {sep['ci95'][1]:.3f}]"
    )
    lines.append(
        f"Fitted slopes per geom quintile: "
        + ", ".join(f"{s:.3f}" for s in slopes)
    )
    parallel = (
        max(slopes) - min(slopes) < 0.15
        if len(slopes) >= 2
        else False
    )
    slope_ok = all(abs(s + 0.5) < 0.15 for s in slopes) if slopes else False
    sep_sig = abs(sep["mean"]) > 2 * sep["std"] if sep["std"] > 0 else False
    lines.append(f"Lines separate (sep ≳ 2σ): {sep_sig}")
    lines.append(f"Slopes mutually parallel (Δslope < 0.15): {parallel}")
    lines.append(f"Slopes consistent with −1/2 (|s+0.5|<0.15): {slope_ok}")

    # Does separation survive at fixed layer?
    layer_seps = []
    for lk, rep in results["by_layer"].items():
        s = rep["headline_logodds_sep_Q1_minus_Q5"]
        layer_seps.append((lk, s["mean"], s["std"]))
    if layer_seps:
        n_same = sum(1 for _, m, s in layer_seps if abs(m) > 2 * s and m * sep["mean"] > 0)
        lines.append(
            f"Layer strata with same-sign sep ≳ 2σ: {n_same}/{len(layer_seps)}"
        )
        survive = n_same >= max(1, len(layer_seps) // 2)
        lines.append(f"Separation survives at fixed layer: {survive}")
    else:
        lines.append("Separation survives at fixed layer: insufficient layer stats")

    w = results["global"].get("window")
    if w:
        ws = w["headline_logodds_sep_Q1_minus_Q5"]
        lines.append(
            f"Secondary (window_count) headline sep: "
            f"{ws['mean']:.3f} ± {ws['std']:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+", help="Parquet/CSV slim row files")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    args = p.parse_args()
    df = load_rows(args.inputs)
    results = analyze_strata(df, args.out_dir, n_boot=args.n_boot)
    summary = summarize(results)
    print(summary)
    (args.out_dir / "SUMMARY.txt").write_text(summary + "\n")
    # Compact JSON without huge arrays duplication
    compact = {
        "global_geom": results["global"]["geom"],
        "global_window_headline": results["global"]
        .get("window", {})
        .get("headline_logodds_sep_Q1_minus_Q5"),
        "layer_headlines": {
            k: v["headline_logodds_sep_Q1_minus_Q5"]
            for k, v in results["by_layer"].items()
        },
        "eta_headlines": {
            k: v["headline_logodds_sep_Q1_minus_Q5"]
            for k, v in results["by_eta"].items()
        },
        "summary": summary,
    }
    (args.out_dir / "SUMMARY.json").write_text(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
