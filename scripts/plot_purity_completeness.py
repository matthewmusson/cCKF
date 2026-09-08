#!/usr/bin/env python3
"""Purity / completeness for DM-matched tracks (Tight / Medium / Fast).

Uses matching-scan dumps under ``experiments/joint_motpe/pt_scan/``:
  - tracksummary_ambi.root
  - particles_selected.root

Definitions (ACTS TrackFinderPerformanceCollector):
  purity       = nMajorityHits / (nMeasurements + nOutliers)
  completeness = nMajorityHits / number_of_hits(majority particle)

Analysis sample: Matched or Duplicate tracks whose majority particle has
true pT ≥ ``--pt-min`` (default 1 GeV), eval events in the dump.

Outputs (``experiments/plots/joint_motpe/``):
  - purity_completeness_hist.png   full distribution over all such tracks
  - purity_completeness_per_event.png  per-event means ± SEM
  - purity_completeness_summary.csv

Usage (from cCKF/):
  python scripts/plot_purity_completeness.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import uproot
except ImportError as e:  # pragma: no cover
    raise SystemExit("uproot required") from e

CLASS_MATCHED = 1
CLASS_DUPLICATE = 2

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "experiments/joint_motpe/pt_scan"
OUT = ROOT / "experiments/plots/joint_motpe"

OPS = {
    "tight": 79,
    "medium": 70,
    "fast": 331,
}
COLORS = {"tight": "#1565C0", "medium": "#2E7D32", "fast": "#E65100"}


def _barcode(vp, vs, pa, ge, sp) -> tuple[int, int, int, int, int]:
    return (int(vp), int(vs), int(pa), int(ge), int(sp))


def _load_particles(path: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    with uproot.open(path) as f:
        tree = f["particles"]
        arrays = tree.arrays(
            [
                "event_id",
                "vertex_primary",
                "vertex_secondary",
                "particle",
                "generation",
                "sub_particle",
                "pt",
                "number_of_hits",
            ],
            library="np",
        )
        for i in range(len(arrays["event_id"])):
            eid = int(arrays["event_id"][i])
            for j in range(len(arrays["pt"][i])):
                key = (
                    eid,
                    _barcode(
                        arrays["vertex_primary"][i][j],
                        arrays["vertex_secondary"][i][j],
                        arrays["particle"][i][j],
                        arrays["generation"][i][j],
                        arrays["sub_particle"][i][j],
                    ),
                )
                out[key] = {
                    "pt": float(arrays["pt"][i][j]),
                    "n_hits": int(arrays["number_of_hits"][i][j]),
                }
    return out


def _load_dm_tracks(
    summ_path: Path,
    particles: dict[tuple, dict],
    pt_min: float,
) -> list[dict]:
    """One row per DM-matched track with majority particle pT ≥ pt_min."""
    rows: list[dict] = []
    with uproot.open(summ_path) as f:
        tree = f["tracksummary"]
        arrays = tree.arrays(
            [
                "event_nr",
                "trackClassification",
                "nMeasurements",
                "nOutliers",
                "nMajorityHits",
                "majorityParticleId_vertex_primary",
                "majorityParticleId_vertex_secondary",
                "majorityParticleId_particle",
                "majorityParticleId_generation",
                "majorityParticleId_sub_particle",
            ],
            library="np",
        )
        n_evt = len(arrays["event_nr"])
        for i in range(n_evt):
            eid = int(arrays["event_nr"][i])
            cls = arrays["trackClassification"][i]
            nmeas = arrays["nMeasurements"][i]
            nout = arrays["nOutliers"][i]
            nmaj = arrays["nMajorityHits"][i]
            vp = arrays["majorityParticleId_vertex_primary"][i]
            vs = arrays["majorityParticleId_vertex_secondary"][i]
            pa = arrays["majorityParticleId_particle"][i]
            ge = arrays["majorityParticleId_generation"][i]
            sp = arrays["majorityParticleId_sub_particle"][i]
            for j in range(len(cls)):
                c = int(cls[j])
                if c not in (CLASS_MATCHED, CLASS_DUPLICATE):
                    continue
                key = (eid, _barcode(vp[j], vs[j], pa[j], ge[j], sp[j]))
                part = particles.get(key)
                if part is None or part["pt"] < pt_min:
                    continue
                denom_p = int(nmeas[j]) + int(nout[j])
                n_hit = part["n_hits"]
                if denom_p <= 0 or n_hit <= 0:
                    continue
                nm = int(nmaj[j])
                rows.append(
                    {
                        "event": eid,
                        "purity": nm / denom_p,
                        "completeness": nm / n_hit,
                        "classification": c,
                    }
                )
    return rows


def _per_event_means(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (event_ids, mean_purity, mean_completeness) per event."""
    by_evt: dict[int, list[dict]] = {}
    for r in rows:
        by_evt.setdefault(r["event"], []).append(r)
    eids = np.array(sorted(by_evt))
    mp = np.array([np.mean([r["purity"] for r in by_evt[e]]) for e in eids])
    mc = np.array([np.mean([r["completeness"] for r in by_evt[e]]) for e in eids])
    return eids, mp, mc


def _sem(x: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    return float(np.std(x, ddof=1) / np.sqrt(n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", type=Path, default=SCAN)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--pt-min", type=float, default=1.0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    series: dict[str, dict] = {}
    for name, trial in OPS.items():
        d = args.scan_dir / f"{name}_t{trial}"
        part = d / "particles_selected.root"
        summ = d / "tracksummary_ambi.root"
        if not part.exists() or not summ.exists():
            raise FileNotFoundError(f"missing dumps in {d}")
        print(f"Loading {name} from {d}")
        particles = _load_particles(part)
        tracks = _load_dm_tracks(summ, particles, args.pt_min)
        eids, mp, mc = _per_event_means(tracks)
        pur = np.array([r["purity"] for r in tracks])
        comp = np.array([r["completeness"] for r in tracks])
        series[name] = {
            "trial": trial,
            "purity": pur,
            "completeness": comp,
            "event_ids": eids,
            "evt_purity": mp,
            "evt_completeness": mc,
            "n_tracks": len(tracks),
            "n_events": len(eids),
            "mean_purity_tracks": float(pur.mean()) if len(pur) else float("nan"),
            "mean_completeness_tracks": float(comp.mean()) if len(comp) else float("nan"),
            "mean_purity_events": float(mp.mean()) if len(mp) else float("nan"),
            "sem_purity_events": _sem(mp),
            "mean_completeness_events": float(mc.mean()) if len(mc) else float("nan"),
            "sem_completeness_events": _sem(mc),
        }
        s = series[name]
        print(
            f"  {name}: n_tracks={s['n_tracks']} n_events={s['n_events']}  "
            f"⟨purity⟩_evt={s['mean_purity_events']:.4f}±{s['sem_purity_events']:.4f} (SEM)  "
            f"⟨compl⟩_evt={s['mean_completeness_events']:.4f}±{s['sem_completeness_events']:.4f} (SEM)"
        )
        print(
            f"         track-pooled ⟨purity⟩={s['mean_purity_tracks']:.4f}  "
            f"⟨compl⟩={s['mean_completeness_tracks']:.4f}  "
            f"min(p,c)=({pur.min():.3f},{comp.min():.3f})  "
            f"frac(p≤0.5)={(pur <= 0.5).mean()*100:.2f}%  "
            f"frac(c≤0.5)={(comp <= 0.5).mean()*100:.2f}%"
        )

    # --- Full-distribution histograms (fraction of tracks per bin) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    bins = np.linspace(0.5, 1.0, 26)
    for name, s in series.items():
        c = COLORS[name]
        n = max(s["n_tracks"], 1)
        # Each track contributes 100/n so bin height = % of that config's tracks
        w = np.full(n, 100.0 / n)
        axes[0].hist(
            s["purity"],
            bins=bins,
            weights=w,
            histtype="step",
            lw=2,
            color=c,
            label=f"{name} (n={s['n_tracks']})",
        )
        axes[1].hist(
            s["completeness"],
            bins=bins,
            weights=w,
            histtype="step",
            lw=2,
            color=c,
            label=f"{name} (n={s['n_tracks']})",
        )
    axes[0].axvline(0.5, color="#9E9E9E", ls="--", lw=1)
    axes[1].axvline(0.5, color="#9E9E9E", ls="--", lw=1)
    axes[0].set_xlabel("purity")
    axes[1].set_xlabel("completeness")
    axes[0].set_ylabel("fraction of tracks (%)")
    axes[0].set_title("Purity (DM-matched tracks)")
    axes[1].set_title("Completeness (DM-matched tracks)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlim(0.48, 1.02)
    fig.suptitle(
        rf"Post-ambi DM-matched tracks · majority $p_T \geq {args.pt_min:g}$ GeV"
        "\n"
        "bin height = % of that config's tracks in the bin "
        "(sums to 100%); "
        r"purity $= n_{\mathrm{maj}}/(n_{\mathrm{meas}}+n_{\mathrm{out}})$; "
        r"completeness $= n_{\mathrm{maj}}/n_{\mathrm{hits}}(t)$",
        fontsize=11,
    )
    fig.tight_layout()
    out_hist = args.out_dir / "purity_completeness_hist.png"
    fig.savefig(out_hist, dpi=150)
    plt.close()
    print(f"Wrote {out_hist}")

    # --- Per-event means ± SEM ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    x = np.arange(len(series))
    width = 0.55
    names = list(series.keys())
    for ax, key_mean, key_sem, title, ylabel in [
        (
            axes[0],
            "mean_purity_events",
            "sem_purity_events",
            "Per-event mean purity",
            r"$\langle\mathrm{purity}\rangle$ (event mean)",
        ),
        (
            axes[1],
            "mean_completeness_events",
            "sem_completeness_events",
            "Per-event mean completeness",
            r"$\langle\mathrm{completeness}\rangle$ (event mean)",
        ),
    ]:
        means = [series[n][key_mean] for n in names]
        sems = [series[n][key_sem] for n in names]
        bars = ax.bar(
            x,
            means,
            width,
            yerr=sems,
            color=[COLORS[n] for n in names],
            edgecolor="black",
            linewidth=0.6,
            capsize=4,
            error_kw={"elinewidth": 1.2},
        )
        for i, n in enumerate(names):
            ax.annotate(
                f"{means[i]:.3f}±{sems[i]:.3f}",
                (x[i], means[i] + sems[i]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )
        # faint per-event scatter
        for i, n in enumerate(names):
            vals = (
                series[n]["evt_purity"]
                if key_mean == "mean_purity_events"
                else series[n]["evt_completeness"]
            )
            jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.15
            ax.scatter(
                np.full(len(vals), x[i]) + jitter,
                vals,
                s=10,
                c=COLORS[n],
                alpha=0.35,
                zorder=3,
            )
        ax.set_xticks(x, names)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0.5, color="#9E9E9E", ls="--", lw=1)
        ax.set_ylim(0.45, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        _ = bars
    fig.suptitle(
        rf"Event-averaged purity / completeness · $N_{{\mathrm{{evt}}}}=32$ · "
        rf"$p_T\geq {args.pt_min:g}$ GeV"
        "\n"
        r"error bars = SEM $= s/\sqrt{N}$ over per-event means",
        fontsize=11,
    )
    fig.tight_layout()
    out_evt = args.out_dir / "purity_completeness_per_event.png"
    fig.savefig(out_evt, dpi=150)
    plt.close()
    print(f"Wrote {out_evt}")

    # CSV summary
    csv_path = args.out_dir / "purity_completeness_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "point",
                "trial",
                "pt_min_GeV",
                "n_tracks",
                "n_events",
                "mean_purity_tracks",
                "mean_completeness_tracks",
                "mean_purity_events",
                "sem_purity_events",
                "mean_completeness_events",
                "sem_completeness_events",
            ]
        )
        for name, s in series.items():
            w.writerow(
                [
                    name,
                    s["trial"],
                    args.pt_min,
                    s["n_tracks"],
                    s["n_events"],
                    f"{s['mean_purity_tracks']:.6f}",
                    f"{s['mean_completeness_tracks']:.6f}",
                    f"{s['mean_purity_events']:.6f}",
                    f"{s['sem_purity_events']:.6f}",
                    f"{s['mean_completeness_events']:.6f}",
                    f"{s['sem_completeness_events']:.6f}",
                ]
            )
    print(f"Wrote {csv_path}")

    # machine-readable JSON for LOG
    summary = {
        name: {
            k: (float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in s.items()
            if k
            not in {
                "purity",
                "completeness",
                "event_ids",
                "evt_purity",
                "evt_completeness",
            }
        }
        for name, s in series.items()
    }
    (args.out_dir / "purity_completeness_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
