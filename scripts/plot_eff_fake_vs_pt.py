#!/usr/bin/env python3
"""Compute ε_DM and f_DM vs true-pT cutoff from matching-scan ROOT dumps.

Requires per op-point directory containing:
  - particles_selected.root  (RootParticleWriter; event_id + barcode + pt)
  - performance_finding_ambi.root  (matchingdetails TTree)
  - tracksummary_ambi.root  (trackClassification + majority t_pT)

Definitions at cutoff pT_min (GeV):
  ε(pT_min) = |{particles in T with pt≥pT_min and matched}|
            / |{particles in T with pt≥pT_min}|
  f(pT_min) = |{tracks not Matched/Duplicate to a particle with pt≥pT_min}|
            / |{all reco tracks}|
  (TrackMatchClassification: Unknown=0, Matched=1, Duplicate=2, Fake=3)

Usage:
  python scripts/plot_eff_fake_vs_pt.py \\
    --scan-dir experiments/joint_motpe/pt_scan \\
    --out-dir experiments/plots/joint_motpe

Writes:
  - eff_fake_vs_pt_cutoff.png/.csv   (ε, f vs pT cutoff)
  - eff_vs_fake_pt_trajectory.png    (all op points: ε vs f trajectories)
  - eff_vs_fake_{tight,medium,fast}.png  (per-OP ε vs f, 0.1 GeV segments)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import uproot
except ImportError as e:  # pragma: no cover
    raise SystemExit("uproot required: pip install uproot") from e


CLASS_MATCHED = 1
CLASS_DUPLICATE = 2


def _barcode_key(vertex_primary, vertex_secondary, particle, generation, sub_particle):
    return (
        int(vertex_primary),
        int(vertex_secondary),
        int(particle),
        int(generation),
        int(sub_particle),
    )


def _load_particles(path: Path) -> dict[tuple, dict]:
    """Flatten RootParticleWriter vectors → {(event, barcode): {pt, ...}}."""
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
            ],
            library="np",
        )
        n_evt = len(arrays["event_id"])
        for i in range(n_evt):
            eid = int(arrays["event_id"][i])
            vp = arrays["vertex_primary"][i]
            vs = arrays["vertex_secondary"][i]
            pa = arrays["particle"][i]
            ge = arrays["generation"][i]
            sp = arrays["sub_particle"][i]
            pt = arrays["pt"][i]
            for j in range(len(pt)):
                key = (eid, _barcode_key(vp[j], vs[j], pa[j], ge[j], sp[j]))
                out[key] = {"pt": float(pt[j])}
    return out


def _load_matching(path: Path) -> dict[tuple, bool] | None:
    """matchingdetails: one row per particle → {(event, barcode): matched}.

    Returns None if the tree is absent (older ACTS writer without the flag).
    """
    out: dict[tuple, bool] = {}
    with uproot.open(path) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        if "matchingdetails" not in keys:
            return None
        tree = f["matchingdetails"]
        arrays = tree.arrays(
            [
                "event_nr",
                "particle_id_vertex_primary",
                "particle_id_vertex_secondary",
                "particle_id_particle",
                "particle_id_generation",
                "particle_id_sub_particle",
                "matched",
            ],
            library="np",
        )
        n = len(arrays["event_nr"])
        for i in range(n):
            key = (
                int(arrays["event_nr"][i]),
                _barcode_key(
                    arrays["particle_id_vertex_primary"][i],
                    arrays["particle_id_vertex_secondary"][i],
                    arrays["particle_id_particle"][i],
                    arrays["particle_id_generation"][i],
                    arrays["particle_id_sub_particle"][i],
                ),
            )
            out[key] = bool(arrays["matched"][i])
    return out


def _load_tracks(path: Path) -> list[dict]:
    """tracksummary: flatten per-track classification + majority truth pT."""
    tracks: list[dict] = []
    with uproot.open(path) as f:
        tree = f["tracksummary"]
        arrays = tree.arrays(
            [
                "event_nr",
                "trackClassification",
                "t_pT",
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
            tpt = arrays["t_pT"][i]
            nmj = arrays["nMajorityHits"][i]
            vp = arrays["majorityParticleId_vertex_primary"][i]
            vs = arrays["majorityParticleId_vertex_secondary"][i]
            pa = arrays["majorityParticleId_particle"][i]
            ge = arrays["majorityParticleId_generation"][i]
            sp = arrays["majorityParticleId_sub_particle"][i]
            for j in range(len(cls)):
                tracks.append(
                    {
                        "event": eid,
                        "classification": int(cls[j]),
                        "t_pT": float(tpt[j]) if np.isfinite(tpt[j]) else float("nan"),
                        "nMajorityHits": int(nmj[j]),
                        "majority": _barcode_key(vp[j], vs[j], pa[j], ge[j], sp[j]),
                    }
                )
    return tracks


def _matching_from_tracks(
    particles: dict[tuple, dict], tracks: list[dict]
) -> dict[tuple, bool]:
    """Fallback ε source when matchingdetails is missing.

    A particle in T is matched if any Matched/Duplicate track has it as
    majorityParticleId (DM primary/duplicate classification).
    """
    matched_keys: set[tuple] = set()
    for tr in tracks:
        if tr["classification"] in (CLASS_MATCHED, CLASS_DUPLICATE):
            matched_keys.add((tr["event"], tr["majority"]))
    return {key: (key in matched_keys) for key in particles}


def metrics_vs_pt(
    particles: dict[tuple, dict],
    matching: dict[tuple, bool],
    tracks: list[dict],
    pt_cuts: np.ndarray,
) -> dict[str, np.ndarray]:
    """ε / f vs true-pT cutoff (LOG DM definitions).

    ε(cut): matched fraction among particles_selected with pt≥cut.

    f(cut): fraction of reco tracks that are *not* Matched/Duplicate to a
    particle in particles_selected with pt≥cut. Soft-matched tracks become
    fake as the cut rises — this is the LOG f_DM, not ACTS's built-in
    fakerate (which only counts Unknown/Fake classifications and stays
    ~flat vs this cut when majority IDs come from the full sim map).
    """
    rows = []
    for key, matched in matching.items():
        if key not in particles:
            continue
        rows.append((particles[key]["pt"], matched))
    if not rows and particles:
        for key, info in particles.items():
            rows.append((info["pt"], bool(matching.get(key, False))))
    if not rows:
        raise RuntimeError("no joined particle/match rows — check ROOT dumps")
    pts = np.array([r[0] for r in rows], dtype=float)
    matched = np.array([r[1] for r in rows], dtype=bool)

    eff = np.zeros_like(pt_cuts, dtype=float)
    fake = np.zeros_like(pt_cuts, dtype=float)
    n_t = np.zeros_like(pt_cuts, dtype=float)
    n_tracks = float(len(tracks)) if tracks else float("nan")

    # Majority truth pT from particles_selected when available
    track_maj_pt = []
    for tr in tracks:
        key = (tr["event"], tr["majority"])
        if key in particles:
            track_maj_pt.append(particles[key]["pt"])
        else:
            track_maj_pt.append(
                tr["t_pT"] if np.isfinite(tr["t_pT"]) else float("nan")
            )
    track_maj_pt = np.asarray(track_maj_pt, dtype=float)
    track_cls = np.asarray([tr["classification"] for tr in tracks], dtype=int)

    for i, cut in enumerate(pt_cuts):
        sel = pts >= cut
        n_t[i] = sel.sum()
        eff[i] = matched[sel].mean() if sel.any() else float("nan")
        if tracks and n_tracks > 0:
            good = (
                (track_cls == CLASS_MATCHED) | (track_cls == CLASS_DUPLICATE)
            ) & (track_maj_pt >= cut)
            fake[i] = 1.0 - good.mean()
        else:
            fake[i] = float("nan")
    return {
        "pt_cut": pt_cuts,
        "efficiency": eff * 100,
        "fakerate": fake * 100,
        "n_T": n_t,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scan-dir",
        type=Path,
        default=Path("experiments/joint_motpe/pt_scan"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/plots/joint_motpe"),
    )
    ap.add_argument("--pt-min", type=float, default=0.15)
    ap.add_argument("--pt-max", type=float, default=5.0)
    ap.add_argument("--n-cuts", type=int, default=50)
    ap.add_argument(
        "--pt-step",
        type=float,
        default=0.1,
        help="Step for ε-vs-f trajectory markers (GeV)",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.scan_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        dirs = {
            name: Path(v["output_dir"]) if Path(v["output_dir"]).is_absolute()
            else args.scan_dir / Path(v["output_dir"]).name
            for name, v in manifest.items()
            if v.get("ok")
        }
        # Prefer local mirror naming name_tNNN
        for name, v in manifest.items():
            if not v.get("ok"):
                continue
            local = args.scan_dir / f"{name}_t{v['trial']}"
            if local.is_dir():
                dirs[name] = local
    else:
        dirs = {p.name.split("_")[0]: p for p in args.scan_dir.glob("*_t*")}

    # 0.1 GeV trajectory grid + denser sampling for the vs-cutoff curves
    pt_step = np.round(
        np.arange(args.pt_min, args.pt_max + 1e-9, args.pt_step), 10
    )
    pt_step = pt_step[(pt_step >= args.pt_min) & (pt_step <= args.pt_max)]
    pt_cuts = np.unique(
        np.concatenate(
            [
                np.linspace(args.pt_min, args.pt_max, args.n_cuts),
                pt_step,
                np.array([0.5, 0.7, 1.0, 1.5, 2.0, 3.0]),
            ]
        )
    )
    colors = {
        "tight": "#1565C0",
        "medium": "#2E7D32",
        "fast": "#E65100",
        "loose": "#6A1B9A",
    }
    series = {}
    for name, d in sorted(dirs.items()):
        part = d / "particles_selected.root"
        perf = d / "performance_finding_ambi.root"
        summ = d / "tracksummary_ambi.root"
        for req in (part, perf, summ):
            if not req.exists():
                raise FileNotFoundError(req)
        print(f"Loading {name} from {d}")
        particles = _load_particles(part)
        matching = _load_matching(perf)
        tracks = _load_tracks(summ)
        if matching is None:
            print("  (no matchingdetails — deriving particle matches from tracksummary)")
            matching = _matching_from_tracks(particles, tracks)
        series[name] = metrics_vs_pt(particles, matching, tracks, pt_cuts)
        print(
            f"  {name}: n_particles={len(particles)} n_match_rows={len(matching)} "
            f"n_tracks={len(tracks)}  ε(1 GeV)={np.interp(1.0, pt_cuts, series[name]['efficiency']):.2f}% "
            f"f(1 GeV)={np.interp(1.0, pt_cuts, series[name]['fakerate']):.3f}%"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for name, s in series.items():
        c = colors.get(name, None)
        axes[0].plot(s["pt_cut"], s["efficiency"], label=name, color=c, lw=2)
        axes[1].plot(s["pt_cut"], s["fakerate"], label=name, color=c, lw=2)
    axes[0].axvline(1.0, color="#9E9E9E", ls="--", lw=1, label="Stage 5 cut")
    axes[1].axvline(1.0, color="#9E9E9E", ls="--", lw=1)
    axes[0].set_ylabel(r"$\varepsilon_{\mathrm{DM}}$ (%)")
    axes[1].set_ylabel(r"$f_{\mathrm{DM}}$ (%)")
    axes[0].set_xlabel(r"true $p_T$ cutoff (GeV)")
    axes[1].set_xlabel(r"true $p_T$ cutoff (GeV)")
    axes[0].set_title("Particle efficiency vs $p_T$ cutoff")
    axes[1].set_title("Track fake rate vs $p_T$ cutoff")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[1].legend(fontsize=9)
    fig.suptitle(
        r"Post-ambi DM metrics vs true $p_T$ cutoff on $T$"
        "\n"
        r"(matching floor $=0.15$ GeV; $f$ counts soft-matched tracks as fake "
        r"— not ACTS built-in Unknown-only fakerate)",
        fontsize=11,
    )
    fig.tight_layout()
    out = args.out_dir / "eff_fake_vs_pt_cutoff.png"
    fig.savefig(out, dpi=160)
    plt.close()
    print(f"Wrote {out}")

    # CSV dump
    csv_path = args.out_dir / "eff_fake_vs_pt_cutoff.csv"
    with csv_path.open("w") as f:
        header = ["pt_cut_GeV"]
        for name in series:
            header += [f"{name}_eff_pct", f"{name}_fake_pct", f"{name}_n_T"]
        f.write(",".join(header) + "\n")
        for i, cut in enumerate(pt_cuts):
            row = [f"{cut:.4f}"]
            for name in series:
                s = series[name]
                row += [
                    f"{s['efficiency'][i]:.4f}",
                    f"{s['fakerate'][i]:.4f}",
                    f"{s['n_T'][i]:.0f}",
                ]
            f.write(",".join(row) + "\n")
    print(f"Wrote {csv_path}")

    # ε vs f trajectories at ΔpT = 0.1 GeV (exact grid values, not interp)
    step_series: dict[str, dict[str, np.ndarray]] = {}
    for name, s in series.items():
        mask = np.isin(np.round(s["pt_cut"], 10), np.round(pt_step, 10))
        # Fallback: nearest-index gather if float uniqueness drifts
        if mask.sum() < len(pt_step) - 1:
            idx = [int(np.argmin(np.abs(s["pt_cut"] - p))) for p in pt_step]
            step_series[name] = {
                "pt_cut": pt_step,
                "efficiency": s["efficiency"][idx],
                "fakerate": s["fakerate"][idx],
            }
        else:
            step_series[name] = {
                "pt_cut": s["pt_cut"][mask],
                "efficiency": s["efficiency"][mask],
                "fakerate": s["fakerate"][mask],
            }

    _plot_eff_vs_fake_combined(step_series, args.out_dir, colors)
    for name, s in step_series.items():
        _plot_eff_vs_fake_single(name, s, args.out_dir)


def _annotate_pt_markers(
    ax: plt.Axes,
    fake: np.ndarray,
    eff: np.ndarray,
    pt: np.ndarray,
    *,
    every: float = 0.5,
    color: str,
) -> None:
    """Label selected pT cutoffs along an ε–f trajectory."""
    for f, e, p in zip(fake, eff, pt):
        if not np.isfinite(f) or not np.isfinite(e):
            continue
        # Annotate integer and half-GeV cuts; always mark 1.0
        if abs(p - 1.0) < 1e-6 or abs(p / every - round(p / every)) < 1e-6:
            ax.annotate(
                f"{p:.1f}",
                (f, e),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=color,
                alpha=0.9,
            )


def _plot_eff_vs_fake_combined(
    series: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
    colors: dict[str, str],
) -> None:
    """One figure: each op point is a line through (f, ε) at ΔpT = 0.1."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for name, s in series.items():
        c = colors.get(name, None)
        ax.plot(
            s["fakerate"],
            s["efficiency"],
            "-o",
            color=c,
            lw=2,
            ms=3.5,
            label=name,
            markevery=1,
        )
        _annotate_pt_markers(
            ax, s["fakerate"], s["efficiency"], s["pt_cut"], color=c or "#333"
        )
        # Highlight Stage-5 1 GeV operating metric
        i1 = int(np.argmin(np.abs(s["pt_cut"] - 1.0)))
        ax.scatter(
            [s["fakerate"][i1]],
            [s["efficiency"][i1]],
            s=80,
            facecolors="none",
            edgecolors=c,
            linewidths=2,
            zorder=5,
        )
    ax.set_xlabel(r"$f_{\mathrm{DM}}$ (%)  (LOG def.; soft-matched → fake)")
    ax.set_ylabel(r"$\varepsilon_{\mathrm{DM}}$ (%)")
    ax.set_title(
        r"Efficiency vs fake rate · trajectory over true $p_T$ cutoff"
        "\n"
        r"(markers every $0.1$ GeV; open circle = $1$ GeV Stage-5 cut)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    out = out_dir / "eff_vs_fake_pt_trajectory.png"
    fig.savefig(out, dpi=160)
    plt.close()
    print(f"Wrote {out}")


def _plot_eff_vs_fake_single(
    name: str,
    s: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """Per–op-point ε vs f: each 0.1 GeV step is its own colored segment."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    pt = s["pt_cut"]
    fake = s["fakerate"]
    eff = s["efficiency"]
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(pt.min()), vmax=float(pt.max()))
    legend_pts = {0.5, 1.0, 1.5, 2.0, 3.0, 5.0}

    # One line segment per 0.1 GeV increment (p_i → p_{i+1})
    for i in range(len(pt) - 1):
        p_mid = 0.5 * (pt[i] + pt[i + 1])
        lab = None
        if any(abs(pt[i] - lp) < 1e-6 for lp in legend_pts):
            lab = f"{pt[i]:.1f}→{pt[i+1]:.1f} GeV"
        ax.plot(
            [fake[i], fake[i + 1]],
            [eff[i], eff[i + 1]],
            "-",
            color=cmap(norm(p_mid)),
            lw=2.4,
            solid_capstyle="round",
            label=lab,
            zorder=2,
        )
    ax.scatter(
        fake,
        eff,
        c=pt,
        cmap=cmap,
        norm=norm,
        s=28,
        zorder=3,
        edgecolors="k",
        linewidths=0.25,
    )
    _annotate_pt_markers(ax, fake, eff, pt, every=0.5, color="#222222")
    i1 = int(np.argmin(np.abs(pt - 1.0)))
    ax.scatter(
        [fake[i1]],
        [eff[i1]],
        s=110,
        facecolors="none",
        edgecolors="crimson",
        linewidths=2.2,
        zorder=5,
        label=r"$p_T\geq 1$ GeV",
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"true $p_T$ cutoff (GeV)")
    ax.set_xlabel(r"$f_{\mathrm{DM}}$ (%)  (LOG def.; soft-matched → fake)")
    ax.set_ylabel(r"$\varepsilon_{\mathrm{DM}}$ (%)")
    ax.set_title(
        f"{name}: ε vs f · separate line per $0.1$ GeV $p_T$ step"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right", ncols=2)
    fig.tight_layout()
    out = out_dir / f"eff_vs_fake_{name}.png"
    fig.savefig(out, dpi=160)
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
