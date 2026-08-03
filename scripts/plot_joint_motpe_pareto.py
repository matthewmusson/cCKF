#!/usr/bin/env python3
"""Pareto / diagnostic plots for Stage 5 joint MO-TPE study.

Mirrors the Stage 3 plot suite under ``experiments/plots/``, adapted for the
joint 10D seeding+CKF search. Uses:

  - ``experiments/joint_motpe/trials.csv``     (opt set [0, 32))
  - ``experiments/joint_motpe/eval_pareto_4d.csv`` (eval set [32, 64))

Outputs land in ``experiments/plots/joint_motpe/``.

Usage (from cCKF/):
    python scripts/plot_joint_motpe_pareto.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ROOT / "experiments/joint_motpe/trials.csv"
EVAL = ROOT / "experiments/joint_motpe/eval_pareto_4d.csv"
OUT = ROOT / "experiments/plots/joint_motpe"

# Operating points chosen from eval-set Pareto (see experiments/LOG.md)
OPS = {
    "tight": 79,  # ε≥90%, min f
    "medium": 70,  # max ε with f<1%
    "loose": 284,  # max ε with f<5%
}
OPS_COLORS = {"tight": "#1565C0", "medium": "#2E7D32", "loose": "#E65100"}


def _load_trials(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("state") != "COMPLETE":
                continue
            try:
                rows.append(
                    {
                        "n": int(r["number"]),
                        "eff": float(r["values_0"]),
                        "fake": float(r["values_1"]),
                        "dup": float(r["values_2"]),
                        "rt": float(r["values_3"]),
                        "branch": int(float(r["params_ckf_numMeasurementsCutOff"])),
                        "nmeas": int(float(r["params_ckf_nMeasurementsMin"])),
                        "holes": int(float(r["params_ckf_maxHolesAndOutliers"])),
                        "chi2": float(r["params_ckf_chi2CutOffMeasurement"]),
                        "seeds": int(float(r["params_num_seeds_per_spm"])),
                        "ptmin": float(r["params_ckf_ptMin"]),
                        "impact": float(r["params_seed_impactMax"]),
                        "seed_pt": float(r["params_seed_minPt"]),
                        "sigma": float(r["params_seed_sigmaScattering"]),
                    }
                )
            except (ValueError, KeyError):
                continue
    return rows


def _load_eval(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "n": int(r["trial_number"]),
                    "opt_eff": float(r["opt_efficiency"]),
                    "opt_fake": float(r["opt_fakerate"]),
                    "eff": float(r["eval_efficiency"]),
                    "fake": float(r["eval_fakerate"]),
                    "dup": float(r["eval_duplicaterate"]),
                    "rt": float(r["eval_wall_per_event"]),
                    "de": float(r["delta_efficiency"]),
                    "df": float(r["delta_fakerate"]),
                    "branch": int(float(r["ckf_numMeasurementsCutOff"])),
                    "nmeas": int(float(r["ckf_nMeasurementsMin"])),
                    "holes": int(float(r["ckf_maxHolesAndOutliers"])),
                    "chi2": float(r["ckf_chi2CutOffMeasurement"]),
                    "seeds": int(float(r["num_seeds_per_spm"])),
                    "ptmin": float(r["ckf_ptMin"]),
                }
            )
    return rows


def _nondom(pts: list[dict], eff_key="eff", fake_key="fake") -> list[dict]:
    out = []
    for p in pts:
        dominated = False
        for q in pts:
            if q["n"] == p["n"]:
                continue
            if (
                q[eff_key] >= p[eff_key]
                and q[fake_key] <= p[fake_key]
                and (q[eff_key] > p[eff_key] or q[fake_key] < p[fake_key])
            ):
                dominated = True
                break
        if not dominated:
            out.append(p)
    return sorted(out, key=lambda x: x[eff_key])


def _mark_ops(ax, pts_by_n: dict, xkey="eff", ykey="fake"):
    for name, num in OPS.items():
        if num not in pts_by_n:
            continue
        p = pts_by_n[num]
        ax.scatter(
            [p[xkey]],
            [p[ykey]],
            c=OPS_COLORS[name],
            s=160,
            marker="*",
            edgecolors="black",
            linewidths=0.6,
            zorder=10,
            label=f"{name} (t{num})",
        )


def plot_eff_vs_fake(trials: list[dict], eval_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Opt set
    ax = axes[0]
    ax.scatter(
        [t["eff"] for t in trials],
        [t["fake"] for t in trials],
        c="#B0BEC5",
        s=22,
        alpha=0.45,
        label="All COMPLETE",
        zorder=2,
    )
    pareto = _nondom(trials)
    ax.plot(
        [p["eff"] for p in pareto],
        [p["fake"] for p in pareto],
        "k-",
        alpha=0.35,
        lw=1,
        zorder=3,
    )
    ax.scatter(
        [p["eff"] for p in pareto],
        [p["fake"] for p in pareto],
        c="#1565C0",
        s=36,
        zorder=4,
        label=f"2D Pareto (n={len(pareto)})",
    )
    # Mark ops using opt metrics from eval CSV trial numbers
    by_n = {t["n"]: t for t in trials}
    _mark_ops(ax, by_n)
    ax.axhline(1.14, color="#9E9E9E", ls="--", lw=1, label="ACTS default f")
    ax.axvline(37.3, color="#9E9E9E", ls=":", lw=1, label="ACTS default ε")
    ax.set_xlabel(r"$\varepsilon_{\mathrm{DM}}$ (%)")
    ax.set_ylabel(r"$f_{\mathrm{DM}}$ (%)")
    ax.set_title("Optimization set [0, 32)")
    ax.set_xlim(45, 100)
    ax.set_ylim(-1, 40)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

    # Eval set (Pareto configs only)
    ax = axes[1]
    ax.scatter(
        [r["opt_eff"] for r in eval_rows],
        [r["opt_fake"] for r in eval_rows],
        c="#90CAF9",
        s=40,
        alpha=0.5,
        label="Opt metrics (same configs)",
        zorder=2,
    )
    ax.scatter(
        [r["eff"] for r in eval_rows],
        [r["fake"] for r in eval_rows],
        c="#C62828",
        s=40,
        alpha=0.75,
        label="Eval metrics [32, 64)",
        zorder=3,
    )
    for r in eval_rows:
        ax.plot(
            [r["opt_eff"], r["eff"]],
            [r["opt_fake"], r["fake"]],
            color="#BDBDBD",
            lw=0.5,
            alpha=0.4,
            zorder=1,
        )
    by_n = {r["n"]: r for r in eval_rows}
    _mark_ops(ax, by_n)
    ax.axhline(1.14, color="#9E9E9E", ls="--", lw=1)
    ax.axvline(37.3, color="#9E9E9E", ls=":", lw=1)
    ax.set_xlabel(r"$\varepsilon_{\mathrm{DM}}$ (%)")
    ax.set_title("Eval set [32, 64) — Pareto configs")
    ax.set_xlim(45, 100)
    ax.set_ylim(-1, 40)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Joint MO-TPE · ColliderML ttbar μ=200 · pT > 1 GeV · DM matching",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT / "pareto_eff_vs_fake.png", dpi=150)
    plt.close()
    print("✓ pareto_eff_vs_fake.png")


def plot_eff_vs_runtime(trials: list[dict], eval_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    ax = axes[0]
    ax.scatter(
        [t["eff"] for t in trials],
        [t["rt"] for t in trials],
        c="#B0BEC5",
        s=22,
        alpha=0.45,
    )
    pareto = _nondom(trials)
    ax.scatter(
        [p["eff"] for p in pareto],
        [p["rt"] for p in pareto],
        c="#1565C0",
        s=40,
        label="2D ε–f Pareto",
    )
    _mark_ops(ax, {t["n"]: t for t in trials}, "eff", "rt")
    ax.set_xlabel(r"$\varepsilon_{\mathrm{DM}}$ (%)")
    ax.set_ylabel("Wall time / event (s)")
    ax.set_title("Opt set")
    ax.set_xlim(45, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(
        [r["eff"] for r in eval_rows],
        [r["rt"] for r in eval_rows],
        c="#C62828",
        s=40,
        alpha=0.75,
    )
    _mark_ops(ax, {r["n"]: r for r in eval_rows}, "eff", "rt")
    ax.set_xlabel(r"$\varepsilon_{\mathrm{DM}}$ (%)")
    ax.set_ylabel("Wall time / event (s)")
    ax.set_title("Eval set (Pareto configs)")
    ax.set_xlim(45, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Efficiency vs runtime · joint MO-TPE", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "pareto_eff_vs_runtime.png", dpi=150)
    plt.close()
    print("✓ pareto_eff_vs_runtime.png")


def plot_fake_vs_runtime(trials: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        [t["fake"] for t in trials],
        [t["rt"] for t in trials],
        c=[t["eff"] for t in trials],
        cmap="viridis",
        s=28,
        alpha=0.7,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\varepsilon_{\mathrm{DM}}$ (%)")
    _mark_ops(ax, {t["n"]: t for t in trials}, "fake", "rt")
    ax.set_xlabel(r"$f_{\mathrm{DM}}$ (%)")
    ax.set_ylabel("Wall time / event (s)")
    ax.set_title("Fake rate vs runtime (opt set, colored by ε)")
    ax.set_xlim(-1, 50)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fake_vs_runtime.png", dpi=150)
    plt.close()
    print("✓ fake_vs_runtime.png")


def _nondom_3d(pts: list[dict]) -> list[dict]:
    """Non-dominated set for ε↑, f↓, runtime↓."""
    out = []
    for p in pts:
        dominated = False
        for q in pts:
            if q["n"] == p["n"]:
                continue
            better_or_eq = (
                q["eff"] >= p["eff"]
                and q["fake"] <= p["fake"]
                and q["rt"] <= p["rt"]
            )
            strictly_better = (
                q["eff"] > p["eff"] or q["fake"] < p["fake"] or q["rt"] < p["rt"]
            )
            if better_or_eq and strictly_better:
                dominated = True
                break
        if not dominated:
            out.append(p)
    return out


def plot_eff_fake_runtime_3d_interactive(trials: list[dict]) -> Path:
    """Interactive browser 3D scatter with range filters + smooth Pareto surface.

    Writes ``pareto_eff_fake_runtime_3d.html``. The surface is a smoothed
    thin-plate RBF fit of wall ≈ s(ε, f) on the currently visible 3D-Pareto
    (+ operating-point) samples — a visualization aid, not a physical model.
    """
    import json

    pareto = _nondom_3d(trials)
    pareto_ns = {p["n"] for p in pareto}
    op_name = {num: name for name, num in OPS.items()}

    points = []
    for t in trials:
        kind = (
            "op"
            if t["n"] in op_name
            else ("pareto" if t["n"] in pareto_ns else "other")
        )
        points.append(
            {
                "n": t["n"],
                "eff": t["eff"],
                "fake": t["fake"],
                "dup": t["dup"],
                "rt": t["rt"],
                "seeds": t["seeds"],
                "branch": t["branch"],
                "nmeas": t["nmeas"],
                "chi2": t["chi2"],
                "kind": kind,
                "op": op_name.get(t["n"]),
            }
        )

    effs = [p["eff"] for p in points]
    fakes = [p["fake"] for p in points]
    rts = [p["rt"] for p in points]
    bounds = {
        "eff": [float(min(effs)), float(max(effs))],
        "fake": [float(min(fakes)), float(max(fakes))],
        "rt": [float(min(rts)), float(max(rts))],
    }

    out = OUT / "pareto_eff_fake_runtime_3d.html"
    payload = json.dumps(
        {"points": points, "bounds": bounds, "opColors": OPS_COLORS}
    )
    # HTML uses doubled braces for literal JS braces; payload injected once.
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Joint MO-TPE · ε / f / wall 3D</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    --bg: #f7f8fa; --panel: #ffffff; --ink: #1a1d23;
    --muted: #5c6570; --line: #d8dde3; --accent: #1565c0;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: var(--bg); color: var(--ink);
  }
  .wrap { display: grid; grid-template-columns: 300px 1fr; height: 100vh; }
  aside {
    background: var(--panel); border-right: 1px solid var(--line);
    padding: 16px 16px 24px; overflow-y: auto;
  }
  h1 { font-size: 15px; font-weight: 650; margin: 0 0 4px; line-height: 1.3; }
  .sub { font-size: 12px; color: var(--muted); margin-bottom: 16px; line-height: 1.35; }
  .group { border-top: 1px solid var(--line); padding: 12px 0; }
  .group h2 {
    font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--muted); margin: 0 0 8px; font-weight: 600;
  }
  .dual { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 6px; }
  .dual > div { display: flex; flex-direction: column; gap: 3px; }
  .dual span { font-size: 11px; color: var(--muted); }
  input[type="number"], input[type="range"] {
    width: 100%; padding: 5px 6px; border: 1px solid var(--line);
    border-radius: 4px; font-size: 12px; font-variant-numeric: tabular-nums;
  }
  .checks label {
    display: flex; gap: 8px; align-items: center;
    font-size: 13px; margin-bottom: 6px; cursor: pointer;
  }
  .hint { font-size: 11px; color: var(--muted); line-height: 1.35; margin-top: 6px; }
  button {
    width: 100%; margin-top: 8px; padding: 8px 10px;
    border: 1px solid var(--line); background: var(--bg);
    border-radius: 6px; font-size: 13px; cursor: pointer;
  }
  button.primary { background: var(--accent); color: white; border-color: var(--accent); }
  #count { margin-top: 10px; font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
  #plot { width: 100%; height: 100vh; }
</style>
</head>
<body>
<div class="wrap">
  <aside>
    <h1>Joint MO-TPE · 3D scatter</h1>
    <div class="sub">Opt set [0, 32) · ε↑ f↓ wall↓ · drag to rotate</div>

    <div class="group">
      <h2>Efficiency ε_DM (%)</h2>
      <div class="dual">
        <div><span>min</span><input id="effMin" type="number" step="0.1"/></div>
        <div><span>max</span><input id="effMax" type="number" step="0.1"/></div>
      </div>
    </div>
    <div class="group">
      <h2>Fake rate f_DM (%)</h2>
      <div class="dual">
        <div><span>min</span><input id="fakeMin" type="number" step="0.01"/></div>
        <div><span>max</span><input id="fakeMax" type="number" step="0.01"/></div>
      </div>
    </div>
    <div class="group">
      <h2>Runtime wall (s/evt)</h2>
      <div class="dual">
        <div><span>min</span><input id="rtMin" type="number" step="0.1"/></div>
        <div><span>max</span><input id="rtMax" type="number" step="0.1"/></div>
      </div>
    </div>

    <div class="group checks">
      <h2>Layers</h2>
      <label><input id="showOther" type="checkbox"/> All COMPLETE</label>
      <label><input id="showPareto" type="checkbox" checked/> 3D Pareto</label>
      <label><input id="showOps" type="checkbox" checked/> Operating points</label>
      <label><input id="showSurface" type="checkbox" checked/> Smooth Pareto surface</label>
      <div class="hint">
        Surface = smoothed thin-plate RBF: wall ≈ s(ε, f) fit to visible
        Pareto (+ ops) points. Refits when filters change. Needs ≥6 unique (ε,f) sites.
      </div>
      <label style="margin-top:8px">Smoothing λ
        <input id="smoothLam" type="range" min="0" max="2" step="0.05" value="0.35"/>
      </label>
      <div class="hint">λ=<span id="lamVal">0.35</span> (higher = smoother)</div>
    </div>

    <button class="primary" id="applyBtn" type="button">Apply filters</button>
    <button id="resetBtn" type="button">Reset ranges</button>
    <div id="count"></div>
  </aside>
  <div id="plot"></div>
</div>
<script>
const DATA = __PAYLOAD__;

function el(id) { return document.getElementById(id); }
function num(id) { return parseFloat(el(id).value); }

function setDefaults() {
  const b = DATA.bounds;
  el("effMin").value = b.eff[0].toFixed(2);
  el("effMax").value = b.eff[1].toFixed(2);
  el("fakeMin").value = b.fake[0].toFixed(3);
  el("fakeMax").value = b.fake[1].toFixed(3);
  el("rtMin").value = b.rt[0].toFixed(2);
  el("rtMax").value = b.rt[1].toFixed(2);
}

function hover(p) {
  const op = p.op ? ` · ${p.op}` : "";
  return `trial ${p.n}${op}<br>` +
    `ε=${p.eff.toFixed(2)}%  f=${p.fake.toFixed(3)}%  ` +
    `d=${p.dup.toFixed(3)}%  wall=${p.rt.toFixed(2)}s/evt<br>` +
    `seeds=${p.seeds}  branch=${p.branch}  nMeas=${p.nmeas}  χ²=${p.chi2.toFixed(1)}`;
}

function filtered() {
  const effLo = num("effMin"), effHi = num("effMax");
  const fakeLo = num("fakeMin"), fakeHi = num("fakeMax");
  const rtLo = num("rtMin"), rtHi = num("rtMax");
  return DATA.points.filter(p =>
    p.eff >= effLo && p.eff <= effHi &&
    p.fake >= fakeLo && p.fake <= fakeHi &&
    p.rt >= rtLo && p.rt <= rtHi
  );
}

function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => row.slice().concat([b[i]]));
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++)
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    const div = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= div;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map(row => row[n]);
}

function phiThinPlate(r) {
  if (r < 1e-12) return 0;
  return r * r * Math.log(r);
}

function uniqueSites(pts) {
  const map = new Map();
  for (const p of pts) {
    const key = p.eff.toFixed(4) + "|" + p.fake.toFixed(4);
    if (!map.has(key) || p.rt < map.get(key).rt) map.set(key, p);
  }
  return [...map.values()];
}

function fitSurface(pts, lam) {
  const sites = uniqueSites(pts);
  if (sites.length < 6) return null;
  const xs = sites.map(p => p.eff);
  const ys = sites.map(p => p.fake);
  const zs = sites.map(p => p.rt);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const dx = Math.max(x1 - x0, 1e-6), dy = Math.max(y1 - y0, 1e-6);
  const X = xs.map(v => (v - x0) / dx);
  const Y = ys.map(v => (v - y0) / dy);
  const n = sites.length;
  const A = Array.from({length: n}, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const r = Math.hypot(X[i] - X[j], Y[i] - Y[j]);
      const v = phiThinPlate(r);
      A[i][j] = v;
      A[j][i] = v;
    }
    A[i][i] += lam;
  }
  const w = solve(A, zs);
  if (!w) return null;
  return { X, Y, w, x0, dx, y0, dy, xMin: x0, xMax: x1, yMin: y0, yMax: y1 };
}

function evalSurface(fit, eff, fake) {
  const x = (eff - fit.x0) / fit.dx;
  const y = (fake - fit.y0) / fit.dy;
  let s = 0;
  for (let i = 0; i < fit.w.length; i++) {
    const r = Math.hypot(x - fit.X[i], y - fit.Y[i]);
    s += fit.w[i] * phiThinPlate(r);
  }
  return s;
}

function surfaceTrace(fitPts) {
  const lam = Math.max(num("smoothLam"), 1e-4);
  el("lamVal").textContent = num("smoothLam").toFixed(2);
  const fit = fitSurface(fitPts, lam);
  if (!fit) return null;
  const nx = 35, ny = 35;
  const xGrid = [], yGrid = [], zGrid = [];
  for (let i = 0; i < nx; i++)
    xGrid.push(fit.xMin + (fit.xMax - fit.xMin) * i / (nx - 1));
  for (let j = 0; j < ny; j++)
    yGrid.push(fit.yMin + (fit.yMax - fit.yMin) * j / (ny - 1));
  const zTrain = fitPts.map(p => p.rt);
  const zLo = Math.min(...zTrain) - 1;
  const zHi = Math.max(...zTrain) + 1;
  for (let j = 0; j < ny; j++) {
    const row = [];
    for (let i = 0; i < nx; i++) {
      let z = evalSurface(fit, xGrid[i], yGrid[j]);
      if (z < zLo || z > zHi || !isFinite(z)) z = null;
      row.push(z);
    }
    zGrid.push(row);
  }
  return {
    type: "surface",
    name: "Smooth wall ≈ s(ε, f)",
    x: xGrid,
    y: yGrid,
    z: zGrid,
    colorscale: "Viridis",
    opacity: 0.72,
    showscale: true,
    colorbar: { title: "wall (s/evt)", x: 1.02 },
    hovertemplate: "ε=%{x:.2f}%<br>f=%{y:.3f}%<br>wall≈%{z:.2f}s<extra></extra>",
  };
}

function buildTraces(pts) {
  const showOther = el("showOther").checked;
  const showPareto = el("showPareto").checked;
  const showOps = el("showOps").checked;
  const showSurface = el("showSurface").checked;
  const other = pts.filter(p => p.kind === "other");
  const pareto = pts.filter(p => p.kind === "pareto");
  const ops = pts.filter(p => p.kind === "op");
  const traces = [];

  if (showSurface) {
    const fitPts = pts.filter(p => p.kind === "pareto" || p.kind === "op");
    const surf = surfaceTrace(fitPts);
    if (surf) traces.push(surf);
  }
  if (showOther && other.length) {
    traces.push({
      type: "scatter3d", mode: "markers",
      name: `All COMPLETE (${other.length})`,
      x: other.map(p => p.eff), y: other.map(p => p.fake), z: other.map(p => p.rt),
      text: other.map(hover), hoverinfo: "text",
      marker: { size: 3, color: "#90A4AE", opacity: 0.45 },
    });
  }
  if (showPareto && pareto.length) {
    traces.push({
      type: "scatter3d", mode: "markers",
      name: `3D Pareto (${pareto.length})`,
      x: pareto.map(p => p.eff), y: pareto.map(p => p.fake), z: pareto.map(p => p.rt),
      text: pareto.map(hover), hoverinfo: "text",
      marker: {
        size: 5, color: pareto.map(p => p.rt), colorscale: "Viridis",
        colorbar: showSurface ? undefined : { title: "wall (s/evt)", x: 1.02 },
        opacity: 0.95,
      },
    });
  }
  if (showOps) {
    for (const p of ops) {
      traces.push({
        type: "scatter3d", mode: "markers",
        name: `${p.op} (t${p.n})`,
        x: [p.eff], y: [p.fake], z: [p.rt],
        text: [hover(p)], hoverinfo: "text",
        marker: {
          size: 8, color: DATA.opColors[p.op] || "#000", symbol: "diamond",
          line: { width: 1, color: "#000" },
        },
      });
    }
  }
  if (!traces.length) {
    traces.push({ type: "scatter3d", mode: "markers", name: "no points", x: [], y: [], z: [], marker: { size: 1 } });
  }
  return traces;
}

const layout = {
  title: { text: "ε_DM · f_DM · wall — scatter + optional smooth surface", font: { size: 14 } },
  scene: {
    xaxis: { title: "ε_DM (%)" },
    yaxis: { title: "f_DM (%)" },
    zaxis: { title: "wall (s/evt)" },
    aspectmode: "manual",
    aspectratio: { x: 1.2, y: 1.0, z: 0.8 },
  },
  legend: { x: 0.01, y: 0.99 },
  margin: { l: 0, r: 0, t: 40, b: 0 },
  paper_bgcolor: "#f7f8fa",
  uirevision: "keep-camera",
};

function render() {
  const pts = filtered();
  const nPareto = pts.filter(p => p.kind === "pareto").length;
  const nOp = pts.filter(p => p.kind === "op").length;
  const nFit = uniqueSites(pts.filter(p => p.kind === "pareto" || p.kind === "op")).length;
  let msg = `Showing ${pts.length} / ${DATA.points.length} · Pareto ${nPareto} · ops ${nOp}`;
  if (el("showSurface").checked)
    msg += nFit >= 6 ? ` · surface fit on ${nFit} sites` : ` · surface needs ≥6 sites (have ${nFit})`;
  el("count").textContent = msg;
  Plotly.react("plot", buildTraces(pts), layout, {responsive: true});
}

setDefaults();
el("showOther").checked = false;
render();
el("applyBtn").addEventListener("click", render);
el("resetBtn").addEventListener("click", () => { setDefaults(); render(); });
["showOther","showPareto","showOps","showSurface"].forEach(id =>
  el(id).addEventListener("change", render));
el("smoothLam").addEventListener("input", () => {
  el("lamVal").textContent = num("smoothLam").toFixed(2);
  render();
});
["effMin","effMax","fakeMin","fakeMax","rtMin","rtMax"].forEach(id => {
  el(id).addEventListener("change", render);
  el(id).addEventListener("keydown", (e) => { if (e.key === "Enter") render(); });
});
</script>
</body>
</html>
"""
    html = html.replace("__PAYLOAD__", payload)
    out.write_text(html, encoding="utf-8")
    print(f"✓ {out.name}")
    return out


def plot_three_objective_bubble(trials: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    # Bubble size ∝ 1/runtime (faster = larger)
    rts = np.array([max(t["rt"], 0.5) for t in trials])
    sizes = 400.0 / rts
    sc = ax.scatter(
        [t["eff"] for t in trials],
        [t["fake"] for t in trials],
        s=sizes,
        c=[t["dup"] for t in trials],
        cmap="coolwarm",
        alpha=0.55,
        edgecolors="white",
        linewidths=0.3,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$d_{\mathrm{DM}}$ (%)")
    _mark_ops(ax, {t["n"]: t for t in trials})
    ax.set_xlabel(r"$\varepsilon_{\mathrm{DM}}$ (%)")
    ax.set_ylabel(r"$f_{\mathrm{DM}}$ (%)")
    ax.set_title("ε vs f · bubble∝1/runtime · color=d_DM (opt set)")
    ax.set_xlim(45, 100)
    ax.set_ylim(-1, 40)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "three_objective_bubble.png", dpi=150)
    plt.close()
    print("✓ three_objective_bubble.png")


def plot_duplicate_rate(trials: list[dict], eval_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.hist([t["dup"] for t in trials], bins=40, color="#5C6BC0", alpha=0.8)
    ax.set_xlabel(r"$d_{\mathrm{DM}}$ (%)")
    ax.set_ylabel("Count")
    ax.set_title("Opt set duplicate rate")
    ax.grid(True, alpha=0.25, axis="y")

    ax = axes[1]
    ax.hist([r["dup"] for r in eval_rows], bins=30, color="#EF5350", alpha=0.8)
    ax.set_xlabel(r"$d_{\mathrm{DM}}$ (%)")
    ax.set_title("Eval Pareto configs")
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle("Track-level duplicate rate (post-ambi)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "duplicate_rate.png", dpi=150)
    plt.close()
    print("✓ duplicate_rate.png")


def plot_nmeas_min_effect(trials: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vals = sorted({t["nmeas"] for t in trials})
    cmap = plt.cm.plasma
    colors = {v: cmap(i / max(len(vals) - 1, 1)) for i, v in enumerate(vals)}

    ax = axes[0]
    for v in vals:
        pts = [t for t in trials if t["nmeas"] == v]
        ax.scatter(
            [p["eff"] for p in pts],
            [p["fake"] for p in pts],
            c=[colors[v]],
            s=28,
            alpha=0.65,
            label=f"nMeas={v}",
        )
    ax.set_xlabel(r"$\varepsilon_{\mathrm{DM}}$ (%)")
    ax.set_ylabel(r"$f_{\mathrm{DM}}$ (%)")
    ax.set_title("ε–f by nMeasurementsMin")
    ax.set_xlim(45, 100)
    ax.set_ylim(-1, 40)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    data = [[t["eff"] for t in trials if t["nmeas"] == v] for v in vals]
    bp = ax.boxplot(data, positions=vals, widths=0.6, patch_artist=True)
    for patch, v in zip(bp["boxes"], vals):
        patch.set_facecolor(colors[v])
        patch.set_alpha(0.6)
    ax.set_xlabel("nMeasurementsMin")
    ax.set_ylabel(r"$\varepsilon_{\mathrm{DM}}$ (%)")
    ax.set_title("Efficiency")
    ax.grid(True, alpha=0.25, axis="y")

    ax = axes[2]
    data = [[t["fake"] for t in trials if t["nmeas"] == v] for v in vals]
    bp = ax.boxplot(data, positions=vals, widths=0.6, patch_artist=True)
    for patch, v in zip(bp["boxes"], vals):
        patch.set_facecolor(colors[v])
        patch.set_alpha(0.6)
    ax.set_xlabel("nMeasurementsMin")
    ax.set_ylabel(r"$f_{\mathrm{DM}}$ (%)")
    ax.set_title("Fake rate")
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle("nMeasurementsMin effect (opt set)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "nmeas_min_effect.png", dpi=150)
    plt.close()
    print("✓ nmeas_min_effect.png")


def plot_branch_cap_effect(trials: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.viridis
    for br in range(1, 6):
        pts = [t for t in trials if t["branch"] == br]
        if not pts:
            continue
        ax.scatter(
            [p["eff"] for p in pts],
            [p["fake"] for p in pts],
            c=[cmap((br - 1) / 4)],
            s=36,
            alpha=0.65,
            label=f"branch={br} (n={len(pts)})",
        )
    ax.set_xlabel(r"$\varepsilon_{\mathrm{DM}}$ (%)")
    ax.set_ylabel(r"$f_{\mathrm{DM}}$ (%)")
    ax.set_title("Branch cap (numMeasurementsCutOff) · opt set")
    ax.set_xlim(45, 100)
    ax.set_ylim(-1, 40)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "branch_cap_effect.png", dpi=150)
    plt.close()
    print("✓ branch_cap_effect.png")


def plot_branch_cap_marginals(trials: list[dict]) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    for br in range(1, 6):
        pts = [t for t in trials if t["branch"] == br]
        if not pts:
            continue
        color = plt.cm.viridis((br - 1) / 4)
        ax1.boxplot(
            [[p["eff"] for p in pts]],
            positions=[br],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color="black", lw=2),
        )
        ax2.boxplot(
            [[p["fake"] for p in pts]],
            positions=[br],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color="black", lw=2),
        )
        rts = [p["rt"] for p in pts if p["rt"] < 60]
        ax3.boxplot(
            [rts],
            positions=[br],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color="black", lw=2),
        )
    for ax, ylab, title in zip(
        (ax1, ax2, ax3),
        (r"$\varepsilon_{\mathrm{DM}}$ (%)", r"$f_{\mathrm{DM}}$ (%)", "Wall s/event"),
        ("Efficiency", "Fake rate", "Runtime"),
    ):
        ax.set_xlabel("numMeasurementsCutOff")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xticks(range(1, 6))
        ax.grid(True, alpha=0.25, axis="y")
    fig.suptitle("Marginal branch-cap effect (opt set)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "branch_cap_marginals.png", dpi=150)
    plt.close()
    print("✓ branch_cap_marginals.png")


def plot_branch_cap_controlled(trials: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    cmap = plt.cm.viridis
    branch_colors = {i: cmap((i - 1) / 4) for i in range(1, 6)}
    for ax, nmeas_val in zip(axes, [6, 7, 9]):
        subset = [t for t in trials if t["nmeas"] == nmeas_val]
        for br in range(1, 6):
            pts = [t for t in subset if t["branch"] == br]
            if pts:
                ax.scatter(
                    [p["eff"] for p in pts],
                    [p["fake"] for p in pts],
                    c=[branch_colors[br]],
                    s=50,
                    alpha=0.65,
                    edgecolors="white",
                    linewidths=0.3,
                    label=f"br={br}" if nmeas_val == 6 else None,
                )
        ax.set_xlabel(r"$\varepsilon_{\mathrm{DM}}$ (%)")
        ax.set_title(f"nMeasurementsMin = {nmeas_val}")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(45, 100)
        ax.set_ylim(-1, 40)
        ax.text(
            0.95,
            0.95,
            f"n={len(subset)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            color="gray",
        )
    axes[0].set_ylabel(r"$f_{\mathrm{DM}}$ (%)")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=branch_colors[i],
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=str(i),
        )
        for i in range(1, 6)
    ]
    fig.legend(
        handles=handles,
        title="Branch Cap",
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
    )
    fig.suptitle(
        "Branch cap controlled for nMeasurementsMin · joint MO-TPE",
        fontsize=13,
    )
    fig.subplots_adjust(right=0.88)
    fig.savefig(OUT / "branch_cap_controlled.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ branch_cap_controlled.png")


def plot_param_marginals(trials: list[dict]) -> None:
    params = [
        ("seeds", "num_seeds_per_spm"),
        ("chi2", r"$\chi^2$ measurement"),
        ("branch", "branch cap"),
        ("nmeas", "nMeasurementsMin"),
        ("holes", "maxHolesAndOutliers"),
        ("ptmin", r"CKF $p_T$ min (GeV)"),
        ("impact", r"seed impactMax (mm)"),
        ("seed_pt", r"seed minPt (GeV)"),
    ]
    # Leave room on the right for a shared colorbar (avoids overlapping axes).
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    sc = None
    for ax, (key, label) in zip(axes.ravel(), params):
        sc = ax.scatter(
            [t[key] for t in trials],
            [t["eff"] for t in trials],
            c=[t["fake"] for t in trials],
            cmap="magma_r",
            s=18,
            alpha=0.65,
            vmin=0,
            vmax=max(40.0, max(t["fake"] for t in trials)),
        )
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\varepsilon_{\mathrm{DM}}$ (%)")
        ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(r"$f_{\mathrm{DM}}$ (%)")
    fig.suptitle("Parameter marginals · ε vs param (color=f) · opt set", fontsize=13)
    fig.savefig(OUT / "pareto_param_marginals.png", dpi=150)
    plt.close()
    print("✓ pareto_param_marginals.png")


def plot_opt_vs_eval_gap(eval_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.hist([r["de"] for r in eval_rows], bins=30, color="#5C6BC0", alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(
        float(np.mean([r["de"] for r in eval_rows])),
        color="#C62828",
        ls="--",
        label="mean",
    )
    ax.set_xlabel(r"$\Delta\varepsilon$ = eval − opt (pp)")
    ax.set_ylabel("Count")
    ax.set_title("Efficiency transfer")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")

    ax = axes[1]
    ax.hist([r["df"] for r in eval_rows], bins=30, color="#EF5350", alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(
        float(np.mean([r["df"] for r in eval_rows])),
        color="#1565C0",
        ls="--",
        label="mean",
    )
    ax.set_xlabel(r"$\Delta f$ = eval − opt (pp)")
    ax.set_title("Fake-rate transfer")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle("Opt→eval generalization (139 Pareto configs)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "opt_vs_eval_gap.png", dpi=150)
    plt.close()
    print("✓ opt_vs_eval_gap.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trials = _load_trials(TRIALS)
    eval_rows = _load_eval(EVAL)
    print(f"Loaded {len(trials)} COMPLETE opt trials, {len(eval_rows)} eval Pareto rows")
    print(f"Writing to {OUT}")

    plot_eff_vs_fake(trials, eval_rows)
    plot_eff_vs_runtime(trials, eval_rows)
    plot_fake_vs_runtime(trials)
    plot_eff_fake_runtime_3d_interactive(trials)
    plot_three_objective_bubble(trials)
    plot_duplicate_rate(trials, eval_rows)
    plot_nmeas_min_effect(trials)
    plot_branch_cap_effect(trials)
    plot_branch_cap_marginals(trials)
    plot_branch_cap_controlled(trials)
    plot_param_marginals(trials)
    plot_opt_vs_eval_gap(eval_rows)
    print("Done.")


if __name__ == "__main__":
    main()
