#!/usr/bin/env python3
"""Drive the (tau_g, tau_v) Pareto sweep on NERSC and emit the sweep CSV.

Produces exactly the columns scripts/pareto_sweep.py expects:

    tau_g, tau_v, efficiency, fake_rate, duplicate_rate_pre_ambi,
    duplicate_rate_post_ambi, runtime_per_event_s, gate_calls, value_calls,
    wall_seconds

Metric definitions follow spec section 3 and CLAUDE.md:
  efficiency               particle-level eps_DM, from trackeff_vs_eta
                           (passed/total over the TEfficiency)
  fake_rate                TRACK-level f_DM, the fakeratio_tracks scalar.
                           NOT fakeratio_particles.
  duplicate_rate_pre_ambi  duplicateratio_tracks from the CKF-stage file,
                           i.e. before ambiguity resolution
  runtime_per_event_s      CKF wall time from the run's timing.csv

Usage
-----
    python scripts/pareto_sweep_nersc.py --runs-dir $SCRATCH/cckf/runs \
        --out $SCRATCH/cckf/results/pareto_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path


def _tefficiency_integral(tf, name):
    """passed/total over a TEfficiency, i.e. the aggregate rate."""
    o = tf.Get(name)
    if not o:
        return None
    try:
        total = o.GetTotalHistogram().Integral()
        passed = o.GetPassedHistogram().Integral()
        return passed / total if total > 0 else None
    except Exception:
        return None


def _scalar(tf, name):
    o = tf.Get(name)
    if not o:
        return None
    try:
        return float(o[0])
    except Exception:
        return None


def read_run(run_dir: Path) -> dict:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kFatal

    out: dict = {}
    ckf = run_dir / "performance_finding_ckf.root"
    ambi = run_dir / "performance_finding_ambi.root"

    if ckf.exists():
        tf = ROOT.TFile.Open(str(ckf))
        out["duplicate_rate_pre_ambi"] = _scalar(tf, "duplicateratio_tracks")
        out["efficiency_pre_ambi"] = _tefficiency_integral(tf, "trackeff_vs_eta")
        tf.Close()
    if ambi.exists():
        tf = ROOT.TFile.Open(str(ambi))
        # Post-ambiguity is the reportable operating point.
        out["efficiency"] = _tefficiency_integral(tf, "trackeff_vs_eta")
        out["fake_rate"] = _scalar(tf, "fakeratio_tracks")
        out["duplicate_rate_post_ambi"] = _scalar(tf, "duplicateratio_tracks")
        tf.Close()

    # Latency and MLP call counts come from the per-event timing CSV written
    # by CckfTrackFindingAlgorithm.
    timing = run_dir / "timing.csv"
    if timing.exists():
        with open(timing) as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            r = rows[-1]
            def num(k, default=0.0):
                try:
                    return float(r.get(k, default) or default)
                except (TypeError, ValueError):
                    return default
            # meas_selection_ns covers the gate; value_inference_ns the value fn.
            out["gate_calls"] = num("gate_inference_calls")
            out["value_calls"] = num("value_inference_calls")
            total_ns = num("meas_selection_ns") + num("value_inference_ns")
            out["runtime_per_event_s"] = total_ns / 1e9
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default=r"sweep_g(?P<g>[0-9p.]+)_v(?P<v>[0-9p.]+)")
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    rx = re.compile(args.pattern)
    fields = ["tau_g", "tau_v", "efficiency", "fake_rate",
              "duplicate_rate_pre_ambi", "duplicate_rate_post_ambi",
              "runtime_per_event_s", "gate_calls", "value_calls", "wall_seconds"]

    rows = []
    for d in sorted(runs.iterdir()):
        if not d.is_dir():
            continue
        m = rx.match(d.name)
        if not m:
            continue
        rec = {k: "" for k in fields}
        rec["tau_g"] = float(m.group("g").replace("p", "."))
        rec["tau_v"] = float(m.group("v").replace("p", "."))
        rec.update({k: v for k, v in read_run(d).items() if k in fields})
        wall = d / "wall_seconds"
        if wall.exists():
            rec["wall_seconds"] = wall.read_text().strip()
        rows.append(rec)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["tau_g"], x["tau_v"])):
            w.writerow(r)
    print(f"wrote {args.out} with {len(rows)} sweep points")
    for r in rows:
        print(f"  tau_g={r['tau_g']} tau_v={r['tau_v']} eff={r['efficiency']} "
              f"fake={r['fake_rate']} dup_pre={r['duplicate_rate_pre_ambi']} "
              f"t={r['runtime_per_event_s']}")


if __name__ == "__main__":
    main()
