"""Collect Pareto fronts and per-run DM metrics on NERSC for the experiment-log
backfill. Host python for the CSVs; shifter+PyROOT for performance ROOT files
(same read as scripts/ehvi_sweep.py:read_run)."""
import csv, glob, os, subprocess, sys

S = "/pscratch/sd/m/mussonm/cckf"
IMG = "ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0"
ROOT_PREFIX = "/spack/opt/spack/linux-x86_64/root-6.38.00-fkp6aauipwq6nh2lkh23427cswpjirnh"
SPACK_PY = "/spack/opt/spack/linux-x86_64/python-3.13.11-awxtqzerpdzhatylv3uagd35ebciqs3o/bin/python3"
SNIP = """
import sys, ROOT
ROOT.gROOT.SetBatch(True); ROOT.gErrorIgnoreLevel = ROOT.kFatal
for p in sys.argv[1:]:
    tf = ROOT.TFile.Open(p)
    if not tf or tf.IsZombie():
        print(p, "UNREADABLE"); continue
    eff = tf.Get("trackeff_vs_eta")
    tot = eff.GetTotalHistogram().Integral(); pas = eff.GetPassedHistogram().Integral()
    fake = float(tf.Get("fakeratio_tracks")[0])
    dup = tf.Get("duplicationRatio_tracks")
    dup = float(dup[0]) if dup else float("nan")
    print(p, f"eff={pas/tot if tot else -1:.4f} fake={fake:.4f} dup={dup:.4f} n_truth={tot:.0f}")
"""

print("=== Pareto CSV fronts ===")
for f in sorted(glob.glob(f"{S}/results/pareto_*.csv")):
    rows = list(csv.DictReader(open(f)))
    if not rows or "efficiency" not in rows[0]:
        print(f"== {os.path.basename(f)}: columns {list(rows[0].keys()) if rows else 'empty'}")
        continue
    rows = [r for r in rows if r.get("efficiency") not in ("", None) and r.get("fake_rate") not in ("", None)]
    pts = [(float(r["efficiency"]), float(r["fake_rate"]), r.get("tau_g"), r.get("tau_v"), r.get("source", "")) for r in rows]
    front = sorted(p for p in pts if not any(q[0] >= p[0] and q[1] <= p[1] and q != p for q in pts))
    nfail = sum(1 for p in pts if str(p[4]).startswith("fail"))
    print(f"== {os.path.basename(f)}: {len(pts)} pts, {len(front)} on front, {nfail} failed")
    for p in front:
        print(f"   eff={p[0]:.4f} fake={p[1]:.4f} g={p[2]} v={p[3]} {p[4]}")

print("\n=== per-run DM metrics (performance_finding_ambi.root) ===")
files = []
for d in ["runs_classical", "runs_nscan", "runs_resolvers", "runs_t3", "runs_maj", "runs_pure"]:
    for r in sorted(glob.glob(f"{S}/{d}/*")):
        for name in ["performance_finding_ambi.root", "performance_finding_ambi_scorebased.root", "performance_finding_ckf.root", "performance_finding.root"]:
            p = os.path.join(r, name)
            if os.path.exists(p):
                files.append(p); break
        else:
            print(f"{r}: no performance ROOT; contents={sorted(os.listdir(r))[:6]}")
if files:
    out = subprocess.run(["shifter", f"--image={IMG}", f"--env=PYTHONPATH={ROOT_PREFIX}/lib/root",
                          f"--env=LD_LIBRARY_PATH={ROOT_PREFIX}/lib/root", "--", SPACK_PY, "-c", SNIP] + files,
                         capture_output=True, text=True)
    print(out.stdout); print(out.stderr[-2000:], file=sys.stderr)
