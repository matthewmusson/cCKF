"""Extract track candidates + truth from a run dir into parquet for offline
ambiguity-resolver experiments.

Reads (all written by the standard harness writers):
  trackstates_ckf.root   — per-track per-state: module ids, local hit position
                           (hit identity), per-state truth barcode
  tracksummary_ckf.root  — per-track: nMeasurements, chi2Sum, NDF, holes,
                           majority particle, nMajorityHits, trackClassification
  performance_finding_ckf.root — matchingdetails tree: the selected truth
                           particles (efficiency denominator) + matched flag

Writes <out>/tracks.parquet and <out>/particles.parquet.
Run inside shifter with spack python + PyROOT.
"""
import sys
from pathlib import Path

import ROOT
import csv

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

run_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else run_dir


def barcode(pri, sec, par, gen, sub):
    return f"{pri}|{sec}|{par}|{gen}|{sub}"


# --- tracksummary: track-level fields -------------------------------------
tf = ROOT.TFile.Open(str(run_dir / "tracksummary_ckf.root"))
ts = tf.Get("tracksummary")
rows = {}
for entry in ts:
    for i in range(len(entry.track_nr)):
        tn = int(entry.track_nr[i])
        rows[tn] = dict(
            track_nr=tn,
            nMeasurements=int(entry.nMeasurements[i]),
            nHoles=int(entry.nHoles[i]),
            nOutliers=int(entry.nOutliers[i]),
            nSharedHits=int(entry.nSharedHits[i]),
            chi2Sum=float(entry.chi2Sum[i]),
            NDF=int(entry.NDF[i]),
            nMajorityHits=int(entry.nMajorityHits[i]),
            majority=barcode(
                entry.majorityParticleId_vertex_primary[i],
                entry.majorityParticleId_vertex_secondary[i],
                entry.majorityParticleId_particle[i],
                entry.majorityParticleId_generation[i],
                entry.majorityParticleId_sub_particle[i],
            ),
            classification=int(entry.trackClassification[i]),
        )
tf.Close()

# --- trackstates: hit identity per track ----------------------------------
tf = ROOT.TFile.Open(str(run_dir / "trackstates_ckf.root"))
st = tf.Get("trackstates")
stateTypes = {}
for entry in st:
    tn = int(entry.track_nr)
    keys = []
    import math
    for i in range(len(entry.volume_id)):
        stype = int(entry.stateType[i])
        stateTypes[stype] = stateTypes.get(stype, 0) + 1
        # 0 = measurement; outliers/holes/material carry no shared hit
        if stype != 0 or math.isnan(entry.l_x_hit[i]):
            continue
        keys.append(
            f"{int(entry.volume_id[i])}:{int(entry.layer_id[i])}:"
            f"{int(entry.module_id[i])}:{entry.l_x_hit[i]:.4f}:{entry.l_y_hit[i]:.4f}"
        )
    if tn in rows:
        rows[tn]["hit_keys"] = ";".join(keys)
        rows[tn]["n_hit_keys"] = len(keys)
tf.Close()

tracks = list(rows.values())
print("stateType counts:", stateTypes)
print("tracks:", len(tracks))
mismatch = sum(1 for r in tracks if r.get("n_hit_keys") != r["nMeasurements"])
print(f"hit-key/nMeas mismatches: {mismatch}")

# --- matchingdetails: selected truth particles ----------------------------
tf = ROOT.TFile.Open(str(run_dir / "performance_finding_ckf.root"))
md = tf.Get("matchingdetails")
prow = []
for entry in md:
    prow.append(dict(
        particle=barcode(
            entry.particle_id_vertex_primary,
            entry.particle_id_vertex_secondary,
            entry.particle_id_particle,
            entry.particle_id_generation,
            entry.particle_id_sub_particle,
        ),
        matched=bool(entry.matched),
    ))
tf.Close()
print("particles:", len(prow), "matched:", sum(1 for r in prow if r["matched"]))

out_dir.mkdir(parents=True, exist_ok=True)
def wcsv(path, recs):
    if not recs:
        return
    keys = list(recs[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)
wcsv(out_dir / "tracks.csv", tracks)
wcsv(out_dir / "particles.csv", prow)
print("wrote", out_dir / "tracks.csv")
