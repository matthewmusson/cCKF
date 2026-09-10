"""Follow one particle through the edm4hep input: its MCParticle entry and its
SimTrackerHit entries in the six ODD tracker readout collections.
Usage: python3 trail_edm4hep.py <edm4hep.root> <event_index> <tpx> <tpy> <tpz>
The particle is identified by the momentum at its first simhit (from
simhits.csv); we pick the generator-status-1 charged MCParticle whose
production momentum is closest.
"""
import sys, uproot, numpy as np, awkward as ak
path, ev = sys.argv[1], int(sys.argv[2])
target = np.array([float(x) for x in sys.argv[3:6]])
t = uproot.open(path)["events"]
def col(c, fields):
    a = t.arrays([f"{c}/{c}.{f}" for f in fields], entry_start=ev, entry_stop=ev + 1, library="ak")[0]
    return {f: ak.to_numpy(a[f"{c}/{c}.{f}"]) for f in fields}
mc = col("MCParticles", ["PDG", "generatorStatus", "simulatorStatus", "charge", "mass", "time",
                         "vertex.x", "vertex.y", "vertex.z", "endpoint.x", "endpoint.y", "endpoint.z",
                         "momentum.x", "momentum.y", "momentum.z", "parents_begin", "parents_end",
                         "daughters_begin", "daughters_end"])
n = len(mc["PDG"])
p = np.stack([mc["momentum.x"], mc["momentum.y"], mc["momentum.z"]], axis=1)
cand = (mc["generatorStatus"] == 1) & (mc["charge"] != 0)
d = np.linalg.norm(p - target, axis=1); d[~cand] = np.inf
i = int(np.argmin(d))
print(f"event {ev}: {n} MCParticles; generatorStatus==1 & charged: {int(cand.sum())}")
print(f"closest to first-simhit momentum {target.round(3).tolist()} GeV: MCParticle index {i}, |dp| = {d[i]:.4f} GeV")
print("MCParticles[%d]:" % i)
for k, v in mc.items():
    print(f"  {k:<22} {v[i]}")
pt = float(np.hypot(p[i, 0], p[i, 1])); eta = float(np.arcsinh(p[i, 2] / pt))
print(f"  derived: pT = {pt:.4f} GeV, eta = {eta:.4f}")
print()
tot = 0
for c in ["PixelBarrelReadout", "PixelEndcapReadout", "ShortStripBarrelReadout",
          "ShortStripEndcapReadout", "LongStripBarrelReadout", "LongStripEndcapReadout"]:
    h = col(c, ["cellID", "eDep", "time", "pathLength", "position.x", "position.y", "position.z",
                "momentum.x", "momentum.y", "momentum.z"])
    rel = t.arrays([f"_{c}_particle/_{c}_particle.index"], entry_start=ev, entry_stop=ev + 1, library="ak")[0]
    idx = ak.to_numpy(rel[f"_{c}_particle/_{c}_particle.index"])
    m = np.where(idx == i)[0]
    print(f"{c}: {len(idx)} hits in the event, {len(m)} from our particle")
    for j in m:
        r = float(np.hypot(h["position.x"][j], h["position.y"][j]))
        print(f"  hit {j:>6}: cellID={h['cellID'][j]}  pos=({h['position.x'][j]:.2f}, {h['position.y'][j]:.2f}, {h['position.z'][j]:.2f}) mm  r={r:.1f}  "
              f"eDep={h['eDep'][j]:.3e} GeV  t={h['time'][j]:.3f} ns  pathLength={h['pathLength'][j]:.4f}  "
              f"p=({h['momentum.x'][j]:.3f}, {h['momentum.y'][j]:.3f}, {h['momentum.z'][j]:.3f})")
    tot += len(m)
print(f"total tracker simhits for this particle in edm4hep: {tot}")
