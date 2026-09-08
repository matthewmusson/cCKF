import sys, glob, time
sys.path.insert(0, "/global/cfs/cdirs/atlas/mussonm/cCKF")
from cckf.cache import build_gate_cache
paths = sorted(glob.glob(sys.argv[1] + "/expanded_event*.parquet"))[:2]
print("testing cache build on:", [p.split("/")[-1] for p in paths], flush=True)
t0 = time.time()
meta = build_gate_cache(paths, sys.argv[2])
print(f"OK secs={time.time()-t0:.1f}")
print("meta:", {k: v for k, v in list(meta.items())[:12]})
