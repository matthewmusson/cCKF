import sys, time
sys.path.insert(0, "/global/cfs/cdirs/atlas/mussonm/cCKF")
from expansion import run_expansion
pilot, ev, out, nch, ci = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
digi = ("/global/cfs/cdirs/atlas/mussonm/ODD_v5/install/share/"
        "OpenDataDetector/config/odd-digi-geometric-config.json")
t0 = time.time()
df = run_expansion(f"{pilot}/trackstates_ckf.root", pilot, ev, out,
                   digi_config_path=digi, n_chunks=nch, chunk_idx=ci)
print(f"ROWS={len(df)} SECS={round(time.time()-t0,1)} CHUNK={ci}/{nch}")
