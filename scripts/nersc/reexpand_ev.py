import sys, time
sys.path.insert(0, "/global/cfs/cdirs/atlas/mussonm/cCKF")
from expansion import run_expansion
pilot, local_ev, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
digi = ("/global/cfs/cdirs/atlas/mussonm/ODD_v5/install/share/"
        "OpenDataDetector/config/odd-digi-geometric-config.json")
t0 = time.time()
df = run_expansion(f"{pilot}/trackstates_ckf.root", pilot, local_ev, out,
                   digi_config_path=digi)
print(f"ROWS={len(df)} COLS={len(df.columns)} SECS={round(time.time()-t0,1)}")
print("WROTE", out)
