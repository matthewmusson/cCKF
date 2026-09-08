import sys, time
sys.path.insert(0, "/global/cfs/cdirs/atlas/mussonm/cCKF")
import expansion as E
pilot, ev = sys.argv[1], int(sys.argv[2])
def t(label, fn, *a, **k):
    t0=time.time(); r=fn(*a,**k); print(f"  {label:<28} {time.time()-t0:7.1f}s", flush=True); return r
print("PROFILE", flush=True)
st  = t("load_trackstates",  E.load_trackstates, f"{pilot}/trackstates_ckf.root", ev)
me  = t("load_measurements", E.load_measurements, pilot, ev)
pc  = t("load_predicted_cov",E.load_predicted_cov, pilot, ev)
ce  = t("load_cells",        E.load_cells, pilot, ev)
sh  = t("load_simhits",      E.load_simhits, pilot, ev)
mm  = t("load_meas_simhit",  E.load_measurement_simhit_map, pilot, ev)
ex  = t("expand_trackstates",E.expand_trackstates, st, me, predicted_cov=pc)
print(f"  -> expanded rows: {len(ex):,}", flush=True)
bh  = t("compute_branch_history", E.compute_branch_history, ex)
ct  = t("cluster_features_table", E.compute_cluster_features_table, ce)
