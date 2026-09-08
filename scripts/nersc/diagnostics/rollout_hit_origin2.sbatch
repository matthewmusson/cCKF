#!/bin/bash
#SBATCH --job-name=rollout_origin2
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=atlas
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
# Follow-up to rollout_hit_origin: the hits CSV 'step' column is written from
# trackStatesReversed(), so step 0 is the OUTERMOST state, max step is the
# launch. Classify EVERY rollout state (holes included) by surface against the
# launching branch, test the propagation direction, and dump examples.
module load python
export PYTHONUNBUFFERED=1
cd /global/cfs/cdirs/atlas/mussonm/cCKF
python3 - <<'PYEOF'
import numpy as np, pandas as pd, sys, time
import pyarrow.parquet as pq
sys.path.insert(0, ".")
from cckf.tier3_walker import classify_event
pd.set_option("display.width", 200)
S = "/pscratch/sd/m/mussonm/cckf"; E = 4
pqp = f"{S}/reexpanded/expanded_event{E:09d}.parquet"
states = classify_event(pqp)
surf = pq.read_table(pqp, columns=["seed_id", "step_k", "surface_id"]).to_pandas()
surf = surf.drop_duplicates(["seed_id", "step_k"]).rename(columns={"surface_id": "module_id"})
states = states.merge(surf, on=["seed_id", "step_k"], how="left")

hits = pd.read_csv(f"{S}/tier3/hits/event{E:09d}-rollout-hits.csv")
wl = pd.read_csv(f"{S}/tier3/worklists/event{E:09d}-rollout-worklist.csv")
wl = wl.rename(columns={"step_k": "k", "geometry_id": "launch_gid"})
hits = hits.merge(wl[["rollout_id", "seed_id", "k", "launch_gid"]], on="rollout_id", how="inner")
anc = states[["seed_id", "step_k", "state_class", "sel_hit"]].rename(
    columns={"step_k": "k", "state_class": "anchor", "sel_hit": "launch_sel_hit"})
hits = hits.merge(anc, on=["seed_id", "k"], how="left")
g = hits["geometry_id"].to_numpy().astype(np.uint64)
hits["vol"] = ((g >> np.uint64(56)) & np.uint64(0xFF)).astype(np.int64)
hits["lay"] = ((g >> np.uint64(36)) & np.uint64(0xFFF)).astype(np.int64)
hits["mod"] = ((g >> np.uint64(8)) & np.uint64(0xFFFFF)).astype(np.int64)
# track order: 0 = launch, increasing outward (undo trackStatesReversed)
nst = hits.groupby("rollout_id")["step"].transform("max")
hits["t"] = nst - hits["step"]

print("launch (t=0) surface == worklist surface:", float((hits.loc[hits.t == 0, "geometry_id"] == hits.loc[hits.t == 0, "launch_gid"]).mean()))
print("launch (t=0) meas_id == branch sel_hit at k (when branch held a hit):",
      float((hits.loc[(hits.t == 0) & (hits.launch_sel_hit >= 0), "meas_id"] == hits.loc[(hits.t == 0) & (hits.launch_sel_hit >= 0), "launch_sel_hit"]).mean()))

bs = states[["seed_id", "step_k", "volume_id", "layer_id", "module_id"]].rename(
    columns={"volume_id": "vol", "layer_id": "lay", "module_id": "mod", "step_k": "surf_step"})
bs = bs.dropna().astype({"mod": np.int64}).drop_duplicates(["seed_id", "vol", "lay", "mod"], keep="first")
hits = hits.merge(bs, on=["seed_id", "vol", "lay", "mod"], how="left")
bh = states.loc[states.sel_hit >= 0, ["seed_id", "step_k", "sel_hit"]].rename(
    columns={"sel_hit": "meas_id", "step_k": "held_step"})
hits = hits.merge(bh, on=["seed_id", "meas_id"], how="left")

ss = hits["surf_step"]; k = hits["k"]
hits["surf_cat"] = np.select(
    [ss.notna() & (ss < k), ss.notna() & (ss == k), ss.notna() & (ss > k)],
    ["branch_past", "launch_surface", "branch_future"], default="never_visited")
hits["is_hit"] = hits.meas_id >= 0
print("\n=== ALL rollout states (holes + hits) by surface category x anchor ===")
print(pd.crosstab([hits["anchor"], hits["is_hit"]], hits["surf_cat"], margins=True).to_string())

# direction test: volume / layer at launch vs at the far end
first = hits[hits.t == 0].set_index("rollout_id")[["vol", "lay"]]
last = hits.loc[hits.t == nst].set_index("rollout_id")[["vol", "lay"]]
d = first.join(last, lsuffix="_launch", rsuffix="_end")
print("\n=== propagation direction (launch -> far end) ===")
print("vol_launch < vol_end:", float((d.vol_launch < d.vol_end).mean()),
      " vol equal:", float((d.vol_launch == d.vol_end).mean()),
      " vol_launch > vol_end:", float((d.vol_launch > d.vol_end).mean()))
same = d[d.vol_launch == d.vol_end]
print("within same volume: lay_launch < lay_end:", float((same.lay_launch < same.lay_end).mean()),
      " equal:", float((same.lay_launch == same.lay_end).mean()),
      " lay_launch > lay_end:", float((same.lay_launch > same.lay_end).mean()))
print("steps per rollout: ", (nst + 1).groupby(hits.rollout_id).first().describe().round(1).to_dict())

# examples: 4 tip rollouts with >=2 hits held by the branch, 2 divergence rollouts
hits["held"] = hits.held_step.notna()
n_held = hits.groupby("rollout_id")["held"].sum()
ex_tip = hits[(hits.anchor == "tip") & hits.rollout_id.isin(n_held[n_held >= 2].index)].rollout_id.unique()[:3]
ex_div = hits[(hits.anchor == "divergence")].rollout_id.unique()[:2]
cols = ["t", "vol", "lay", "mod", "meas_id", "surf_cat", "held_step"]
for rid in list(ex_tip) + list(ex_div):
    r = hits[hits.rollout_id == rid].sort_values("t")
    sid = int(r.seed_id.iloc[0]); kk = int(r.k.iloc[0]); a = r.anchor.iloc[0]
    b = states[states.seed_id == sid].sort_values("step_k")
    print(f"\n--- rollout {rid}: seed {sid}, launch k={kk} ({a}) ---")
    print("branch states (step_k, vol, lay, mod, sel_hit, class):")
    print(b[["step_k", "volume_id", "layer_id", "module_id", "sel_hit", "state_class"]].to_string(index=False))
    print("rollout states (t=0 launch):")
    print(r[cols].to_string(index=False))
PYEOF
echo "ORIGIN2_RC=$?"
