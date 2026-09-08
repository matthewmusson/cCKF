#!/bin/bash
#SBATCH --job-name=rollout_origin
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=atlas
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
# Where do rollout hits come from? Event 4, unbounded (n=inf) rollouts.
# Every ACCEPTED rollout hit is classified against the launching branch:
#   launch_same_hit        step 0, the hit the branch itself holds at k
#   launch_other_hit       step 0, a different hit on the launch surface
#   past_hit_recollected   step>0, a hit the branch holds at step <= k
#   branch_pick_after_k    step>0, same hit the branch picked at step > k
#   branch_surf_after_k    step>0, branch visited that surface after k but
#                          picked a different hit / holed there
#   branch_surf_before_k   step>0, surface the branch visited at step <= k
#   skipped_before_tip     surface the branch never visited, but the
#                          rollout reaches it before its last branch-matched
#                          surface (re-propagation finds MORE)
#   past_tip               surface the branch never visited, after the last
#                          branch-matched surface (beyond the branch's end)
module load python
export PYTHONUNBUFFERED=1
cd /global/cfs/cdirs/atlas/mussonm/cCKF
python3 - <<'PYEOF'
import numpy as np, pandas as pd, sys, time
import pyarrow.parquet as pq
sys.path.insert(0, ".")
from cckf.tier3_walker import classify_event
S = "/pscratch/sd/m/mussonm/cckf"; E = 4
pqp = f"{S}/reexpanded/expanded_event{E:09d}.parquet"
t0 = time.time()
states = classify_event(pqp)
surf = pq.read_table(pqp, columns=["seed_id", "step_k", "surface_id"]).to_pandas().rename(columns={"surface_id": "module_id"})
surf = surf.drop_duplicates(["seed_id", "step_k"])
states = states.merge(surf, on=["seed_id", "step_k"], how="left")
print(f"states {len(states):,} classified in {time.time()-t0:.0f}s")

hits = pd.read_csv(f"{S}/tier3/hits/event{E:09d}-rollout-hits.csv")
wl = pd.read_csv(f"{S}/tier3/worklists/event{E:09d}-rollout-worklist.csv")
wl = wl.rename(columns={"step_k": "k", "geometry_id": "launch_gid"})
hits = hits.merge(wl[["rollout_id", "seed_id", "k", "launch_gid"]], on="rollout_id", how="inner")
anc = states[["seed_id", "step_k", "state_class"]].rename(columns={"step_k": "k", "state_class": "anchor"})
hits = hits.merge(anc, on=["seed_id", "k"], how="left")
g = hits["geometry_id"].to_numpy().astype(np.uint64)
hits["vol"] = ((g >> np.uint64(56)) & np.uint64(0xFF)).astype(np.int64)
hits["lay"] = ((g >> np.uint64(36)) & np.uint64(0xFFF)).astype(np.int64)
hits["mod"] = ((g >> np.uint64(8)) & np.uint64(0xFFFFF)).astype(np.int64)
print(f"rollout rows {len(hits):,}, rollouts {hits.rollout_id.nunique():,}, "
      f"accepted {(hits.meas_id>=0).sum():,}")
print("step-0 surface == launch surface:", float((hits.loc[hits.step==0, "geometry_id"] == hits.loc[hits.step==0, "launch_gid"]).mean()))

# branch surfaces (vol, lay, mod) -> step visited; branch hits -> step held
bs = states[["seed_id", "step_k", "volume_id", "layer_id", "module_id"]].rename(
    columns={"volume_id": "vol", "layer_id": "lay", "module_id": "mod", "step_k": "surf_step"})
bs = bs.dropna().astype({"mod": np.int64}).drop_duplicates(["seed_id", "vol", "lay", "mod"], keep="first")
hits = hits.merge(bs, on=["seed_id", "vol", "lay", "mod"], how="left")
bh = states.loc[states.sel_hit >= 0, ["seed_id", "step_k", "sel_hit"]].rename(
    columns={"sel_hit": "meas_id", "step_k": "held_step"})
hits = hits.merge(bh, on=["seed_id", "meas_id"], how="left")

# last rollout step that lands on a branch surface visited AFTER k
after = hits["surf_step"].notna() & (hits["surf_step"] > hits["k"])
last_match = hits.loc[after].groupby("rollout_id")["step"].max().rename("last_match")
hits = hits.merge(last_match, on="rollout_id", how="left")
hits["last_match"] = hits["last_match"].fillna(0)

acc = hits[hits.meas_id >= 0].copy()
k = acc["k"]; st = acc["step"]; hs = acc["held_step"]; ss = acc["surf_step"]
cat = np.select(
    [
        (st == 0) & (hs == k),
        (st == 0),
        hs.notna() & (hs <= k),
        hs.notna() & (hs > k),
        ss.notna() & (ss > k),
        ss.notna() & (ss <= k),
        st < acc["last_match"],
    ],
    ["launch_same_hit", "launch_other_hit", "past_hit_recollected",
     "branch_pick_after_k", "branch_surf_after_k", "branch_surf_before_k",
     "skipped_before_tip"],
    default="past_tip",
)
acc["cat"] = cat

print("\n=== accepted rollout hits by category x anchor class ===")
print(pd.crosstab(acc["cat"], acc["anchor"], margins=True).to_string())

per = acc.groupby(["rollout_id", "anchor"])["cat"].value_counts().unstack(fill_value=0).reset_index()
for c in ["launch_same_hit", "launch_other_hit", "past_hit_recollected", "skipped_before_tip", "past_tip"]:
    if c not in per: per[c] = 0
print("\n=== per-rollout distributions ===")
for a in ["tip", "divergence"]:
    p = per[per.anchor == a]
    print(f"[{a}] rollouts {len(p):,}")
    print("  launch_same_hit per rollout:", p["launch_same_hit"].value_counts().sort_index().to_dict())
    print("  past_hit_recollected      :", p["past_hit_recollected"].value_counts().sort_index().head(6).to_dict())
    print("  skipped_before_tip        :", p["skipped_before_tip"].value_counts().sort_index().head(8).to_dict())
    print("  past_tip                  :", p["past_tip"].value_counts().sort_index().head(8).to_dict())

# suffix branches (never diverge): tip rollouts only
has_div = states.groupby("seed_id")["state_class"].apply(lambda s: (s == "divergence").any())
suffix = set(has_div.index[~has_div])
ps = per[(per.anchor == "tip") & per.rollout_id.isin(wl.loc[wl.seed_id.isin(suffix), "rollout_id"])]
ps = ps.copy(); ps["extra"] = ps["launch_same_hit"] + ps["launch_other_hit"] + ps["past_hit_recollected"] + ps["past_tip"]
print(f"\n=== suffix-branch tip rollouts: {len(ps):,} ===")
print("  extra hits (launch + recollected + past_tip) per rollout:", ps["extra"].value_counts().sort_index().head(8).to_dict())
print("  past_tip per rollout:", ps["past_tip"].value_counts().sort_index().head(8).to_dict())
# where do past-tip hits live (volume)?
pt = acc[(acc.cat == "past_tip") & acc.rollout_id.isin(ps.rollout_id)]
print("  past_tip hits by volume:", pt["vol"].value_counts().sort_index().to_dict())
print("  tip-state volume of those branches:",
      states[(states.state_class == "tip") & states.seed_id.isin(suffix)]["volume_id"].value_counts().sort_index().to_dict())
PYEOF
echo "ORIGIN_RC=$?"
