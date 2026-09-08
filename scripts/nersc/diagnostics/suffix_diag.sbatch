#!/bin/bash
#SBATCH --job-name=suffix_diag
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=atlas
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
# Diagnose the 44% truth-suffix disagreement on event 4: per-state sign of
# (tier3 - tier2), and its relation to the tip rollout's n_findable.
module load python
export PYTHONUNBUFFERED=1
cd /global/cfs/cdirs/atlas/mussonm/cCKF
export CCKF_STAGE1_BASE=$SCRATCH/cckf/modal_backup/results
python3 - <<'PYEOF'
import numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from cckf.tier3_walker import classify_event
from cckf.tier3_stitch import rollout_futures
from cckf import tier3_inputs
from cckf.stage1_map import csv_dir_for
import importlib
st = importlib.import_module("scripts.stitch_tier3") if False else None
S = "/pscratch/sd/m/mussonm/cckf"; E = 4
pq = f"{S}/reexpanded/expanded_event{E:09d}.parquet"
states = classify_event(pq)
hits = pd.read_csv(f"{S}/tier3/hits/event{E:09d}-rollout-hits.csv")
wl = pd.read_csv(f"{S}/tier3/worklists/event{E:09d}-rollout-worklist.csv")
fut = rollout_futures(hits, wl)
t3 = pd.read_parquet(f"{S}/tier3_targets/vstar_nsig10_event{E:09d}.parquet")
# tier-2 via the driver's pure path
sys.path.insert(0, "scripts")
import stitch_tier3 as drv
t2 = drv._recompute_tier2(pq, str(csv_dir_for(E)), E)
has_div = states.groupby("seed_id")["state_class"].apply(lambda s: (s == "divergence").any())
suffix = has_div.index[~has_div]
j = t3[t3.seed_id.isin(suffix)].merge(t2[["seed_id","step_k","vstar_t2"]], on=["seed_id","step_k"])
j["diff"] = j.vstar_tier3 - j.vstar_t2
print("suffix states:", len(j), " tier3>tier2:", int((j["diff"] > 0.01).sum()), " tier3<tier2:", int((j["diff"] < -0.01).sum()), " equal:", int((j["diff"].abs() <= 0.01).sum()))
# tip rollout findable per suffix seed
tips = states[(states.state_class == "tip") & states.seed_id.isin(suffix)][["seed_id","step_k"]]
tipf = tips.merge(fut[["seed_id","step_k","n_findable","n_future_holes"]], on=["seed_id","step_k"], how="left")
per_seed = j.groupby("seed_id")["diff"].agg(["max","min","mean"]).merge(tipf[["seed_id","n_findable","n_future_holes"]], on="seed_id", how="left")
print("per-seed: tip n_findable distribution among DISAGREEING seeds (max|diff|>0.01):")
dis = per_seed[per_seed[["max","min"]].abs().max(axis=1) > 0.01]
agr = per_seed[per_seed[["max","min"]].abs().max(axis=1) <= 0.01]
print("  disagreeing seeds:", len(dis), " tip n_findable value_counts:", dis.n_findable.value_counts().head(8).to_dict())
print("  agreeing seeds:   ", len(agr), " tip n_findable value_counts:", agr.n_findable.value_counts().head(8).to_dict())
print("  corr(max diff, tip n_findable):", float(np.corrcoef(dis["max"].fillna(0), dis.n_findable.fillna(0))[0,1]) if len(dis) > 2 else "n/a")
# does the diff scale like k / N_total?  show diff * N_total for disagreeing states
nt = tier3_inputs.n_total_true(pq, str(csv_dir_for(E)), E)
jj = j.merge(nt, on="seed_id")
jj["diff_x_N"] = jj["diff"] * jj["N_total_true"]
print("diff * N_total_true (should cluster at integers if it's a hit-count offset):")
print(jj.loc[jj["diff"].abs() > 0.01, "diff_x_N"].round(2).value_counts().head(10).to_dict())
# where along the branch does disagreement live: fraction disagreeing by step position from tip
j2 = j.merge(states[["seed_id","step_k","state_class"]], on=["seed_id","step_k"])
j2["is_tip"] = j2.state_class == "tip"
print("disagree rate at tip states:", float((j2[j2.is_tip]["diff"].abs() > 0.01).mean()), " at non-tip:", float((j2[~j2.is_tip]["diff"].abs() > 0.01).mean()))
PYEOF
echo "DIAG_RC=$?"
