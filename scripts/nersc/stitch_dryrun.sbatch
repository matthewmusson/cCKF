#!/bin/bash
#SBATCH --job-name=stitch_dry
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=atlas
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
# Task 5 acceptance dry run: stitch event 4 against the UNBOUNDED reference
# rollouts with NSIG=10 so the truth-suffix gate fires (tier-2 recomputed
# from the parquet; disagreement must be < 1%).
module load python
export PYTHONUNBUFFERED=1
cd /global/cfs/cdirs/atlas/mussonm/cCKF
export CCKF_STAGE1_BASE=$SCRATCH/cckf/modal_backup/results
python3 scripts/stitch_tier3.py 4 10 $SCRATCH/cckf --hits-dir $SCRATCH/cckf/tier3/hits
echo "STITCH_DRY_RC=$?"
