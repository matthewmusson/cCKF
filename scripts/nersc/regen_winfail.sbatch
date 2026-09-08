#!/bin/bash
#SBATCH --job-name=regen_wf
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=atlas
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
# Stage 3 of the tight/fast regeneration: uncensored winfail accumulation
# over the 8 regenerated events, NO --emulate (every cut already happened
# for real inside ACTS in stage 1).
module load python
export PYTHONUNBUFFERED=1
cd /global/cfs/cdirs/atlas/mussonm/cCKF
export CCKF_STAGE1_BASE=$SCRATCH/cckf/modal_backup/results
R=$SCRATCH/cckf/regen_$REGEN_CONFIG
seq 0 7 | xargs -P 8 -I{} python3 scripts/winfail_uncensored.py {} $SCRATCH/cckf \
  --parquet-dir $R/reexpanded --csv-dir $R --out-dir $R/winfail_unc
echo "REGEN_WF_DONE $REGEN_CONFIG $(ls $R/winfail_unc/*.npz 2>/dev/null | wc -l)/8"
