#!/bin/bash
#SBATCH --job-name=winfail_unc
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
module load python
cd /global/cfs/cdirs/atlas/mussonm/cCKF
export CCKF_STAGE1_BASE=$SCRATCH/cckf/modal_backup/results
seq 0 31 | xargs -P 8 -I{} python3 scripts/winfail_uncensored.py {} $SCRATCH/cckf
echo "DONE $(ls $SCRATCH/cckf/winfail_unc/*.npz | wc -l)/32"
