#!/bin/bash
#SBATCH --job-name=winfail_emu
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=atlas
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
module load python
export PYTHONUNBUFFERED=1
cd /global/cfs/cdirs/atlas/mussonm/cCKF
export CCKF_STAGE1_BASE=$SCRATCH/cckf/modal_backup/results
seq 0 31 | xargs -P 8 -I{} python3 scripts/winfail_uncensored.py {} $SCRATCH/cckf --emulate $EMU_CONFIG
echo "DONE $(ls $SCRATCH/cckf/winfail_unc_emu_$EMU_CONFIG/*.npz 2>/dev/null | wc -l)/32"
