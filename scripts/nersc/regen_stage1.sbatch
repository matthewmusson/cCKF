#!/bin/bash
#SBATCH --job-name=regen_s1
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
set -o pipefail
cd /global/cfs/cdirs/atlas/mussonm/cCKF
mkdir -p $SCRATCH/cckf/regen_$REGEN_CONFIG
srun -n1 -c 128 --mem=0 ./scripts/run_cckf_nersc.sh \
  --config configs/_regen_$REGEN_CONFIG.yaml \
  --input-file $SCRATCH/cckf/modal_backup/events/edm4hep.root \
  --events 8 --output $SCRATCH/cckf/regen_$REGEN_CONFIG
echo "REGEN_DONE $REGEN_CONFIG rc=$?"
