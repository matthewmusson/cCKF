#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
set -o pipefail
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
REPO=/global/cfs/cdirs/atlas/mussonm/cCKF
cd $REPO
srun -n1 -c128 --mem=0 shifter --image=$IMG --env=PYTHONPATH=$REPO --env=OMP_NUM_THREADS=128 -- \
  $V/bin/python scripts/train_value.py \
    --train-cache $SCRATCH/cckf/vcache_v3/train --val-cache $SCRATCH/cckf/vcache_v3/val \
    --out-dir $SCRATCH/cckf/models_v3/value_t2_maj --device cpu
echo "VTRAIN_EXIT=$?"
