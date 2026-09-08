#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
set -o pipefail
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
REPO=/global/cfs/cdirs/atlas/mussonm/cCKF
srun -n1 -c128 --mem=0 shifter --image=$IMG --env=PYTHONPATH=$REPO -- \
  $V/bin/python $REPO/scripts/build_caches_nersc.py \
    --parquet-dir $SCRATCH/cckf/reexpanded \
    --out-dir $SCRATCH/cckf/caches_v3 \
    --splits train,val,cal && echo GATE_CACHES_OK || echo GATE_CACHES_FAILED
