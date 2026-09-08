#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --account=atlas
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
set -o pipefail
SPLIT=$1
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
REPO=/global/cfs/cdirs/atlas/mussonm/cCKF
srun -n1 -c128 --mem=0 shifter --image=$IMG --env=PYTHONPATH=$REPO -- \
  $V/bin/python $REPO/scripts/build_caches_nersc.py \
    --parquet-dir $SCRATCH/cckf/reexpanded --out-dir $SCRATCH/cckf/caches_v3 \
    --splits $SPLIT && echo GATE_${SPLIT}_OK || echo GATE_${SPLIT}_FAILED
srun -n1 -c128 --mem=0 shifter --image=$IMG --env=PYTHONPATH=$REPO \
  --env=CCKF_STAGE1_BASE=$SCRATCH/cckf/modal_backup/results -- \
  $V/bin/python $REPO/scripts/build_value_cache.py \
    --split $SPLIT --parquet-dir $SCRATCH/cckf/reexpanded \
    --out-dir $SCRATCH/cckf/vcache_v3/$SPLIT && echo VCACHE_${SPLIT}_OK || echo VCACHE_${SPLIT}_FAILED
