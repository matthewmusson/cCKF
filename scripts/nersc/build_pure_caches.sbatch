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
for SPLIT in train val cal; do
  srun -n1 -c128 --mem=0 shifter --image=$IMG --env=PYTHONPATH=$REPO -- \
    $V/bin/python $REPO/scripts/build_gate_cache.py \
      --split $SPLIT --parquet-dir $SCRATCH/cckf/reexpanded \
      --out-dir $SCRATCH/cckf/caches_v3_pure/$SPLIT --pure-seeds-only \
    && echo "PURE_GATE_${SPLIT}_OK" || { echo "PURE_GATE_${SPLIT}_FAILED"; exit 1; }
  srun -n1 -c128 --mem=0 shifter --image=$IMG --env=PYTHONPATH=$REPO \
    --env=CCKF_STAGE1_BASE=$SCRATCH/cckf/modal_backup/results -- \
    $V/bin/python $REPO/scripts/build_value_cache.py \
      --split $SPLIT --parquet-dir $SCRATCH/cckf/reexpanded \
      --out-dir $SCRATCH/cckf/vcache_v3_pure/$SPLIT --pure-seeds-only \
    && echo "PURE_VCACHE_${SPLIT}_OK" || { echo "PURE_VCACHE_${SPLIT}_FAILED"; exit 1; }
done
