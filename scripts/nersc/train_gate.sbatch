#!/bin/bash
#SBATCH --job-name=cckf_gate
#SBATCH --constraint=gpu
#SBATCH --account=atlas_g
set -o pipefail
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
CACHE=$1; OUT=$2; ARM=${3:-A}
cd /global/cfs/cdirs/atlas/mussonm/cCKF
srun -n1 shifter --image=$IMG --module=gpu -- $V/bin/python scripts/train_gate.py \
    --train-cache $CACHE/train --val-cache $CACHE/val --out-dir $OUT \
    --sampler $ARM --device cuda
echo "TRAIN_EXIT=$?"
