#!/bin/bash
# Retrain the gate on re-expanded data. GPU: the atlas_g allocation is
# untouched (100 node-hours) whereas atlas CPU is the scarce one.
#SBATCH --job-name=cckf_train_gate
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --qos=regular
#SBATCH --account=atlas_g
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
set -o pipefail
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
CACHE=$1; OUT=$2; ARM=${3:-A}
cd /global/cfs/cdirs/atlas/mussonm/cCKF
srun -n1 shifter --image=$IMG --module=gpu -- $V/bin/python scripts/train_gate.py \
    --train-cache $CACHE/train --val-cache $CACHE/val --out-dir $OUT \
    --sampler $ARM --device cuda
echo "TRAIN_EXIT=$?"
