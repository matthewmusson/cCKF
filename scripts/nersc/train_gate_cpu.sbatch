#!/bin/bash
set -o pipefail
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
REPO=/global/cfs/cdirs/atlas/mussonm/cCKF
cd $REPO
srun -n1 shifter --image=$IMG --env=PYTHONPATH=$REPO --env=OMP_NUM_THREADS=128 -- \
  $V/bin/python scripts/train_gate.py \
  --train-cache $1/train --val-cache $1/val --out-dir $2 \
  --sampler A --device cpu
echo "TRAIN_EXIT=$?"
