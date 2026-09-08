#!/bin/bash
set -o pipefail
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
cd /global/cfs/cdirs/atlas/mussonm/cCKF
srun -n1 shifter --image=$IMG -- $V/bin/python scripts/train_gate.py \
  --train-cache $1/train --val-cache $1/val --out-dir $2 \
  --sampler A --device cpu --max-epochs 1
echo "SMOKE_EXIT=$?"
