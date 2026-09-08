#!/bin/bash
set -o pipefail
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
REPO=/global/cfs/cdirs/atlas/mussonm/cCKF
EV=$1; PILOT=$2; IN=$3; OUT=$4
cd $REPO
srun -n1 shifter --image=$IMG --env=PYTHONPATH=$REPO -- $V/bin/python scripts/patch_is_selected.py \
  --parquet "$IN" --root "$PILOT/trackstates_ckf.root" --event-id "$EV" --out "$OUT"
echo "PATCH_EXIT=$?"
