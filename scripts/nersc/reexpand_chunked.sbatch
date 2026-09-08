#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --account=atlas
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
set -o pipefail
PILOT=$1; EV=$2; OUT=$3; NCH=$4; CI=$5
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
mkdir -p $(dirname $OUT)
srun -n1 -c 128 --mem=0 shifter --image=$IMG -- \
  $V/bin/python $SCRATCH/cckf/reexpand_chunk.py "$PILOT" "$EV" "$OUT" "$NCH" "$CI"
echo "EXPAND_EXIT=$?"
