#!/bin/bash
#SBATCH --job-name=rx
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --account=atlas
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
set -o pipefail
PILOT=$1; LOCAL_EV=$2; OUT=$3
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
mkdir -p $(dirname $OUT)
shifter --image=$IMG -- $V/bin/python $SCRATCH/cckf/reexpand_ev.py "$PILOT" "$LOCAL_EV" "$OUT"
echo "EXPAND_EXIT=$?"
