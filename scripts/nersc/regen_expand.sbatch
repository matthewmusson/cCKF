#!/bin/bash
#SBATCH --job-name=regen_exp
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
# Stage 2 of the tight/fast regeneration: expand the 8 regen events with
# the same run_expansion the August re-expansion used (shifter + Modal venv,
# single-shot per event; regen events are far below the chunking threshold).
set -o pipefail
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
PILOT=$SCRATCH/cckf/regen_$REGEN_CONFIG
OUT=$PILOT/reexpanded
mkdir -p $OUT
for E in 0 1 2 3 4 5 6 7; do
  P=$(printf "%s/expanded_event%09d.parquet" $OUT $E)
  [ -s "$P" ] && { echo "SKIP ev=$E exists"; continue; }
  srun -n1 -c 128 --mem=0 shifter --image=$IMG -- \
    $V/bin/python $SCRATCH/cckf/reexpand_ev.py "$PILOT" "$E" "$P" \
    || { echo "EXPAND_FAILED ev=$E"; exit 1; }
done
echo "REGEN_EXPAND_DONE $REGEN_CONFIG $(ls $OUT/*.parquet | wc -l)/8"
