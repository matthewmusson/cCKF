#!/bin/bash
#SBATCH --job-name=reexpand
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=atlas
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
set -o pipefail
PILOT=$1; LOCAL_EV=$2; OUT=$3
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
mkdir -p $(dirname $OUT)
( while true; do echo "$(date +%H:%M:%S) $(awk '/MemAvailable/{printf "avail_GB=%.1f",$2/1048576}' /proc/meminfo)"; sleep 5; done ) \
  > $SCRATCH/cckf/logs/mem_reexpand_${SLURM_JOB_ID}.log 2>&1 &
MON=$!
srun -n1 -c 128 --mem=0 shifter --image=$IMG -- \
  $V/bin/python $SCRATCH/cckf/reexpand_ev.py "$PILOT" "$LOCAL_EV" "$OUT"
rc=$?; kill $MON 2>/dev/null; echo "EXPAND_EXIT=$rc"; exit $rc
