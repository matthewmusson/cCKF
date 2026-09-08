#!/bin/bash
#SBATCH --job-name=cckf_in
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=atlas
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
set -o pipefail
cd /global/cfs/cdirs/atlas/mussonm/cCKF
CFG=$1; NAME=$2; DATA=$3
OUT_BASE=${4:-$SCRATCH/cckf/runs}
mkdir -p $OUT_BASE/$NAME
( while true; do echo "$(date +%H:%M:%S) $(awk '/MemAvailable/{printf "avail_GB=%.1f",$2/1048576}' /proc/meminfo)"; sleep 2; done ) \
  > $SCRATCH/cckf/logs/mem_${NAME}_${SLURM_JOB_ID}.log 2>&1 &
MON=$!
srun -n1 -c 128 --mem=0 ./scripts/run_cckf_nersc.sh --config configs/$CFG \
     --input-file $DATA --events 1 --output $OUT_BASE/$NAME
rc=$?; kill $MON 2>/dev/null; echo "RUN_EXIT=$rc"; exit $rc
