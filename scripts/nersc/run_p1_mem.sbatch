#!/bin/bash
#SBATCH --job-name=cckf_mem
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=atlas
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
set -o pipefail
DATA=/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/ttbar/v1/runs/0/edm4hep.root
cd /global/cfs/cdirs/atlas/mussonm/cCKF
CFG=$1; NAME=$2
mkdir -p $SCRATCH/cckf/runs/$NAME
# Sample node-wide memory every 2s so we can see the shape of the growth.
( while true; do
    echo "$(date +%H:%M:%S) $(awk '/MemAvailable/{printf "avail_GB=%.1f", $2/1048576}' /proc/meminfo)"
    sleep 2
  done ) > $SCRATCH/cckf/logs/mem_${NAME}_${SLURM_JOB_ID}.log 2>&1 &
MONPID=$!
srun -n1 -c 128 --mem=0 ./scripts/run_cckf_nersc.sh --config configs/$CFG \
     --input-file $DATA --events 1 --output $SCRATCH/cckf/runs/$NAME
rc=$?
kill $MONPID 2>/dev/null
echo "RUN_EXIT=$rc"
exit $rc
