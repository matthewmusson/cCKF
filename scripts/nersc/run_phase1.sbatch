#!/bin/bash
#SBATCH --job-name=cckf_p1
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
# --mem=0 requests ALL node memory. Without it srun caps the step at roughly
# one core's share and a mu=200 ttbar event OOMs (job 57538970).
set -o pipefail
DATA=/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/ttbar/v1/runs/0/edm4hep.root
cd /global/cfs/cdirs/atlas/mussonm/cCKF
CFG=$1; NAME=$2
mkdir -p $SCRATCH/cckf/runs/$NAME
srun -n1 -c 128 --mem=0 ./scripts/run_cckf_nersc.sh --config configs/$CFG \
     --input-file $DATA --events 1 --output $SCRATCH/cckf/runs/$NAME
rc=$?
echo "RUN_EXIT=$rc"
exit $rc     # propagate, so sacct does not report COMPLETED on a failed run
