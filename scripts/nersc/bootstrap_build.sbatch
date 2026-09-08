#!/bin/bash
#SBATCH --job-name=cckf_bootstrap
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --time=01:00:00
#SBATCH --output=%h_bootstrap_%j.out
cd /global/cfs/cdirs/atlas/mussonm/cCKF
export JOBS=64
srun -n1 ./scripts/build_cckf_nersc.sh --install
