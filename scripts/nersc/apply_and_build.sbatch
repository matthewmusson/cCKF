#!/bin/bash
#SBATCH --job-name=cckf_rebuild
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --time=01:30:00
#SBATCH --output=%h_rebuild_%j.out
# Task 2 (window-conditioned tier-3): re-apply the cCKF integration patch
# (new TruthRolloutAlgorithm Config field -> pybind member list) against the
# patched ACTS source on scratch, then run the same incremental build the
# August tier-3 binary came from.
set -o pipefail
export CCKF_REPO=/global/cfs/cdirs/atlas/mussonm/cCKF
export ACTS_SOURCE=$SCRATCH/cckf/acts/src
export JOBS=64
echo "== apply_cckf_integration.sh =="
bash $CCKF_REPO/scripts/apply_cckf_integration.sh || { echo "APPLY_FAILED rc=$?"; exit 1; }
echo "== build_cckf_nersc.sh --install =="
cd $CCKF_REPO
srun -n1 ./scripts/build_cckf_nersc.sh --install || { echo "BUILD_FAILED rc=$?"; exit 1; }
echo "REBUILD_DONE"
