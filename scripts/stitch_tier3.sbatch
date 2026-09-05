#!/bin/bash
#SBATCH --job-name=stitch_t3
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=atlas
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#
# Tier-3 stitch driver, batched over all 32 events for one window (NSIG).
# Clones scripts/winfail_unc.sbatch's 8-way parallel pattern. For NSIG == 10
# each worker's truth-suffix gate can exit nonzero (scripts/stitch_tier3.py);
# xargs does not abort the batch on a single worker's failure, so check each
# event's stdout/stderr for SUFFIX_CHECK / "SUFFIX GATE FAILED" after the
# job completes.
#
# Env parameters:
#   NSIG      required. Rollout acceptance window passed to stitch_tier3.py
#             (e.g. 3, 5, 10).
#   HITS_DIR  optional. Overrides the per-window hits directory (passed as
#             --hits-dir); used for the unbounded dry run with
#             $SCRATCH/cckf/tier3/hits. Unset -> stitch_tier3.py's default
#             {scratch}/tier3_nsig{NSIG}/hits.
#
# Invocation:
#   NSIG=10 sbatch scripts/stitch_tier3.sbatch
#   NSIG=10 HITS_DIR=$SCRATCH/cckf/tier3/hits sbatch scripts/stitch_tier3.sbatch
set -o pipefail

: "${NSIG:?set NSIG (e.g. NSIG=10) before submitting}"

module load python
export PYTHONUNBUFFERED=1
cd /global/cfs/cdirs/atlas/mussonm/cCKF
export CCKF_STAGE1_BASE=$SCRATCH/cckf/modal_backup/results

HITS_ARG=""
if [ -n "$HITS_DIR" ]; then
  HITS_ARG="--hits-dir $HITS_DIR"
fi

seq 0 31 | xargs -P 8 -I{} python3 scripts/stitch_tier3.py {} $NSIG $SCRATCH/cckf $HITS_ARG

echo "DONE $(ls $SCRATCH/cckf/tier3_targets/vstar_nsig${NSIG}_event*.parquet 2>/dev/null | wc -l)/32"
