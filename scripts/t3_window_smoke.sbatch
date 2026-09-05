#!/bin/bash
#SBATCH --job-name=t3wsmoke
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#
# Behavioral smoke for the tier-3 rollout chi2 acceptance window
# (rollout_window_nsigma). Runs event 1's rollout stage TWICE from the SAME
# existing worklist -- unbounded (nsigma 0) and windowed (nsigma 3) -- into
# separate scratch dirs, then diffs per-rollout n_findable.
#
# Reuses $SCRATCH/cckf/tier3/worklists/ read-only; writes nothing under
# $SCRATCH/cckf/tier3/ (the reference unbounded hits stay untouched).
#
# Invocation mirrors the rollout stage of $SCRATCH/cckf/tier3_gen.sbatch:
# same image/venv, same CCKF_STAGE1_BASE, skip = E % 2 = 1 for event 1.
#
# Tier 3 (infrastructure).
#
#   sbatch scripts/t3_window_smoke.sbatch
set -o pipefail

E=1
E9=$(printf "%09d" $E)
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
REPO=/global/cfs/cdirs/atlas/mussonm/cCKF
S=$SCRATCH/cckf
export CCKF_STAGE1_BASE=$S/modal_backup/results
cd $REPO || exit 2

WL=$S/tier3/worklists/event${E9}-rollout-worklist.csv
[ -s "$WL" ] || { echo "MISSING_WORKLIST $WL"; exit 3; }

PILOT=$(shifter --image=$IMG --env=PYTHONPATH=$REPO --env=CCKF_STAGE1_BASE=$CCKF_STAGE1_BASE -- \
  $V/bin/python -c "from cckf.stage1_map import pilot_dir_for; print(pilot_dir_for($E))")
[ -d "$PILOT" ] || { echo "MISSING_PILOT $PILOT"; exit 3; }

# rollout_max: 200 keeps each arm to a couple of minutes; the window's effect
# shows up in the first handful of rollouts.
for ARM in 0 3; do
  OUT=$S/tier3_smoke_w${ARM}/hits
  mkdir -p "$OUT"
  CFG=configs/_t3_smoke_w${ARM}.yaml
  sed -e "s/^skip: .*/skip: $((E % 2))/" \
      -e "s|^rollout_csv_dir: .*|rollout_csv_dir: $PILOT|" \
      -e "s|^rollout_output_dir: .*|rollout_output_dir: $OUT|" \
      -e "s/^rollout_max: .*/rollout_max: 200/" \
      $REPO/configs/_t3_full_ev1.yaml > $REPO/$CFG
  # rollout_window_nsigma is not in the template; append it.
  echo "rollout_window_nsigma: ${ARM}.0" >> $REPO/$CFG

  echo "=== rollout arm nsigma=${ARM} (cfg $CFG) ==="
  srun -n1 -c128 --mem=0 ./scripts/run_cckf_nersc.sh --config $CFG \
    --input-file $S/modal_backup/events/edm4hep.root --events 1 \
    --output $S/runs_t3_smoke_w${ARM}
  rc=$?
  HITS=$OUT/event${E9}-rollout-hits.csv
  [ -s "$HITS" ] || { echo "MISSING_HITS $HITS"; exit 5; }
  echo "ARM=${ARM} EXIT=$rc hits_rows=$(wc -l < $HITS)"
done

echo "=== comparison ==="
module load python 2>/dev/null
python3 $REPO/scripts/t3_window_smoke_check.py \
  $S/tier3_smoke_w0/hits/event${E9}-rollout-hits.csv \
  $S/tier3_smoke_w3/hits/event${E9}-rollout-hits.csv
