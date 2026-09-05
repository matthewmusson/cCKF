#!/bin/bash
#SBATCH --job-name=t3rolln
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=atlas
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --array=0-31
#
# Multi-window tier-3 rollout generation (Task 4 of the
# window-conditioned tier-3 value plan, 2026-09-03).
#
# Regenerates the ROLLOUT stage only (never the walker) for every event,
# at a caller-chosen chi2 acceptance window NSIG, re-using the worklists
# already produced by $SCRATCH/cckf/tier3_gen.sbatch. The reference
# unbounded hits at $SCRATCH/cckf/tier3/hits/ are read-only inputs here
# and are never written to.
#
# This clones the rollout half of $SCRATCH/cckf/tier3_gen.sbatch (same
# image/venv/srun invocation, same pilot resolution via
# cckf.stage1_map.pilot_dir_for, same LOCAL skip = E % 2 for the 2-event
# pilot dirs) and adds one knob: rollout_window_nsigma, wired to the CKF
# TrackStateCreator via Task 2 (config key added in commit 131112a,
# rebuilt/installed at $SCRATCH/cckf/acts/install).
#
# Required env:
#   NSIG   chi2 acceptance window multiplier (float; e.g. 3, 5, 10).
#          Passed through as rollout_window_nsigma: ${NSIG}.0.
#
# Output contract (one file per array task):
#   $SCRATCH/cckf/tier3_nsig${NSIG}/hits/event{E:09d}-rollout-hits.csv
#
# Rendered per-event configs are written under $SCRATCH, never into the
# CFS configs/ dir (that copy is reserved for the unbounded tier3_gen
# configs and must not be clobbered by a sweep over NSIG).
#
# Invocation (submitted by the controller, not by this agent):
#   NSIG=3  sbatch scripts/tier3_rollout_n.sbatch
#   NSIG=5  sbatch scripts/tier3_rollout_n.sbatch
#   NSIG=10 sbatch scripts/tier3_rollout_n.sbatch
#
# Single-event smoke (Step 2 of the task brief), before the full array:
#   NSIG=3  sbatch --array=0-0 scripts/tier3_rollout_n.sbatch
set -o pipefail

: "${NSIG:?NSIG env var required, e.g. NSIG=3 sbatch scripts/tier3_rollout_n.sbatch}"

E=$SLURM_ARRAY_TASK_ID
E9=$(printf "%09d" "$E")
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
REPO=/global/cfs/cdirs/atlas/mussonm/cCKF
S=$SCRATCH/cckf
export CCKF_STAGE1_BASE=$S/modal_backup/results
cd "$REPO" || exit 2

# Worklist: produced once by tier3_gen.sbatch's walker stage, reused as-is
# for every NSIG. Never rewritten here.
WL=$S/tier3/worklists/event${E9}-rollout-worklist.csv
[ -s "$WL" ] || { echo "MISSING_WORKLIST $WL"; exit 3; }

# Pilot dir: same 2-event-range resolution tier3_gen uses. The per-event
# CFS config _t3_ev${E}.yaml already has this baked in, but we re-resolve
# it directly (rather than depending on that file's contents) so this job
# has no CFS write dependency of its own.
PILOT=$(shifter --image=$IMG --env=PYTHONPATH=$REPO --env=CCKF_STAGE1_BASE=$CCKF_STAGE1_BASE -- \
  $V/bin/python -c "from cckf.stage1_map import pilot_dir_for; print(pilot_dir_for($E))")
[ -d "$PILOT" ] || { echo "MISSING_PILOT $PILOT"; exit 3; }

OUT=$S/tier3_nsig${NSIG}/hits
CFGDIR=$S/tier3_nsig${NSIG}/configs
mkdir -p "$OUT" "$CFGDIR"

# Render this event's rollout config from the shared template, same sed
# substitutions as tier3_gen.sbatch (skip, rollout_csv_dir) plus the two
# new knobs this task owns: rollout_output_dir (isolate per-NSIG output)
# and rollout_window_nsigma (Task 2's acceptance window). worklist_dir is
# set explicitly too, even though the template default already points at
# the shared worklists dir, so the contract doesn't silently depend on
# that default never changing.
CFG=$CFGDIR/_t3_ev${E}.yaml
sed -e "s/^skip: .*/skip: $((E % 2))/" \
    -e "s|^rollout_csv_dir: .*|rollout_csv_dir: $PILOT|" \
    -e "s|^rollout_output_dir: .*|rollout_output_dir: $OUT|" \
    -e "s|^rollout_worklist_dir: .*|rollout_worklist_dir: $S/tier3/worklists|" \
    $REPO/configs/_t3_full_ev1.yaml > "$CFG"
# rollout_window_nsigma is not in the template; append it (mirrors
# scripts/t3_window_smoke.sbatch's arm rendering from Task 2's smoke job).
echo "rollout_window_nsigma: ${NSIG}.0" >> "$CFG"
[ -s "$CFG" ] || { echo "MISSING_CONFIG $CFG"; exit 3; }

echo "=== rollout event $E  nsigma=$NSIG  cfg=$CFG ==="
srun -n1 -c128 --mem=0 ./scripts/run_cckf_nersc.sh --config "$CFG" \
  --input-file "$S/modal_backup/events/edm4hep.root" --events 1 \
  --output "$S/runs_t3_nsig${NSIG}/t3_ev${E}"
rc=$?

HITS=$OUT/event${E9}-rollout-hits.csv
[ -s "$HITS" ] || { echo "MISSING_HITS $HITS"; exit 5; }
echo "T3ROLLN_EXIT=$rc nsig=$NSIG event=$E hits_rows=$(wc -l < "$HITS")"
exit $rc
