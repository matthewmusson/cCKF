#!/bin/bash
#SBATCH --job-name=t3gen
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
set -o pipefail
E=$SLURM_ARRAY_TASK_ID
E9=$(printf "%09d" $E)
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
REPO=/global/cfs/cdirs/atlas/mussonm/cCKF
S=$SCRATCH/cckf
export CCKF_STAGE1_BASE=$S/modal_backup/results
cd $REPO

PILOT=$(shifter --image=$IMG --env=PYTHONPATH=$REPO --env=CCKF_STAGE1_BASE=$CCKF_STAGE1_BASE -- \
  $V/bin/python -c "from cckf.stage1_map import pilot_dir_for; print(pilot_dir_for($E))")
PARQUET=$S/reexpanded/expanded_event${E9}.parquet
TS=$PILOT/trackstates_ckf.root
WL=$S/tier3/worklists/event${E9}-rollout-worklist.csv
for f in "$PARQUET" "$TS" "$PILOT/detectors.csv"; do
  [ -e "$f" ] || { echo "MISSING_INPUT $f"; exit 3; }
done

echo "=== walker event $E ==="
shifter --image=$IMG --env=PYTHONPATH=$REPO --env=CCKF_STAGE1_BASE=$CCKF_STAGE1_BASE --env=OMP_NUM_THREADS=64 -- \
  $V/bin/python cckf/tier3_walker.py --parquet "$PARQUET" \
    --trackstates-root "$TS" --event-id $E \
    --detectors-csv "$PILOT/detectors.csv" --worklist-out "$WL"
rc=$?; [ $rc -eq 0 ] || { echo "WALKER_EXIT=$rc"; exit $rc; }
[ -s "$WL" ] || { echo "EMPTY_WORKLIST"; exit 4; }

echo "=== rollout event $E ==="
CFG=configs/_t3_ev${E}.yaml
sed -e "s/^skip: .*/skip: $((E % 2))/" \
    -e "s|^rollout_csv_dir: .*|rollout_csv_dir: $PILOT|" \
    $REPO/configs/_t3_full_ev1.yaml > $REPO/$CFG
srun -n1 -c128 --mem=0 ./scripts/run_cckf_nersc.sh --config $CFG \
  --input-file $S/modal_backup/events/edm4hep.root --events 1 \
  --output $S/runs_t3/t3_ev${E}
rc=$?
HITS=$S/tier3/hits/event${E9}-rollout-hits.csv
[ -s "$HITS" ] || { echo "MISSING_HITS $HITS"; exit 5; }
echo "T3GEN_EXIT=$rc hits_rows=$(wc -l < $HITS)"
exit $rc
