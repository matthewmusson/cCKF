#!/bin/bash
# Drive the 12 sweep points through debug QOS, 5 at a time. cCKF runs take
# 6-11 min so they fit the 29-minute cap, unlike expansion.
REPO=/global/cfs/cdirs/atlas/mussonm/cCKF
MODIN=$SCRATCH/cckf/modal_backup/events/edm4hep.root
while true; do
  done_n=0
  for g in 0.3 0.5 0.7 0.9; do for v in 0.1 0.2 0.4; do
    tag="sweep_g${g//./p}_v${v//./p}"
    out=$SCRATCH/cckf/runs/$tag/performance_finding_ambi.root
    if [ -s "$out" ]; then done_n=$((done_n+1)); continue; fi
    squeue -u $USER -h -o "%j" | grep -qx "$tag" && continue
    inflight=$(squeue -u $USER -h -o "%j" | grep -c "^sweep_")
    [ "$inflight" -ge 5 ] && continue
    sbatch --parsable --qos=debug --time=00:29:00 --nodes=1 --constraint=cpu \
      --account=atlas --cpus-per-task=128 --mem=0 --job-name=$tag \
      --output=$SCRATCH/cckf/logs/${tag}_%j.out \
      $SCRATCH/cckf/run_p1_input.sbatch "_${tag}.yaml" "$tag" "$MODIN" >/dev/null 2>&1
  done; done
  echo "[$(date +%H:%M:%S)] sweep done=$done_n/12 inflight=$(squeue -u $USER -h -o '%j'|grep -c '^sweep_')"
  [ "$done_n" -ge 12 ] && { echo "SWEEP COMPLETE"; break; }
  sleep 90
done
