#!/bin/bash
# Chunk-aware re-expansion driver. Replaces wave_driver.sh (2 rogue copies
# killed 2026-08-25). Differences:
#   - every event is chunked (unchunked re-expansion now exceeds both the
#     29-min debug wall and node memory: rx1 timed out at 29:00 with 9.2 GB
#     free once endcap candidates were restored)
#   - tops up to the QOS submit cap instead of assuming it can queue freely
#   - merges chunk parts into the single-file layout (build_value_cache
#     reads one file per event) and deletes parts
#   - self-terminating; logs to $SCRATCH/cckf/logs/chunk_driver.log
set -u
O=$SCRATCH/cckf/reexpanded
D=$SCRATCH/cckf/modal_backup/results
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
MAXQ=8   # top-up ceiling; sbatch failures from the real cap are tolerated

declare -A PILOT=( [0]=pilot_1786524194 [1]=pilot_1786524194 [2]=pilot_1786524971 [3]=pilot_1786524971
 [4]=pilot_1786525888 [5]=pilot_1786525888 [6]=pilot_1786527639 [7]=pilot_1786527639
 [8]=pilot_1786529841 [9]=pilot_1786529841 [10]=pilot_1786531084 [11]=pilot_1786531084
 [12]=pilot_1786532999 [13]=pilot_1786532999 [14]=pilot_1786534329 [15]=pilot_1786534329
 [16]=pilot_1786536577 [17]=pilot_1786536577 [18]=pilot_1786538022 [19]=pilot_1786538022
 [20]=pilot_1786539360 [21]=pilot_1786539360 [22]=pilot_1786540365 [23]=pilot_1786540365
 [24]=pilot_1786541815 [25]=pilot_1786541815 [26]=pilot_1786542563 [27]=pilot_1786542563
 [28]=pilot_1786543789 [29]=pilot_1786543789 [30]=pilot_1786547065 [31]=pilot_1786547065 )
# chunks per event: memory-bound five at 8 (prior 4-chunk profile was proven
# at half the rows); >=2GB originals at 3; rest at 2. Event 1 already in
# flight as a 2-chunk pilot.
nch() { case $1 in 5|7|14|17|18) echo 8;; 0|2|3) echo 4;; *) echo 2;; esac }

log() { echo "[$(date +%H:%M:%S)] $*"; }

while true; do
  done_n=0; inflight=$(squeue -u $USER -h -o "%j" | grep -c "^rx" || true)
  for ev in $(seq 0 31); do
    single=$O/expanded_event$(printf "%09d" $ev).parquet
    if [ -s "$single" ] && [ "$(stat -c%s "$single")" -gt 100000000 ]; then done_n=$((done_n+1)); continue; fi
    N=$(nch $ev)
    # all parts present -> merge (idempotent; parts deleted on success)
    all_parts=1
    for ci in $(seq 0 $((N-1))); do
      [ -s "$O/expanded_event$(printf "%09d" $ev)_p${ci}.parquet" ] || { all_parts=0; break; }
    done
    if [ "$all_parts" = 1 ] && ! squeue -u $USER -h -o "%j" | grep -qx "rx${ev}"; then
      log "merging event $ev ($N parts)"
      if shifter --image=$IMG -- $V/bin/python /global/cfs/cdirs/atlas/mussonm/merge_parts.py "$single" >> $SCRATCH/cckf/logs/merge_ev${ev}.log 2>&1; then
        rm -f $O/expanded_event$(printf "%09d" $ev)_p*.parquet
        log "merged event $ev OK"
      else
        rm -f "$single"; log "MERGE FAILED event $ev (see merge_ev${ev}.log)"
      fi
      continue
    fi
    # submit missing parts, up to the top-up ceiling
    for ci in $(seq 0 $((N-1))); do
      part=$O/expanded_event$(printf "%09d" $ev)_p${ci}.parquet
      [ -s "$part" ] && continue
      squeue -u $USER -h -o "%j %o" 2>/dev/null | grep -q "rx${ev} .*_p${ci}\.parquet" && continue
      # job-name+comment dedup is imprecise across slurm versions; use a tag file
      tag=$SCRATCH/cckf/logs/.sub_ev${ev}_p${ci}
      if [ -f "$tag" ]; then
        # resubmit only if no rxN job carries this part and the part is absent
        jid=$(cat "$tag")
        squeue -h -j "$jid" -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|COMPLETING" && continue
      fi
      [ "$inflight" -ge $MAXQ ] && continue
      jid=$(sbatch --parsable --qos=debug --time=00:29:00 --nodes=1 --constraint=cpu --account=atlas \
        --cpus-per-task=128 --mem=0 --job-name=rx${ev} \
        --output=$SCRATCH/cckf/logs/rxc_ev${ev}_p${ci}_%j.out \
        $SCRATCH/cckf/reexpand_chunked.sbatch $D/${PILOT[$ev]} $ev "$part" $N $ci 2>/dev/null) || continue
      echo "$jid" > "$tag"; inflight=$((inflight+1)); log "submitted ev$ev p$ci ($jid)"
    done
  done
  log "done=$done_n/32 inflight=$inflight"
  [ "$done_n" -ge 32 ] && { log "ALL 32 COMPLETE"; break; }
  sleep 120
done
