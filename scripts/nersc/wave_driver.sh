#!/bin/bash
# Keep up to MAXJOBS debug-QOS expansions in flight until all 32 exist.
# debug schedules in ~75s where regular/shared estimate 6 hours, but caps at
# 5 concurrent and 30 min wall, so the work is driven in waves.
#
# NOTE: pilot dirs use GLOBAL event numbering (pilot_1786524971 holds
# event000000002/3, not 0/1). Passing a local index silently looked for a
# nonexistent CSV and failed in load_measurements.
MAXJOBS=5
D=$SCRATCH/cckf/modal_backup/results
OUTD=$SCRATCH/cckf/reexpanded
declare -A PILOT=( [0]=pilot_1786524194 [1]=pilot_1786524194 [2]=pilot_1786524971 [3]=pilot_1786524971
 [4]=pilot_1786525888 [5]=pilot_1786525888 [6]=pilot_1786527639 [7]=pilot_1786527639
 [8]=pilot_1786529841 [9]=pilot_1786529841 [10]=pilot_1786531084 [11]=pilot_1786531084
 [12]=pilot_1786532999 [13]=pilot_1786532999 [14]=pilot_1786534329 [15]=pilot_1786534329
 [16]=pilot_1786536577 [17]=pilot_1786536577 [18]=pilot_1786538022 [19]=pilot_1786538022
 [20]=pilot_1786539360 [21]=pilot_1786539360 [22]=pilot_1786540365 [23]=pilot_1786540365
 [24]=pilot_1786541815 [25]=pilot_1786541815 [26]=pilot_1786542563 [27]=pilot_1786542563
 [28]=pilot_1786543789 [29]=pilot_1786543789 [30]=pilot_1786547065 [31]=pilot_1786547065 )
mkdir -p $OUTD
while true; do
  done_n=0; launched=0
  for ev in $(seq 0 31); do
    f=$OUTD/expanded_event$(printf "%09d" $ev).parquet
    if [ -s "$f" ] && [ "$(stat -c%s "$f")" -gt 100000000 ]; then done_n=$((done_n+1)); continue; fi
    squeue -u $USER -h -o "%j" 2>/dev/null | grep -qx "rx${ev}" && continue
    inflight=$(squeue -u $USER -h -o "%j" 2>/dev/null | grep -c "^rx")
    [ "$inflight" -ge $MAXJOBS ] && break
    P=$D/${PILOT[$ev]}
    # global event id, matching the CSV filenames in that pilot dir
    sbatch --parsable --qos=debug --time=00:29:00 --nodes=1 --constraint=cpu \
      --account=atlas --cpus-per-task=128 --mem=0 --job-name=rx${ev} \
      --output=$SCRATCH/cckf/logs/rx_ev${ev}_%j.out \
      $SCRATCH/cckf/reexpand.sbatch "$P" "$ev" "$f" >/dev/null 2>&1 && launched=$((launched+1))
  done
  echo "[$(date +%H:%M:%S)] done=$done_n/32 launched=$launched inflight=$(squeue -u $USER -h -o '%j'|grep -c '^rx')"
  [ "$done_n" -ge 32 ] && { echo "ALL 32 COMPLETE"; break; }
  sleep 90
done
