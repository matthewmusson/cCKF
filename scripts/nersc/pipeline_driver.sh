#!/bin/bash
# End-to-end, self-healing: expansions -> caches -> gate retrain -> sweep.
# Resubmits died/OOM'd events automatically, escalating to track_nr chunking
# when a whole-event attempt is OOM-killed (a 503 GB node is not enough for
# the largest events now that the valid-mask fix quadrupled the state count).
set -u
REPO=/global/cfs/cdirs/atlas/mussonm/cCKF
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
RX=$SCRATCH/cckf/reexpanded; CACHE=$SCRATCH/cckf/cache_v2
MODEL=$SCRATCH/cckf/models_v2/gate_A; W=$SCRATCH/cckf/weights_v2
D=$SCRATCH/cckf/modal_backup/results
declare -A P=( [0]=pilot_1786524194 [1]=pilot_1786524194 [2]=pilot_1786524971 [3]=pilot_1786524971
 [4]=pilot_1786525888 [5]=pilot_1786525888 [6]=pilot_1786527639 [7]=pilot_1786527639
 [8]=pilot_1786529841 [9]=pilot_1786529841 [10]=pilot_1786531084 [11]=pilot_1786531084
 [12]=pilot_1786532999 [13]=pilot_1786532999 [14]=pilot_1786534329 [15]=pilot_1786534329
 [16]=pilot_1786536577 [17]=pilot_1786536577 [18]=pilot_1786538022 [19]=pilot_1786538022
 [20]=pilot_1786539360 [21]=pilot_1786539360 [22]=pilot_1786540365 [23]=pilot_1786540365
 [24]=pilot_1786541815 [25]=pilot_1786541815 [26]=pilot_1786542563 [27]=pilot_1786542563
 [28]=pilot_1786543789 [29]=pilot_1786543789 [30]=pilot_1786547065 [31]=pilot_1786547065 )
log(){ echo "[$(date +%H:%M:%S)] $*"; }
have(){ local e=$1 f=$RX/expanded_event$(printf "%09d" $e).parquet
  [ -s "$f" ] && [ "$(stat -c%s "$f")" -gt 100000000 ] && return 0
  local n=$(ls $RX/expanded_event$(printf "%09d" $e)_p*.parquet 2>/dev/null | wc -l)
  [ "$n" -ge 4 ] && return 0; return 1; }

while true; do
  n=0; for e in $(seq 0 31); do have $e && n=$((n+1)); done
  log "expansion $n/32"
  [ "$n" -ge 32 ] && break
  for e in $(seq 0 31); do
    have $e && continue
    squeue -u $USER -h -o "%j" | grep -q "^rx${e}\b\|^rx${e}c" && continue
    # was the last attempt OOM-killed? then chunk it
    last=$(ls -t $SCRATCH/cckf/logs/rx_ev${e}_*.out 2>/dev/null | head -1)
    if [ -n "$last" ] && grep -q "Out Of Memory\|oom_kill" "$last" 2>/dev/null; then
      log "  ev$e OOM previously -> chunking into 4"
      for ci in 0 1 2 3; do
        [ -s "$RX/expanded_event$(printf "%09d" $e)_p${ci}.parquet" ] && continue
        sbatch --qos=regular --time=03:00:00 --job-name=rx${e}c${ci} \
          --output=$SCRATCH/cckf/logs/rx_ev${e}_c${ci}_%j.out \
          $SCRATCH/cckf/reexpand_chunked.sbatch "$D/${P[$e]}" "$e" \
          "$RX/expanded_event$(printf "%09d" $e)_p${ci}.parquet" 4 "$ci" >/dev/null 2>&1
      done
    else
      log "  ev$e resubmit whole"
      sbatch --qos=regular --time=03:00:00 --nodes=1 --constraint=cpu --account=atlas \
        --cpus-per-task=128 --mem=0 --job-name=rx${e} \
        --output=$SCRATCH/cckf/logs/rx_ev${e}_%j.out \
        $SCRATCH/cckf/reexpand.sbatch "$D/${P[$e]}" "$e" \
        "$RX/expanded_event$(printf "%09d" $e).parquet" >/dev/null 2>&1
    fi
  done
  sleep 300
done
log "ALL 32 EXPANDED"
shifter --image=$IMG -- $V/bin/python $REPO/scripts/build_caches_nersc.py \
  --parquet-dir $RX --out-dir $CACHE > $SCRATCH/cckf/logs/cache_build.log 2>&1
log "cache rc=$?"
jid=$(sbatch --parsable --output=$SCRATCH/cckf/logs/train_gate_%j.out \
      $REPO/scripts/train_gate_nersc.sbatch $CACHE $MODEL A 2>/dev/null)
log "train job $jid"
while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 120; done
[ -f "$MODEL/gate_model.pt" ] || { log "NO MODEL PRODUCED"; exit 1; }
mkdir -p $W
shifter --image=$IMG -- $V/bin/python $REPO/scripts/export_weights.py \
  --model $MODEL/gate_model.pt --out $W/gate.bin >> $SCRATCH/cckf/logs/export.log 2>&1
cp /global/cfs/cdirs/atlas/mussonm/cckf_weights/value.bin $W/ 2>/dev/null
log "RETRAIN COMPLETE -> $W"
