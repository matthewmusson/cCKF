#!/bin/bash
# One event end-to-end: sequential chunks -> merge -> cleanup.
# Args: PILOT_DIR EVENT_ID OUT_SINGLE N_CHUNKS
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --account=atlas
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
set -o pipefail
PILOT=$1; EV=$2; OUT=$3; NCH=$4
# already done (e.g. a backup job started after the primary finished)
[ -s "$OUT" ] && { echo "SINGLE_EXISTS ev=$EV, exiting"; exit 0; }
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
mkdir -p $(dirname $OUT)
for ci in $(seq 0 $((NCH-1))); do
  PART=${OUT%.parquet}_p${ci}.parquet
  [ -s "$PART" ] && { echo "SKIP chunk $ci (exists)"; continue; }
  srun -n1 -c 128 --mem=0 shifter --image=$IMG -- \
    $V/bin/python $SCRATCH/cckf/reexpand_chunk.py "$PILOT" "$EV" "$PART" "$NCH" "$ci" \
    || { echo "CHUNK_FAILED ci=$ci"; exit 1; }
done
shifter --image=$IMG -- $V/bin/python /global/cfs/cdirs/atlas/mussonm/merge_parts.py "$OUT" \
  || { echo "MERGE_FAILED"; rm -f "$OUT"; exit 1; }
rm -f ${OUT%.parquet}_p*.parquet
echo "EVENT_DONE ev=$EV rows=$(stat -c%s $OUT)"
