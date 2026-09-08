#!/bin/bash
# Pull the 32 expanded training Parquets from the Modal volume to SCRATCH.
# Resumable: skips any event whose file is already present and non-trivial.
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
MODAL=/global/cfs/cdirs/atlas/mussonm/venvs/modal/bin/modal
DEST=$SCRATCH/cckf/expanded
for i in $(seq 0 31); do
  f=$(printf "expanded_event%09d.parquet" $i)
  if [ -s "$DEST/$f" ] && [ "$(stat -c%s "$DEST/$f")" -gt 1000000 ]; then
    echo "[$(date +%H:%M:%S)] skip $f ($(du -h $DEST/$f | cut -f1))"; continue
  fi
  echo "[$(date +%H:%M:%S)] pulling $f ..."
  shifter --image=$IMG -- $MODAL volume get --force surp-acts-data \
      "results/train32/expanded/$f" "$DEST/$f" >/dev/null 2>&1 \
    && echo "[$(date +%H:%M:%S)]   ok  $(du -h $DEST/$f | cut -f1)" \
    || echo "[$(date +%H:%M:%S)]   FAILED $f"
done
echo "[$(date +%H:%M:%S)] DONE. total: $(du -sh $DEST | cut -f1), files: $(ls $DEST | wc -l)"
