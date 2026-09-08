#!/bin/bash
# Full mirror of the surp-acts-data Modal volume to SCRATCH.
# Resumable: a path already present with content is skipped.
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
MODAL=/global/cfs/cdirs/atlas/mussonm/venvs/modal/bin/modal
DEST=$SCRATCH/cckf/modal_backup

pull() {  # $1 = volume path
  local p="$1" out="$DEST/$(dirname "$1")"
  if [ -e "$DEST/$p" ] && [ -n "$(ls -A "$DEST/$p" 2>/dev/null)" ]; then
    echo "[$(date +%H:%M:%S)] skip $p"; return
  fi
  mkdir -p "$out"
  if shifter --image=$IMG -- $MODAL volume get --force surp-acts-data "$p" "$out/" >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] ok   $p ($(du -sh "$DEST/$p" 2>/dev/null | cut -f1))"
  else
    echo "[$(date +%H:%M:%S)] FAIL $p"
  fi
}
export -f pull; export DEST IMG MODAL

# Top-level entries other than results/, then every results/ subdir.
{ echo cache; echo models; echo weights; echo analysis; echo optimizer; \
  echo events; echo material_generation; echo odd-material-maps.root;
  sed 's|^|results/|' $SCRATCH/cckf/results_dirs.txt; } \
  | xargs -P 6 -I{} bash -c 'pull "$@"' _ {}

echo "[$(date +%H:%M:%S)] MIRROR DONE  total=$(du -sh $DEST | cut -f1)"
