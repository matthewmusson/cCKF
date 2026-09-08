#!/bin/bash
D=$SCRATCH/cckf/modal_backup
A=/global/cfs/cdirs/atlas/mussonm/cckf_archive
mkdir -p $A/results/train32
for item in models optimizer analysis weights; do
  echo "[$(date +%H:%M:%S)] copying $item ..."
  cp -a $D/$item $A/ && echo "[$(date +%H:%M:%S)]   ok $(du -sh $A/$item | cut -f1)"
done
echo "[$(date +%H:%M:%S)] copying train32/expanded (31G) ..."
cp -a $D/results/train32/expanded $A/results/train32/ \
  && echo "[$(date +%H:%M:%S)]   ok $(du -sh $A/results/train32/expanded | cut -f1)"
echo "[$(date +%H:%M:%S)] ARCHIVE DONE  total=$(du -sh $A | cut -f1)"
