#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=atlas
#SBATCH --time=00:29:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
set -o pipefail
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
V=/global/cfs/cdirs/atlas/mussonm/venvs/modal
REPO=/global/cfs/cdirs/atlas/mussonm/cCKF
EV1=$SCRATCH/cckf/reexpanded/expanded_event000000001.parquet
echo "=== PILOT 1: gate cache on event 1 ==="
srun -n1 -c128 --mem=0 shifter --image=$IMG --env=PYTHONPATH=$REPO -- $V/bin/python -c "
from cckf.cache import build_gate_cache
import json
meta = build_gate_cache(['$EV1'], '$SCRATCH/cckf/pilot_gate_cache_ev1')
print('GATE_CACHE_OK', json.dumps(meta)[:400])
" || echo GATE_CACHE_FAILED
echo "=== PILOT 2: tier-2 value cache on event 1 ==="
srun -n1 -c128 --mem=0 shifter --image=$IMG --env=PYTHONPATH=$REPO -- $V/bin/python $REPO/scripts/build_value_cache.py \
  --split train --only-events 1 \
  --parquet-dir $SCRATCH/cckf/reexpanded \
  --csv-dir $SCRATCH/cckf/modal_backup/results/pilot_1786524194 \
  --out-dir $SCRATCH/cckf/pilot_value_cache_ev1 && echo VALUE_CACHE_OK || echo VALUE_CACHE_FAILED
