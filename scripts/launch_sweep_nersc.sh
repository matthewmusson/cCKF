#!/bin/bash
# Launch the (tau_g, tau_v) Pareto sweep on NERSC.
#
# One cCKF run per grid point. Results are collected by
# scripts/pareto_sweep_nersc.py into the CSV that scripts/pareto_sweep.py
# consumes, giving the four axes the sweep exists to trade off:
# efficiency, fake rate, latency, and pre-ambiguity duplicate rate.
#
# Usage:
#   ./scripts/launch_sweep_nersc.sh <weights_dir> <event> [runs_dir]
set -euo pipefail

WEIGHTS=${1:?usage: launch_sweep_nersc.sh <weights_dir> <event> [runs_dir]}
EVENT=${2:?}
RUNS=${3:-$SCRATCH/cckf/runs}
REPO=/global/cfs/cdirs/atlas/mussonm/cCKF
MODIN=$SCRATCH/cckf/modal_backup/events/edm4hep.root

# tau_g low was the failure mode on Modal: gate=0.3 caused branch explosion
# and timed out at one hour. The box pre-filter should have tamed it, but each
# point carries a wall-clock cap so one runaway cannot eat the queue -- a
# timeout is recorded as a data point, not a crash.
TAU_G=${TAU_G:-"0.3 0.5 0.7 0.9"}
TAU_V=${TAU_V:-"0.1 0.2 0.4"}

mkdir -p "$RUNS" "$SCRATCH/cckf/logs"
n=0
for g in $TAU_G; do
  for v in $TAU_V; do
    tag="sweep_g${g//./p}_v${v//./p}"
    cfg="configs/_${tag}.yaml"
    sed -e "s/^cckf_gate_threshold: .*/cckf_gate_threshold: $g/" \
        -e "s/^cckf_value_threshold: .*/cckf_value_threshold: $v/" \
        -e "s|^cckf_gate_weights: .*|cckf_gate_weights: $WEIGHTS/gate.bin|" \
        -e "s|^cckf_value_weights: .*|cckf_value_weights: $WEIGHTS/value.bin|" \
        -e "s/^skip: .*/skip: $EVENT/" \
        "$REPO/configs/nersc_cckf_full_dm.yaml" > "$REPO/$cfg"
    sbatch --parsable --qos=regular --time=01:00:00 \
      --output="$SCRATCH/cckf/logs/${tag}_%j.out" \
      "$SCRATCH/cckf/run_p1_input.sbatch" "_${tag}.yaml" "$tag" "$MODIN" >/dev/null
    n=$((n+1))
  done
done
echo "launched $n sweep points over tau_g={$TAU_G} x tau_v={$TAU_V} on event $EVENT"
