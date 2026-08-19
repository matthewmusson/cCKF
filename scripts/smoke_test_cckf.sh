#!/bin/bash
# End-to-end cCKF smoke test on Modal (Task 9).
#
# Verifies the full pipeline compiles and runs: patched ACTS build with
# CckfTrackFindingAlgorithm, randomly-initialized gate/value weight blobs in
# the correct binary format, and a short run of the cCKF-enabled
# reconstruction chain (cckf_envelope.yaml, cckf: true) on a couple of
# events from the held-out test set [32, 64) (spec §6.1).
#
# This is a compile-and-run smoke test, not a tracking-quality check —
# the weight blobs are random, so the metrics printed at the end
# (efficiency/fake/duplicate rate) are expected to be poor. What matters is:
#   - the run completes without crashing
#   - timing.csv (cCKF per-event timing) has one row per event
#   - gate_inference_calls > 0 and value_inference_calls > 0 in that CSV
#   - performance_finding_ckf.root / performance_finding_ambi.root exist
#
# Run each step in order; each `modal run` call blocks until that step
# finishes (or fails) before moving to the next.
set -eo pipefail

cd "$(dirname "$0")/.."

echo "=== Step 1/3: Build patched ACTS (includes CckfTrackFindingAlgorithm) ==="
modal run modal_build_acts.py::build_acts --force

echo
echo "=== Step 2/3: Generate random gate/value weight blobs ==="
modal run modal_build_acts.py::generate_dummy_weights

echo
echo "=== Step 3/3: Run cCKF on 2 test events ==="
modal run modal_build_acts.py::run_cckf --events 2 --gate-threshold 0.5 --value-threshold 0.1

echo
echo "Smoke test complete. Inspect the printed metrics dict above for:"
echo "  - cckf_timing_per_event: 2 rows, gate_inference_calls > 0, value_inference_calls > 0"
echo "  - summary.pre_ambi / summary.post_ambi: present (efficiency/fake/duplicate rates)"
