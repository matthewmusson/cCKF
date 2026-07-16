#!/bin/bash
# Phase 1: Run a single ttbar pu200 event through digi+reco to validate the pipeline.
#
# Usage (interactive node):
#   salloc --nodes=1 --time=00:30:00 --constraint=cpu --qos=interactive --account=atlas
#   bash run_single_event.sh [--run-id 0] [--config baseline]
#
# Usage (login node, quick test with 1 event):
#   bash run_single_event.sh --events 1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"

RUN_ID="${1:-0}"
CONFIG_NAME="${2:-baseline}"
N_EVENTS="${3:-10}"

# Parse named args
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-id) RUN_ID="$2"; shift 2;;
        --config) CONFIG_NAME="$2"; shift 2;;
        --events) N_EVENTS="$2"; shift 2;;
        *) shift;;
    esac
done

INPUT_FILE="${CCKF_DATA}/ttbar/v1/runs/${RUN_ID}/edm4hep.root"
CONFIG_FILE="${SCRIPT_DIR}/configs/${CONFIG_NAME}.yaml"
OUTPUT_DIR="${CCKF_SCRATCH}/validation/${CONFIG_NAME}_run${RUN_ID}"

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "ERROR: edm4hep file not found: ${INPUT_FILE}"
    exit 1
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: config file not found: ${CONFIG_FILE}"
    echo "Available configs:"
    ls "${SCRIPT_DIR}/configs/"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "CKF Single-Event Validation"
echo "========================================"
echo "Input:   ${INPUT_FILE}"
echo "Config:  ${CONFIG_FILE} (${CONFIG_NAME})"
echo "Output:  ${OUTPUT_DIR}"
echo "Events:  ${N_EVENTS}"
echo "========================================"

# Copy our digi_and_reco.py into the output dir so we can mount it
# (the container's built-in script may differ from what we need)
cp "${SCRIPT_DIR}/digi_and_reco.py" "${OUTPUT_DIR}/digi_and_reco.py"
cp "${CONFIG_FILE}" "${OUTPUT_DIR}/config.yaml"

shifter --image="${CCKF_IMAGE}" -- \
    python3 "${OUTPUT_DIR}/digi_and_reco.py" \
        --config "${OUTPUT_DIR}/config.yaml" \
        --input-file "${INPUT_FILE}" \
        --output "${OUTPUT_DIR}" \
        --output-subdir results \
        --digi --reco

echo ""
echo "========================================"
echo "Results"
echo "========================================"

RESULTS_DIR="${OUTPUT_DIR}/results"
if [[ -f "${RESULTS_DIR}/performance_finding_ckf.root" ]]; then
    echo "SUCCESS: performance_finding_ckf.root generated"
    ls -lh "${RESULTS_DIR}"/performance_*.root
    ls -lh "${RESULTS_DIR}"/tracksummary_*.root 2>/dev/null
else
    echo "WARNING: performance_finding_ckf.root not found"
    echo "Output contents:"
    find "${OUTPUT_DIR}" -type f | head -20
fi
