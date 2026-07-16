#!/bin/bash
#SBATCH --job-name=ckf_sweep
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=0-11
#
# CKF parameter sweep on ttbar pu200.
# Each array task runs one config on one edm4hep file (128 events).
#
# Submit:  sbatch sweep.sbatch
# Monitor: squeue -u $USER

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"

mkdir -p "${SCRIPT_DIR}/logs"

# --- Sweep configurations ---
# Coordinate-wise sweep strategy:
#   Sweep 1 (tasks 0-4):  chi2CutOffMeasurement at default holes/branches
#   Sweep 2 (tasks 5-8):  numMeasurementsCutOff at best chi2
#   Sweep 3 (tasks 9-11): nMeasurementsMin at best chi2 + best branches
#
# Config files are pre-generated in configs/ by generate_sweep_configs.py

CONFIGS=($(ls -1 "${SCRIPT_DIR}/configs/sweep_"*.yaml 2>/dev/null | sort))

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "ERROR: No sweep config files found. Run generate_sweep_configs.py first."
    exit 1
fi

if [[ ${SLURM_ARRAY_TASK_ID} -ge ${#CONFIGS[@]} ]]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} exceeds number of configs (${#CONFIGS[@]}). Exiting."
    exit 0
fi

CONFIG_FILE="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
CONFIG_NAME="$(basename "${CONFIG_FILE}" .yaml)"

# Use run 0 (128 events) for the sweep
RUN_ID=0
INPUT_FILE="${CCKF_DATA}/ttbar/v1/runs/${RUN_ID}/edm4hep.root"
OUTPUT_DIR="${CCKF_SCRATCH}/sweep/${CONFIG_NAME}"

mkdir -p "${OUTPUT_DIR}"
cp "${SCRIPT_DIR}/digi_and_reco.py" "${OUTPUT_DIR}/digi_and_reco.py"
cp -r "${SCRIPT_DIR}/utils" "${OUTPUT_DIR}/utils"
cp "${CONFIG_FILE}" "${OUTPUT_DIR}/config.yaml"

echo "========================================"
echo "CKF Sweep — Task ${SLURM_ARRAY_TASK_ID}"
echo "Config:  ${CONFIG_NAME}"
echo "Input:   ${INPUT_FILE}"
echo "Output:  ${OUTPUT_DIR}"
echo "========================================"

START_TIME=$(date +%s)

cckf_shifter_run "${OUTPUT_DIR}/digi_and_reco.py" \
    --config "${OUTPUT_DIR}/config.yaml" \
    --input-file "${INPUT_FILE}" \
    --output "${OUTPUT_DIR}" \
    --output-subdir results \
    --digi --reco

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "Completed in ${ELAPSED}s"
ls -lh "${OUTPUT_DIR}/results/performance_finding_ckf.root" 2>/dev/null && echo "CKF performance: OK" || echo "CKF performance: MISSING"
ls -lh "${OUTPUT_DIR}/results/performance_finding_ambi.root" 2>/dev/null && echo "Ambi performance: OK" || echo "Ambi performance: MISSING"
