#!/bin/bash
#SBATCH --job-name=ckf_optim
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --output=logs/optimizer_%j.out
#SBATCH --error=logs/optimizer_%j.err
#
# Bayesian optimization of CKF + seeding parameters.
# Each trial runs digi+reco via shifter (~60-90s per trial for 128 events).
#
# Submit:
#   sbatch optimizer.sbatch
#   sbatch optimizer.sbatch --export=ALL,OPT_MODE=multi,OPT_VARS=ckf
#
# Resume:
#   sbatch optimizer.sbatch --export=ALL,RESUME_DIR=bayesian_ckf_results

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"

mkdir -p "${SCRIPT_DIR}/logs"

# Configurable via --export
N_SEED="${N_SEED:-10}"
N_GUIDED="${N_GUIDED:-40}"
OPT_VARS="${OPT_VARS:-all}"
OPT_MODE="${OPT_MODE:-single}"
CHANNEL="${CHANNEL:-ttbar}"
RUN_ID="${RUN_ID:-0}"
EVENTS="${EVENTS:-128}"
RESUME_DIR="${RESUME_DIR:-}"

OUTPUT_DIR="${CCKF_SCRATCH}/bayesian_ckf_results"
mkdir -p "${OUTPUT_DIR}"

ARGS=(
    --opt-vars "${OPT_VARS}"
    --n-seed-trials "${N_SEED}"
    --n-guided-trials "${N_GUIDED}"
    --output-dir "${OUTPUT_DIR}"
    --channel "${CHANNEL}"
    --run-id "${RUN_ID}"
    --events-per-trial "${EVENTS}"
    --verbose
)

if [[ "${OPT_MODE}" == "multi" ]]; then
    ARGS+=(--multi-objective)
fi

if [[ -n "${RESUME_DIR}" ]]; then
    ARGS+=(--resume-from "${RESUME_DIR}")
fi

echo "========================================"
echo "CKF Bayesian Optimization"
echo "========================================"
echo "Variables:  ${OPT_VARS}"
echo "Mode:       ${OPT_MODE}"
echo "Trials:     ${N_SEED} seed + ${N_GUIDED} guided"
echo "Channel:    ${CHANNEL}, run ${RUN_ID}"
echo "Events:     ${EVENTS} per trial"
echo "Output:     ${OUTPUT_DIR}"
echo "========================================"

# The optimizer runs on the compute node with xopt/botorch from the venv.
# Each trial spawns a shifter subprocess for the ACTS reconstruction.
source /global/cfs/cdirs/atlas/mussonm/venvs/cckf-opt/bin/activate
python3 "${SCRIPT_DIR}/optimizer.py" "${ARGS[@]}"

echo ""
echo "Optimization complete. Results in ${OUTPUT_DIR}"
