#!/bin/bash
# Run the cCKF (or baseline CKF) pipeline on NERSC against OUR patched ACTS.
#
# setup_env.sh points at the stock spack ACTS in the container, which has no
# CckfTrackFindingAlgorithm. This script points at the patched install built
# by scripts/build_cckf_nersc.sh instead.
#
# USAGE
#   ./scripts/run_cckf_nersc.sh --config configs/nersc_cckf_full.yaml \
#                               --events 1 --output $SCRATCH/cckf/runs/full
#
# Extra env knobs, used by the SIGSEGV investigation
# (docs/superpowers/plans/2026-08-24-sigsegv-systematic-debug.md):
#   MALLOC_DEBUG=1   set MALLOC_CHECK_=3 and MALLOC_PERTURB_=42, enable cores
#   ASAN=1           load the ASan runtime and set ASAN_OPTIONS
#
# Tier 3 (infrastructure).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CCKF_ACTS_ROOT="${CCKF_ACTS_ROOT:-${SCRATCH}/cckf/acts}"
ACTS_INSTALL="${ACTS_INSTALL:-${CCKF_ACTS_ROOT}/install}"
ODD_INSTALL="${ODD_INSTALL:-/global/cfs/cdirs/atlas/mussonm/ODD_v5/install}"
CCKF_IMAGE="${CCKF_IMAGE:-ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0}"
MALLOC_DEBUG="${MALLOC_DEBUG:-0}"
ASAN="${ASAN:-0}"

if [[ ! -f "${ACTS_INSTALL}/lib/libActsExamplesTrackFinding.so" ]]; then
    echo "ERROR: no patched ACTS install at ${ACTS_INSTALL}" >&2
    echo "       Run scripts/build_cckf_nersc.sh --install first." >&2
    exit 1
fi

echo "cCKF NERSC run"
echo "  acts install : ${ACTS_INSTALL}"
echo "  odd install  : ${ODD_INSTALL}"
echo "  args         : $*"
[[ "${MALLOC_DEBUG}" == "1" ]] && echo "  MALLOC_DEBUG : on"
[[ "${ASAN}" == "1" ]] && echo "  ASAN         : on"
echo

shifter --image="${CCKF_IMAGE}" \
    --env=ACTS_INSTALL="${ACTS_INSTALL}" \
    --env=ODD_INSTALL="${ODD_INSTALL}" \
    --env=REPO_ROOT="${REPO_ROOT}" \
    --env=MALLOC_DEBUG="${MALLOC_DEBUG}" \
    --env=ASAN="${ASAN}" \
    --env=RUN_ARGS="$*" \
    -- bash -s <<'CONTAINER_EOF'
set -euo pipefail

# Our patched ACTS python bindings, plus the spack py-* packages (numpy,
# pyyaml, ...). The py-* glob deliberately excludes spack's ACTS python
# directory, whose json.py shadows stdlib json.
PY_SITES=$(ls -d /spack/opt/spack/linux-x86_64/py-*/lib/python*/site-packages 2>/dev/null | tr '\n' ':')
export PYTHONPATH="${ACTS_INSTALL}/python:${REPO_ROOT}:${PY_SITES%:}"

# ODD's factory library must precede everything so getOpenDataDetector finds
# libOpenDataDetector.so, and our ACTS libs must precede spack's stock ACTS.
export LD_LIBRARY_PATH="${ACTS_INSTALL}/lib:${ODD_INSTALL}/lib:${LD_LIBRARY_PATH:-}"
export ODD_PATH="${ODD_INSTALL}/share/OpenDataDetector"

PY=$(ls -d /spack/opt/spack/linux-x86_64/python-3.13*/bin/python3 | head -1)

if [[ "${MALLOC_DEBUG}" == "1" ]]; then
    # Abort at the first detected heap inconsistency rather than whenever the
    # damage happens to become fatal, and poison freed memory so a
    # use-after-free reads garbage instead of plausible stale data.
    export MALLOC_CHECK_=3
    export MALLOC_PERTURB_=42
    ulimit -c unlimited || true
fi

if [[ "${ASAN}" == "1" ]]; then
    ASAN_LIB=$(ls /usr/lib/gcc/x86_64-linux-gnu/*/libasan.so 2>/dev/null | head -1)
    # detect_leaks=0: ACTS leaks by design at exit and the noise buries the
    # signal we actually want.
    export ASAN_OPTIONS="detect_leaks=0:abort_on_error=1:print_stacktrace=1:halt_on_error=1"
    export LD_PRELOAD="${ASAN_LIB}"
fi

# Sanity: the algorithm must be importable before we spend a run on it.
"${PY}" -c "
import acts.examples, sys
if not hasattr(acts.examples, 'CckfTrackFindingAlgorithm'):
    sys.exit('FATAL: CckfTrackFindingAlgorithm missing from bindings')
" || exit 1

cd "${REPO_ROOT}"
# shellcheck disable=SC2086
exec "${PY}" digi_and_reco.py ${RUN_ARGS}
CONTAINER_EOF
