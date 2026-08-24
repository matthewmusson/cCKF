#!/bin/bash
# Incremental patched-ACTS build for cCKF on NERSC Perlmutter (shifter).
#
# WHY THIS EXISTS
# ---------------
# scripts/build_patched_acts.sh clones ACTS, re-runs cmake, and rebuilds from
# scratch on EVERY invocation -- ~20 min per cycle. Since the only thing that
# actually changes between cycles is a handful of header-only files under
# acts_patches/cckf/ plus one .cpp, nearly all of that is wasted.
#
# This script splits the work in two:
#
#   bootstrap  (once)    clone + instrumentation patch + CMakeLists surgery +
#                        cmake configure. Guarded by a stamp file because the
#                        CMakeLists patching is NOT idempotent -- it appends a
#                        target_include_directories() block and sed-inserts a
#                        source entry, so running it twice corrupts the build.
#   sync+build (each)    rsync only the cCKF files, then make the two targets
#                        that include them.
#
# Touching acts_patches/cckf/*.hpp recompiles exactly the translation units
# that include them (CckfTrackFindingAlgorithm.cpp and the pybind TU) plus a
# relink: order 1-2 minutes rather than 20.
#
# USAGE
#   # Grab an interactive node so SLURM queue wait does not dominate a
#   # 2-minute build. Keep the window SHORT and exit when done: the CPU
#   # allocation is 200 node-hours total (Iris, 2026-08-24) and an idle
#   # salloc bills by wall clock whether or not you are compiling.
#   salloc -N 1 -C cpu -q interactive -t 00:45:00 -A atlas
#   ./scripts/build_cckf_nersc.sh              # sync + build (the fast path)
#   ./scripts/build_cckf_nersc.sh --install    # also `make install`
#   ./scripts/build_cckf_nersc.sh --bootstrap  # force full re-bootstrap
#   ./scripts/build_cckf_nersc.sh --targets "ActsExamplesTrackFinding"
#
# Tier 3 (infrastructure).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Build on SCRATCH: Lustre handles the many-small-files compile pattern far
# better than CFS/GPFS. SCRATCH is purge-eligible but only after weeks of no
# access, which is irrelevant on this timescale. Install lands on SCRATCH too;
# override CCKF_ACTS_ROOT to put it on CFS if you need it to outlive a purge.
CCKF_ACTS_ROOT="${CCKF_ACTS_ROOT:-${SCRATCH}/cckf/acts}"
ACTS_SOURCE="${ACTS_SOURCE:-${CCKF_ACTS_ROOT}/src}"
ACTS_BUILD="${ACTS_BUILD:-${CCKF_ACTS_ROOT}/build}"
ACTS_INSTALL="${ACTS_INSTALL:-${CCKF_ACTS_ROOT}/install}"
STAMP="${ACTS_BUILD}/.cckf-bootstrap-complete"

BASE_COMMIT="${BASE_COMMIT:-4de1dcbbb2b8d8b6f14ec2c974d9b3a622028c01}"
CCKF_IMAGE="${CCKF_IMAGE:-ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0}"

# ActsExamplesTrackFinding holds CckfTrackFindingAlgorithm.cpp; the bindings
# target re-includes our header through the PUBLIC include dirs the bootstrap
# adds. If the bindings target name differs in this ACTS revision, discover it
# with:  make help | grep -i python
BUILD_TARGETS="${BUILD_TARGETS:-ActsExamplesTrackFinding ActsPythonBindings}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 16)}"

DO_BOOTSTRAP=0
DO_INSTALL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bootstrap) DO_BOOTSTRAP=1; shift ;;
        --install)   DO_INSTALL=1;   shift ;;
        --targets)   BUILD_TARGETS="$2"; shift 2 ;;
        --jobs)      JOBS="$2";          shift 2 ;;
        -h|--help)   sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -d "${REPO_ROOT}/acts_patches" ]]; then
    echo "ERROR: ${REPO_ROOT}/acts_patches not found. Run from the cCKF repo." >&2
    exit 1
fi

mkdir -p "${CCKF_ACTS_ROOT}"

# ---------------------------------------------------------------------------
# The container payload. Everything below runs INSIDE shifter.
#
# Passed via environment rather than string interpolation so paths containing
# spaces or shell metacharacters cannot break out of the quoting.
# ---------------------------------------------------------------------------
run_in_container() {
    shifter --image="${CCKF_IMAGE}" \
        --env=ACTS_SOURCE="${ACTS_SOURCE}" \
        --env=ACTS_BUILD="${ACTS_BUILD}" \
        --env=ACTS_INSTALL="${ACTS_INSTALL}" \
        --env=REPO_ROOT="${REPO_ROOT}" \
        --env=BASE_COMMIT="${BASE_COMMIT}" \
        --env=STAMP="${STAMP}" \
        --env=BUILD_TARGETS="${BUILD_TARGETS}" \
        --env=JOBS="${JOBS}" \
        --env=DO_BOOTSTRAP="${DO_BOOTSTRAP}" \
        --env=DO_INSTALL="${DO_INSTALL}" \
        -- bash -s <<'CONTAINER_EOF'
set -euo pipefail

# ccache turns a re-bootstrap from ~20 min into ~3-4 min by reusing object
# files across source trees. Absent in some image builds, so probe rather
# than assume.
if command -v ccache >/dev/null 2>&1; then
    export CCACHE_DIR="${ACTS_BUILD}/../ccache"
    mkdir -p "${CCACHE_DIR}"
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    echo "ccache: enabled ($(ccache -s 2>/dev/null | head -1))"
else
    echo "ccache: not present, continuing without it"
fi

# -------------------------------------------------------------------------
# BOOTSTRAP -- clone, patch, configure. Runs once.
# -------------------------------------------------------------------------
if [[ "${DO_BOOTSTRAP}" == "1" || ! -f "${STAMP}" ]]; then
    echo "=== BOOTSTRAP (one-time: clone + patch + configure) ==="

    if [[ "${DO_BOOTSTRAP}" == "1" ]]; then

        # start from a clean source tree rather than re-patching in place.
        echo "--bootstrap given: removing existing source tree"
        rm -rf "${ACTS_SOURCE}"
        rm -f "${STAMP}"
    fi

    # NOTE: the clone happens on the HOST, before this container runs. The
    # image's git cannot do HTTPS -- git-remote-https resolves libcurl-gnutls
    # against a spack nghttp2 that lacks
    # nghttp2_option_set_no_rfc9113_leading_and_trailing_ws_validation, and
    # dies with a symbol lookup error. Nothing about cloning needs the
    # container anyway; only cmake and make do.
    if [[ ! -d "${ACTS_SOURCE}/.git" ]]; then
        echo "ERROR: ${ACTS_SOURCE} has no git clone. The host-side clone" >&2
        echo "       step should have run before entering the container." >&2
        exit 1
    fi
    echo "Source tree present (cloned on host), reusing it."

    # Headers, CMakeLists surgery and the pybind registration all live in
    # scripts/apply_cckf_integration.sh, shared with build_patched_acts.sh.
    # Keeping one copy matters most for the pybind block: it enumerates every
    # Config field by name, and a field missing there is silently ignored at
    # runtime rather than erroring.
    ACTS_SOURCE="${ACTS_SOURCE}" CCKF_REPO="${REPO_ROOT}" \
        bash "${REPO_ROOT}/scripts/apply_cckf_integration.sh"

    echo "=== cmake configure ==="
    mkdir -p "${ACTS_BUILD}"
    cd "${ACTS_BUILD}"
    CMAKE_PREFIX_PATH=$(ls -d /spack/opt/spack/linux-x86_64/*/ | grep -v acts-main | tr '\n' ';')
    export CMAKE_PREFIX_PATH

    # podio generates the ActsPodioEdm datamodel with a Python script that
    # imports yaml. Two constraints, both learned the hard way:
    #
    #  1. PyYAML lives only in spack's py-pyyaml prefix, which is not on any
    #     default path, so cmake's Python fails with ModuleNotFoundError.
    #  2. PYTHONPATH must contain ONLY that path. spack's ACTS python
    #     directory ships a json.py that shadows the stdlib json module and
    #     breaks podio codegen (see build_patched_acts.sh).
    #
    # The site-packages is built for python3.13, so cmake must also be told to
    # use spack's 3.13 rather than the container's system python.
    YAML_SITE=$(ls -d /spack/opt/spack/linux-x86_64/py-pyyaml-*/lib/python*/site-packages 2>/dev/null | head -1)
    SPACK_PY=$(ls -d /spack/opt/spack/linux-x86_64/python-3.13*/bin/python3 2>/dev/null | head -1)
    if [[ -z "${YAML_SITE}" || -z "${SPACK_PY}" ]]; then
        echo "ERROR: could not locate spack py-pyyaml or python3.13" >&2
        echo "  YAML_SITE=${YAML_SITE:-<empty>}  SPACK_PY=${SPACK_PY:-<empty>}" >&2
        exit 1
    fi
    export PYTHONPATH="${YAML_SITE}"
    echo "  podio codegen python: ${SPACK_PY}"
    echo "  PYTHONPATH (yaml only): ${PYTHONPATH}"

    cmake "${ACTS_SOURCE}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${ACTS_INSTALL}" \
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
        -DPython_EXECUTABLE="${SPACK_PY}" \
        -DPython3_EXECUTABLE="${SPACK_PY}" \
        -DCMAKE_CXX_STANDARD=20 \
        -DACTS_BUILD_UNITTESTS=OFF \
        -DACTS_BUILD_EXAMPLES=ON \
        -DACTS_BUILD_EXAMPLES_DD4HEP=ON \
        -DACTS_BUILD_EXAMPLES_EDM4HEP=ON \
        -DACTS_BUILD_EXAMPLES_GEANT4=ON \
        -DACTS_BUILD_PLUGIN_DD4HEP=ON \
        -DACTS_BUILD_PLUGIN_EDM4HEP=ON \
        -DACTS_BUILD_PLUGIN_ROOT=ON \
        -DACTS_BUILD_EXAMPLES_ROOT=ON \
        -DACTS_BUILD_PLUGIN_JSON=ON \
        -DACTS_BUILD_PLUGIN_GEANT4=ON \
        -DACTS_BUILD_PYTHON_BINDINGS=ON \
        -DACTS_BUILD_ODD=OFF \
        -DACTS_BUILD_FATRAS=ON \
        -DACTS_BUILD_FATRAS_GEANT4=ON \
        -DACTS_BUILD_ALIGNMENT=OFF \
        -DACTS_BUILD_BENCHMARKS=OFF

    # ACTS_BUILD_UNITTESTS flipped to OFF relative to build_patched_acts.sh:
    # nothing in cCKF consumes them and they are a large share of build time.

    touch "${STAMP}"
    echo "=== Bootstrap complete ==="
fi

# -------------------------------------------------------------------------
# SYNC -- copy only changed cCKF files into the source tree.
#
# --checksum compares content, not timestamps: an unchanged header is skipped
# entirely and keeps its old mtime, so make does not consider it dirty. Only
# genuinely edited files get a fresh mtime and trigger recompilation.
# -------------------------------------------------------------------------
echo "=== Syncing cCKF sources ==="
rsync -a --checksum --itemize-changes \
    "${REPO_ROOT}"/acts_patches/cckf/*.hpp \
    "${ACTS_SOURCE}/Examples/Framework/include/ActsExamples/cckf/"
rsync -a --checksum --itemize-changes \
    "${REPO_ROOT}/acts_patches/ActsExamples/TrackFinding/CckfTrackFindingAlgorithm.hpp" \
    "${ACTS_SOURCE}/Examples/Algorithms/TrackFinding/include/ActsExamples/TrackFinding/"
rsync -a --checksum --itemize-changes \
    "${REPO_ROOT}/acts_patches/ActsExamples/TrackFinding/CckfTrackFindingAlgorithm.cpp" \
    "${ACTS_SOURCE}/Examples/Algorithms/TrackFinding/src/"

# -------------------------------------------------------------------------
# BUILD
# -------------------------------------------------------------------------
cd "${ACTS_BUILD}"
echo "=== make -j${JOBS} ${BUILD_TARGETS} ==="
SECONDS=0
make -j"${JOBS}" ${BUILD_TARGETS}
echo "=== Build finished in ${SECONDS}s ==="

if [[ "${DO_INSTALL}" == "1" ]]; then
    echo "=== make install ==="
    make -j"${JOBS}" install
    echo "Installed to ${ACTS_INSTALL}"
fi
CONTAINER_EOF
}

echo "cCKF NERSC build"
echo "  repo    : ${REPO_ROOT}"
echo "  source  : ${ACTS_SOURCE}"
echo "  build   : ${ACTS_BUILD}"
echo "  install : ${ACTS_INSTALL}"
echo "  image   : ${CCKF_IMAGE}"
echo "  targets : ${BUILD_TARGETS}"
echo

if ! command -v shifter >/dev/null 2>&1; then
    echo "ERROR: shifter not found. Are you on a Perlmutter login/compute node?" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Host-side clone.
#
# Must NOT run inside shifter: the image's git-remote-https resolves
# libcurl-gnutls against a spack nghttp2 missing
# nghttp2_option_set_no_rfc9113_leading_and_trailing_ws_validation, so any
# HTTPS clone dies with a symbol lookup error. The host git is fine.
# ---------------------------------------------------------------------------
if [[ "${DO_BOOTSTRAP}" == "1" ]]; then
    echo "--bootstrap: removing existing source tree"
    rm -rf "${ACTS_SOURCE}" "${STAMP}"
fi

if [[ ! -d "${ACTS_SOURCE}/.git" ]]; then
    echo "=== Cloning ACTS @ ${BASE_COMMIT} (host git) ==="
    mkdir -p "$(dirname "${ACTS_SOURCE}")"
    git clone --filter=blob:none https://github.com/acts-project/acts.git \
        "${ACTS_SOURCE}"
    git -C "${ACTS_SOURCE}" checkout "${BASE_COMMIT}"
    echo "=== Applying instrumentation patch ==="
    git -C "${ACTS_SOURCE}" apply "${REPO_ROOT}/instrumentation.patch"
    echo "Clone + patch complete."
fi

run_in_container
