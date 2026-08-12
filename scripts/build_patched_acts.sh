#!/bin/bash
set -eo pipefail

# Build patched ACTS from source inside the ghcr.io/opendatadetector container.
# Expected inputs:
#   /app/instrumentation.patch  — git format-patch from 4 cCKF commits
# Outputs:
#   /opt/acts-build/            — cmake build tree (test binaries live here)
#   /opt/acts-install/          — installed ACTS (lib, python, bin, include)

ACTS_SOURCE="${ACTS_SOURCE:-/opt/acts-src}"
ACTS_BUILD="${ACTS_BUILD:-/opt/acts-build}"
ACTS_INSTALL="${ACTS_INSTALL:-/opt/acts-install}"
BASE_COMMIT="${BASE_COMMIT:-4de1dcbbb2b8d8b6f14ec2c974d9b3a622028c01}"
BUILD_TARGET="${BUILD_TARGET:-all}"

# Capture Modal Python's pyyaml location BEFORE spack overrides PATH.
# Podio's code generator (EDM4HEP build) needs 'import yaml' under spack Python.
YAML_SITE=$(python3 -c "import yaml, os; print(os.path.dirname(os.path.dirname(yaml.__file__)))" 2>/dev/null || true)
if [ -n "$YAML_SITE" ]; then
    export PYTHONPATH="${YAML_SITE}:${PYTHONPATH:-}"
    echo "PYTHONPATH includes yaml from: $YAML_SITE"
fi

echo "=== Setting up spack environment ==="

# Find the spack ACTS install
ACTS_DIR=$(ls -d /spack/opt/spack/linux-x86_64/acts-main-*/ 2>/dev/null | head -1)
if [ -z "$ACTS_DIR" ]; then
    echo "ERROR: cannot find spack ACTS install"
    exit 1
fi
echo "Spack ACTS dir: $ACTS_DIR"

# Source the ACTS env to get cmake/gcc/etc. on PATH.
# Disable nounset temporarily — spack scripts reference undefined vars.
set +u
source "${ACTS_DIR}bin/this_acts_withdeps.sh" 2>/dev/null || true
set -u

# Clean PYTHONPATH: spack's ACTS python dir has json.py that shadows stdlib json,
# breaking podio code generation. Keep only the Modal pyyaml path.
if [ -n "${YAML_SITE:-}" ]; then
    export PYTHONPATH="$YAML_SITE"
else
    unset PYTHONPATH 2>/dev/null || true
fi

echo "cmake: $(which cmake 2>/dev/null || echo NOT FOUND)"
echo "gcc: $(which gcc 2>/dev/null || echo NOT FOUND)"
echo "git: $(which git 2>/dev/null || echo NOT FOUND)"

# CMAKE_PREFIX_PATH: all spack packages EXCEPT acts itself (we're rebuilding it)
CMAKE_PREFIX_PATH=$(ls -d /spack/opt/spack/linux-x86_64/*/ | grep -v acts-main | tr '\n' ';')
export CMAKE_PREFIX_PATH
echo "CMAKE_PREFIX_PATH has $(echo "$CMAKE_PREFIX_PATH" | tr ';' '\n' | wc -l) entries"

echo "=== Cloning ACTS (partial clone) ==="
git clone --filter=blob:none https://github.com/acts-project/acts.git "$ACTS_SOURCE"
cd "$ACTS_SOURCE"
git checkout "$BASE_COMMIT"

echo "=== Applying instrumentation patches ==="
git apply /app/instrumentation.patch
echo "Patches applied. HEAD is now:"
git log --oneline -5

echo "=== Configuring cmake ==="
mkdir -p "$ACTS_BUILD"
cd "$ACTS_BUILD"

cmake "$ACTS_SOURCE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$ACTS_INSTALL" \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DCMAKE_CXX_STANDARD=20 \
    -DACTS_BUILD_UNITTESTS=ON \
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

echo "=== Building target: ${BUILD_TARGET} ==="
make -j$(nproc) $BUILD_TARGET

echo "=== Build complete ==="
if [ "$BUILD_TARGET" = "all" ]; then
    echo "Installing..."
    make install
    echo "Installed to $ACTS_INSTALL"
fi
