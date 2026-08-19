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

echo "=== Applying cCKF integration files ==="
# Copy cCKF headers and sources (Tasks 1-5, /app/acts_patches on the Modal
# image) into the ACTS source tree, then patch the build system so they
# compile and are exposed to Python. Done via cp/sed/python3 here rather
# than a second git patch — simpler to keep in sync as acts_patches grows.

CCKF_FRAMEWORK_INC="$ACTS_SOURCE/Examples/Framework/include/ActsExamples/cckf"
CCKF_TRACKFINDING_INC="$ACTS_SOURCE/Examples/Algorithms/TrackFinding/include/ActsExamples/TrackFinding"
CCKF_TRACKFINDING_SRC="$ACTS_SOURCE/Examples/Algorithms/TrackFinding/src"

mkdir -p "$CCKF_FRAMEWORK_INC"
cp /app/acts_patches/cckf/*.hpp "$CCKF_FRAMEWORK_INC/"
cp /app/acts_patches/ActsExamples/TrackFinding/CckfTrackFindingAlgorithm.hpp \
   "$CCKF_TRACKFINDING_INC/"
cp /app/acts_patches/ActsExamples/TrackFinding/CckfTrackFindingAlgorithm.cpp \
   "$CCKF_TRACKFINDING_SRC/"
echo "Copied cCKF headers/sources into ACTS source tree."

echo "=== Patching Examples/Algorithms/TrackFinding/CMakeLists.txt ==="
TRACKFINDING_CMAKE="$ACTS_SOURCE/Examples/Algorithms/TrackFinding/CMakeLists.txt"

# a) Add the new source file to the acts_add_library() source list.
#    (-i.bak / rm rather than bare -i: portable across BSD and GNU sed.)
sed -i.bak 's|src/TrackFindingAlgorithm\.cpp|src/TrackFindingAlgorithm.cpp\n    src/CckfTrackFindingAlgorithm.cpp|' \
    "$TRACKFINDING_CMAKE"
rm -f "${TRACKFINDING_CMAKE}.bak"

# c) Include paths. CckfTrackFindingAlgorithm.cpp includes its own header as
#    bare "CckfTrackFindingAlgorithm.hpp" (found via same-directory lookup
#    for files in this target's own include/ActsExamples/TrackFinding dir —
#    NOT satisfied by the target's existing PUBLIC .../include dir, which
#    only exposes the "ActsExamples/TrackFinding/..." spelling), and the
#    cckf/*.hpp headers as bare "cckf/CckfXxx.hpp" (those live under
#    Examples/Framework/include/ActsExamples/cckf/, one level below the
#    Framework target's PUBLIC .../include dir). Neither resolves via the
#    include paths ExamplesTrackFinding already has, so add both directories
#    explicitly rather than editing the .cpp's #include directives.
cat >> "$TRACKFINDING_CMAKE" <<'CMAKE_EOF'

# cCKF integration: CckfTrackFindingAlgorithm.cpp uses bare #include paths
# ("CckfTrackFindingAlgorithm.hpp", "cckf/CckfXxx.hpp") that aren't resolved
# by ExamplesTrackFinding's existing include directories, so add the two
# containing directories directly.
target_include_directories(
    ActsExamplesTrackFinding
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/ActsExamples/TrackFinding
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/include/ActsExamples
)
# SensorLookup.hpp includes <nlohmann/json.hpp>. ACTS's top-level CMake
# already calls find_package(nlohmann_json), so the target just needs to
# be linked to pick up its include directory.
target_link_libraries(ActsExamplesTrackFinding PRIVATE nlohmann_json::nlohmann_json)
CMAKE_EOF
echo "CMakeLists.txt patched."

echo "=== Patching Python/Examples/src/TrackFinding.cpp ==="
python3 - "$ACTS_SOURCE/Python/Examples/src/TrackFinding.cpp" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    text = f.read()

marker = 'CckfTrackFindingAlgorithm'
if marker in text:
    print("TrackFinding.cpp already patched, skipping")
    sys.exit(0)

include_anchor = (
    '#include "ActsExamples/TrackFinding/TrackFindingAlgorithm.hpp"'
)
new_include = (
    '#include "ActsExamples/TrackFinding/CckfTrackFindingAlgorithm.hpp"\n'
    + include_anchor
)
if include_anchor not in text:
    raise SystemExit(f"ERROR: include anchor not found in {path}")
text = text.replace(include_anchor, new_include, 1)

# Insert the CckfTrackFindingAlgorithm binding block right after the
# existing TrackFindingAlgorithm block closes (mirrors its structure).
block_anchor = (
    "        useJosephFormulation, constrainToVolumeIds, "
    "endOfWorldVolumeIds);\n  }\n"
)
if block_anchor not in text:
    raise SystemExit(f"ERROR: binding block anchor not found in {path}")

cckf_block = '''
  {
    using Alg = CckfTrackFindingAlgorithm;
    auto [alg, c] = declareAlgorithm<Alg, IAlgorithm>(mex, "CckfTrackFindingAlgorithm");
    alg.def_static("makeTrackFinderFunction",
      [](std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry,
         std::shared_ptr<const Acts::MagneticFieldProvider> magneticField,
         Acts::Logging::Level level) {
        return Alg::makeTrackFinderFunction(
            std::move(trackingGeometry), std::move(magneticField),
            *Acts::getDefaultLogger("CckfTrackFinding", level));
      });
    py::class_<Alg::TrackFinderFunction,
               std::shared_ptr<Alg::TrackFinderFunction>>(alg, "TrackFinderFunction");
    ACTS_PYTHON_STRUCT(c, inputMeasurements, inputInitialTrackParameters, inputSeeds,
        outputTracks, trackingGeometry, magneticField, findTracks,
        measurementSelectorCfg, trackSelectorCfg, maxSteps, twoWay,
        reverseSearch, seedDeduplication, stayOnSeed, pixelVolumeIds,
        stripVolumeIds, maxPixelHoles, maxStripHoles, trimTracks,
        useJosephFormulation, constrainToVolumeIds, endOfWorldVolumeIds,
        gateWeightsPath, valueWeightsPath, gateThreshold, valueThreshold,
        gateMaxCandidates, inputClusters, outputTimingPath, digiConfigPath);
  }
'''

text = text.replace(block_anchor, block_anchor + cckf_block, 1)

with open(path, "w") as f:
    f.write(text)

print("TrackFinding.cpp patched: include + CckfTrackFindingAlgorithm binding added.")
PYEOF

echo "=== cCKF integration complete ==="

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
