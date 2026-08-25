#!/bin/bash
# Apply the cCKF integration into an ACTS source tree. Idempotent.
#
# WHY THIS IS ITS OWN FILE
# ------------------------
# This logic used to live inline in build_patched_acts.sh. Once a second
# builder appeared (build_cckf_nersc.sh) it had to be shared rather than
# copied, because the pybind block below enumerates every
# CckfTrackFindingAlgorithm::Config field by name. A field added to the C++
# but missing from that list does NOT produce an error -- the build succeeds
# and the setting is silently ignored, running with its default. Two copies
# of the list means two chances to forget, and no signal when you do.
#
# INPUTS (environment)
#   ACTS_SOURCE  path to the patched ACTS source tree
#   CCKF_REPO    path to the cCKF repo (provides acts_patches/)
#
# All three steps are safe to re-run: the file copy is a plain cp, the
# CMakeLists edit is grep-guarded, and the pybind patch checks for its own
# marker before touching anything.

set -euo pipefail

: "${ACTS_SOURCE:?ACTS_SOURCE must be set}"
: "${CCKF_REPO:?CCKF_REPO must be set}"

if [[ ! -d "${ACTS_SOURCE}" ]]; then
    echo "ERROR: ACTS_SOURCE does not exist: ${ACTS_SOURCE}" >&2
    exit 1
fi
if [[ ! -d "${CCKF_REPO}/acts_patches" ]]; then
    echo "ERROR: ${CCKF_REPO}/acts_patches not found" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Copy cCKF headers and sources into the ACTS source tree
# ---------------------------------------------------------------------------
echo "=== Applying cCKF integration files ==="
CCKF_FRAMEWORK_INC="${ACTS_SOURCE}/Examples/Framework/include/ActsExamples/cckf"
CCKF_TF_INC="${ACTS_SOURCE}/Examples/Algorithms/TrackFinding/include/ActsExamples/TrackFinding"
CCKF_TF_SRC="${ACTS_SOURCE}/Examples/Algorithms/TrackFinding/src"

mkdir -p "${CCKF_FRAMEWORK_INC}"
cp "${CCKF_REPO}"/acts_patches/cckf/*.hpp "${CCKF_FRAMEWORK_INC}/"
cp "${CCKF_REPO}/acts_patches/ActsExamples/TrackFinding/CckfTrackFindingAlgorithm.hpp" \
   "${CCKF_TF_INC}/"
cp "${CCKF_REPO}/acts_patches/ActsExamples/TrackFinding/CckfTrackFindingAlgorithm.cpp" \
   "${CCKF_TF_SRC}/"
echo "Copied cCKF headers/sources into ACTS source tree."

# ---------------------------------------------------------------------------
# 2. Patch Examples/Algorithms/TrackFinding/CMakeLists.txt
#
# NOT naturally idempotent -- it sed-inserts a source entry and appends a
# block -- so it is grep-guarded.
# ---------------------------------------------------------------------------
TRACKFINDING_CMAKE="${ACTS_SOURCE}/Examples/Algorithms/TrackFinding/CMakeLists.txt"
if grep -q "CckfTrackFindingAlgorithm.cpp" "${TRACKFINDING_CMAKE}"; then
    echo "CMakeLists.txt already patched, skipping."
else
    echo "=== Patching Examples/Algorithms/TrackFinding/CMakeLists.txt ==="
    # -i.bak / rm rather than bare -i: portable across BSD and GNU sed.
    sed -i.bak 's|src/TrackFindingAlgorithm\.cpp|src/TrackFindingAlgorithm.cpp\n    src/CckfTrackFindingAlgorithm.cpp|' \
        "${TRACKFINDING_CMAKE}"
    rm -f "${TRACKFINDING_CMAKE}.bak"

    # Include paths: CckfTrackFindingAlgorithm.cpp includes its own header as
    # a bare "CckfTrackFindingAlgorithm.hpp", and the cckf headers as bare
    # "cckf/CckfXxx.hpp" which live under Framework/include. Neither resolves
    # via the include paths ExamplesTrackFinding already has. PUBLIC so the
    # Python bindings target (which includes our header) inherits them.
    cat >> "${TRACKFINDING_CMAKE}" <<'CMAKE_EOF'

# cCKF: the gate/value MLP kernel (MlpInference.hpp, Eigen products) is
# instantiated only in this translation unit. Perlmutter CPU nodes are
# EPYC 7763 (Zen 3): AVX2+FMA. Eigen's x86-64 baseline is SSE2, so without
# an arch flag the vectorized kernel runs at less than half throughput.
# Per-source rather than global: a global flag change would force a full
# ACTS rebuild; this recompiles one file.
set_source_files_properties(src/CckfTrackFindingAlgorithm.cpp
    PROPERTIES COMPILE_OPTIONS "-march=znver3")

# cCKF integration: CckfTrackFindingAlgorithm.hpp transitively includes
# cckf headers (SensorLookup.hpp, etc.) that live under Framework/include.
target_include_directories(
    ActsExamplesTrackFinding
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/ActsExamples/TrackFinding
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/include/ActsExamples
)
# SensorLookup.hpp includes <nlohmann/json.hpp>. ACTS's top-level CMake
# already calls find_package(nlohmann_json), so the target just needs to be
# linked to pick up its include directory. PUBLIC because our header
# transitively includes SensorLookup.hpp.
target_link_libraries(ActsExamplesTrackFinding PUBLIC nlohmann_json::nlohmann_json)
CMAKE_EOF
    echo "CMakeLists.txt patched."
fi

# ---------------------------------------------------------------------------
# 3. Register CckfTrackFindingAlgorithm with the Python bindings
#
# pybind11 exposure is not automatic: without this block the C++ compiles
# into the library but acts.examples.CckfTrackFindingAlgorithm does not
# exist, and digi_and_reco.py's addCckfTracks fails with AttributeError.
#
# THE CONFIG FIELD LIST IN ACTS_PYTHON_STRUCT BELOW MUST STAY IN SYNC WITH
# CckfTrackFindingAlgorithm::Config. Adding a field there and not here is
# silent -- no build error, the setting just never reaches C++.
# ---------------------------------------------------------------------------
echo "=== Patching Python/Examples/src/TrackFinding.cpp ==="
python3 - "${ACTS_SOURCE}/Python/Examples/src/TrackFinding.cpp" <<'PYEOF'
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

# Insert the CckfTrackFindingAlgorithm binding right after the existing
# TrackFindingAlgorithm block closes (mirrors its structure).
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
        gateMaxCandidates, gateChi2Ceiling, gateWindowNSigma, inputClusters,
        outputTimingPath, digiConfigPath);
  }
'''

text = text.replace(block_anchor, block_anchor + cckf_block, 1)

with open(path, "w") as f:
    f.write(text)

print("TrackFinding.cpp patched: include + CckfTrackFindingAlgorithm binding added.")
PYEOF

echo "=== cCKF integration complete ==="
