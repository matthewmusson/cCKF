#!/bin/bash
# Dev convenience: apply the cCKF integration (headers + CMake/Python wiring)
# to a local ACTS checkout and print a `git diff`, without running a full
# Modal build. Useful for reviewing exactly what build_patched_acts.sh does
# to the ACTS tree before shipping a change to acts_patches/.
#
# Mirrors the "Applying cCKF integration files" section of
# scripts/build_patched_acts.sh — keep the two in sync.
#
# Usage:
#   scripts/generate_cckf_patch.sh <acts-source-dir>
#
# <acts-source-dir> must be a clean git checkout of ACTS (no local changes),
# already checked out to the commit build_patched_acts.sh uses
# (see BASE_COMMIT there) with instrumentation.patch already applied, if you
# want a diff that isolates just the cCKF layer.

set -eo pipefail

ACTS_SRC="${1:?Usage: $0 <acts-source-dir>}"
PATCH_DIR="$(cd "$(dirname "$0")/.." && pwd)/acts_patches"

if [ ! -d "$ACTS_SRC/.git" ]; then
    echo "ERROR: $ACTS_SRC does not look like a git checkout (no .git dir)"
    exit 1
fi

echo "=== Copying cCKF headers and sources into $ACTS_SRC ==="
mkdir -p "$ACTS_SRC/Examples/Framework/include/ActsExamples/cckf"
cp "$PATCH_DIR"/cckf/*.hpp \
   "$ACTS_SRC/Examples/Framework/include/ActsExamples/cckf/"
cp "$PATCH_DIR"/ActsExamples/TrackFinding/CckfTrackFindingAlgorithm.hpp \
   "$ACTS_SRC/Examples/Algorithms/TrackFinding/include/ActsExamples/TrackFinding/"
cp "$PATCH_DIR"/ActsExamples/TrackFinding/CckfTrackFindingAlgorithm.cpp \
   "$ACTS_SRC/Examples/Algorithms/TrackFinding/src/"

echo "=== Patching Examples/Algorithms/TrackFinding/CMakeLists.txt ==="
TRACKFINDING_CMAKE="$ACTS_SRC/Examples/Algorithms/TrackFinding/CMakeLists.txt"
sed -i.bak 's|src/TrackFindingAlgorithm\.cpp|src/TrackFindingAlgorithm.cpp\n    src/CckfTrackFindingAlgorithm.cpp|' \
    "$TRACKFINDING_CMAKE"
rm -f "${TRACKFINDING_CMAKE}.bak"

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
CMAKE_EOF

echo "=== Patching Python/Examples/src/TrackFinding.cpp ==="
python3 - "$ACTS_SRC/Python/Examples/src/TrackFinding.cpp" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    text = f.read()

marker = "CckfTrackFindingAlgorithm"
if marker in text:
    print("TrackFinding.cpp already patched, skipping")
    sys.exit(0)

include_anchor = '#include "ActsExamples/TrackFinding/TrackFindingAlgorithm.hpp"'
new_include = (
    '#include "ActsExamples/TrackFinding/CckfTrackFindingAlgorithm.hpp"\n'
    + include_anchor
)
if include_anchor not in text:
    raise SystemExit(f"ERROR: include anchor not found in {path}")
text = text.replace(include_anchor, new_include, 1)

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

cd "$ACTS_SRC"
git add -A -- Examples/Framework/include/ActsExamples/cckf \
              Examples/Algorithms/TrackFinding \
              Python/Examples/src/TrackFinding.cpp
git diff --cached > /tmp/cckf_integration.patch
git reset --quiet -- Examples/Framework/include/ActsExamples/cckf \
                     Examples/Algorithms/TrackFinding \
                     Python/Examples/src/TrackFinding.cpp

echo "Patch written to /tmp/cckf_integration.patch"
