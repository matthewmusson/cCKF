#!/bin/bash
# Setup environment for ACTS on NERSC Perlmutter via shifter.
#
# Usage:
#   source setup_env.sh          — sets env vars only (for sourcing in other scripts)
#   ./setup_env.sh --test        — runs a quick ACTS import test inside the container

export CCKF_IMAGE="ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0"
export CCKF_DATA="/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter"
export CCKF_SCRATCH="${SCRATCH}/cckf"
export CCKF_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The digi_and_reco.py script location inside the container
export CCKF_DIGI_RECO="/workspace/scripts/simulation/digi_and_reco.py"

mkdir -p "${CCKF_SCRATCH}"

if [[ "$1" == "--test" ]]; then
    echo "Testing ACTS import inside shifter container..."
    shifter --image="${CCKF_IMAGE}" -- python3 -c "
import acts
print('ACTS version:', acts.__version__)
from acts.examples.reconstruction import addCKFTracks, CkfConfig
print('CKF bindings: OK')
from acts.examples.odd import getOpenDataDetector
print('ODD geometry: OK')
print('All checks passed.')
"
    echo ""
    echo "Testing edm4hep data access..."
    ls -lh "${CCKF_DATA}/ttbar/v1/runs/0/edm4hep.root" 2>/dev/null && echo "Data access: OK" || echo "Data access: FAILED"
fi
