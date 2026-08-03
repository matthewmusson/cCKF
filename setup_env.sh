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

# Inside the container: spack-installed Python 3.13 (matches ACTS bindings)
export CCKF_PYTHON="/spack/opt/spack/linux-x86_64/python-3.13.11-awxtqzerpdzhatylv3uagd35ebciqs3o/bin/python3"
export CCKF_ACTS_DIR="/spack/opt/spack/linux-x86_64/acts-main-udwtnx3aoh5lh6s76slc2fzc5szhwe7y"

# ODD geometry: built from OpenDataDetector/OpenDataDetector v5.0.0
# (matches the version used to generate ColliderML datasets)
export ODD_PATH="/global/cfs/cdirs/atlas/mussonm/ODD_v5/install/share/OpenDataDetector"

mkdir -p "${CCKF_SCRATCH}"

# Helper: run a Python script inside the shifter container with correct ACTS paths.
# The spack build puts acts module files in $CCKF_ACTS_DIR/python/ without an
# acts/ subdirectory, so we symlink it at runtime. We also add all spack
# Python 3.13 site-packages to PYTHONPATH (pyyaml, numpy, dd4hep, podio, etc.).
cckf_shifter_run() {
    shifter --image="${CCKF_IMAGE}" -- bash -c '
        source '"${CCKF_ACTS_DIR}"'/bin/this_acts_withdeps.sh 2>/dev/null
        rm -rf /tmp/acts_pypath && mkdir -p /tmp/acts_pypath
        ln -sf '"${CCKF_ACTS_DIR}"'/python /tmp/acts_pypath/acts
        SPACK_PYPATH=$(find /spack/opt -path "*/python3.13/site-packages" -type d 2>/dev/null | tr "\n" ":")
        export PYTHONPATH="/tmp/acts_pypath:${SPACK_PYPATH}${PYTHONPATH:+:$PYTHONPATH}"
        export LD_LIBRARY_PATH="/global/cfs/cdirs/atlas/mussonm/ODD_v5/install/lib:${LD_LIBRARY_PATH}"
        export ODD_PATH="'"${ODD_PATH}"'"
        '"${CCKF_PYTHON}"' "$@"
    ' -- "$@"
}
export -f cckf_shifter_run

if [[ "${1:-}" == "--test" ]]; then
    echo "Testing ACTS import inside shifter container..."
    cckf_shifter_run -c "
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
