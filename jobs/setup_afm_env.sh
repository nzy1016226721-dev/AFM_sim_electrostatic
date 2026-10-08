#!/bin/bash
# One-time Fir setup for the AFM runtime environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AFM_VENV="${AFM_VENV:-${HOME}/afm_env}"
WHEELHOUSE="${AFM_WHEELHOUSE:-/cvmfs/soft.computecanada.ca/custom/python/wheelhouse}"

module load "${AFM_PYTHON_MODULE:-python}"
module load "${AFM_SCIPY_MODULE:-scipy-stack}"

if [[ -e "${AFM_VENV}" && ! -f "${AFM_VENV}/bin/activate" ]]; then
    echo "ERROR: AFM_VENV exists but is not a Python virtual environment: ${AFM_VENV}" >&2
    exit 2
fi

if [[ ! -f "${AFM_VENV}/bin/activate" ]]; then
    mkdir -p "$(dirname "${AFM_VENV}")"
    virtualenv --system-site-packages "${AFM_VENV}"
fi
source "${AFM_VENV}/bin/activate"

if [[ ! -d "${WHEELHOUSE}" ]]; then
    echo "ERROR: Alliance Python wheelhouse not found: ${WHEELHOUSE}" >&2
    exit 2
fi

# Install only packages not supplied by scipy-stack.  --no-index prevents an
# accidental network/PyPI dependency and makes the setup reproducible on Fir.
python -m pip install --no-index --find-links="${WHEELHOUSE}" numba psutil

cd "${PROJECT_ROOT}"
python -c 'import numpy, scipy, numba, psutil; print("AFM environment ready:", "NumPy", numpy.__version__, "SciPy", scipy.__version__, "Numba", numba.__version__, "psutil", psutil.__version__)'
echo "Environment: ${AFM_VENV}"
