#!/usr/bin/env bash
# No-allocation dependency/config/resource check for the distributed solver.
# Usage:
#   bash jobs/preflight_afm_mpi.sh CONFIG.json RANKS RANKS_PER_NODE NODE_MEMORY_GIB
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_ARG="${1:-afm_config_nm_mpi_16384_template.json}"
PLAN_RANKS="${2:-}"
PLAN_RANKS_PER_NODE="${3:-}"
PLAN_NODE_MEMORY_GIB="${4:-}"
AFM_VENV="${AFM_VENV:-${HOME}/afm_env}"

if [[ "${CONFIG_ARG}" = /* ]]; then
    CONFIG_PATH="${CONFIG_ARG}"
elif [[ -f "${PROJECT_ROOT}/${CONFIG_ARG}" ]]; then
    CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_ARG}"
else
    echo "ERROR: MPI JSON configuration not found: ${CONFIG_ARG}" >&2
    exit 2
fi

module load "${AFM_PYTHON_MODULE:-python}"
module load "${AFM_SCIPY_MODULE:-scipy-stack}"
module load "${AFM_MPI4PY_MODULE:-mpi4py}"
if [[ ! -f "${AFM_VENV}/bin/activate" ]]; then
    echo "ERROR: AFM virtual environment not found: ${AFM_VENV}" >&2
    echo "Run bash jobs/setup_afm_mpi_env.sh first." >&2
    exit 2
fi
source "${AFM_VENV}/bin/activate"
cd "${PROJECT_ROOT}"

python - "${CONFIG_PATH}" <<'PY'
import sys
import mpi4py, numba, numpy, psutil, scipy
from simulation.mpi_config import load_mpi_config
from simulation.mpi_main import batch_main_mpi  # noqa: F401
from simulation.numerics import (
    diagnostic_iteration_limit,
    resolve_residual_tolerance_mode,
    resolve_solver_dtype,
)

_, cfg = load_mpi_config(sys.argv[1])
dtype = resolve_solver_dtype(cfg.get("solver_dtype", "float32"))
mode = resolve_residual_tolerance_mode(
    cfg.get("residual_tolerance_mode", "absolute")
)
shape = tuple(int(cfg["grid_resolution"][key]) for key in ("nx", "ny", "nz"))
diagnostic = cfg.get("diagnostic")
if diagnostic:
    initial = int(cfg.get("mpi", {}).get(
        "initial_grid_level", cfg.get("initial_grid_level", min(shape))
    ))
    current = tuple(min(initial, axis) for axis in shape)
    while True:
        diagnostic_iteration_limit(diagnostic, current, solver_dtype=dtype)
        if current == shape:
            break
        current = tuple(min(axis * 2, target) for axis, target in zip(current, shape))
print(
    "MPI imports OK:", mpi4py.__version__, numba.__version__, numpy.__version__,
    scipy.__version__, psutil.__version__, "dtype:", dtype.name,
    "residual mode:", mode, "diagnostic:", diagnostic or "none",
)
PY

PLAN_ARGS=("${CONFIG_PATH}" --plan)
if [[ -n "${PLAN_RANKS}" ]]; then
    PLAN_ARGS+=(--ranks "${PLAN_RANKS}")
fi
if [[ -n "${PLAN_RANKS_PER_NODE}" ]]; then
    PLAN_ARGS+=(--ranks-per-node "${PLAN_RANKS_PER_NODE}")
fi
if [[ -n "${PLAN_NODE_MEMORY_GIB}" ]]; then
    PLAN_ARGS+=(--node-memory-gib "${PLAN_NODE_MEMORY_GIB}")
fi
python run_mpi.py "${PLAN_ARGS[@]}"
