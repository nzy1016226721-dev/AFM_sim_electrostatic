#!/bin/bash
# Validate one AFM JSON and the Fir Python environment without running a
# simulation or allocating its grid. The first argument is optional:
#   bash jobs/preflight_afm.sh [config.json]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_ARG="${1:-afm_config_nm.json}"
AFM_VENV="${AFM_VENV:-${HOME}/afm_env}"

if [[ "${CONFIG_ARG}" = /* ]]; then
    CONFIG_PATH="${CONFIG_ARG}"
elif [[ -f "${PROJECT_ROOT}/${CONFIG_ARG}" ]]; then
    CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_ARG}"
elif [[ -f "${SCRIPT_DIR}/${CONFIG_ARG}" ]]; then
    CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_ARG}"
else
    echo "ERROR: JSON configuration not found: ${CONFIG_ARG}" >&2
    exit 2
fi
CONFIG_PATH="$(cd -- "$(dirname -- "${CONFIG_PATH}")" && pwd)/$(basename -- "${CONFIG_PATH}")"

module load "${AFM_PYTHON_MODULE:-python}"
module load "${AFM_SCIPY_MODULE:-scipy-stack}"
if [[ ! -f "${AFM_VENV}/bin/activate" ]]; then
    echo "ERROR: AFM virtual environment not found: ${AFM_VENV}" >&2
    echo "Run bash jobs/setup_afm_env.sh first." >&2
    exit 2
fi
source "${AFM_VENV}/bin/activate"
cd "${PROJECT_ROOT}"

python - "${CONFIG_PATH}" "${SLURM_CPUS_PER_TASK:-}" <<'PY'
import sys
from pathlib import Path

import matplotlib
import numba
import numpy
import psutil
import scipy

# Import the complete run path, not merely its third-party dependencies.  This
# catches partial uploads such as a newer main_loop.py paired with an older
# io_utils.py before a high-memory Slurm job is submitted.
from simulation.main_loop import batch_main  # noqa: F401
from simulation.mpi_config import load_afm_config
from simulation.numerics import (
    diagnostic_iteration_limit,
    resolve_residual_tolerance_mode,
    resolve_solver_dtype,
)

config_path = Path(sys.argv[1])
slurm_cpus = sys.argv[2]
_, cfg = load_afm_config(config_path)

grid = cfg.get("grid_resolution")
if not isinstance(grid, dict):
    raise SystemExit("grid_resolution must be an object with nx, ny, nz")
shape = tuple(int(grid[key]) for key in ("nx", "ny", "nz"))
if any(axis < 2 for axis in shape):
    raise SystemExit(f"grid dimensions must be >= 2: {shape}")
threads = int(cfg.get("cpu_threads", 1))
if threads < 1:
    raise SystemExit(f"cpu_threads must be >= 1: {threads}")
if slurm_cpus and threads > int(slurm_cpus):
    raise SystemExit(f"cpu_threads={threads} exceeds SLURM_CPUS_PER_TASK={slurm_cpus}")
for key in ("res_tol_main", "res_tol_zoom"):
    if key in cfg and float(cfg[key]) <= 0:
        raise SystemExit(f"{key} must be positive")
dtype = resolve_solver_dtype(cfg.get("solver_dtype", "float32"))
mode = resolve_residual_tolerance_mode(
    cfg.get("residual_tolerance_mode", "absolute")
)
diagnostic = cfg.get("diagnostic")
if diagnostic:
    initial = int(cfg.get("initial_grid_level", min(shape)))
    current = tuple(min(initial, axis) for axis in shape)
    while True:
        diagnostic_iteration_limit(diagnostic, current, solver_dtype=dtype)
        if current == shape:
            break
        current = tuple(min(axis * 2, target) for axis, target in zip(current, shape))

print("AFM preflight OK")
print("Python packages:", numpy.__version__, scipy.__version__, numba.__version__, psutil.__version__, matplotlib.__version__)
print("Config:", config_path)
print("Grid:", shape, "solver threads:", threads, "Slurm CPUs:", slurm_cpus or "not allocated")
print("Tolerance:", cfg.get("res_tol_main", "default"), "zoom:", cfg.get("res_tol_zoom", "default"))
print("Numerics:", dtype.name, "residual mode:", mode, "diagnostic:", diagnostic or "none")
PY
