#!/usr/bin/env bash
# One generic distributed-memory AFM launcher; the first argument is the JSON.
#
# The script deliberately does not hide node/task/memory choices in a fixed
# selector. Supply them to sbatch so they can be changed after running the
# no-allocation planner. The packaged 16384^3 single-case template plans for:
#
# sbatch --nodes=256 --ntasks=4096 --ntasks-per-node=16 \
#   --cpus-per-task=12 --mem-per-cpu=4G jobs/run_afm_mpi.sh \
#   afm_config_nm_mpi_16384_template.json
#
# This is a memory-feasible example, not a promise that a 256-node request is
# schedulable or allowed by the allocation. Run preflight_afm_mpi.sh first.
#SBATCH --job-name=afm-mpi
#SBATCH --time=4-00:00:00
#SBATCH --account=rrg-hongguo-ad
#SBATCH --output=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this launcher with sbatch; do not execute it on a login node." >&2
    exit 2
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
SUBMIT_DIR="$(cd -- "${SUBMIT_DIR}" && pwd)"
if [[ -n "${AFM_PACKAGE_ROOT:-}" ]]; then
    PROJECT_ROOT="${AFM_PACKAGE_ROOT}"
    [[ "${PROJECT_ROOT}" = /* ]] || PROJECT_ROOT="${SUBMIT_DIR}/${PROJECT_ROOT}"
    PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
elif [[ -f "${SUBMIT_DIR}/run_mpi.py" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../run_mpi.py" ]]; then
    PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
    echo "ERROR: cannot locate run_mpi.py from SLURM_SUBMIT_DIR=${SUBMIT_DIR}." >&2
    exit 2
fi

CONFIG_ARG="${1:-${AFM_CONFIG:-afm_config_nm_mpi_16384_template.json}}"
if [[ $# -gt 0 ]]; then
    shift
fi
if [[ "${CONFIG_ARG}" = /* ]]; then
    CONFIG_PATH="${CONFIG_ARG}"
elif [[ -f "${PROJECT_ROOT}/${CONFIG_ARG}" ]]; then
    CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_ARG}"
elif [[ -f "${SUBMIT_DIR}/${CONFIG_ARG}" ]]; then
    CONFIG_PATH="${SUBMIT_DIR}/${CONFIG_ARG}"
else
    echo "ERROR: MPI JSON configuration not found: ${CONFIG_ARG}" >&2
    exit 2
fi

if [[ "${SLURM_NTASKS:-1}" -lt 2 ]]; then
    echo "ERROR: distributed mode needs at least two MPI tasks; submit with --ntasks." >&2
    exit 2
fi
if [[ "${SLURM_CPUS_PER_TASK:-0}" -lt 1 ]]; then
    echo "ERROR: submit with --cpus-per-task matching JSON cpu_threads." >&2
    exit 2
fi
if [[ "${AFM_ENFORCE_4G_PER_CPU:-1}" == "1" ]]; then
    if [[ -z "${SLURM_MEM_PER_CPU:-}" || "${SLURM_MEM_PER_CPU}" -ne 4096 ]]; then
        echo "ERROR: submit with --mem-per-cpu=4G to preserve the 4 GiB/core charging ratio." >&2
        echo "Observed SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-unset} MiB." >&2
        exit 2
    fi
fi

cd "${PROJECT_ROOT}"
if [[ -n "${AFM_OUTPUT_ROOT:-}" ]]; then
    export AFM_JOB_OUTPUT_ROOT="${AFM_OUTPUT_ROOT%/}/job_${SLURM_JOB_ID}"
else
    export AFM_JOB_OUTPUT_ROOT="${PROJECT_ROOT}/outputs/job_${SLURM_JOB_ID}"
fi
mkdir -p "${AFM_JOB_OUTPUT_ROOT}"

export AFM_NONINTERACTIVE=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

module load "${AFM_PYTHON_MODULE:-python}"
module load "${AFM_SCIPY_MODULE:-scipy-stack}"
module load "${AFM_MPI4PY_MODULE:-mpi4py}"
AFM_VENV="${AFM_VENV:-${HOME}/afm_env}"
if [[ ! -f "${AFM_VENV}/bin/activate" ]]; then
    echo "ERROR: AFM virtual environment not found: ${AFM_VENV}" >&2
    echo "Run bash jobs/setup_afm_mpi_env.sh once, or set AFM_VENV." >&2
    exit 2
fi
source "${AFM_VENV}/bin/activate"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_THREADS="$("${PYTHON_BIN}" -c 'import sys; from simulation.mpi_config import load_mpi_config; print(int(load_mpi_config(sys.argv[1])[1].get("cpu_threads", 1)))' "${CONFIG_PATH}")"
if [[ "${CONFIG_THREADS}" -ne "${SLURM_CPUS_PER_TASK}" ]]; then
    echo "ERROR: JSON cpu_threads=${CONFIG_THREADS} but Slurm --cpus-per-task=${SLURM_CPUS_PER_TASK}." >&2
    exit 2
fi

TASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-${SLURM_TASKS_PER_NODE:-}}"
TASKS_PER_NODE="${TASKS_PER_NODE%%(*}"
TASKS_PER_NODE="${TASKS_PER_NODE%%,*}"
if [[ -z "${TASKS_PER_NODE}" || ! "${TASKS_PER_NODE}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: cannot determine tasks per node; submit with --ntasks-per-node." >&2
    exit 2
fi
if [[ -n "${SLURM_MEM_PER_NODE:-}" ]]; then
    NODE_MEMORY_GIB="$((SLURM_MEM_PER_NODE / 1024))"
else
    NODE_MEMORY_GIB="$((SLURM_MEM_PER_CPU * SLURM_CPUS_ON_NODE / 1024))"
fi

echo "============================================================"
echo "AFM distributed-memory MPI job"
echo "Job ID            = ${SLURM_JOB_ID}"
echo "Package root       = ${PROJECT_ROOT}"
echo "Configuration      = ${CONFIG_PATH}"
echo "Output root        = ${AFM_JOB_OUTPUT_ROOT}"
echo "Nodes/tasks        = ${SLURM_JOB_NUM_NODES:-?}/${SLURM_NTASKS}"
echo "Tasks per node     = ${TASKS_PER_NODE}"
echo "Threads per rank   = ${SLURM_CPUS_PER_TASK}"
echo "Memory per CPU MiB = ${SLURM_MEM_PER_CPU:-unset}"
echo "Python             = $(command -v "${PYTHON_BIN}")"
"${PYTHON_BIN}" -c 'import mpi4py, numba, numpy, psutil, scipy; print("mpi4py", mpi4py.__version__, "NumPy", numpy.__version__, "SciPy", scipy.__version__, "Numba", numba.__version__, "psutil", psutil.__version__)'
echo "============================================================"

# Fail before grid allocation if the requested rank packing cannot fit in the
# requested node memory. The runtime repeats this check using physical RAM.
"${PYTHON_BIN}" run_mpi.py "${CONFIG_PATH}" --plan \
    --ranks "${SLURM_NTASKS}" \
    --ranks-per-node "${TASKS_PER_NODE}" \
    --node-memory-gib "${NODE_MEMORY_GIB}"

if [[ "${AFM_MPI_RETURN_AFTER_RUN:-0}" == "1" ]]; then
    # Dedicated parity wrappers may execute two layouts sequentially inside a
    # single allocation and compare their artifacts afterward.
    srun --kill-on-bad-exit=1 --cpu-bind=cores \
        "${PYTHON_BIN}" run_mpi.py "${CONFIG_PATH}" \
        --output-dir "${AFM_JOB_OUTPUT_ROOT}" --no-plot "$@"
else
    exec srun --kill-on-bad-exit=1 --cpu-bind=cores \
        "${PYTHON_BIN}" run_mpi.py "${CONFIG_PATH}" \
        --output-dir "${AFM_JOB_OUTPUT_ROOT}" --no-plot "$@"
fi
