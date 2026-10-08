#!/bin/bash
# Single AFM Slurm launcher. Choose the solver configuration at submission:
#
#   sbatch run_afm.sh afm_config_nm.json
#   sbatch run_afm.sh ../afm_config_case2.json
#
# The JSON is deliberately a positional argument. One launcher can therefore
# run any AFM configuration without maintaining one .sh selector per JSON.
#SBATCH --job-name=afm
#SBATCH --cpus-per-task=35
#SBATCH --mem=140G
#SBATCH --time=15:00:00
#SBATCH --account=rrg-hongguo-ad
# Keep Slurm stdout/stderr out of the package root and out of each scientific
# result directory. `%x` tracks an explicit --job-name override.
#SBATCH --output=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this launcher with sbatch, not by executing it directly." >&2
    exit 2
fi

# Slurm stages the batch script in a spool directory before running it. The
# submit directory is the stable location of the uploaded package and must be
# used instead of dirname(BASH_SOURCE[0]). The two supported layouts are:
#   package/       sbatch jobs/run_afm.sh config.json
#   package/jobs/  sbatch run_afm.sh config.json
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
SUBMIT_DIR="$(cd -- "${SUBMIT_DIR}" && pwd)"

if [[ -n "${AFM_PACKAGE_ROOT:-}" ]]; then
    PROJECT_ROOT="${AFM_PACKAGE_ROOT}"
    if [[ "${PROJECT_ROOT}" != /* ]]; then
        PROJECT_ROOT="${SUBMIT_DIR}/${PROJECT_ROOT}"
    fi
    PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
elif [[ -f "${SUBMIT_DIR}/run_all.py" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../run_all.py" ]]; then
    PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
    echo "ERROR: could not locate run_all.py from SLURM_SUBMIT_DIR=${SUBMIT_DIR}." >&2
    echo "Submit from the package root or its jobs directory, or set AFM_PACKAGE_ROOT." >&2
    exit 2
fi

if [[ ! -f "${PROJECT_ROOT}/run_all.py" || ! -d "${PROJECT_ROOT}/simulation" ]]; then
    echo "ERROR: invalid AFM package root: ${PROJECT_ROOT}" >&2
    exit 2
fi

# The first argument selects exactly one JSON. If omitted, the canonical
# configuration is used. Remaining non-interactive arguments are passed to
# run_all.py; --no-plot and --output-dir remain available when explicitly
# needed.
CONFIG_ARG="${1:-${AFM_CONFIG:-afm_config_nm.json}}"
if [[ "${CONFIG_ARG}" == -* ]]; then
    echo "ERROR: the first argument must be a JSON configuration filename/path." >&2
    echo "Usage: sbatch run_afm.sh [config.json] [run_all.py options]" >&2
    exit 2
fi
if [[ $# -gt 0 ]]; then
    shift
fi
RUN_ARGS=("$@")

for arg in "${RUN_ARGS[@]}"; do
    if [[ "${arg}" == "--interactive" || "${arg}" == "--plot" ]]; then
        echo "ERROR: ${arg} is not allowed in a Slurm job." >&2
        echo "Batch execution is headless and non-interactive; omit it or use --no-plot." >&2
        exit 2
    fi
done

resolve_config() {
    local requested="$1"
    local candidate
    if [[ "${requested}" = /* ]]; then
        candidate="${requested}"
        [[ -f "${candidate}" ]] && { printf '%s\n' "$(cd -- "$(dirname -- "${candidate}")" && pwd)/$(basename -- "${candidate}")"; return 0; }
    else
        # A filename normally lives at the package root. Also accept paths
        # relative to the directory from which sbatch was invoked.
        for candidate in "${PROJECT_ROOT}/${requested}" "${SUBMIT_DIR}/${requested}"; do
            if [[ -f "${candidate}" ]]; then
                printf '%s\n' "$(cd -- "$(dirname -- "${candidate}")" && pwd)/$(basename -- "${candidate}")"
                return 0
            fi
        done
    fi
    return 1
}

if ! CONFIG_PATH="$(resolve_config "${CONFIG_ARG}")"; then
    echo "ERROR: JSON configuration not found: ${CONFIG_ARG}" >&2
    echo "Package root: ${PROJECT_ROOT}" >&2
    exit 2
fi
case "${CONFIG_PATH}" in
    *.json|*.JSON) ;;
    *) echo "ERROR: configuration must be a .json file: ${CONFIG_PATH}" >&2; exit 2 ;;
esac

cd "${PROJECT_ROOT}"

# Keep generated data in the package's conventional ``outputs`` directory by
# default. AFM_OUTPUT_ROOT can be supplied with sbatch --export when a project
# or scratch location is explicitly requested.
if [[ -n "${AFM_OUTPUT_ROOT:-}" ]]; then
    export AFM_JOB_OUTPUT_ROOT="${AFM_OUTPUT_ROOT%/}/job_${SLURM_JOB_ID}"
else
    export AFM_JOB_OUTPUT_ROOT="${PROJECT_ROOT}/outputs/job_${SLURM_JOB_ID}"
fi
if ! mkdir -p "${AFM_JOB_OUTPUT_ROOT}"; then
    echo "ERROR: cannot create job output directory: ${AFM_JOB_OUTPUT_ROOT}" >&2
    echo "Set AFM_OUTPUT_ROOT to a different writable path." >&2
    exit 2
fi

export AFM_NONINTERACTIVE=1
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Use the generic Alliance module names by default, matching the known-good
# v15 launcher. Site/version-specific modules can be selected at submission:
#   sbatch --export=ALL,AFM_PYTHON_MODULE=python/3.12,AFM_SCIPY_MODULE=scipy-stack/2025a ...
module load "${AFM_PYTHON_MODULE:-python}"
module load "${AFM_SCIPY_MODULE:-scipy-stack}"

# The environment is intentionally required but never created in a compute
# job. Run jobs/setup_afm_env.sh once on Fir, or set AFM_VENV explicitly.
AFM_VENV="${AFM_VENV:-${HOME}/afm_env}"
if [[ ! -f "${AFM_VENV}/bin/activate" ]]; then
    echo "ERROR: AFM virtual environment not found: ${AFM_VENV}" >&2
    echo "Run bash jobs/setup_afm_env.sh once on Fir, or export AFM_VENV=/path/to/venv." >&2
    exit 2
fi
source "${AFM_VENV}/bin/activate"

python_bin="${PYTHON_BIN:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${python_bin}" >&2
    exit 2
fi

echo "============================================================"
echo "AFM Alliance Slurm job"
echo "SLURM_JOB_ID    = ${SLURM_JOB_ID}"
echo "SLURM_SUBMIT_DIR= ${SUBMIT_DIR}"
echo "Package root    = ${PROJECT_ROOT}"
echo "Configuration   = ${CONFIG_PATH}"
echo "Output root     = ${AFM_JOB_OUTPUT_ROOT}"
echo "CPU allocation  = ${SLURM_CPUS_PER_TASK:-1}"
echo "Python          = $(command -v "${python_bin}")"
"${python_bin}" --version
"${python_bin}" -c 'import numpy, scipy, numba, psutil; print("NumPy", numpy.__version__, "SciPy", scipy.__version__, "Numba", numba.__version__, "psutil", psutil.__version__)'
echo "============================================================"

# Batch jobs must not open preview/plot windows. If the caller did not make
# an explicit --no-plot choice, append it. Invoke Python directly (as in the
# known-good v15 launcher); this avoids execute-bit problems on helper shell
# files and does not require srun to locate a staged script.
has_plot_flag=0
for arg in "${RUN_ARGS[@]}"; do
    [[ "${arg}" == "--no-plot" ]] && has_plot_flag=1
done
if [[ ${has_plot_flag} -eq 0 ]]; then
    RUN_ARGS+=("--no-plot")
fi

exec "${python_bin}" run_all.py "${CONFIG_PATH}" "${RUN_ARGS[@]}"
