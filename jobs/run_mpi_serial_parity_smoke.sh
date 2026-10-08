#!/usr/bin/env bash
# Bounded two-rank regression for the MPI/serial numerical contract.
# Submit from the package root:
#   sbatch jobs/run_mpi_serial_parity_smoke.sh
# Results are isolated under outputs/validation/job_<id>/.
#SBATCH --job-name=afm-mpi-parity
#SBATCH --partition=cpubase_bycore_b5
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:15:00
#SBATCH --account=rrg-hongguo-ad
#SBATCH --output=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this wrapper with sbatch." >&2
    exit 2
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
SUBMIT_DIR="$(cd -- "${SUBMIT_DIR}" && pwd)"
if [[ -f "${SUBMIT_DIR}/run_mpi.py" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../run_mpi.py" ]]; then
    PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
    echo "ERROR: could not locate AFM package from SLURM_SUBMIT_DIR=${SUBMIT_DIR}." >&2
    exit 2
fi

module load "${AFM_PYTHON_MODULE:-python}"
module load "${AFM_SCIPY_MODULE:-scipy-stack}"
module load "${AFM_MPI4PY_MODULE:-mpi4py}"
AFM_VENV="${AFM_VENV:-${HOME}/afm_env}"
if [[ ! -f "${AFM_VENV}/bin/activate" ]]; then
    echo "ERROR: AFM virtual environment not found: ${AFM_VENV}" >&2
    exit 2
fi
source "${AFM_VENV}/bin/activate"

cd "${PROJECT_ROOT}"
CONFIG="${PROJECT_ROOT}/tests/data/afm_config_mpi_parity.json"
VALIDATION_ROOT="${PROJECT_ROOT}/outputs/validation/job_${SLURM_JOB_ID}"
MPI_ROOT="${VALIDATION_ROOT}/mpi"
SERIAL_ROOT="${VALIDATION_ROOT}/serial"
REPORT="${VALIDATION_ROOT}/mpi_serial_comparison.json"
mkdir -p "${PROJECT_ROOT}/outputs/slurm_logs" "${MPI_ROOT}" "${SERIAL_ROOT}"

echo "Running corrected two-rank MPI parity case: ${CONFIG}"
srun --ntasks=2 --ntasks-per-node=2 --cpus-per-task=1 \
    python run_mpi.py "${CONFIG}" --output-dir "${MPI_ROOT}"

echo "Running established shared-memory reference case"
python run_all.py "${CONFIG}" --output-dir "${SERIAL_ROOT}" --no-plot

MPI_NPY="${MPI_ROOT}/afm_config_mpi_parity/afm_phi_1_0nm_-1.00V.npy"
SERIAL_NPY="${SERIAL_ROOT}/afm_config_mpi_parity/afm_phi_1_0nm_-1.00V.npy"
if [[ ! -f "${MPI_NPY}" || ! -f "${SERIAL_NPY}" ]]; then
    echo "ERROR: expected parity NPY files were not produced." >&2
    find "${VALIDATION_ROOT}" -maxdepth 4 -type f -print >&2
    exit 3
fi

python postprocessing/compare_mpi_npy.py \
    "${SERIAL_NPY}" "${MPI_NPY}" \
    --atol 2e-5 --rtol 2e-5 --report "${REPORT}"

echo "MPI_SERIAL_PARITY_PASS report=${REPORT}"
