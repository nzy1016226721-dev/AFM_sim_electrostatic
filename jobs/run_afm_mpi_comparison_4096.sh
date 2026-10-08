#!/usr/bin/env bash
# One-case 4096^3 distributed-MPI decomposition-parity trial.
#
# Both 512-rank layouts solve the same production geometry, position, voltage,
# hierarchy, and 1e-6 tolerance.  Their 201^3 physical cuts are then compared
# bit-for-bit.  The deliberately different Cartesian boundaries exercise halo
# exchange and prolongation independently of a particular array partition.
#
# 512 CPUs * 4 GiB/CPU = 2 TiB requested, matching Alliance core-equivalent
# accounting.  The 2048^3 completed trial measured 2.744 GiB maximum rank RSS
# for the same 512^3 local block; the conservative planner estimates 3.65 GiB.
# Submit from /home/nizy/afm_parallel only after explicit approval:
#   sbatch jobs/run_afm_mpi_comparison_4096.sh
#SBATCH --job-name=afm-mpi-4096-compare
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=16
#SBATCH --ntasks=512
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --account=rrg-hongguo-ad
#SBATCH --output=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this wrapper with sbatch, not by executing it directly." >&2
    exit 2
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
SUBMIT_DIR="$(cd -- "${SUBMIT_DIR}" && pwd)"
if [[ -f "${SUBMIT_DIR}/run_mpi.py" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../run_mpi.py" ]]; then
    PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
    echo "ERROR: could not locate AFM MPI package from SLURM_SUBMIT_DIR=${SUBMIT_DIR}." >&2
    exit 2
fi

CONFIG_A="afm_config_nm_mpi_comparison_4096_layout_8x8x8.json"
CONFIG_B="afm_config_nm_mpi_comparison_4096_layout_16x8x4.json"
JOB_OUTPUT="${PROJECT_ROOT}/outputs/job_${SLURM_JOB_ID}"
export AFM_OUTPUT_ROOT="${PROJECT_ROOT}/outputs"
export AFM_MPI_RETURN_AFTER_RUN=1

cd "${PROJECT_ROOT}"
bash jobs/run_afm_mpi.sh "${CONFIG_A}"
bash jobs/run_afm_mpi.sh "${CONFIG_B}"

module load "${AFM_PYTHON_MODULE:-python}"
module load "${AFM_SCIPY_MODULE:-scipy-stack}"
AFM_VENV="${AFM_VENV:-${HOME}/afm_env}"
if [[ ! -f "${AFM_VENV}/bin/activate" ]]; then
    echo "ERROR: AFM virtual environment not found for comparison: ${AFM_VENV}" >&2
    exit 2
fi
source "${AFM_VENV}/bin/activate"
PYTHON_BIN="${PYTHON_BIN:-python}"

CUT_NAME="afm_phi_1_0nm_-1.00V_cut_from_grid4096x4096x4096.npy"
REFERENCE="${JOB_OUTPUT}/afm_config_nm_mpi_comparison_4096_layout_8x8x8/${CUT_NAME}"
CANDIDATE="${JOB_OUTPUT}/afm_config_nm_mpi_comparison_4096_layout_16x8x4/${CUT_NAME}"
REPORT="${JOB_OUTPUT}/mpi_layout_comparison_4096.json"

"${PYTHON_BIN}" postprocessing/compare_mpi_npy.py \
    "${REFERENCE}" "${CANDIDATE}" \
    --atol 0 --rtol 0 --chunk-planes 1 --report "${REPORT}"

echo "SUCCESS: both 4096^3 MPI layouts converged and their saved cuts are bitwise identical."
echo "Comparison report: ${REPORT}"
