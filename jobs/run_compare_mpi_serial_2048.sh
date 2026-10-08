#!/usr/bin/env bash
# Compare the completed 2048^3 MPI single-case cut against an equivalently
# configured current-code serial control. Submit with an afterok dependency:
#
#   sbatch --dependency=afterok:<serial_job_id> \
#     --export=ALL,AFM_SERIAL_JOB_ID=<serial_job_id>,AFM_MPI_JOB_ID=<mpi_job_id> \
#     jobs/run_compare_mpi_serial_2048.sh
#
# A numerical mismatch is written to the JSON report and does not make this
# diagnostic job fail. Infrastructure/input failures still exit non-zero.
#SBATCH --job-name=afm-compare-2048
#SBATCH --partition=cpubase_bycore_b5
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --account=rrg-hongguo-ad
#SBATCH --output=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this script with sbatch, not by executing it directly." >&2
    exit 2
fi

: "${AFM_SERIAL_JOB_ID:?ERROR: set AFM_SERIAL_JOB_ID to the serial control Slurm job ID.}"
: "${AFM_MPI_JOB_ID:?ERROR: set AFM_MPI_JOB_ID to the completed MPI trial Slurm job ID.}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
SUBMIT_DIR="$(cd -- "${SUBMIT_DIR}" && pwd)"
if [[ -f "${SUBMIT_DIR}/run_all.py" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../run_all.py" ]]; then
    PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
    echo "ERROR: could not locate AFM package from SLURM_SUBMIT_DIR=${SUBMIT_DIR}." >&2
    exit 2
fi
cd "${PROJECT_ROOT}"

module load "${AFM_PYTHON_MODULE:-python}"
module load "${AFM_SCIPY_MODULE:-scipy-stack}"
AFM_VENV="${AFM_VENV:-${HOME}/afm_env}"
if [[ ! -f "${AFM_VENV}/bin/activate" ]]; then
    echo "ERROR: AFM virtual environment not found: ${AFM_VENV}" >&2
    exit 2
fi
source "${AFM_VENV}/bin/activate"

serial_dir="${PROJECT_ROOT}/outputs/job_${AFM_SERIAL_JOB_ID}/afm_config_nm_serial_control_2048"
mpi_dir="${PROJECT_ROOT}/outputs/job_${AFM_MPI_JOB_ID}/afm_config_nm_mpi_trial_2048"
serial_file="$(find "${serial_dir}" -maxdepth 1 -type f -name '*_0nm_-1.00V_cut_from_grid2048x2048x2048.npy' -print -quit)"
mpi_file="$(find "${mpi_dir}" -maxdepth 1 -type f -name '*_0nm_-1.00V_cut_from_grid2048x2048x2048.npy' -print -quit)"
report="${serial_dir}/mpi_serial_comparison_current_code.json"
residual_log="${serial_dir}/residual_history.csv"

if [[ -z "${serial_file}" || -z "${mpi_file}" ]]; then
    echo "ERROR: required cut NPY is missing." >&2
    echo "Serial directory: ${serial_dir}" >&2
    echo "MPI directory:    ${mpi_dir}" >&2
    exit 2
fi

# A cut emitted after the serial solver's per-level time limit is not a valid
# parity reference. The final row is the final hierarchy level's residual;
# require it to meet the same tolerance declared in the control JSON before
# interpreting a numerical comparison.
if [[ ! -s "${residual_log}" ]]; then
    echo "ERROR: serial residual log is missing: ${residual_log}" >&2
    exit 2
fi
python - "${PROJECT_ROOT}/afm_config_nm_serial_control_2048.json" "${residual_log}" <<'PY'
import csv
import json
import math
import sys

config_path, residual_path = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    tolerance = float(json.load(handle)["res_tol_main"])
with open(residual_path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit("ERROR: serial residual log has no data rows")
last = rows[-1]
residual = float(last["residual_avg"])
if not math.isfinite(residual) or residual > tolerance:
    raise SystemExit(
        f"ERROR: serial final residual {residual:.6e} exceeds tolerance {tolerance:.6e}; "
        "refusing to use an unconverged serial field as the MPI reference"
    )
print(f"Serial final residual: {residual:.6e} (tolerance {tolerance:.6e})")
PY

echo "Serial reference: ${serial_file}"
echo "MPI candidate:    ${mpi_file}"
echo "Report:           ${report}"

set +e
python postprocessing/compare_mpi_npy.py \
    "${serial_file}" "${mpi_file}" \
    --atol 2e-5 --rtol 2e-5 --report "${report}"
comparison_rc=$?
set -e

case "${comparison_rc}" in
    0)
        echo "COMPARISON_RESULT=PASS"
        ;;
    1)
        echo "COMPARISON_RESULT=NUMERIC_MISMATCH"
        echo "The report was written; inspect its max/rms differences before changing the solver."
        ;;
    *)
        echo "ERROR: comparison program failed with exit code ${comparison_rc}." >&2
        exit "${comparison_rc}"
        ;;
esac
