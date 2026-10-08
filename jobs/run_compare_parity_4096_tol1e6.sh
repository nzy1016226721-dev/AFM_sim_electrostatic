#!/usr/bin/env bash
# Compare only the saved QD-centred 4096^3 physical cuts from the matched
# current-code serial and MPI jobs. Numerical acceptance is abs <= 1e-6 with
# rtol=0; an independent zero-tolerance report records exact parity.
#SBATCH --job-name=afm-compare4096-t6
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:15:00
#SBATCH --account=rrg-hongguo-ad
#SBATCH --output=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.err

set -euo pipefail

: "${AFM_SERIAL_JOB_ID:?ERROR: AFM_SERIAL_JOB_ID is required.}"
: "${AFM_MPI_JOB_ID:?ERROR: AFM_MPI_JOB_ID is required.}"

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
PYTHON_BIN="${PYTHON_BIN:-python}"

VOLTAGE_TAG="${1:?ERROR: supply minus1V or minus9V}"
case "${VOLTAGE_TAG}" in
    minus1V) VOLTAGE_FIXED="-1.00"; VOLTAGE_LOG="-1.000000" ;;
    minus9V) VOLTAGE_FIXED="-9.00"; VOLTAGE_LOG="-9.000000" ;;
    *) echo "ERROR: unsupported voltage tag: ${VOLTAGE_TAG}" >&2; exit 2 ;;
esac
SERIAL_STEM="afm_config_nm_parity_4096_tol1e6_serial_${VOLTAGE_TAG}"
MPI_STEM="afm_config_nm_parity_4096_tol1e6_mpi_${VOLTAGE_TAG}"
SERIAL_DIR="${PROJECT_ROOT}/outputs/job_${AFM_SERIAL_JOB_ID}/${SERIAL_STEM}"
MPI_DIR="${PROJECT_ROOT}/outputs/job_${AFM_MPI_JOB_ID}/${MPI_STEM}"
CUT_PATTERN="*_0nm_${VOLTAGE_FIXED}V_cut_from_grid4096x4096x4096.npy"
SERIAL_FILE="$(find "${SERIAL_DIR}" -maxdepth 1 -type f -name "${CUT_PATTERN}" -print -quit)"
MPI_FILE="$(find "${MPI_DIR}" -maxdepth 1 -type f -name "${CUT_PATTERN}" -print -quit)"
MPI_TIMING="${MPI_DIR}/mpi_logs/move_0nm_V${VOLTAGE_LOG}/mg_timing_mpi.csv"
SERIAL_RESIDUAL="${SERIAL_DIR}/residual_history.csv"
REPORT_DIR="${PROJECT_ROOT}/outputs/job_${SLURM_JOB_ID}/parity_4096_tol1e6_${VOLTAGE_TAG}"
EXACT_REPORT="${REPORT_DIR}/comparison_exact.json"
ACCEPTANCE_REPORT="${REPORT_DIR}/comparison_atol_1e-6.json"
mkdir -p "${REPORT_DIR}"

if [[ -z "${SERIAL_FILE}" || -z "${MPI_FILE}" ]]; then
    echo "ERROR: one or both cut NPY files are missing." >&2
    echo "Serial directory: ${SERIAL_DIR}" >&2
    echo "MPI directory:    ${MPI_DIR}" >&2
    exit 2
fi
if [[ ! -s "${SERIAL_RESIDUAL}" || ! -s "${MPI_TIMING}" ]]; then
    echo "ERROR: convergence evidence is missing." >&2
    exit 2
fi

"${PYTHON_BIN}" - \
    "${PROJECT_ROOT}/${SERIAL_STEM}.json" \
    "${SERIAL_RESIDUAL}" "${MPI_TIMING}" <<'PY'
import csv
import math
import sys
from simulation.mpi_config import load_afm_config

config_path, serial_path, mpi_path = sys.argv[1:]
tolerance = float(load_afm_config(config_path)[1]["res_tol_main"])
if tolerance != 1e-6:
    raise SystemExit("ERROR: this trial requires res_tol_main=1e-6")

with open(serial_path, newline="", encoding="utf-8") as handle:
    serial_rows = list(csv.DictReader(handle))
with open(mpi_path, newline="", encoding="utf-8") as handle:
    mpi_rows = list(csv.DictReader(handle))
if mpi_rows and (tuple(int(mpi_rows[-1][k]) for k in ("Nx", "Ny", "Nz")) != (4096, 4096, 4096)
                 or any(r["result"] != "converged" for r in mpi_rows)):
    raise SystemExit("ERROR: MPI hierarchy incomplete or a level failed convergence")
if not serial_rows or not mpi_rows:
    raise SystemExit("ERROR: an empty convergence log cannot validate a field")
serial_residual = float(serial_rows[-1]["residual_avg"])
mpi_residual = float(mpi_rows[-1]["residual"])
mpi_result = mpi_rows[-1]["result"]
if not math.isfinite(serial_residual) or serial_residual > tolerance:
    raise SystemExit(
        f"ERROR: serial residual {serial_residual:.12e} exceeds {tolerance:.12e}"
    )
if mpi_result != "converged" or not math.isfinite(mpi_residual) or mpi_residual > tolerance:
    raise SystemExit(
        f"ERROR: MPI final result={mpi_result}, residual={mpi_residual:.12e}, "
        f"tolerance={tolerance:.12e}"
    )
print(f"SERIAL_FINAL_RESIDUAL={serial_residual:.12e}")
print(f"MPI_FINAL_RESIDUAL={mpi_residual:.12e}")
print(f"SOLVER_TOLERANCE={tolerance:.12e}")
PY

echo "Serial cut: ${SERIAL_FILE}"
echo "MPI cut:    ${MPI_FILE}"

set +e
"${PYTHON_BIN}" postprocessing/compare_mpi_npy.py \
    "${SERIAL_FILE}" "${MPI_FILE}" --atol 0 --rtol 0 \
    --chunk-planes 1 --report "${EXACT_REPORT}"
exact_rc=$?
set -e

"${PYTHON_BIN}" postprocessing/compare_mpi_npy.py \
    "${SERIAL_FILE}" "${MPI_FILE}" --atol 1e-6 --rtol 0 \
    --chunk-planes 1 --report "${ACCEPTANCE_REPORT}"

"${PYTHON_BIN}" - "${SERIAL_FILE}" "${MPI_FILE}" "${REPORT_DIR}" <<'PY'
import hashlib, json, pathlib, sys
from postprocessing.compare_mpi_npy import compare_npy_fields
a,b,out = map(pathlib.Path,sys.argv[1:])
summary = json.loads((out/"comparison_atol_1e-6.json").read_text())
if summary["reference_dtype"] != "float32" or summary["reference_shape"] != [200,200,200]:
    raise SystemExit("ERROR: expected a float32 physical cut with shape 200x200x200")
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()
ha,hb=sha(a),sha(b)
proof={"reference_sha256":ha,"candidate_sha256":hb,"byte_identical":ha==hb}
(out/"file_hashes.json").write_text(json.dumps(proof,indent=2)+"\n")
print("BYTE_IDENTICAL="+str(ha==hb))
# Additional accuracy evidence for -1 V: compare against the completed 1e-5 MPI cut.
if "_minus1V" in str(b.parent):
    old=pathlib.Path("outputs/job_57894384/afm_config_nm_serial_mpi_comparison_mpi_4096")/b.name
    if old.exists():
        delta=compare_npy_fields(old,b,atol=1e-6,rtol=0)
        (out/"tolerance_change_1e5_to_1e6.json").write_text(json.dumps(delta,indent=2)+"\n")
        print("TOLERANCE_CHANGE_MAX_ABS="+str(delta["max_abs_diff"]))
PY

if [[ ${exact_rc} -eq 0 ]]; then
    echo "COMPARISON_RESULT=EXACT_VALUES (see file_hashes.json for byte identity)"
else
    echo "COMPARISON_RESULT=PASS_WITHIN_1E-6"
fi
echo "EXACT_REPORT=${EXACT_REPORT}"
echo "ACCEPTANCE_REPORT=${ACCEPTANCE_REPORT}"
