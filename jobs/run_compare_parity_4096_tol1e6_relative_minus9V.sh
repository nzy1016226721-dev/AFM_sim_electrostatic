#!/usr/bin/env bash
# Validate serial/MPI parity for the -9 V relative-drive 4096^3 physical cuts.
#SBATCH --job-name=afm-c4096-rel9-t6
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
: "${AFM_SERIAL_JOB_ID:?ERROR: AFM_SERIAL_JOB_ID is required}"
: "${AFM_MPI_JOB_ID:?ERROR: AFM_MPI_JOB_ID is required}"

SUBMIT_DIR="$(cd -- "${SLURM_SUBMIT_DIR:-${PWD}}" && pwd)"
if [[ -f "${SUBMIT_DIR}/run_all.py" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../run_all.py" ]]; then
    PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
    echo "ERROR: could not locate AFM package." >&2
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

SERIAL_STEM=afm_config_nm_parity_4096_tol1e6_relative_serial_minus9V
MPI_STEM=afm_config_nm_parity_4096_tol1e6_relative_mpi_minus9V
SERIAL_DIR="${PROJECT_ROOT}/outputs/job_${AFM_SERIAL_JOB_ID}/${SERIAL_STEM}"
MPI_DIR="${PROJECT_ROOT}/outputs/job_${AFM_MPI_JOB_ID}/${MPI_STEM}"
SERIAL_FILE="$(find "${SERIAL_DIR}" -maxdepth 1 -type f -name '*_0nm_-9.00V_cut_from_grid4096x4096x4096.npy' -print -quit 2>/dev/null || true)"
MPI_FILE="$(find "${MPI_DIR}" -maxdepth 1 -type f -name '*_0nm_-9.00V_cut_from_grid4096x4096x4096.npy' -print -quit 2>/dev/null || true)"
SERIAL_RESIDUAL="${SERIAL_DIR}/residual_history.csv"
MPI_TIMING="${MPI_DIR}/mpi_logs/move_0nm_V-9.000000/mg_timing_mpi.csv"
REPORT_DIR="${PROJECT_ROOT}/outputs/job_${SLURM_JOB_ID}/parity_4096_tol1e6_relative_minus9V"
mkdir -p "${REPORT_DIR}"

if [[ -z "${SERIAL_FILE}" || -z "${MPI_FILE}" ]]; then
    echo "ERROR: one or both relative-drive cut files are missing." >&2
    exit 2
fi

"${PYTHON_BIN}" - \
    "${PROJECT_ROOT}/${SERIAL_STEM}.json" \
    "${PROJECT_ROOT}/${MPI_STEM}.json" \
    "${SERIAL_RESIDUAL}" "${MPI_TIMING}" <<'PY'
import csv
import math
import sys
from simulation.mpi_config import load_afm_config

serial_cfg_path, mpi_cfg_path, serial_path, mpi_path = sys.argv[1:]
serial_cfg = load_afm_config(serial_cfg_path)[1]
mpi_cfg = load_afm_config(mpi_cfg_path)[1]
for label, cfg in (("serial", serial_cfg), ("MPI", mpi_cfg)):
    if float(cfg["res_tol_main"]) != 1e-6:
        raise SystemExit(f"ERROR: {label} tolerance is not 1e-6")
    if cfg.get("residual_tolerance_mode") != "relative_drive":
        raise SystemExit(f"ERROR: {label} is not using relative_drive mode")
    if float(cfg["v_start"]) != -9 or float(cfg["v_stop"]) != -9:
        raise SystemExit(f"ERROR: {label} is not the -9 V case")

with open(serial_path, newline="", encoding="utf-8") as handle:
    serial_rows = list(csv.DictReader(handle))
with open(mpi_path, newline="", encoding="utf-8") as handle:
    mpi_rows = list(csv.DictReader(handle))
if not serial_rows or not mpi_rows:
    raise SystemExit("ERROR: convergence evidence is empty")
if tuple(int(mpi_rows[-1][key]) for key in ("Nx", "Ny", "Nz")) != (4096, 4096, 4096):
    raise SystemExit("ERROR: MPI hierarchy did not reach 4096^3")
if any(row["result"] != "converged" for row in mpi_rows):
    raise SystemExit("ERROR: one or more MPI levels did not converge")
serial_residual = float(serial_rows[-1]["residual_avg"])
mpi_residual = float(mpi_rows[-1]["residual"])
if not math.isfinite(serial_residual) or serial_residual > 1e-6:
    raise SystemExit(f"ERROR: serial normalized residual={serial_residual:.12e}")
if not math.isfinite(mpi_residual) or mpi_residual > 1e-6:
    raise SystemExit(f"ERROR: MPI normalized residual={mpi_residual:.12e}")
print(f"SERIAL_NORMALIZED_RESIDUAL={serial_residual:.12e}")
print(f"MPI_NORMALIZED_RESIDUAL={mpi_residual:.12e}")
print(f"SERIAL_PHYSICAL_RESIDUAL_BOUND_V={9*serial_residual:.12e}")
print(f"MPI_PHYSICAL_RESIDUAL_BOUND_V={9*mpi_residual:.12e}")
PY

EXACT_REPORT="${REPORT_DIR}/comparison_exact.json"
ACCEPTANCE_REPORT="${REPORT_DIR}/comparison_atol_1e-6.json"
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
import hashlib
import json
import pathlib
import sys
import numpy as np

serial_path, mpi_path, report_dir = map(pathlib.Path, sys.argv[1:])
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

serial_hash = sha(serial_path)
mpi_hash = sha(mpi_path)
(report_dir / "file_hashes.json").write_text(json.dumps({
    "serial_sha256": serial_hash,
    "mpi_sha256": mpi_hash,
    "byte_identical": serial_hash == mpi_hash,
}, indent=2) + "\n")
print(f"BYTE_IDENTICAL={serial_hash == mpi_hash}")

minus1 = pathlib.Path(
    "outputs/job_58724507/afm_config_nm_parity_4096_tol1e6_mpi_minus1V/"
    "afm_phi_1_0nm_-1.00V_cut_from_grid4096x4096x4096.npy"
)
if minus1.exists():
    reference = np.load(minus1, mmap_mode="r")
    candidate = np.load(mpi_path, mmap_mode="r")
    if reference.shape != candidate.shape:
        raise SystemExit("ERROR: -1 V and -9 V cut shapes differ")
    maximum = 0.0
    square_sum = 0.0
    count = 0
    for plane in range(reference.shape[0]):
        expected = np.multiply(reference[plane], np.float32(9.0))
        difference = candidate[plane].astype(np.float64) - expected.astype(np.float64)
        maximum = max(maximum, float(np.max(np.abs(difference))))
        square_sum += float(np.sum(difference * difference, dtype=np.float64))
        count += difference.size
    report = {
        "reference_minus1": str(minus1.resolve()),
        "candidate_minus9": str(mpi_path.resolve()),
        "relation": "minus9 = float32(9 * minus1)",
        "max_abs_diff_V": maximum,
        "rms_abs_diff_V": (square_sum / count) ** 0.5,
        "value_count": count,
        "passed_exact": maximum == 0.0,
    }
    (report_dir / "linearity_minus9_vs_9x_minus1.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"LINEARITY_MAX_ABS_V={maximum:.12e}")
PY

if [[ ${exact_rc} -eq 0 ]]; then
    echo "COMPARISON_RESULT=EXACT_VALUES"
else
    echo "COMPARISON_RESULT=PASS_WITHIN_1E-6"
fi
echo "REPORT_DIR=${REPORT_DIR}"
