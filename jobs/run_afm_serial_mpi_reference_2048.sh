#!/usr/bin/env bash
# Single-node, shared-memory reference for the current 2048^3 MPI parity gate.
# The test tolerance is 1e-5 to leave substantial margin inside one hour; the
# field-comparison acceptance threshold remains the stricter 1e-6.
#SBATCH --job-name=afm-serial-ref-2048
#SBATCH --partition=cpubase_bycore_b5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
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
if [[ -f "${SUBMIT_DIR}/run_all.py" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../run_all.py" ]]; then
    PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
    echo "ERROR: could not locate AFM package from SLURM_SUBMIT_DIR=${SUBMIT_DIR}." >&2
    exit 2
fi

exec bash "${PROJECT_ROOT}/jobs/run_afm.sh" \
    "afm_config_nm_serial_mpi_comparison_serial_2048.json" "$@"
