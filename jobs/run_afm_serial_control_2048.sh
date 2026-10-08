#!/usr/bin/env bash
# One-position, one-voltage serial control for validating the MPI 2048^3
# result against the current package revision.  Submit from the package root:
#   sbatch jobs/run_afm_serial_control_2048.sh
#
# 96 CPUs and 384 GiB preserve the required 4 GiB-per-core allocation ratio.
# The one-hour wall limit is deliberate: this is a bounded parity control, not
# the 108-case production sweep.
#SBATCH --job-name=afm-serial-2048
#SBATCH --partition=cpubase_bycore_b5
#SBATCH --cpus-per-task=96
#SBATCH --mem=384G
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
    "afm_config_nm_serial_control_2048.json" "$@"
