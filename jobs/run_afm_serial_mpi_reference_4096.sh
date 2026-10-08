#!/usr/bin/env bash
# Single-node shared-memory reference for the prepared 4096^3 parity gate.
# The scientific case and 1e-5 solver tolerance match the validated 2048^3
# gate. Only the 100 nm x 100 nm x 100 nm physical cut is retained.
# 2048^3 serial job 57843357 measured 211.73 GiB peak RSS. Cubic scaling
# predicts about 1694 GiB at 4096^3; 2560 GiB adds about 51% headroom.
# This is an estimate until a complete 4096^3 run records its actual peak.
# b1 supports this three-hour job on all eight large-memory nodes; b5
# unnecessarily restricted eligibility to two nodes at the 2026-09-07 check.
#SBATCH --job-name=afm-serial-ref-4096
#SBATCH --partition=cpularge_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=2560G
#SBATCH --time=03:00:00
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
    "afm_config_nm_serial_mpi_comparison_serial_4096.json" "$@"
