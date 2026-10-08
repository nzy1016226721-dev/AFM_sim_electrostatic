#!/usr/bin/env bash
# 4096^3, 1e-6 average residual, minus1V; 512 ranks / 16 nodes.
# Validated 79 GiB node snapshots and 3.743 GiB task MaxRSS at 1e-5.
# 128 GiB per node retains the validated rank packing and memory headroom.
#SBATCH --job-name=afm-m4096-t6-minus1V
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

exec bash "${PROJECT_ROOT}/jobs/run_afm_mpi.sh" \
    "afm_config_nm_parity_4096_tol1e6_mpi_minus1V.json" "$@"
