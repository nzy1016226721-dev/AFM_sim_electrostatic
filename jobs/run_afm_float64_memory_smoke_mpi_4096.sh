#!/usr/bin/env bash
# Diagnostic only: float64 2048^3 x 10 iterations, then 4096^3 x 20.
# 512 ranks are spread over 32 nodes. Two CPUs/rank preserve the 4 GiB/core ratio.
#SBATCH --job-name=afm-m4096-f64mem
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=32
#SBATCH --ntasks=512
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --account=rrg-hongguo-ad
#SBATCH --output=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.err

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this wrapper with sbatch." >&2
    exit 2
fi
SUBMIT_DIR="$(cd -- "${SLURM_SUBMIT_DIR:-${PWD}}" && pwd)"
if [[ -f "${SUBMIT_DIR}/run_mpi.py" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../run_mpi.py" ]]; then
    PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
    echo "ERROR: could not locate AFM MPI package." >&2
    exit 2
fi
exec bash "${PROJECT_ROOT}/jobs/run_afm_mpi.sh" \
    afm_config_nm_float64_memory_smoke_mpi_4096.json "$@"
