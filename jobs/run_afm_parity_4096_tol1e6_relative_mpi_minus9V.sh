#!/usr/bin/env bash
# Relative-drive 1e-6 MPI parity candidate at -9 V.
#SBATCH --job-name=afm-m4096-rel9-t6
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=16
#SBATCH --ntasks=512
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:20:00
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
    afm_config_nm_parity_4096_tol1e6_relative_mpi_minus9V.json "$@"
