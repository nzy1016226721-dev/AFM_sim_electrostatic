#!/usr/bin/env bash
# Diagnostic only: float64 2048^3 x 10 iterations, then 4096^3 x 20.
#SBATCH --job-name=afm-s4096-f64mem
#SBATCH --partition=cpularge_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=4096G
#SBATCH --time=03:00:00
#SBATCH --account=rrg-hongguo-ad
#SBATCH --output=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.err

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this wrapper with sbatch." >&2
    exit 2
fi
SUBMIT_DIR="$(cd -- "${SLURM_SUBMIT_DIR:-${PWD}}" && pwd)"
if [[ -f "${SUBMIT_DIR}/run_all.py" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../run_all.py" ]]; then
    PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
    echo "ERROR: could not locate AFM package." >&2
    exit 2
fi
exec bash "${PROJECT_ROOT}/jobs/run_afm.sh" \
    afm_config_nm_float64_memory_smoke_serial_4096.json "$@"
