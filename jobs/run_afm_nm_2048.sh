#!/usr/bin/env bash
# Full AFM nanometre-coordinate production sweep at 2048^3.
#
# This wrapper deliberately owns only Slurm resources and the fixed
# configuration name.  The canonical non-interactive runtime implementation
# remains jobs/run_afm.sh, avoiding a second copy of environment/output logic.
#
# Submit from /home/nizy/afm_parallel:
#   sbatch jobs/run_afm_nm_2048.sh
#
# The 384 GiB / 96 CPU request follows the 4 GiB-per-core charging ratio and
# leaves headroom above the 2048^3 diagnostic's approximately 275 GiB peak.
#SBATCH --job-name=afm-nm-2048
#SBATCH --partition=cpubase_bycore_b5
#SBATCH --cpus-per-task=96
#SBATCH --mem=384G
#SBATCH --time=24:00:00
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

exec "${PROJECT_ROOT}/jobs/run_afm.sh" "afm_config_nm_production_2048.json" "$@"
