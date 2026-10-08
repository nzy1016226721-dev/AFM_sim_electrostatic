#!/usr/bin/env bash
# Relative-drive 1e-6 parity reference at -9 V. Unit-bias solve, physical output.
# The -1 V reference completed in 1:53:19 with a 1693 GiB peak.
#SBATCH --job-name=afm-s4096-rel9-t6
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
    afm_config_nm_parity_4096_tol1e6_relative_serial_minus9V.json "$@"
