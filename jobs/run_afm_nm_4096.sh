#!/usr/bin/env bash
# Full AFM nanometre-coordinate production sweep at 4096^3.
#
# This wrapper fixes the resource request and configuration name while routing
# execution through the canonical non-interactive jobs/run_afm.sh launcher.
# It is uploaded now but must be submitted only after the direct 4096^3 ->
# 8192^3 memory trial supplies successful runtime evidence.
#
# Submit from /home/nizy/afm_parallel:
#   sbatch jobs/run_afm_nm_4096.sh
#
# A 4096^3 solve is expected to exceed base-node memory.  Fir high-memory
# nodes have 192 physical CPUs and 6 TiB RAM, so 4 TiB / 192 CPUs is the
# safe one-node topology limit even though the ideal 4 GiB-per-core billing
# ratio would require more CPUs than one Python process can use on that node.
#SBATCH --job-name=afm-nm-4096
#SBATCH --partition=cpularge_bynode_b5
#SBATCH --cpus-per-task=192
#SBATCH --mem=4096G
#SBATCH --time=4-00:00:00
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

exec "${PROJECT_ROOT}/jobs/run_afm.sh" "afm_config_nm_production_4096.json" "$@"
