#!/usr/bin/env bash
# Idempotently submit the relative -9 V parity gate and both float64 memory probes.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p outputs/slurm_logs outputs/submissions/relative_minus9_float64_20260909
LEDGER=outputs/submissions/relative_minus9_float64_20260909/jobs.tsv
exec 9>outputs/submissions/relative_minus9_float64_20260909/submit.lock
flock -n 9 || { echo "ERROR: another submission is in progress" >&2; exit 2; }

SERIAL_REL_CFG=afm_config_nm_parity_4096_tol1e6_relative_serial_minus9V.json
MPI_REL_CFG=afm_config_nm_parity_4096_tol1e6_relative_mpi_minus9V.json
SERIAL_F64_CFG=afm_config_nm_float64_memory_smoke_serial_4096.json
MPI_F64_CFG=afm_config_nm_float64_memory_smoke_mpi_4096.json

bash jobs/preflight_afm.sh "${SERIAL_REL_CFG}"
bash jobs/preflight_afm_mpi.sh "${MPI_REL_CFG}" 512 32 128
bash jobs/preflight_afm.sh "${SERIAL_F64_CFG}"
bash jobs/preflight_afm_mpi.sh "${MPI_F64_CFG}" 512 16 128

sbatch --test-only jobs/run_afm_parity_4096_tol1e6_relative_serial_minus9V.sh
sbatch --test-only jobs/run_afm_parity_4096_tol1e6_relative_mpi_minus9V.sh
sbatch --test-only --export=ALL,AFM_SERIAL_JOB_ID=1,AFM_MPI_JOB_ID=2 \
    jobs/run_compare_parity_4096_tol1e6_relative_minus9V.sh
sbatch --test-only jobs/run_afm_float64_memory_smoke_serial_4096.sh
sbatch --test-only jobs/run_afm_float64_memory_smoke_mpi_4096.sh

submit_once() {
    local role="$1"
    shift
    local existing jobid
    existing="$(awk -F '\t' -v role="${role}" '$1==role {print $2}' "${LEDGER}" 2>/dev/null || true)"
    if [[ -n "${existing}" ]]; then
        [[ "${existing}" =~ ^[0-9]+$ ]] || {
            echo "ERROR: malformed ledger entry for ${role}" >&2
            return 2
        }
        printf '%s\n' "${existing}"
        return
    fi
    jobid="$(sbatch --parsable "$@")"
    jobid="${jobid%%;*}"
    [[ "${jobid}" =~ ^[0-9]+$ ]] || {
        echo "ERROR: ambiguous sbatch result for ${role}: ${jobid}" >&2
        return 2
    }
    printf '%s\t%s\n' "${role}" "${jobid}" >> "${LEDGER}"
    printf '%s\n' "${jobid}"
}

relative_serial="$(submit_once relative_serial \
    jobs/run_afm_parity_4096_tol1e6_relative_serial_minus9V.sh)"
relative_mpi="$(submit_once relative_mpi \
    jobs/run_afm_parity_4096_tol1e6_relative_mpi_minus9V.sh)"
relative_compare="$(submit_once relative_compare \
    --dependency="afterok:${relative_serial}:${relative_mpi}" \
    --kill-on-invalid-dep=yes \
    --export="ALL,AFM_SERIAL_JOB_ID=${relative_serial},AFM_MPI_JOB_ID=${relative_mpi}" \
    jobs/run_compare_parity_4096_tol1e6_relative_minus9V.sh)"
float64_serial="$(submit_once float64_serial \
    jobs/run_afm_float64_memory_smoke_serial_4096.sh)"
float64_mpi="$(submit_once float64_mpi \
    jobs/run_afm_float64_memory_smoke_mpi_4096.sh)"

echo "RELATIVE_MINUS9_SERIAL_JOB_ID=${relative_serial}"
echo "RELATIVE_MINUS9_MPI_JOB_ID=${relative_mpi}"
echo "RELATIVE_MINUS9_COMPARE_JOB_ID=${relative_compare}"
echo "FLOAT64_SERIAL_SMOKE_JOB_ID=${float64_serial}"
echo "FLOAT64_MPI_SMOKE_JOB_ID=${float64_mpi}"
cat "${LEDGER}"
