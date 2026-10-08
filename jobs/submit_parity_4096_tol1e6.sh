#!/usr/bin/env bash
# Login-node submission for two independently gated voltage cases.
# Retry resumes from the package-local ledger; it never intentionally duplicates jobs.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p outputs/slurm_logs outputs/submissions/parity_4096_tol1e6_20260908
LEDGER="outputs/submissions/parity_4096_tol1e6_20260908/jobs.tsv"
exec 9>outputs/submissions/parity_4096_tol1e6_20260908/submit.lock
flock -n 9 || { echo 'ERROR: another submission is in progress' >&2; exit 2; }

for voltage in minus1V minus9V; do
    bash jobs/preflight_afm.sh "afm_config_nm_parity_4096_tol1e6_serial_${voltage}.json"
    bash jobs/preflight_afm_mpi.sh "afm_config_nm_parity_4096_tol1e6_mpi_${voltage}.json" 512 32 128
    sbatch --test-only "jobs/run_afm_parity_4096_tol1e6_serial_${voltage}.sh"
    sbatch --test-only "jobs/run_afm_parity_4096_tol1e6_mpi_${voltage}.sh"
    sbatch --test-only --export=ALL,AFM_SERIAL_JOB_ID=1,AFM_MPI_JOB_ID=2 \
        jobs/run_compare_parity_4096_tol1e6.sh "${voltage}"
done

submit_once() {
    local role="$1" voltage="$2" found jobid
    shift 2
    found="$(awk -F '\t' -v r="${role}" -v v="${voltage}" '$1==r && $2==v {print $3}' "${LEDGER}" 2>/dev/null || true)"
    if [[ -n "${found}" ]]; then
        [[ "${found}" =~ ^[0-9]+$ ]] || { echo 'ERROR: malformed submission ledger' >&2; return 2; }
        printf '%s\n' "${found}"
        return
    fi
    jobid="$(sbatch --parsable "$@")"
    jobid="${jobid%%;*}"
    [[ "${jobid}" =~ ^[0-9]+$ ]] || { echo "ERROR: ambiguous sbatch response: ${jobid}; inspect queue before retry" >&2; return 2; }
    printf '%s\t%s\t%s\n' "${role}" "${voltage}" "${jobid}" >> "${LEDGER}"
    printf '%s\n' "${jobid}"
}

for voltage in minus1V minus9V; do
    serial_job="$(submit_once serial "${voltage}" "jobs/run_afm_parity_4096_tol1e6_serial_${voltage}.sh")"
    mpi_job="$(submit_once mpi "${voltage}" "jobs/run_afm_parity_4096_tol1e6_mpi_${voltage}.sh")"
    compare_job="$(submit_once comparison "${voltage}" \
        --job-name="afm-c4096-t6-${voltage}" \
        --dependency="afterok:${serial_job}:${mpi_job}" --kill-on-invalid-dep=yes \
        --export="ALL,AFM_SERIAL_JOB_ID=${serial_job},AFM_MPI_JOB_ID=${mpi_job}" \
        jobs/run_compare_parity_4096_tol1e6.sh "${voltage}")"
    echo "${voltage}: serial=${serial_job} mpi=${mpi_job} comparison=${compare_job}"
done
cat "${LEDGER}"
