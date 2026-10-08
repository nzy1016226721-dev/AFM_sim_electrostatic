#!/usr/bin/env bash
# Login-node orchestrator: preflight and submit the two one-hour solver jobs,
# then a small comparison job that starts only if both solvers succeed.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p outputs/slurm_logs

bash jobs/preflight_afm.sh afm_config_nm_serial_mpi_comparison_serial_2048.json
bash jobs/preflight_afm_mpi.sh \
    afm_config_nm_serial_mpi_comparison_mpi_2048.json 64 32 256

sbatch --test-only jobs/run_afm_serial_mpi_reference_2048.sh
sbatch --test-only jobs/run_afm_serial_mpi_candidate_2048.sh
sbatch --test-only \
    --export=ALL,AFM_SERIAL_JOB_ID=1,AFM_MPI_JOB_ID=2 \
    jobs/run_compare_serial_mpi_current_2048.sh

serial_job="$(sbatch --parsable jobs/run_afm_serial_mpi_reference_2048.sh)"
mpi_job="$(sbatch --parsable jobs/run_afm_serial_mpi_candidate_2048.sh)"
comparison_job="$(sbatch --parsable \
    --dependency="afterok:${serial_job}:${mpi_job}" \
    --kill-on-invalid-dep=yes \
    --export="ALL,AFM_SERIAL_JOB_ID=${serial_job},AFM_MPI_JOB_ID=${mpi_job}" \
    jobs/run_compare_serial_mpi_current_2048.sh)"

echo "AFM_SERIAL_JOB_ID=${serial_job}"
echo "AFM_MPI_JOB_ID=${mpi_job}"
echo "AFM_COMPARISON_JOB_ID=${comparison_job}"
squeue -j "${serial_job},${mpi_job},${comparison_job}" \
    -o "%.18i %.28j %.2t %.10M %.10l %.6D %.6C %.12m %R"
