#!/usr/bin/env bash
# Login-node orchestrator for the prepared 4096^3 serial/MPI cut-only gate.
# Both solvers must exit successfully before comparison.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p outputs/slurm_logs

bash jobs/preflight_afm.sh afm_config_nm_serial_mpi_comparison_serial_4096.json
bash jobs/preflight_afm_mpi.sh \
    afm_config_nm_serial_mpi_comparison_mpi_4096.json 512 32 128

sbatch --test-only jobs/run_afm_serial_mpi_reference_4096.sh
sbatch --test-only jobs/run_afm_serial_mpi_candidate_4096.sh
sbatch --test-only \
    --export=ALL,AFM_SERIAL_JOB_ID=1,AFM_MPI_JOB_ID=2 \
    jobs/run_compare_serial_mpi_current_4096.sh

serial_job="$(sbatch --parsable jobs/run_afm_serial_mpi_reference_4096.sh)"
mpi_job="$(sbatch --parsable jobs/run_afm_serial_mpi_candidate_4096.sh)"
comparison_job="$(sbatch --parsable \
    --dependency="afterok:${serial_job}:${mpi_job}" \
    --kill-on-invalid-dep=yes \
    --export="ALL,AFM_SERIAL_JOB_ID=${serial_job},AFM_MPI_JOB_ID=${mpi_job}" \
    jobs/run_compare_serial_mpi_current_4096.sh)"

echo "AFM_SERIAL_JOB_ID=${serial_job}"
echo "AFM_MPI_JOB_ID=${mpi_job}"
echo "AFM_COMPARISON_JOB_ID=${comparison_job}"
squeue -j "${serial_job},${mpi_job},${comparison_job}" \
    -o "%.18i %.28j %.2t %.10M %.10l %.6D %.6C %.12m %R"
