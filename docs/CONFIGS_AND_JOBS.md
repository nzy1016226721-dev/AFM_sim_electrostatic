# Configuration and job catalogue

October 7, 2026 source inventory. The 27 root JSONs and all distinct executable
job scripts are retained unchanged. Similar names are intentional variants,
not duplicate physics to merge. Inherited settings resolve relative to each
file. Read the [numerical contract](../DOCUMENTATION.md) before editing them.
Submitting any script below launches actual work and requires suitable resources.

## Config families

| Purpose | Configurations | Matching entry/script |
| --- | --- | --- |
| Canonical shared-memory sweep | [afm_config_nm.json](../afm_config_nm.json) | `run_all.py` / `jobs/run_afm.sh` |
| Production 2048/4096 | [2048](../afm_config_nm_production_2048.json), [4096](../afm_config_nm_production_4096.json) | `run_afm_nm_2048.sh`, `run_afm_nm_4096.sh` |
| Shared-memory 2048 control | [control](../afm_config_nm_serial_control_2048.json) | `run_afm_serial_control_2048.sh` |
| 2048 MPI trial | [trial](../afm_config_nm_mpi_trial_2048.json) | `run_afm_mpi_trial_2048.sh` |
| 16384 MPI template | [template](../afm_config_nm_mpi_16384_template.json) | `run_afm_mpi.sh`; plan before any allocation |
| 2048 serial/MPI comparison | [base](../afm_config_nm_serial_mpi_comparison_2048_base.json), [serial](../afm_config_nm_serial_mpi_comparison_serial_2048.json), [MPI](../afm_config_nm_serial_mpi_comparison_mpi_2048.json) | `submit_serial_mpi_comparison_2048.sh` |
| 4096 serial/MPI comparison | [base](../afm_config_nm_serial_mpi_comparison_4096_base.json), [serial](../afm_config_nm_serial_mpi_comparison_serial_4096.json), [MPI](../afm_config_nm_serial_mpi_comparison_mpi_4096.json) | `submit_serial_mpi_comparison_4096.sh` |
| 4096 MPI decomposition comparison | [base](../afm_config_nm_mpi_comparison_4096_base.json), [8×8×8](../afm_config_nm_mpi_comparison_4096_layout_8x8x8.json), [16×8×4](../afm_config_nm_mpi_comparison_4096_layout_16x8x4.json) | `run_afm_mpi_comparison_4096.sh` |
| Absolute 4096, native tolerance 1e-6 | [base](../afm_config_nm_parity_4096_tol1e6_base.json), [serial −1](../afm_config_nm_parity_4096_tol1e6_serial_minus1V.json), [MPI −1](../afm_config_nm_parity_4096_tol1e6_mpi_minus1V.json), [serial −9](../afm_config_nm_parity_4096_tol1e6_serial_minus9V.json), [MPI −9](../afm_config_nm_parity_4096_tol1e6_mpi_minus9V.json) | `submit_parity_4096_tol1e6.sh` |
| Relative-drive −9 parity | [serial](../afm_config_nm_parity_4096_tol1e6_relative_serial_minus9V.json), [MPI](../afm_config_nm_parity_4096_tol1e6_relative_mpi_minus9V.json) | `submit_relative_minus9_and_float64_smokes_20260909.sh` |
| Historical 800 local parity | [serial](../afm_config_nm_local_comparison_serial_800.json), [MPI](../afm_config_nm_local_comparison_mpi_800.json) | `run_local_mpi_serial_comparison_800.ps1` |
| Memory-only / opt-in float64 diagnostics | [memory](../afm_memory_test.json), [float64 serial](../afm_config_nm_float64_memory_smoke_serial_4096.json), [float64 MPI](../afm_config_nm_float64_memory_smoke_mpi_4096.json) | `run_memory_test.sh`, `run_afm_float64_memory_smoke_*_4096.sh`; not convergence evidence |

Do not retune or automatically run these large studies as part of installation.
Configured tolerances, voltages, thread counts, layouts and diagnostic choices
are deliberately preserved. A −9-V relative-drive run does not have the same
native absolute-residual interpretation as the absolute-mode −1-V run.

## Jobs: all 34 shell scripts

Setup/preflight and generic launch:

- [setup_afm_env.sh](../jobs/setup_afm_env.sh), [setup_afm_mpi_env.sh](../jobs/setup_afm_mpi_env.sh)
- [preflight_afm.sh](../jobs/preflight_afm.sh), [preflight_afm_mpi.sh](../jobs/preflight_afm_mpi.sh)
- [run_afm.sh](../jobs/run_afm.sh), [run_afm_mpi.sh](../jobs/run_afm_mpi.sh)
- [run_mpi_serial_parity_smoke.sh](../jobs/run_mpi_serial_parity_smoke.sh)

Production, memory and direct controls:

- [run_afm_nm_2048.sh](../jobs/run_afm_nm_2048.sh), [run_afm_nm_4096.sh](../jobs/run_afm_nm_4096.sh)
- [run_afm_serial_control_2048.sh](../jobs/run_afm_serial_control_2048.sh), [run_memory_test.sh](../jobs/run_memory_test.sh)
- [run_afm_mpi_trial_2048.sh](../jobs/run_afm_mpi_trial_2048.sh), [run_afm_mpi_comparison_4096.sh](../jobs/run_afm_mpi_comparison_4096.sh)

Serial/MPI comparison families:

- [run_afm_serial_mpi_reference_2048.sh](../jobs/run_afm_serial_mpi_reference_2048.sh), [run_afm_serial_mpi_candidate_2048.sh](../jobs/run_afm_serial_mpi_candidate_2048.sh)
- [run_compare_serial_mpi_current_2048.sh](../jobs/run_compare_serial_mpi_current_2048.sh), [run_compare_mpi_serial_2048.sh](../jobs/run_compare_mpi_serial_2048.sh)
- [submit_serial_mpi_comparison_2048.sh](../jobs/submit_serial_mpi_comparison_2048.sh)
- [run_afm_serial_mpi_reference_4096.sh](../jobs/run_afm_serial_mpi_reference_4096.sh), [run_afm_serial_mpi_candidate_4096.sh](../jobs/run_afm_serial_mpi_candidate_4096.sh)
- [run_compare_serial_mpi_current_4096.sh](../jobs/run_compare_serial_mpi_current_4096.sh), [submit_serial_mpi_comparison_4096.sh](../jobs/submit_serial_mpi_comparison_4096.sh)

Absolute/relative-drive parity and opt-in float64 diagnostics:

- [run_afm_parity_4096_tol1e6_serial_minus1V.sh](../jobs/run_afm_parity_4096_tol1e6_serial_minus1V.sh), [run_afm_parity_4096_tol1e6_mpi_minus1V.sh](../jobs/run_afm_parity_4096_tol1e6_mpi_minus1V.sh)
- [run_afm_parity_4096_tol1e6_serial_minus9V.sh](../jobs/run_afm_parity_4096_tol1e6_serial_minus9V.sh), [run_afm_parity_4096_tol1e6_mpi_minus9V.sh](../jobs/run_afm_parity_4096_tol1e6_mpi_minus9V.sh)
- [run_compare_parity_4096_tol1e6.sh](../jobs/run_compare_parity_4096_tol1e6.sh), [submit_parity_4096_tol1e6.sh](../jobs/submit_parity_4096_tol1e6.sh)
- [run_afm_parity_4096_tol1e6_relative_serial_minus9V.sh](../jobs/run_afm_parity_4096_tol1e6_relative_serial_minus9V.sh), [run_afm_parity_4096_tol1e6_relative_mpi_minus9V.sh](../jobs/run_afm_parity_4096_tol1e6_relative_mpi_minus9V.sh)
- [run_compare_parity_4096_tol1e6_relative_minus9V.sh](../jobs/run_compare_parity_4096_tol1e6_relative_minus9V.sh)
- [run_afm_float64_memory_smoke_serial_4096.sh](../jobs/run_afm_float64_memory_smoke_serial_4096.sh), [run_afm_float64_memory_smoke_mpi_4096.sh](../jobs/run_afm_float64_memory_smoke_mpi_4096.sh)
- [submit_relative_minus9_and_float64_smokes_20260909.sh](../jobs/submit_relative_minus9_and_float64_smokes_20260909.sh)

The retained [PowerShell 800³ launcher](../jobs/run_local_mpi_serial_comparison_800.ps1)
is separate from these shell scripts. Distinct requests/wrappers have different
resources, layouts and checks; they are not byte-identical redundant copies.

## Dated study notes

- [Production preparation](../jobs/PRODUCTION_RUNS.md)
- [Absolute 4096 parity and completed/failed history](../jobs/PARITY_4096_TOL1E6.md)
- [Relative-drive/float64 diagnostic history](../jobs/RELATIVE_MINUS9_AND_FLOAT64_SMOKE.md)
- [Historical serial 4096 resource measurement](../jobs/SERIAL_4096_RESOURCE_NOTE.md)
- [September local parity evidence](../MPI_PARITY_VALIDATION_20260902.md)

Job-state, disk-headroom, timing and upload statements in these retained notes
are dated observations, not live status. Outputs are local/cluster evidence,
not shipped source. All submitters must be reviewed for account/partition/path
and present resource availability before use.
