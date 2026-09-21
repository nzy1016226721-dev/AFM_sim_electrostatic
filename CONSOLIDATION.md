# Parallel AFM package consolidation

Consolidated on 2026-08-29 from the parallel-solver variants found in
`C:\Users\NiN\Downloads`.

## Selected baseline

The newest `afm_trial_parallel_ready.tar` was selected as the baseline. Its
simulation and postprocessing Python files were byte-identical to the
`afm_package_alliance_fir_1024_trial_parallel` folder, while its environment
scripts contained the later `/home/<user>/afm_env` default requested for Fir.
The older `afm_package_alliance_fir_parallel` folder had the same solver core
but older configuration, launcher, and documentation revisions.

Only one production configuration and one production Slurm launcher are kept:

- `afm_config_nm.json`: 1024^3 Fir configuration (`cpu_threads=35`; the
  launcher allocates 35 CPUs with 140G, matching 4G/core accounting).
- `jobs/run_afm.sh`: generic Slurm entry point accepting any JSON path.
- `afm_memory_test.json` and `jobs/run_memory_test.sh`: diagnostic-only,
  package-local per-level memory test (1024^3 through 16384^3), using the
  single-node Fir high-memory allocation needed to reach the 4096^3 target;
  it saves no full fields and is not a production sweep.

The duplicate `run_afm_nm.sh`, the obsolete `_run_afm_common.sh` helper,
generated outputs, caches, logs, backup files, and large NumPy arrays were not
migrated. The generic launcher is self-contained so a staged Slurm script never
tries to execute a helper from the temporary spool directory.

## Additional hardening

- Removed the solver's unnecessary pandas runtime dependency; convergence CSV
  plotting now uses the Python standard library.
- Made `psutil` explicit in the main dependency file because the canonical
  configuration enables memory tracking.
- Added deterministic tests for the Numba Jacobi kernel, residual reductions,
  exact JSON selection, and job-specific output routing.
- Added `.gitignore` rules for generated result and cache files.

The full 1024^3 scientific workload cannot be validated on this laptop without
the Fir memory allocation. Local validation covers compilation, imports,
parallel-kernel equivalence on representative small grids, CLI/config routing,
and shell syntax. The supplied `jobs/preflight_afm.sh` validates the cluster
modules, virtual environment, thread count, grid, and tolerances before a full
submission.
