# Relative -9 V parity and float64 memory probes

Dated September experiments retained October 7, 2026. Results are historical,
not newly run convergence tests or changes to production defaults. See the
[current package catalogue](../docs/CONFIGS_AND_JOBS.md).

This workflow contains two independent experiments. It does not change the
default float32/absolute-residual behavior of existing configurations.

## Relative-drive -9 V parity

The serial and MPI jobs solve the linear dielectric problem after dividing all
Dirichlet voltages by the largest absolute drive (9 V). They therefore converge
against `res_tol_main = 1e-6` in normalized units, then multiply the completed
potential back to physical volts before evaluating or saving it. The dependent
comparison requires equal finite 200x200x200 float32 cuts within absolute
`1e-6` V and separately records exact-array and byte-level identity. If the
earlier completed -1 V MPI cut is present, the report also checks the expected
linear relation `phi(-9 V) = 9 * phi(-1 V)` after float32 rounding.

- Serial: one large-memory node, 192 threads, 2560 GiB, 3 hours.
- MPI: 512 ranks over 16 nodes, 32 ranks/node, 4 GiB/rank, 20 minutes.
- Output: `outputs/job_<id>/<config-stem>/`.
- Comparison: `outputs/job_<id>/parity_4096_tol1e6_relative_minus9V/`.

## Float64 4096 memory smoke tests

These are diagnostic allocation/throughput probes, not converged scientific
solutions. Both start directly at 2048 cubed, execute exactly 10 weighted-Jacobi
iterations, prolongate to 4096 cubed, and execute exactly 20 iterations. The
solver field, dielectric coefficients, work buffers, halo fields, and
prolongation remain float64. Each run saves only the compact physical 100 nm
cut; no full-grid or intermediate NPY is written.

- Serial: one large-memory node, 192 threads, 4096 GiB, 3 hours.
- MPI: 512 ranks over 32 nodes, 16 ranks/node, 2 CPUs/rank and 4 GiB/CPU,
  totaling 4 TiB while preserving the Alliance 4 GiB/core-equivalent ratio.
- Serial evidence: `memory_live_rss.csv`, `memory_usage_log.csv`, and
  `float64_memory_smoke_levels.csv` in the run-result directory.
- MPI evidence: `memory_usage_mpi.csv` and `mg_timing_mpi.csv` under the case's
  `mpi_logs` directory. The `faces_ready_with_epsilon` stage captures the
  intended peak before the temporary epsilon halo is released.

Submit all five jobs once from the package root:

```bash
bash jobs/submit_relative_minus9_and_float64_smokes_20260909.sh
```

The script preflights and scheduler-validates all jobs before submission, then
records the IDs in
`outputs/submissions/relative_minus9_float64_20260909/jobs.tsv`. Re-running it
uses the recorded IDs and does not duplicate jobs.

Batch stdout/stderr are kept outside scientific result subdirectories at
`outputs/slurm_logs/<job-name>-<job-id>.out/.err`.
