# 4096³ serial/MPI comparisons at 1e-6: -1 V and -9 V

Dated September study/evidence retained October 7, 2026; this publication does
not rerun its campaigns or refresh job status. See the
[current package catalogue](../docs/CONFIGS_AND_JOBS.md).

The two voltage cases use independent serial, MPI and dependent comparison
jobs. The frozen base JSON preserves the geometry of the successful 1e-5
comparison. Each case starts from the same initial condition independently:
there is no warm start from the other voltage.

- `res_tol_main = res_tol_zoom = 1e-6`; zoom disabled.
- Seven levels: 64→128→256→512→1024→2048→4096.
- No iteration cap and no per-level timeout; Slurm wall time is the time cap.
- Only the 200³ physical cut is saved, not the 4096³ field or intermediate fields.
- Serial: 192 threads on one node, 2560 GiB, 12 hours, `cpularge_bynode_b2`.
- MPI: 512 ranks, 16 nodes, 32 ranks/node, one CPU/rank, 4 GiB/CPU,
  128 GiB/node (2 TiB total), one hour, `cpubase_bycore_b1`.
- Comparison: one CPU, 4 GiB, 15 minutes, `cpubase_bycore_b1`.

The serial memory request retains about 51% headroom above the measured
1693 GiB peak. Its longer time limit allows slower convergence while keeping
all eight large-memory nodes eligible. MPI retains the validated rank layout
and memory request. One hour is a test budget, not a measured completion
prediction. Both jobs exit when finished; reserved time is not idle runtime.

Serial is the existing multithreaded run_all.py path; MPI uses run_mpi.py.
Each solver/voltage has a JSON and a matching submission shell wrapper.
The filenames include `serial`/`mpi`, `4096`, `tol1e6`, and `minus1V`/`minus9V`.

Submit the complete workflow from the Fir package root:

```bash
bash jobs/submit_parity_4096_tol1e6.sh
```

The script preflights every case and keeps an idempotent submission ledger
at `outputs/submissions/parity_4096_tol1e6_20260908/jobs.tsv`. Rerunning it
reuses recorded job IDs; it does not automatically retry failed simulations.
If submission output is ambiguous, inspect the queue before retrying.

The comparison runs only after both corresponding solvers exit successfully.
It rejects non-converged final residuals, missing cuts, incomplete MPI levels,
non-finite values, mismatched shapes/dtypes, and differences over 1e-6
(absolute; rtol=0). An independent zero-tolerance report and SHA256 report
record exact numerical and byte-level identity. For -1 V it also records the
change from the completed 1e-5 MPI cut when that file is available; that
accuracy diagnostic does not change the serial/MPI agreement gate.

Results: `outputs/job_<id>/<config-stem>/`.
Batch stdout/stderr: `outputs/slurm_logs/<job-name>-<id>.out/.err`.
Comparison: `outputs/job_<id>/parity_4096_tol1e6_<voltage>/`.

## Completed absolute-mode result and -9 V follow-up

The absolute-residual -1 V pair completed with an exactly identical serial/MPI
cut. At -9 V, however, both float32 implementations reached an approximately
`1.9e-6` numerical residual floor at 1024 cubed and exhausted their wall times,
so their dependent comparison could not run. The follow-up configuration uses
the explicit `relative_drive` mode documented in
`RELATIVE_MINUS9_AND_FLOAT64_SMOKE.md`: it solves the normalized unit-drive
problem and rescales only after convergence. Existing configurations retain
absolute mode unless they opt in explicitly.
