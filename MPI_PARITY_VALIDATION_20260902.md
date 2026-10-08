# MPI numerical-parity validation — 2026-09-02

**Historical validation snapshot.** The 800³ parity measurements remain useful
evidence, but this file predates the current dynamic large-grid hierarchy policy.
The current implementation selects 6–8 levels (2048/4096/8192 start at 64³;
16384 starts at 128³) unless a config explicitly overrides the initial level.
The prepared Fir jobs described here were not thereby completed; consult
`../docs/CURRENT_STATE.md` and `README_MPI.md` for current contracts/status.

## Result

The current distributed solver matches the shared-memory solver bit-for-bit
for the completed production-geometry 800³ validation. The comparison loaded
and checked all 512,000,000 float32 values; maximum absolute, RMS, and maximum
relative differences were all zero, with zero non-finite values.

This is local Windows/MS-MPI evidence. The prepared 4096³ test has not been
uploaded or run on Fir.

## Corrections validated

- Jacobi denominator, numerator, relaxation, and final float32 rounding now
  follow the same order in serial and MPI kernels.
- Serial and MPI convergence checks use the same ten-iteration cadence.
- Distributed trilinear prolongation matches SciPy's endpoint-aligned
  coordinates, C-order eight-corner accumulation, float64 intermediate
  arithmetic, and final float32 cast.
- The upper physical endpoint no longer reads an invalid outer ghost, even
  with zero interpolation weight.
- Local tip axes/masking use the same `linspace` coordinates and geometry
  expression as the serial implementation.
- Normal non-plotting shared-memory solves use two Jacobi solution buffers and
  recompute the denominator, eliminating persistent denominator and residual
  volumes without changing results.

## Completed evidence

| Gate | Decomposition | Compared values | Maximum absolute difference | Result |
|---|---:|---:|---:|---|
| Random trilinear refinement vs SciPy | local kernel | multiple shapes | 0 | bitwise identical |
| 64³ serial vs MPI | 2x2x2 | 262,144 | 0 | bitwise identical |
| 64³ serial vs MPI | 4x2x1 | 262,144 | 0 | bitwise identical |
| 128³ strict serial vs optimized serial vs MPI | 2x2x2 | 2,097,152 | 0 | bitwise identical |
| 800³ optimized serial vs MPI | 2x2x2 | 512,000,000 | 0 | bitwise identical |

The 128³ gate anchors the optimized shared-memory path to the original strict
one-thread NumPy reference. The 800³ run used 16 shared-memory threads versus
8 MPI ranks with 2 threads per rank, so both legs used 16 logical CPUs.

| Level | Iterations | Final RMS residual | Serial time (s) | MPI time (s) |
|---:|---:|---:|---:|---:|
| 64³ | 12,740 | 9.958040299765e-7 | 6.731 | 4.469 |
| 128³ | 8,360 | 9.986570396463e-7 | 28.438 | 12.109 |
| 256³ | 5,880 | 9.991773343767e-7 | 160.395 | 70.313 |
| 512³ | 1,230 | 9.985952849938e-7 | 221.921 | 123.828 |
| 800³ | 690 | 9.957622957734e-7 | 411.714 | 262.578 |

Total reported case runtime was 894.00 s for shared memory and 508.19 s for
MPI. Measured 800³ peaks were 10.136 GiB for the shared-memory process and
9.542 GiB summed over all MPI ranks (1.207 GiB maximum per rank).

Evidence files (moved on this laptop to `archive_old/outputs/`, checked 2026-09-27):

- `archive_old/outputs/local_mpi_comparison_800/comparison_800.json`
- `archive_old/outputs/local_mpi_comparison_800/serial/afm_config_nm_local_comparison_serial_800/`
- `archive_old/outputs/local_mpi_comparison_800/mpi/afm_config_nm_local_comparison_mpi_800/`

The complete Python regression suite reports 47 passing tests; both relevant
Bash launchers pass `bash -n`.

## Current-code 2048³ Fir confirmation — prepared, not yet completed

The upload gate uses a one-node 96-thread serial reference and a two-node,
64-rank x 2-thread MPI candidate. Both run one position and -1 V, start at
64³, use a bounded diagnostic residual tolerance of 1e-5, disable zoom and
full-field output, and save only the same QD-centred physical cut. The
comparison writes an exact report plus a required `atol=1e-6, rtol=0` report.

## Prepared 4096³ Fir gate — not run

`jobs/run_afm_mpi_comparison_4096.sh` sequentially runs the same 4096³
one-position, -1 V, no-zoom case with 8x8x8 and 16x8x4 process grids, then
compares their 201³ physical cuts at zero absolute and relative tolerance.
At the time this snapshot was written, both hierarchies were configured to
start at 512³. Current defaults use the dynamic policy above; this historical
job description does not establish a current default or completed run.

The staged request is 16 nodes, 512 tasks, 32 tasks/node, 1 CPU/task,
4 GiB/CPU, and one hour. It therefore requests 512 CPUs and 2 TiB while
preserving the 4 GiB/core charging rule. No-allocation planning passed for
both layouts at 128 GiB/node: conservative node peaks are 116.94 and
117.10 GiB. These figures are planning estimates, not completed runtime
evidence.
