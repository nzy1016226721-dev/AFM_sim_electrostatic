# Lossless RAM release: validation and provenance — October 9, 2026

## What this release contains

The standalone JSON entry points now default to lossless `ram_compact` epsilon
planes/packed masks, `in_place` snapshot Jacobi and `scalar` native residual
reduction. All 34 normal configurations record that policy and disable solver
plotting. Two explicitly named float64 diagnostic presets remain standard-mode
opt-outs. [RAM_SAVING.md](../RAM_SAVING.md) owns the configuration contract.

The numerical implementation comes from the separately backed-up, sealed
`ram_compact_20261008` trial. Its 181-file source-index SHA-256 is
`767b85854138085811a84bdeca706c597d165f19bf934c22294526ee13e1e78c`.
The preserved pre-RAM standalone Git baseline is
[`7ec0e647d35332362d73f4bcf75cc56f550abefe`](https://github.com/nzy1016226721-dev/AFM_sim_electrostatic/commit/7ec0e647d35332362d73f4bcf75cc56f550abefe).
The new source manifest records checkout bytes, not numerical certificates or
an environment lockfile. The cancelled October 4–6 V-cycle/certified-refinement
work is not part of this release.

## Completed large shared-memory trials

Both physical minus-one-volt trials used float32 potentials and epsilon, eight
permitted P logical processors and native absolute RMS tolerance `1e-6`.
The original and compact full NPY files and complete native residual histories
were independently replayed and matched exactly, including iteration counts.
Peaks below are observed process samples, not payload estimates/reservations.

| Grid | Original private / RSS GiB | Compact private / RSS GiB | Reduction private / RSS | Measured runtime qualification |
| --- | --- | --- | --- | --- |
| 512 cubed | 1.495846 / 1.513062 | 0.773624 / 1.114285 | 48.28% / 26.36% | Original is the matching prefix of the 1024 reference, not a separately timed full/save case; finest solve about 3.80x slower |
| 1024 cubed | 10.678780 / 10.117413 | 4.814049 / 4.953442 | 54.92% / 51.04% | 1189.272803 to 3805.195881 s end-to-end, about 3.20x; finest solve about 3.25x |

| Grid | Native final RMS V | Native maximum V | Finest iterations | Full NPY SHA-256 |
| --- | --- | --- | --- | --- |
| 512 cubed | 9.98591332344922e-7 | 1.9550323486328125e-5 | 990 | `1aa4661327217586eed208f2cb7d6253cb2012ff3885238e5dd3df5fb2cbcba4` |
| 1024 cubed | 9.83720784003256e-7 | 3.9577484130859375e-5 | 290 | `b4e2f489ed7015f7139c3dd72e87a01aee0cf256e2a70e7e9e5abbeceeeffcc9` |

RMS is the existing mean convergence statistic, not a signed mean or a bound
on pointwise solution error. Maximum residuals exceed RMS as shown. These
trials tested the sealed candidate; new 512/1024 solves were not repeated for
local installation or Git publication.

## Completed installed-package checks

The maintained standalone was checked in fresh local namespaces after promotion:

- Selected regression: **173 passed, one opt-in float64 diagnostic deselected**.
- Genuine two-rank MPI replay: **37 cases**, comprising 36 fixed-iteration,
  noncubic diagnostic cases and one native-converged physical64 case. All
  **111 saved MPI NPYs** match the immutable dense/compact references; local
  ownership, exchanged halos, native reports and residual histories match.
  Coverage includes different split axes, uneven partitions, coarse/fine epsilon,
  both phi updates and all residual selectors. Fixed-iteration diagnostics are
  equivalence evidence, not convergence evidence.
- Normal shared and MPI64 JSON launchers: omitted selectors resolve correctly;
  full/cut outputs, movement/voltage/grid names and coordinates pass. Both
  converge at RMS `9.984822755296167e-7` and maximum `2.86102294921875e-6` V.
  Shared/MPI full payloads are exact despite the existing NPY v1/v2 header
  distinction; cuts are byte-identical. Nominal cut requests snap to solver nodes.
- Eric's local-only visualization: read-only/finite sanity, three installed PNGs
  and retained-node coordinates pass; XY/XZ figures were visually inspected.
  Visualization is separate from solver execution and refuses MPI/Slurm contexts.
- Integrity: pre-change 126-file/seven-document backup, all original config
  physics/resource/runtime values, 181 sealed candidate files, 1,798 independent
  source files, 137 old arrays, six original/candidate large fields/checkpoints
  and 16 archived Eric originals pass. The pre-existing 36-entry tracked dirty
  roster is unchanged. All fresh workers closed and awake leases restored.

One installed analysis attempt failed on an incorrect residual-log path, not
on numerical output. Its receipt and pre-fix analysis helper are preserved.
A fresh corrected audit passed without a solver change or numerical rerun.
The earlier isolated phase also preserves its failed MPI harness call-contract
attempt and successful separately named correction. Failures were not erased.

## Limitations and reproducibility

Small MPI private memory was **4.76% higher** from setup/JIT overhead. Large
distributed memory/performance, inter-node Fir validation and combined/hybrid
ports remain unexecuted. Do not infer universal MPI RAM savings from the large
shared-memory measurements. Material plane repetition is problem-dependent;
all-unique float32 planes compress poorly. Temporary file-backed pages count
toward RSS, require writable disk space and are not free memory.

The unguarded 1024 trial had no memory/time admission controls, as explicitly
requested for that case. It did not remove maintained launcher safeguards.
No unrelated application, persistent power/pagefile/environment setting,
licensed QTCAD run or Fir job was changed by these tests or this source release.

The ordinary development-workspace reports are
`docs/AFM_LOSSLESS_MEMORY_20261008.md` and
`docs/AFM_RAM_DEFAULT_PROMOTION_20261009.md`. Replayable outputs are retained in
the original trial and `afm_parallel/outputs/ram_default_promotion_20261009/`,
including `audit_numerical02.json` and `audit_integrity01.json`. Large arrays,
logs, environments and recoverable archives are intentionally not Git payloads.
The October 9 publication has a separate source/branch/checkout receipt; see
[the branch review](BRANCH_REVIEW_20261009.md). Re-running scientific checks
requires their recorded physical inputs, environment and preserved references,
not just these summary hashes.
