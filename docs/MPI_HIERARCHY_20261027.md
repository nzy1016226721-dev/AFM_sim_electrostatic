# MPI hierarchy admission correction — October27,2026 UTC

Four2048 plateau-study parents failed while refining128→256 on a3×3×5
process grid. Planning had checked memory and final-grid ownership, while
the job's six-rank qualification used a different process grid. A queued
4096 grid6×6×10 was independently found to fail already at64→128.
The failure was a missing coarse sample beyond one halo, not evidence of
out-of-memory or failure of the requested target-grid residual tolerance.

The combined bundle already had a full-hierarchy validation function. This
standalone correction ports that check without replacing the interpolation
kernel. Both `run_mpi.py --plan` and the actual distributed hierarchy now
check all axis partitions at every declared transfer before field allocation.
Automatic grid selection considers only compatible factorizations; runtime
uses the same selection as planning when no grid is explicitly supplied.
If none works, it raises an actionable error rather than allocating the grid.
An explicitly supplied incompatible grid is rejected, never silently changed.

This validates the actual endpoint-aligned index calculation, including
clipped rectangular levels. Divisibility alone is insufficient; some factors
of three work. No arbitrary-rank redistribution or wider halo is introduced.
Snapshot Jacobi, precision, geometry, coarsening, boundary precedence,
residual definitions, reductions and saved-field contracts are unchanged.

## Validation and operational boundary

The new regression reproduces the old planner admitting the memory-fitting
45-rank layout, and passes after the fix. Coverage includes both observed
bad layouts, insufficient initial ownership, supported large layouts,
clipped/noncubic levels, unsupported factorizations and planner/runtime
automatic-selection agreement. The complete selected float32 regression is
275 passed / one opt-in float64 diagnostic deselected. A harness launch from the
workspace root failed collection because package imports were absent; that
receipt is retained, and the corrected package-root invocation passes.

An isolated genuine six-rank qualification transfers64→128→256 and compares
all owned values and valid internal face/edge/corner ghosts bit-for-bit with
an independent root-only SciPy reference. Exact global ownership is checked.
The experimental replacements additionally run that qualification at each
parent's actual48-rank or384-rank layout before their own large solve; each
parent has an independent24-hour total budget. Failure of one does not
prevent other parents from starting. No large-grid convergence, RAM peak or
precision outcome is implied by source, planning or small-grid qualification.

The standalone and bundled engines remain separately maintained. The bundle
already rejects unsafe layouts; it was not bulk-synchronized. Old failed
attempts, held-job controls, local backups and frozen releases are preserved.

Four genuine two-rank shared/MPI comparisons also pass: compact RMS, compact
BOTH, standard BOTH and compact regional BOTH. Full/cut float32 payloads and
cut-coordinate sidecars are exact, ownership and internal halos agree, and
native RMS/maximum histories match within CSV formatting precision.
