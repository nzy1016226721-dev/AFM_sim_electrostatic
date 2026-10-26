# Stopping controls and residual meanings

The implementation is `simulation/stopping.py`, with bounded regional and
precision reductions in `simulation/stopping_stats.py`. Existing configs keep
their original RMS policy unless they explicitly declare `stopping`. Storage
and stopping are independent: scalar residual storage supports the new region
reductions without allocating a whole residual volume or an old phi volume.

## Native residual and drive normalization

The global RMS is the square root of the squared native stencil defects summed
over free interior nodes divided by their count; maximum is the largest
absolute defect on those nodes. Boundary/fixed nodes are excluded. These are
operator residuals, not phi changes or certified voltage errors. RMS is not an
unnormalized L2 norm, and no automatic grid-point divisor is applied.

`residual_tolerance_mode: absolute` is retained as the compatibility default.
Explicit `relative_drive` normalizes all declared Dirichlet voltages by their
common maximum magnitude, applies every stopping threshold in that normalized
problem, and rescales fields to volts. Zero drive uses a scale of one. Reports
record the drive scale and physical/native norms. This is drive-relative
normalization, not `||r||/||b||`, and does not bound `A^-1 r`.

## Strict norm controls

This illustrates an explicit policy, not a newly calibrated default or an
instruction to launch a large preset:

```json
{
  "residual_tolerance_mode": "relative_drive",
  "stopping": {
    "criterion": "both",
    "rms_tol": 1e-6,
    "max_tol": 1e-5,
    "scope": "finest",
    "confirmation_checks": 3,
    "max_iterations": 20000
  }
}
```

| Key | Meaning and default |
| --- | --- |
| `criterion` | `rms` (default), `maximum`, or `both`; requested values must be strictly below their thresholds |
| `rms_tol`, `max_tol` | Positive thresholds; each omitted value inherits the solver tolerance |
| `scope` | `finest` default; `all_levels` explicitly applies this policy and cap to every hierarchy level |
| `confirmation_checks` | Consecutive passing magnitude checks, default one; successful candidates request dense checks |
| `max_iterations` | Optional positive per-active-level update cap; a cap exit is not convergence |

The legacy top-level `max_iter` remains an ignored compatibility input in
ordinary solves. Use the explicit cap above. Changing `scope` also changes
which coarse levels see that cap; preserve study controls when reproducing
older runs. Runtime limits remain effective independently of the new policy.

## Cut and QD residuals

`stopping.regions` accepts `cut` and `qd`. Each requested region must contain
at least one positive `rms_tol` or `max_tol`. Regional checks are additional
to the selected global criterion. For example,
`"regions": {"qd": {"rms_tol": 1e-6, "max_tol": 1e-5}}`.

Cut membership follows saved-cut physical context/retained-node selection.
QD membership includes free interior nodes adjacent to at least one physical
last-block-wins epsilon=15 cell center; it is not a test that the coarsened
epsilon average equals 15. It is not an authoritative analytic-cylinder FEM
geometry. Empty/missing regions reject rather than silently pass. Low-level
callers must supply valid `stopping_context`; JSON launchers provide their
physical cut and moved block context. MPI counts owned nodes once, excluding
halos, then reduces statistics globally.

## Residual history and observational detectors

`history` operates on changes in scalar RMS/maximum residual norms,
not changes in phi and not a pointwise residual-history array. It retains a
bounded scalar window. `mode` is `off` (default), `observe`, or `require`.
`window` defaults to three observations; `probe_updates` defaults to ten.
`require` needs explicit positive `mean_tol` and `max_tol`, applies mean and
maximum absolute norm changes, and requires one-update observation spacing.
A changed sampling cadence resets history rather than treating ten-update
differences as single-update differences.

`stagnation` and `precision` each accept `mode: off | observe | stop`, a
`window` (default five, minimum three), and optional nonnegative
`relative_improvement` and `motion_tol`. `stop` requires both thresholds;
they have no universal calibrated defaults. A flat window can return
`stagnated`. Precision observation recomputes stencil evaluation in float64
on the same float32 field and compares evaluation noise; an eligible plateau
can return `precision_limited`. Stored production phi remains float32.
This diagnoses residual evaluation on the current field, not discretization
error, the ultimate float32 solution floor, or maximum phi error.

Keep both detectors observational until calibrated for the intended mesh,
geometry and cadence. A repeated float32 maximum at a few checks is not proof
of stagnation: a completed Fir pilot still improved its maximum by 24.62%
over updates 1970–2261. The later offline robust/diminishing-returns proposal
has not been added to this solver.

## Outcomes and field admission

Reports distinguish `converged`, `time_limit`, `iteration_limit`, `stagnated`,
`precision_limited`, `invalid_region` and `invalid_numerics`. Only `converged`
passes strict requested norms and any required history/confirmation gates.
Opt-in shared high-level runs require convergence by default; MPI also defaults
to requiring convergence. An explicit diagnostic opt-out can retain an
unqualified finest field, labelled with `trial_` for an active stopping policy.
Coarse opt-in policy failures still block finer refinement. Inspect level
reports and histories with every saved field; a filename alone is insufficient.

The historical no-policy shared path can save after its runtime limit. Fixed
iteration diagnostics, time-limited cuts and qualified strict fields must
retain their distinct classifications. Read [current validation and scientific
limits](VALIDATION_20261026.md) before selecting a new stopping default.
