# Adapted local-only plotting merge — October 9, 2026

This release consolidates `afm-mpi-qol-plotting-integration` into maintained
`afm-parallelized-mpi`, preserving both histories. Its merge parents are the
[RAM release `3b7ba49`](https://github.com/nzy1016226721-dev/AFM_sim_electrostatic/commit/3b7ba49af617150e752748a7c34296de27392f36)
and [plotting contribution `2be34ad`](https://github.com/nzy1016226721-dev/AFM_sim_electrostatic/commit/2be34adda9ea3948f82287a4b5cfa26b62631986).
The separate earlier V-cycle/certified-refinement upgrade remains cancelled.

## Retained contribution and single implementation

Eric Friesen Waldner's plotting ideas and the integration branch's familiar
function names/menu options remain credited and available. Comparative lines,
XY/XZ/YZ planes, cut-preferred file discovery and two-click profiles use the
current local modules rather than a second independent NPY/coordinate reader.
`postprocessing/` compatibility functions adapt to the canonical
`local_visualization/` reader, renderers and selected-plane sampler.

The investigation found that the raw merge's existing tests passed while it
could ignore coordinate receipts, accept a same-shaped wrong JSON, run under
Slurm/MPI and overwrite a saved PNG. The adapted release tests and fixes those
contracts. Source indices/node bounds are validated; optional JSONs only
validate saved coordinates; absent receipts require explicit node bounds.
Mappings close on success and failure, selected lines/planes are detached,
post-menu guards run before prompts and figure saves preserve existing files.

Read [local plotting usage and explicit compatibility changes](MPI_PLOTTING.md)
before updating callers. In particular, single-plane metadata is no longer a
live field, and missing-receipt JSON guessing is deliberately removed.

## Solver and RAM preservation

All `simulation/`, shipped configuration, dependency and job-script files are
byte-exact to the maintained RAM release. Normal configs still select lossless
`ram_compact`, in-place snapshot Jacobi, scalar native RMS/max and solver plots
off; named float64 diagnostics remain exceptions. Float32 phi/epsilon,
material-reference/coarsening policy, geometry, voltages, tolerances, filenames,
rank ownership/halos and convergence arithmetic are unchanged.

The only launcher changes are local post-menu entries/guards. Ordinary shared
and MPI execution remains headless. `.gitattributes` and `.gitignore` now have
explicit LF rules so a fresh Windows checkout preserves manifest bytes under
automatic line-ending conversion. The generated source manifest is rebuilt
from the final selected payload, rather than choosing either parent's stale
manifest. Legacy numerical CRLF preservation remains explicit and unchanged.

## Validation and boundaries

The selected full suite and focused plotting tests cover:

- Original noncubic cut/node geometry, positive/zero/negative movement and
  inherited configuration validation, full/cut plotting and interactive events.
- Wrong same-shaped JSONs, malformed receipts/source provenance, integer input,
  explicit-bound conflicts and missing receipts.
- Local-only imports/calls and post-menu rejection before prompting; ordinary
  simulation launcher help remains usable under an MPI environment.
- Detached selections, normal/error/partial-multifile mapping and figure
  cleanup, safe repeated saves and bounded float64 selected-plane sampling.

The final selected development suite passed **228 tests**, with one opt-in
float64 diagnostic deselected. The initial contract-first selection deliberately
failed23 checks (two passed); its preserved red result demonstrates the gaps
before the adapters were implemented. Focused intermediate suites passed45
then60 tests; subsequent guard/display-error additions are in the final suite.

Small shared-memory (eight threads) and genuine two-rank MPI64 (four threads per
rank) launcher checks also passed, reaching native RMS approximately
`9.98482e-7 V` at minus-one volt. Each check's full/cut NPY bytes and complete
native residual history match its **own corresponding** previous maintained
result. This is not a claim that shared and MPI full volumes are identical to
each other. Rank0 owns32x64x64 of the global64x64x64 finest grid; ranks use
disjoint processor sets, not independent full-grid solver copies.

The first local MPI launcher failed before ranks started with an IPC pipe error;
its failed receipt is retained. A fresh, otherwise identical outside-sandbox
attempt passed. It is not a solver/RAM failure or a changed numerical setting.
All observed test processes closed and their fresh awake leases restored.
These are compatibility/regression checks, not new RAM/performance benchmarks.
The full development report/receipts reside in the
workspace's `docs/AFM_PLOTTING_MERGE_RELEASE_20261009.md` and
`afm_parallel_trials/github_plot_merge_20261009/outputs/`; they are not runtime
dependencies or bundled numerical data.

No new512/1024 solve, inter-node/Fir test, licensed QTCAD calculation or
combined/hybrid port belongs to this integration. Existing large-grid RAM
evidence retains its [original limits](RAM_VALIDATION_20261009.md). Headless
event tests do not establish a real interactive desktop GUI session.

The deleted enhanced branch was fully ancestral and archived before removal.
The plotting feature branch is retained for now; its Git ancestry is preserved
by this merge. Master/Eric/default-branch histories are not replaced.
