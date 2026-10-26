# Standalone AFM state — October 26, 2026

This is the standalone source release for the `afm-parallelized-mpi` branch of
[AFM_sim_electrostatic](https://github.com/nzy1016226721-dev/AFM_sim_electrostatic/tree/afm-parallelized-mpi).
Its preceding public baseline is
`8c4819d6f2ac2797774303502a740a50dd9858c0` (October 9). The user authorized
direct branch publication after the local source audit; see
[publication provenance and unchanged qualification limits](PUBLICATION_20261026.md).

## Implemented behavior and qualification

| Feature | Current behavior | Evidence boundary |
| --- | --- | --- |
| Shared-memory and distributed AFM | Weighted snapshot Jacobi on a coarse-to-fine hierarchy; configurable thread/rank layouts and halo exchange | Small regressions and dated genuine MPI replays; publication does not execute a cluster test |
| Default RAM storage | `ram_compact`, `in_place`, `scalar`; float32 phi and lossless float32 epsilon planes, packed/predicate masks | Exact saved-field/history comparisons in the dated RAM campaign; compression depends on geometry |
| Alternative storage | `ram_first`, `standard`; explicit buffered phi and row/volume residual storage | Separate storage choices, not three different physical equations |
| Partial spatial declarations | Omitted/null axes cover the domain; an omitted gate voltage is zero | Supported JSON/direct-function cases are exercised by range regressions |
| Geometry and movement | Physical-nm normalization, signed motion, fixed-block selection, cone/rounded-pyramid tips and ordered blocks/gates | Existing raster and 512-cell reference/coarsening policy is retained |
| Tolerance normalization | Default `absolute`; explicit `relative_drive` solves a common drive-normalized problem then restores volts | Neither is a pointwise phi-error certificate |
| Stopping controls | Opt-in RMS, maximum or both; cut/QD checks, residual-norm history, bounded observation, explicit caps | See [exact schema and outcomes](STOPPING_MODES.md); no new stopping default is promoted |
| Saved fields | C-order xyz float32 NPY, full/cut saves, movement/voltage/source-grid names, actual-coordinate sidecars | Retained-node coordinates are authoritative; interpolation is analysis, not solver output |
| Postprocessing | Eric's adapted plotting/menu/profile tools and coordinate-aware readers | Local visualization/sanity checks after results are downloaded; excluded from MPI execution |

## Source relationships

The standalone engine, combined AFM/QTCAD bundle and hybrid releases are
deliberately separate packages. The latest local review found identical
`solver.py` and `main_loop.py` in standalone, the combined AFM engine and the
frozen plateau-study engine, but other modules differ. This is not whole-tree
equivalence. The hybrid launcher still selects its immutable October 10 RAM
release, which lacks the newer stopping module. A new hybrid port/release needs
its own validation; publication of this candidate does not perform that port.

The upload audit's fresh module inventory finds 19 identical/six different
simulation modules in the combined engine, plus its two bundle-only modules;
all 25 standalone simulation modules match the frozen plateau engine. The
selected hybrid release has 13 identical/ten different common modules and no
`stopping.py` or `stopping_stats.py`. Module equality is narrower than release,
control, environment or completed scientific-result equivalence.

Cluster experiments run frozen study payloads and controls, not whatever files
are subsequently edited here. Job journals, source seals, original arrays,
crashes and failed gates remain in the development workspace. Their absence
from this source checkout does not make them obsolete. No remote state was
refreshed while preparing this candidate.

## Changes since the public baseline

Before this documentation consolidation, the maintained local package had
eight changed simulation modules and four added stopping/test modules relative
to the public baseline; its existing JSONs, jobs and plotting bytes matched.
The upload audit records the final exact file delta, including documentation
and the source verifier added for this candidate. Existing physical presets,
resource requests, numerical kernels and defaults are unchanged by the
consolidation itself.

The October 4–6 V-cycle/certified-refinement upgrade was cancelled and reverted.
Its proposed authoritative-cylinder geometry and new coarsening policy are not
implemented features of this release. Historical shared-memory zoom code is
retained, disabled in normal presets and unsupported in RAM/MPI modes.

Eric's October 26 serial changes were reviewed separately. Movement and edge
fixes already exist here; his iteration-cap return path and interface changes
need repair before selective porting. Dividing RMS tolerance by the square
root of the grid-point count changes the residual norm criterion; it does not
establish maximum potential error. His latest solver is not merged into this
candidate. The earlier adapted plotting integration retains its provenance in
[the plotting record](PLOTTING_MERGE_20261009.md).

## Remaining work

The [candidate upload audit](UPLOAD_AUDIT_20261026.md) and subsequent publication
record distinguish source publication from scientific qualification. Keep new detector thresholds opt-in;
large-array stopping, field/domain/spacing convergence and MPI qualification
are separate scientific gates. The offline diminishing-returns proposal is
not an implemented production stopping mode. A local test pass does not close
an earlier failed cluster qualifier or certify microvolt field accuracy.
