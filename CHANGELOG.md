# AFM Simulation Package — Change Log

## 2026-10-07 — Complete standalone source distribution

- Published the restored standalone modules, postprocessing, tests and distinct
  launch scripts previously omitted from `afm-parallelized-mpi`. No numerical
  source, JSON preset, executable script or test behaviour changed.
- Archived two superseded comparison drivers, the stale August consolidation
  notice and a historical manual upload manifest; removed embedded legacy
  serial source/caches and the old v15 ZIP from the active Git payload.
- Added package-local navigation, current configuration/job documentation,
  LF text attributes and a SHA-256 source manifest. Originals remain in a
  verified local archive and the old Git revision remains in history.
- The October 4–6 numerical upgrade remains cancelled; this is packaging and
  documentation maintenance, not renewed solver/certified-refinement work.

See [the consolidation record](docs/PACKAGE_CONSOLIDATION_20261007.md).

## 2026-09-14 - Pyramid AFM tip with square base

- Changed the default AFM tip cross-section from the legacy circular cone disk
  to a pyramid: each horizontal slice is a square whose four side planes are
  tangent to the original cone along their centre generators, so the square
  circumscribes the cone disk and the configured `aspect_ratio` still measures
  the opening angle at every side-plane centre. The four vertical edges where
  the planes meet are filleted with the apex curvature radius `R`; within one
  apex radius of the tip apex the fillet consumes the flats and the section
  stays circular, so the apex itself is unchanged.
- Reinterpreted `r_tip`/`r_tip_nm` as the distance from the tip axis to one
  side of the square base edge (a half-side rather than a radius). The base
  height `z_base` formula is unchanged. Existing numeric values are kept.
- Added optional `"tip_shape": "pyramid" | "cone"` JSON key (default
  `"pyramid"`), normalized in `coordinates.normalize_config` and threaded
  through serial, distributed, zoom, preview, and field-line builders.
  `"cone"` reproduces the legacy circular mask bit-for-bit. The canonical
  `afm_config_nm.json` now sets `"tip_shape": "pyramid"` explicitly.
- Implemented the cross-section once in `simulation/numerics.py`
  (`pyramid_tip_half_side`, `pyramid_tip_corner_radius`, `pyramid_tip_inside`)
  and shared it between `simulation/solver.py::build_downward_pointing_tip`
  and `simulation/mpi_domain.py::build_local_tip_mask`, so serial and
  distributed masks stay bit-identical by construction.
- Documented the geometry in `API_Docstrings.md` and `DOCUMENTATION.md`.

Validation:
- New `tests/test_tip_pyramid.py` (9 tests): square base with R-fillets,
  circular sub-R apex slices equal to cone, `z_base` matches the cone formula,
  side-centre slope `ds/dz == dρ/dz`, cone voxels are a strict subset of
  pyramid voxels, serial-vs-distributed agreement with domain-comparable
  `r_tip`, config default/rejection checks. Full suite: 61 passing.
- Headless Agg render of a base cross-section confirmed square-vs-circle.

## 2026-09-09 - Relative-drive convergence and float64 memory diagnostics

- Added an opt-in `relative_drive` residual mode for linear homogeneous
  electrostatic cases. All Dirichlet voltages are divided by their largest
  absolute drive during the solve and the final field is rescaled to physical
  volts before sampling or saving. Existing JSON files remain in absolute mode.
- Added matched serial/MPI -9 V 4096^3 parity jobs at normalized tolerance
  `1e-6`, plus a dependent cut comparator and optional -1 V linearity check.
- Added an opt-in float64 solver path for both shared-memory and distributed
  stencils, dielectric faces, prolongation, residuals, and MPI buffers.
- Added a guarded diagnostic mode that runs exactly 10 iterations at 2048^3
  and 20 at 4096^3. Fixed iteration counts remain invalid for normal production
  jobs and for float32 configurations.
- Added dtype-aware MPI memory planning and peak-stage logging, with a 32-node
  float64 layout that requests 4 GiB per allocated core-equivalent.
- Added regression tests for normalized-drive scaling and exact diagnostic
  iteration counts. The local package suite contains 52 passing tests.

## 2026-09-01 - Distributed-memory MPI main-grid solver

- Corrected the production MPI dielectric hierarchy. The first MPI version
  rasterized every coarse level directly, whereas the established solver
  volume-averages a 512^3 material reference through the coarse hierarchy.
  Rank zero now constructs that bounded coarse field once per level and sends
  only each rank's epsilon halo; levels finer than the reference remain fully
  rank-local. This removes a discretization mismatch without introducing a
  global 4096^3--16384^3 array.
- Added a regression geometry whose coarse dielectric interfaces fail under
  direct rasterization, exact serial/MPI face-coefficient tests, and a bounded
  two-rank serial-versus-MPI Slurm parity job at 2e-5 absolute/relative
  tolerance.
- Recorded the pre-fix 2048^3 evidence rather than treating convergence alone
  as equivalence: both jobs completed, but the cut fields differed by
  2.09597e-3 V maximum and 5.77314e-4 V RMS. After the coarse-epsilon fix,
  Fir jobs 57783896/57783897 completed the full two-node 2048^3 comparison:
  all 8,000,000 float32 cut values passed 2e-5 absolute/relative tolerance,
  with 2.16961e-5 V maximum and 1.31789e-5 V RMS difference.
- Added a 3-D Cartesian `mpi4py` domain decomposition for one global AFM
  solve, with face/edge/corner halo propagation and globally reduced residual
  convergence. This reduces per-rank memory instead of merely distributing
  independent voltages.
- Added distributed local tip/gate/material construction and endpoint-aligned
  coarse-to-fine trilinear prolongation without gathering a global field.
- Added bounded physical-cut gathering and collective MPI-IO output for valid
  C-order float32 `.npy` files.
- Added no-allocation rank/node memory planning plus a second runtime check
  against physical node RAM before large-grid allocation.
- Added `run_mpi.py`, `jobs/run_afm_mpi.sh`, MPI environment/preflight scripts,
  `requirements-mpi.txt`, a two-rank smoke configuration, and a compact 16384³
  single-case template inheriting the established 4096 physical geometry.
- Kept zoom disabled in MPI mode because the existing SciPy zoom pipeline is
  still full-array; the launcher fails clearly if it is enabled.
- Made convergence mandatory by default so an iteration/time limit cannot be
  reported as a successful production result or silently feed a finer level.
- The initial 8³ -> 16³ MS-MPI smoke covered halo exchange and MPI-IO but used
  direct material rasterization and therefore did not exercise the production
  coarse-reference contract. It must not be used as evidence for 2048³ or
  inter-node numerical equivalence.
- Changed the default hierarchy start to 512³ when a target axis exceeds
  2048, while keeping the 64³ start through 2048³ and retaining explicit
  diagnostic overrides.
- Added a one-position/one-voltage MPI 2048³ parity configuration, a two-node
  64-rank Fir trial wrapper, and a memory-mapped NPY comparator for the
  serial-versus-MPI output check. The validated layout uses 32 ranks x 2
  threads and 256 GiB per node, preserving the 4 GiB-per-core charge ratio
  while passing the planner's 80% memory-safety check.
- Replaced rounded fractional `cx/cy/cz` movement filename tags with signed
  physical-nm path offsets: `_0nm` at the reference centre, `_0.001nm` for a
  positive offset, and `-0.001nm` for a negative offset. The same convention
  now covers serial, zoom, MPI, and MPI log case names; the post-processing
  parser remains compatible with archived fractional names.

## 2026-08-30 - Movement-centered cut output and source-grid filenames

- Enabled a 100 x 100 x 100 nm saved cut in every package JSON. The cut follows
  each movement center in x/y.
- Set movement-relative offsets to `[-50, 50, -50, 50, -20, 80]`. With the
  configured movement z=20 nm, those offsets save the absolute z=0..100 nm range.
- Cut filenames retain configuration, movement position, voltage, and zoom metadata
  and now append the original uncut array shape, such as
  `_cut_from_grid512x512x512.npy`.

## 2026-08-30 - Per-level Fir memory test

- Added `afm_memory_test.json` for 1024, 2048, 4096, 8192, and 16384 target
  grids with `1e-5` residual tolerances, a 120-second multigrid limit, one
  voltage/position, no zoom, no plots, and no full-field NPY saves.
- Added `jobs/run_memory_test.sh`, a self-contained, package-root-aware Slurm
  launcher that runs each target in a separate `srun` child, records per-level
  peak RSS/timing logs, and emits level-aware OOM/timeout/failure diagnostics to
  stderr.
- Sized the memory-test job for the 4096^3 target on Fir: one
  `cpularge_bynode_b1` node, 192 CPUs, 4096G RAM, and a one-hour wall-time
  limit. The single-process solver cannot distribute its full-grid arrays over
  multiple nodes; the allocation therefore uses all physical CPUs on the
  available 6-TB node.
- Converted the active memory-test JSON to one focused 8192^3 target so its
  internal hierarchy records the 2048^3 and 4096^3 stages without separately
  repeating lower targets. Added a 55-minute child deadline to leave Slurm
  headroom inside the one-hour batch limit and emit a level-aware deadline
  marker instead of ending only as a scheduler TIMEOUT.
- Reworked the active trial to begin directly at 4096^3 and advance to 8192^3,
  using 1e-4 tolerances and a 20-minute per-level runtime cap. Added flushed
  one-second live RSS sampling so a partial 4096^3 or 8192^3 stage leaves
  usable memory evidence even when it does not return normally.
- Changed the normal launcher's default batch output root to the package-local
  `outputs/job_<jobid>/<config-name>/`; `AFM_OUTPUT_ROOT` remains an explicit
  override for project/scratch storage.

## 2026-08-30 - Fir resource-accounting alignment

- Aligned the canonical 140G request with the Alliance 4G/core rule by
  requesting 35 CPUs per task.
- Set the canonical JSON `cpu_threads` to 35 so the solver can use the
  allocated CPU count without an avoidable CPU/memory mismatch.

## 2026-08-30 - Generic Slurm launcher and Fir path hardening

- Replaced the fixed-configuration selector with one self-contained
  `jobs/run_afm.sh`; submit any JSON as its first argument.
- Added the working Fir RAS account directive `rrg-hongguo-ad`, while allowing
  `sbatch --account=<account>` to override it.
- Resolved the package from `SLURM_SUBMIT_DIR`/`AFM_PACKAGE_ROOT` instead of
  the staged Slurm script path, fixing the `/localscratch/.../slurm_script`
  helper-not-found failure.
- Removed the executable helper-shell dependency and invoke Python directly,
  avoiding permission-denied errors caused by lost execute bits.
- Generalized preflight to accept any JSON and to validate its configured
  `cpu_threads` against the allocated CPUs.

## 2026-08-29 - Consolidated parallel release

- Consolidated the original parallel folder, the 1024-trial folder, and the
  final ready tar into `D:\afm_pack_v1\afm_parallel`.
- Kept one canonical configuration and one production Slurm launcher.
- Removed the duplicate `run_afm_nm.sh`, generated outputs, caches, logs,
  backup files, and NumPy result arrays from the release.
- Removed the unnecessary pandas dependency from convergence plotting.
- Made psutil an explicit runtime dependency because memory tracking is enabled
  in the canonical Fir configuration.
- Added deterministic parallel-kernel and runtime/config tests.

## 2026-08-29 - FIR 1024³ / 32-core trial configuration

- Updated the canonical `afm_config_nm.json` for the confirmed FIR trial:
  - `grid_resolution = 1024 × 1024 × 1024`
  - `res_tol_main = 1e-6`
  - `res_tol_zoom = 1e-6`
  - `v_start = -1.0`, `v_stop = -5.0`, `v_step = 0.5`
  - `cpu_threads = 32`
- Updated `jobs/run_afm.sh` and `jobs/run_afm_nm.sh` to request `32` CPUs, `140G`, and a `15:00:00` wall-time limit.
- Removed the obsolete `afm_config_nm_0nm.json` configuration and `jobs/run_afm_nm_0nm.sh` selector.
- Updated Alliance documentation and API/runtime guidance to describe the current 32-core trial resources and canonical configuration.
- No numerical solver algorithm changes were introduced in this configuration-only update; the existing Numba CPU-parallel solver remains in use.


## Launcher/config-selection update

- Interactive mode now enumerates all AFM JSON configurations in the working directory and selects one exact file.
- Removed legacy JSON base-name/suffix lookup.
- Non-interactive `python run_all.py <config.json>` runs only that JSON; omitting the filename selects the newest AFM JSON automatically.
- Added `--output-dir` and interactive output-root selection.
- Alliance/Slurm output now defaults to `$SCRATCH` or `$PROJECT` job-specific storage instead of `$HOME`.

Validation performed: Python compilation of launcher/simulation/postprocessing modules; AFM config discovery with newest-file ordering; explicit JSON resolution; rejection of extensionless/base-name lookups.

This file records the evolution of the current AFM simulation package relative to the
original uploaded package:

`afm_package_alliance_ready_full(1).zip`

The current package is maintained as a cleaned, Alliance-ready development line.
Each subsequent improvement should add a new dated/versioned entry to this file rather
than rewriting the history.

> Scope note: This changelog describes changes made to the uploaded package during
> this ChatGPT-assisted development session. It does not claim that every historical
> change in the upstream Git repository is represented here.

---

## Baseline — Original upload

**Source:** `afm_package_alliance_ready_full(1).zip`

The original upload contained the broader AFM package, including simulation code,
presimulation/postprocessing material, multiple configuration variants, generated
or auxiliary files, and legacy artifacts.

The original package also used the earlier configuration model in which physical
domain dimensions and/or fractional/intermediate configuration information could be
specified separately, and the later development work identified several areas that
needed consolidation for the Alliance-ready workflow.

---

## v1 — Alliance-ready package cleanup

**Goal:** create a clean release-oriented package from the original upload.

Changes:
- Removed unnecessary backup files (`.bak`, `.bak2`) and generated/temporary data.
- Removed alternate and obsolete configuration variants.
- Removed generated CSV/material/cache/output artifacts from the distributable package.
- Retained the functional AFM simulation code and useful postprocessing tools.
- Retained Alliance execution support and `requirements.txt`.
- Added/updated package-level README and documentation.
- Added API documentation/docstrings for public simulation functionality.
- Removed obsolete/broken aggregation code where it referenced functionality that was
  not actually present in the package.
- Added basic import/smoke validation.

The package was deliberately kept more complete than the first very aggressively
stripped version, preserving useful `presimulation/`/postprocessing functionality
needed by the workflow.

---

## v2 — Conservative cleanup

**Goal:** avoid removing functional workflow components merely because they were not
directly imported by the main solver.

Changes:
- Rebased the clean package on the larger original upload rather than the previously
  stripped-down package.
- Preserved functional postprocessing utilities.
- Preserved Alliance job/launch support.
- Removed obsolete/generated material and configuration clutter.
- Explicitly excluded the GitHub-only Joule/conductivity subsystem.
- Updated documentation and API references to match the retained package.

### Explicitly NOT included
The GitHub `joule.py` / conductivity / sigma functionality identified during the
earlier repository comparison was intentionally not merged into this development
line.

---

## v3 — Zoom boundary-condition modes

Added a configurable zoom boundary mode under the zoom-related configuration:

```json
"clamp": true
```

### `clamp: true`
Preserves the existing zoom behavior:
- inherited potential from the previous level is interpolated onto the new grid;
- voltage masks remain fixed;
- the outer zoom boundary is clamped/fixed using the inherited boundary values.

### `clamp: false`
Adds the natural-boundary mode:
- inherited potential is still interpolated from the previous level;
- voltage masks inside the new cut remain Dirichlet constraints at their configured
  voltages;
- the outer boundary is not added as a fixed Dirichlet boundary;
- the outer boundary uses the solver's natural homogeneous Neumann treatment;
- inherited values provide the initial solution on the new level.

Both canonical configuration files were updated to explicitly contain the new setting,
with `true` retained as the compatibility/default setting at that stage.

---

## v4 — Per-level memory usage logging

Added optional memory tracking for individual main and zoom solver levels.

Changes:
- Added `simulation/memory.py`.
- Tracks peak process RSS while a solver level is running.
- Writes `memory_usage_log.csv` to the same output directory as other simulation output.
- CSV columns:
  - `level resolution`
  - `memory cost(in GB)`
- Distinguishes main levels and zoom magnifications (for example `main ...`,
  `zoom 2x (...)`, `zoom 4x (...)`).
- Added `psutil` support and memory-tracking documentation.

---

## v5 — Memory tracking made optional/separable

The memory profiler was separated from the production simulation path.

Changes:
- Removed unconditional/top-level imports of `simulation.memory` from the core solver
  modules.
- Memory tracking is now lazily imported only when explicitly enabled.
- Added a configuration switch:
  ```json
  "memory_tracking": false
  ```
- Normal Alliance runs do not import the memory-tracking module when disabled.
- Removed `psutil` from the core requirements.
- Added `requirements-memory.txt` for the optional memory feature.
- Kept all memory-specific implementation in `simulation/memory.py`.
- Verified that importing the core simulation modules does not load the memory module
  when tracking is disabled.

This keeps memory instrumentation optional without burdening production Alliance runs.

---

## v6 — Physical-coordinate geometry and presimulation redesign

**Goal:** eliminate fractional/intermediate geometry configuration and make physical
geometry independent of main-grid resolution.

Changes:
- Introduced an explicit physical coordinate system with an origin in the main grid,
  initially represented by an origin fraction such as `[0.5, 0.5, 0.0]`.
- Converted physical geometry from fractional grid coordinates to nanometres relative
  to that origin.
- Added internal nm-to-grid-index transformation logic.
- Removed the need for a separate fractional configuration JSON.
- Converted voltage-gate, dielectric-block, and AFM-tip geometry to physical nm
  coordinates.
- Changed movement geometry to physical distances.
- Removed the old `offsets_nm` sweep field from the solver configuration.
- Added a presimulation module that generates one JSON per tip-z offset.
- Generated offset files use readable suffixes such as:
  `afm_config_nm_-10nm.json`,
  `afm_config_nm_0nm.json`,
  `afm_config_nm_+10nm.json`.
- The solver can process the generated same-base JSON files in numerical offset order.
- Generated presimulation JSON files are not part of the distributable package.

---

## v7 — Single canonical configuration and zoom-grid compatibility

Changes:
- Removed `afm_config_1.json`.
- `afm_config_nm.json` became the only canonical configuration.
- Removed remaining references to the fractional configuration from launch scripts,
  documentation, and workflow code.
- Adjusted zoom-grid handling so zoom magnification represents physical resolution
  rather than requiring a larger cubic array.
- Improved target-grid handling for zoom levels so requested magnification is compatible
  with the current solver implementation.

---

## v8 — Single voxel-size physical scale

Replaced explicit physical domain lengths with one voxel-size parameter.

Canonical configuration now uses:

```json
"grid_resolution": {
    "nx": 512,
    "ny": 512,
    "nz": 512
},
"voxel_nm3": 0.5
```

Meaning:
- one main-grid voxel has an edge length of `0.5 nm`;
- physical dimensions are derived internally as:
  `nx * voxel_nm3`, `ny * voxel_nm3`, `nz * voxel_nm3`.

Thus:
- `512^3` at `0.5 nm` gives `256 nm × 256 nm × 256 nm`;
- changing grid resolution changes the represented physical domain without rewriting
  every physical geometry parameter.

Additional changes:
- Removed `Lx_nm`, `Ly_nm`, and `Lz_nm` from the canonical JSON.
- Updated coordinate conversion to derive dimensions from grid resolution and voxel size.
- Updated zoom calculations to use voxel scale and physical magnification.

---

## v9 — Non-cubic grid compatibility

This revision audited physical geometry and grid handling for non-cubic grids such as:

```json
"grid_resolution": {
    "nx": 256,
    "ny": 256,
    "nz": 100
},
"voxel_nm3": 0.5
```

which represents:

`128 nm × 128 nm × 50 nm`.

Changes:
- Voltage-gate coordinates are converted independently along x/y/z.
- Epsilon/dielectric block ranges are converted independently along x/y/z.
- AFM tip geometry is constructed in physical nm coordinates so an isotropic physical
  tip remains isotropic on rectangular domains.
- Tip radius and z-position remain physical quantities independent of grid aspect ratio.
- Movement distances are evaluated in physical nm rather than fractional-domain distance.
- Main-grid refinement no longer assumes all three dimensions can be doubled together
  indefinitely.
- Zoom/interpolation target shapes are handled independently per axis.
- Postprocessing utilities derive physical dimensions from:
  `grid_resolution + voxel_nm3`.
- Removed dependence on the old explicit `Lx_nm/Ly_nm/Lz_nm` configuration fields.
- Tested/import-checked the current Python modules after the non-cubic changes.

---

## Current package policy

The current development line intentionally keeps:

- only `afm_config_nm.json` as the canonical configuration;
- physical geometry in nm relative to the configured physical origin;
- `voxel_nm3` as the single main-grid physical scale parameter;
- non-cubic `nx`, `ny`, `nz` support;
- optional memory tracking outside the core import path;
- optional zoom boundary behavior via `clamp`;
- presimulation-generated tip-offset JSON files;
- Alliance-ready launch/documentation support.

The following are intentionally NOT part of this development line:
- GitHub-only Joule/conductivity/sigma functionality;
- obsolete fractional configuration JSONs;
- generated output/cache files;
- backup/temporary configuration files.

---

## Future entries

For each future modification, append a new section with:
1. version/date,
2. motivation,
3. files/modules changed,
4. behavioral/API changes,
5. configuration changes,
6. compatibility notes,
7. validation/tests performed.

Do not delete earlier entries. This file is intended to provide a cumulative audit
trail from the original `afm_package_alliance_ready_full(1).zip`.

---

## v11 — Deterministic high-resolution dielectric material hierarchy and memory cleanup (2026-08-26)

**Goal:** eliminate arbitrary coarse-grid dielectric assignment and reproduce the
high-resolution epsilon-to-coarse-grid averaging behavior of the earlier material
pipeline, while keeping the current JSON/movement architecture and controlling RAM
usage.

Changes:
- Reworked `simulation/materials.py` around a deterministic high-resolution epsilon
  reference field generated directly from the normalized JSON dielectric-block
  distribution.
- The reference field respects JSON block ordering, so the existing later-block
  precedence is retained at the high-resolution material level.
- Coarse solver-cell epsilon values are obtained by **box/volume averaging** the
  high-resolution material field into the exact `(nx-1, ny-1, nz-1)` solver-cell
  shape.
- A coarse cell that spans multiple dielectric values therefore receives the
  averaged epsilon rather than an arbitrary single block value.
- The averaging is deterministic and does not randomly select a material value.
- The same material construction is performed after block movement has been applied,
  so moved dielectric structures are reflected in every level's epsilon field.
- Added temporary file-backed `.npy`/memmap reference generation to mirror the
  historical high-resolution-NPY workflow without keeping the entire reference
  field resident in RAM.
- The temporary epsilon reference is deleted after the simulation completes.
- The coarse material reference resolution is configurable under:
  ```json
  "epsilon_material": {
      "reference_resolution": 512,
      "method": "high_resolution_volume_average"
  }
  ```
- The canonical configuration now uses a 512-voxel reference resolution by default,
  preserving substantially more of the physical block information for the 8/16/32/
  64/128 coarse levels than the earlier 128-reference implementation.
- Non-cubic grids remain supported; reference and target dimensions are handled per
  axis.
- Fine levels larger than the configured material reference are rasterized directly
  rather than being incorrectly downsampled from a coarser material field.
- Updated postprocessing sanity checks to use the same material-building path rather
  than the former single-value cell assignment helper.
- Added explicit garbage collection of level-local epsilon arrays, boundary masks,
  gate masks, and temporary interpolation arrays before proceeding to the next main
  level. Only the potential needed for the next level and the final output state are
  intentionally retained.
- Zoom epsilon fields are also released after each zoom level; temporary residual
  coefficient arrays are explicitly deleted after residual plotting.
- Updated material API documentation to describe the high-resolution reference,
  volume averaging, and temporary file-backed behavior.

### Validation
- Verified deterministic repeatability of the same block distribution.
- Verified a coarse cell straddling two dielectric regions receives the corresponding
  weighted average instead of a randomly selected epsilon value.
- Verified non-cubic epsilon output shapes such as `255 x 255 x 99` for a
  `256 x 256 x 100` potential grid.
- Ran a main-solver smoke test through multiple non-cubic refinement levels.
- Compiled all simulation Python modules after the material and cleanup changes.

---

## v12 — Large-grid startup and independent physical NPY output controls

Changes:
- Main multigrid startup now begins at `64 x 64 x 64` whenever any requested
  final main-grid axis is greater than 512; smaller/equal targets retain the
  historical `8 x 8 x 8` start. Each axis still advances independently to its
  requested final resolution.
- Added independent top-level JSON controls:
  ```json
  "save_cut": false,
  "save_full": false
  ```
- Added `save_cut_box_nm`, a physical `(Lx, Ly, Lz)` box size in nm. The box
  center follows the current movement center for every movement position.
- Final main-grid output can now be saved as either a full `.npy`, a physical-box
  cut `.npy`, both, or neither.
- Final zoom output follows the same independent controls. The cut is mapped from
  the final zoom field's physical bounds, so it remains centered on the moving
  physical position rather than on a fixed array index.
- `save_cut` and `save_full` do not depend on one another. Both may be enabled,
  either may be enabled, or both may be disabled. With both disabled, no final
  main/zoom NPY is persisted; plotting during the run remains available.
- Added reusable physical-cut/full-NPY output helpers to `simulation/io_utils.py`.
- Removed unconditional final NPY writes from the batch simulation path.
- Added cleanup of the temporary initial zoom crop immediately after interpolation.
- Updated documentation/API references for the new controls and physical cut semantics.

Validation:
- Compiled all simulation modules.
- Verified independent full/cut output combinations using synthetic 3-D arrays.
- Verified physical cut slicing for rectangular fields.
- Verified large-grid startup selection logic for targets above and below 512.
---

## v13 — Off-centre, movement-relative physical NPY cuts

Changes:
- Changed `save_cut_box_nm` from a three-value box size `(Lx, Ly, Lz)` to six
  signed physical offsets `[xmin, xmax, ymin, ymax, zmin, zmax]` relative to the
  current movement centre.
- Example: `[-1, 20, -5, 5, 10, 15]` defines an off-centre physical box around
  the movement centre.
- The cut centre continues to follow the current physical movement centre for
  both final main and final zoom outputs.
- Added input normalization so the API accepts either a flat six-value list or
  three axis pairs.
- Out-of-bounds requests are clipped independently on x/y/z to the available
  physical field, so boxes near a simulation boundary remain safe.
- If a requested cut has no intersection with the available field, the cut is
  skipped cleanly instead of raising an indexing error.
- Updated the canonical `afm_config_nm.json` and documentation examples.
- Kept `save_cut` and `save_full` independent.



---

## v14 — Original-config-faithful canonical configuration

**Goal:** make the sole `afm_config_nm.json` reproduce the original uploaded
configuration as closely as possible while using the new physical-coordinate schema.

Changes:
- Replaced the original `Lx_nm/Ly_nm/Lz_nm = 256` representation with
  `voxel_nm3 = 0.5` and `512 × 512 × 512`, which represents the same
  `256 nm × 256 nm × 256 nm` physical domain.
- Preserved the original solver tolerances, voltage sweep, runtime limit, tip
  parameters, aspect ratio, output CSV name, fixed block selection, plotting settings,
  and 512³ grid.
- Converted the original fractional movement path to physical, origin-relative nm:
  `[0, 0, 20]` → `[50, 50, 20]` with `7.071 nm` spacing.
- Converted the original full-plane voltage gate to origin-relative physical
  coordinates `x,y = -128 … +128 nm`, `z = 0 nm`.
- Converted every original dielectric block's x/y coordinates from the old absolute
  0–256 nm frame to the new origin-centered frame by subtracting `(128, 128, 0)`;
  z coordinates are unchanged.
- Preserved the original single `offsets_nm: [0.0]` behavior through the new
  presimulation mechanism using `tip_z_offsets_nm: [0.0]`.
- Preserved the original full-potential-save behavior with `save_full: true`.
- Kept `save_cut: false`; `save_cut_box_nm` is empty because no cut was requested
  by the original configuration.
- Kept zoom disabled, matching the original upload, while retaining the current
  compatible zoom parameters and `clamp` field.
- Kept optional memory tracking disabled by default.
- Retained the current high-resolution epsilon volume-averaging material workflow.

This entry is intended to make the numerical intent of the original uploaded JSON
explicit while removing obsolete fractional/absolute-domain configuration fields.

---

## v15 — Configuration layout and natural zoom boundary default

Changes:
- Reordered `afm_config_nm.json` so the large `blocks_nm` definition is the final
  top-level section.
- Moved `Vgate`/`Vgate_nm` immediately before `blocks_nm`.
- Set `zoom_simulation.clamp` to `false`, enabling the natural/Neumann outer-boundary
  zoom mode in the canonical configuration.
- No physical geometry, grid resolution, movement, solver tolerance, or dielectric
  values were otherwise changed.

## 2026-08-28 - CPU-parallel solver

- Added a Numba `prange` CPU implementation of the weighted-Jacobi 3-D dielectric stencil.
- Added top-level JSON `cpu_threads` control; `1` selects the original NumPy reference path and values greater than 1 use the parallel kernel.
- Kept BLAS/OpenMP thread counts at one so Numba owns solver CPU parallelism.
- Alliance job templates now request 8 CPUs per task and use scratch/project output roots instead of `$HOME` when available.
- Added `compare_old_new.py` to run one source JSON through the old package, new 1-CPU reference mode, and new multi-CPU mode and compare `.npy` files bit-for-bit.

### Correctness and benchmark follow-up

- Reworked the Numba stencil to read a full previous-field snapshot and apply
  the relaxation update only after the `prange` kernel returns. The prior
  in-place neighbor read/write pattern was a race and did not implement a
  valid Jacobi step.
- Forwarded `cpu_threads` from the batch launcher into the main-grid solver and
  zoom solver; the setting now controls actual command-line/config runs.
- Avoided allocating the snapshot in serial reference mode.
- Extended `compare_old_new.py` with wall-clock and summed solver timing,
  persistent `comparison_timing.csv` output, and bounded equal-iteration/single-
  position benchmark options.

Validation on 2026-08-28:

- Nonuniform 32³ solver: serial and 8-thread potentials were exactly equal,
  finite, and deterministic after 40 iterations.
- Bounded 512³ run at `tol=1e-6`, 50 iterations per level, one position:
  old vs new serial and new serial vs new 8-thread `.npy` outputs were both
  bit-for-bit equal. The final bounded-run residual was `1.569204e-5`, so this
  benchmark was intentionally not presented as a converged `1e-6` solve.
- Measured new serial vs new 8-thread speedup: `4.350x` by summed solver time
  and `2.152x` by end-to-end wall time, including Numba startup and I/O.
- Parallel residual diagnostics aligned within `1.47e-8` for the average and
  `1.09e-8` for the maximum residual; the corrected max reduction no longer
  reports false zeros.
