# AFM Simulation Package — Change Log

This file records the evolution of the current AFM simulation package relative to the
original uploaded package:

`afm_package`

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
