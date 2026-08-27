# AFM Simulation Package — Documentation

## 1. Purpose

This package solves the electrostatic potential of an AFM-tip / dielectric /
gate geometry on a 3-D finite-difference grid. The current implementation
supports spatially varying dielectric blocks, moving blocks and voltage gates,
multilevel zoom simulations, residual logging, and visualization.

This release intentionally excludes the conductivity/sigma/Joule-heating
subsystem from the GitHub version. Do not expect `joule.py`, conductivity
material maps, current-density calculations, or Joule-power files in this
branch.

## 2. Directory structure

```text
afm_package/
├── afm_config_nm.json          # supported distributed config
├── run_all.py                 # CLI/interactive launcher
├── requirements.txt
├── jobs/run_afm.sh            # Alliance/Slurm launcher
├── simulation/
│   ├── main_loop.py           # public simulation orchestration
│   ├── solver.py              # masked multigrid electrostatic solver
│   ├── materials.py           # dielectric-cell generation
│   ├── io_utils.py            # masks, logs, output helpers
│   ├── plotting.py             # potential/residual visualisation
│   ├── zoom.py                 # movement-aware multilevel zoom
│   └── runtime.py              # IDE/CLI/HPC runtime policy
└── postprocessing/             # optional analysis and diagnostics
```

## 3. Configuration model

The simulator consumes a JSON dictionary. `afm_config_nm.json` is the only
distributed configuration. Physical size is controlled by `voxel_nm3` and
`grid_resolution`; no separate `Lx_nm`, `Ly_nm`, or `Lz_nm` entries are needed. A configuration can define the grid,
voltage sweep, AFM tip, voltage-gate geometry, dielectric blocks, movement,
zoom, plotting, and output behavior.

Important optional runtime settings include:

```json
"plotting": {
  "enabled": true,
  "disable_in_non_ide": true
},
"output_dir_mode": "default"
```

`output_dir_mode` accepts `default`, `config`, or `job_config`. The latter uses
`SLURM_JOB_ID` when running on Alliance.

## 4. Main API

### `simulation.main_loop.run_afm_simulation`

Runs one electrostatic simulation for a supplied grid/configuration state.
It builds the AFM tip mask, voltage-gate mask, dielectric-cell map, solves the
masked Poisson problem, logs residuals/timing, and optionally generates plots.

### `simulation.main_loop.batch_main`

Loads one configuration or a set of matching configuration files and performs
the configured movement/voltage sweep. Prefer passing an explicit JSON path
for reproducible batch jobs.

### Movement API

The movement-aware functions in `main_loop.py` support:

- translating dielectric blocks;
- translating voltage gates;
- computing centre positions;
- preserving relative offsets between moving objects;
- previewing geometry before a long run.

### `simulation.zoom.run_zoom_simulation`

Runs the multilevel zoom workflow. The current local implementation supports
movement-centred zooming, deterministic crop indices, cascade mode, residual
plots, and saving intermediate levels.

### `simulation.solver.mg_3d_masked`

Core masked 3-D multigrid electrostatic solver. It handles the tip/gate masks,
spatially varying dielectric coefficients, Neumann boundaries, convergence
monitoring, and residual logging.

### `simulation.materials`

The dielectric material builder is deterministic and volume-averaged. The JSON
`blocks_nm` distribution is normalized to fractional coordinates, rasterized onto
a temporary high-resolution voxel reference, and box-averaged onto each solver
level's `(nx-1, ny-1, nz-1)` cell grid. This prevents coarse cells that contain
multiple dielectric values from receiving an arbitrary single material value.

`epsilon_material.reference_resolution` controls the reference resolution (the
canonical config uses 512). The reference is file-backed temporarily so a large
reference field does not remain resident in RAM. It is removed after the simulation.

`generate_eps_level(phi_shape, blocks, reference_shape, reference)` — Build a
deterministic level-specific epsilon array.

`build_eps_reference_memmap(reference_shape, blocks)` — Generate the temporary
high-resolution NPY/memmap material reference.

`average_reference_to_cells(reference, target_cells)` — Box-average the reference
field onto a target solver-cell grid.

`release_eps_reference(path, mmap)` — Flush and remove the temporary reference.

`generate_eps_cell(...)` — Backward-compatible wrapper around `generate_eps_level`.

### `simulation.io_utils.make_gate_mask`

Constructs the voltage-gate boundary mask used by the electrostatic solver.

## 5. Output files

Simulation output is generated in the configured `output_dir`. Typical files
include:

- `afm_phi_*.npy` — solved electrostatic potential;
- `residual_history.csv` — multigrid residual history;
- `mg_timing_log.csv` — solver timing information;
- `memory_usage_log.csv` — peak process memory for each main/zoom simulation level; columns are `level resolution` and `memory cost(in GB)`;
- zoom-level potential NPY files;
- optional potential/residual figures.

These are runtime artifacts and are not committed to this release branch.

## 6. Postprocessing

The retained postprocessing modules operate on electrostatic simulation
outputs. They include potential plotting, electric-field diagnostics,
field-line plotting, potential maps, sanity checks, capacitance checks, and
QD lever-arm analysis.

The obsolete `master_post.py` aggregator and the missing `integrate_power.py`
path are intentionally not shipped. The launcher exposes only the currently
available postprocessing functions.

## 7. API documentation

See `API_Docstrings.md` for the public API inventory and docstring-oriented
reference. Public functions in the simulation package document their inputs,
outputs, side effects, and important assumptions directly in their source.

## 8. Dependencies

`requirements.txt` lists:

- NumPy
- SciPy
- Matplotlib
- pandas

## 9. Design boundary

This branch is deliberately narrower than the upstream GitHub implementation:

**Included:** electrostatic potential, dielectric variation, AFM movement,
voltage gates, multigrid solving, zoom, visualization, residual/timing logs,
and electrostatic postprocessing.

**Excluded:** conductivity (`sigma`), Joule heating, current density,
power-density calculation, Joule energy/cycle calculations, sigma caches, and
the associated presimulation generators.


## Zoom boundary modes

The ``zoom_simulation.clamp`` setting selects how the outer boundary of each zoomed domain is treated:

- ``"clamp": true`` — historical mode. The six outer faces are fixed (Dirichlet) and their values are inherited from the previous grid through the linear interpolation used to initialize the zoom level.
- ``"clamp": false`` — natural-boundary mode. The configured voltage masks inside the cut remain fixed at their configured voltages, while the six outer faces are treated with homogeneous Neumann boundary conditions. The initial potential is still inherited from the previous level through linear interpolation.

Example:

```json
"zoom_simulation": {
    "enabled": true,
    "zoom_factor": 2,
    "zoom_limit": 4,
    "clamp": false,
    "cut": {
        "x_range": [0.25, 0.75],
        "y_range": [0.25, 0.75],
        "z_range": [0.0, 0.5]
    }
}
```

The default is ``true`` for backward compatibility.



## Memory usage logging

Each main multigrid level and each zoom level is sampled for process resident memory while its solver is running. The peak resident memory is appended to `memory_usage_log.csv` in the same output directory as the other simulation results. The CSV has exactly two columns:

```text
level resolution,memory cost(in GB)
main 32x32x32,0.123456
zoom 2x (64x64x64),0.234567
zoom 4x (128x128x128),0.456789
```

Memory tracking is an optional component. It is loaded lazily only when `memory_tracking` is enabled; install `requirements-memory.txt` to enable it. Memory is reported in GiB (1024^3 bytes) under the requested `memory cost(in GB)` column name.


## Physical coordinates and tip-z presimulation

All AFM geometry in the canonical JSON configurations is expressed in nanometres
relative to `coordinate_system.origin_fraction` in the main grid. The default
origin is `[0.5, 0.5, 0.0]`, i.e. the centre of the bottom XY plane. The solver
converts these physical coordinates to fractional grid coordinates internally,
using the single `voxel_nm3` entry together with `grid_resolution`. For example,
`voxel_nm3: 0.5` and a `512x512x512` main grid define a 256 nm physical domain
on each axis. Changing the main grid to `2048x2048x2048` automatically enlarges
the physical domain to 1024 nm per axis without changing any geometry coordinates.

Use `blocks_nm` and `Vgate_nm` for dielectric and voltage-mask geometry. An axis
range may be omitted when the object spans the entire main-domain axis; this is
especially useful for substrate/full-domain gates because the range automatically
follows a larger simulation domain.

Tip dimensions use `tip_z_nm`, `R_nm`, and `r_tip_nm`. Movement uses
`movement.start_nm`, `movement.end_nm`, and `movement.spacing_nm`.

### Physical voxel scale

The canonical configuration uses one physical scale entry:

```json
"grid_resolution": {"nx": 512, "ny": 512, "nz": 512},
"voxel_nm3": 0.5
```

`voxel_nm3` is the edge length of one main-grid voxel in nanometres (the name
is retained exactly as the public configuration key). The physical domain is
derived as `nx*voxel_nm3`, `ny*voxel_nm3`, and `nz*voxel_nm3`. Thus the example
represents a 256 nm × 256 nm × 256 nm main simulation space. Changing only
`grid_resolution` automatically changes the physical size while all `*_nm`
geometry remains fixed relative to the configured origin.

Zoom levels use the same physical voxel scale: `zoom_factor: 2` means a voxel
edge half as large as the main grid, and `zoom_factor: 4` means one quarter.
For the standard half-domain cuts, a 512³ main array produces 512³ arrays at
2x and 4x, but those arrays represent progressively smaller physical regions.

### Tip-z sweep

The old `offsets_nm` field is removed. Put the requested physical sweep in the
base configuration:

```json
"presimulation": {
  "tip_z_offsets_nm": [-10, -5, 0, 5, 10]
}
```

Run:

```text
python run_all.py presim afm_config_nm.json
```

This generates:

```text
afm_config_nm_-10nm.json
afm_config_nm_-5nm.json
afm_config_nm_0nm.json
afm_config_nm_+5nm.json
afm_config_nm_+10nm.json
```

Each generated file contains the corresponding absolute `tip_z_nm`; the
presimulation block itself is removed. The solver can then run the generated
series by using the common base name:

```text
python run_all.py sim afm_config_nm
```

The solver loads the generated JSONs in increasing offset order. No fractional
intermediate configuration is required.

### Physical units and rectangular main grids

The canonical configuration uses `voxel_nm3` as the edge length of one main-grid
voxel in nanometres. The physical domain is derived independently on each axis
from `nx`, `ny`, and `nz`; no `Lx_nm`, `Ly_nm`, or `Lz_nm` values are required in
the JSON. All `*_nm` geometry (dielectric blocks, voltage gates, tip dimensions,
tip position, and movement coordinates) is converted axis-by-axis from the
physical origin. AFM tip dimensions are constructed directly in physical
coordinates, so an isotropic tip remains isotropic on non-cubic grids such as
`256 x 256 x 100`.

Main-grid refinement also supports rectangular targets. Each axis is refined
independently and the interpolation is forced to the exact requested shape.
Zoom levels compute their target node count independently for x/y/z, so 2x and
4x magnification remain compatible with non-cubic main grids.



## Large-grid startup and final NPY saving

If any requested final main-grid dimension exceeds 512, the main solver starts at
`64 x 64 x 64` instead of `8 x 8 x 8`. Subsequent levels advance independently per
axis, allowing rectangular targets such as `2048 x 1024 x 512`.

Final NPY persistence is controlled independently by:

```json
"save_cut": false,
"save_full": false,
"save_cut_box_nm": [-32.0, 32.0, -32.0, 32.0, -32.0, 32.0]
```

`save_full` saves complete final main/zoom arrays. `save_cut` saves a physical box
relative to the current movement position. `save_cut_box_nm` contains six signed
offsets `[xmin, xmax, ymin, ymax, zmin, zmax]` in nm. For example,
`[-1, 20, -5, 5, 10, 15]` selects x=`center_x-1..center_x+20` nm,
y=`center_y-5..center_y+5` nm, and z=`center_z+10..center_z+15` nm. The requested
box is clipped independently on each axis if it extends beyond the physical field.
Both controls may be enabled or disabled independently.
