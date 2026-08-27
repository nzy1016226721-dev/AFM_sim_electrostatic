# AFM Simulation Package — Alliance-ready release

This branch is a cleaned release of the current local AFM electrostatic
simulation code. It keeps the current movement-aware solver/zoom implementation
and the useful postprocessing/diagnostic tools, while removing generated data,
backup configurations, obsolete configuration variants, and deprecated
presimulation/Joule-heating code.

## Included configurations

Only these configuration files are intentionally distributed:

- `afm_config_nm.json` — the sole canonical nanometre-coordinate AFM simulation configuration.

## Run a simulation

From inside `afm_package/`:

```bash
python run_all.py afm_config_nm.json
```

Legacy simulation syntax is retained:

```bash
python run_all.py sim afm_config_nm
```

The Python API is:

```python
from simulation.main_loop import batch_main
batch_main("afm_config_nm.json")
```

## Alliance / Slurm

Submit the supplied launcher with:

```bash
sbatch jobs/run_afm.sh afm_config_nm.json
```

The launcher is configured for a conservative initial resource request. Tune
CPU, memory, and wall time after measuring the target grid sizes.

## Scope

This release is **electrostatic only**. It deliberately does **not** include
GitHub's conductivity/sigma/Joule-heating subsystem. There is no `joule.py`,
no conductivity material cache, and no Joule-power output path.

The retained material model is dielectric-only (`epsilon`). The solver still
supports spatially varying dielectric blocks and the current movement-aware
AFM geometry workflow.

### Dielectric material resolution

Dielectric blocks are not assigned to coarse cells by picking whichever block
happens to contain a representative grid point. The solver builds a temporary
high-resolution epsilon reference directly from the JSON `blocks_nm` entries,
including the current movement-adjusted positions, and volume-averages that
reference onto each coarse solver level. This is especially important for the
8/16/32/64/128 levels, where one coarse cell can span multiple dielectric
regions.

The canonical setting is:

```json
"epsilon_material": {
  "reference_resolution": 512,
  "method": "high_resolution_volume_average"
}
```

The reference is file-backed temporarily and removed after the run, so it does
not remain as a large resident array between levels.

## Package contents

```text
afm_package/
└── afm_config_nm.json
├── run_all.py
├── requirements.txt
├── jobs/run_afm.sh
├── simulation/
│   ├── main_loop.py
│   ├── solver.py
│   ├── materials.py
│   ├── io_utils.py
│   ├── plotting.py
│   ├── zoom.py
│   └── runtime.py
└── postprocessing/
    ├── plot_npy.py
    ├── field_calculator.py
    ├── field_lines.py
    ├── potential_map.py
    ├── sanity_check.py
    ├── capacitance_sanity_check.py
    ├── lever_arm_calc.py
    └── npy_utils.py
```

Generated files such as `residual_history.csv`, `mg_timing_log.csv`, NPY
potentials, and figures are created at runtime and are intentionally not part
of the repository release.



## Zoom grid compatibility

The distributed configuration uses a 512×512×512 main grid with `zoom_factor: 2` and `zoom_limit: 4`. With the standard half-domain cuts (`0.25–0.75` in x/y and `0–0.5` in z), the solver produces exact nominal 2× and 4× physical-resolution levels. For a 512³ main grid these levels are also 512³ arrays, but each level covers a progressively smaller physical domain. The zoom code computes the target sample count from `voxel_nm3` and physical zoom magnification, so the same configuration remains valid if the main grid is changed to another compatible power-of-two resolution such as 1024³ or 2048³.

The zoom implementation does not rely on a fractional/intermediate configuration file. `zoom_factor` controls the magnification between levels, while `zoom_limit` controls the highest requested magnification.

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



## Final NPY output controls

The canonical JSON independently controls persistent final potential arrays:

```json
"save_cut": false,
"save_full": false,
"save_cut_box_nm": [-32.0, 32.0, -32.0, 32.0, -32.0, 32.0]
```

- `save_full=true` saves the complete final main and final zoom arrays.
- `save_cut=true` saves a physical nm box from the final main and final zoom arrays.
- `save_cut_box_nm` contains six signed offsets `[xmin, xmax, ymin, ymax, zmin, zmax]` in nm
  relative to the current physical movement center. For example,
  `[-1, 20, -5, 5, 10, 15]` means x=`center_x-1..center_x+20` nm,
  y=`center_y-5..center_y+5` nm, and z=`center_z+10..center_z+15` nm.
  The requested box is clipped to the available physical field when it crosses a boundary.
- The controls are independent. With both false, final main/zoom NPY files are not
  written, while interactive plotting can still run.

For targets larger than 512 in any main-grid axis, the multigrid hierarchy starts at
`64^3` and then grows independently along x/y/z until the requested target is reached.
