## Distributed-memory MPI solver

Large single grids can now be decomposed over multiple nodes with
`run_mpi.py` and `jobs/run_afm_mpi.sh`. This path uses a 3-D Cartesian MPI
grid, halo exchange, global convergence reductions, hybrid Numba threads per
rank, bounded cut gathering, and MPI-IO for optional full `.npy` output. It is
the required path for grids whose solver arrays exceed one node's memory.

See `README_MPI.md` for environment setup, no-allocation memory planning, the
two-rank cluster smoke test, current limitations, and the packaged 16384³
single-case template. The original `run_all.py` path remains available for
single-node serial/Numba jobs.

## CPU-parallel solver

The 3-D dielectric relaxation update is CPU-parallelized with Numba. Set the
number of solver CPU threads in the AFM JSON with the top-level `cpu_threads`
value. No terminal CPU-thread argument is required. For example:

```json
"cpu_threads": 32
```

`cpu_threads: 1` uses the original NumPy reference path. Values greater than 1
use a snapshot-based fused `prange` stencil: the kernel reads a read-only copy
of the previous field, then relaxation is applied separately after the kernel
returns. This preserves Jacobi semantics and avoids a read/write race between
threads. The batch launcher forwards the setting to both main-grid and zoom
solves. On Alliance/FIR, the requested value is capped
to the Slurm `CPUS_PER_TASK` allocation, so the JSON cannot accidentally use
more CPUs than the job requested. BLAS/OpenMP thread counts remain at one so
Numba owns the solver parallelism.

The distributed production launcher requests 35 CPUs for its 140-GiB memory
request (4 GiB per core). Each JSON controls its own `cpu_threads` value; the
current canonical JSON uses 35 so the allocated cores are available to Numba.
Keep that value no higher than the `#SBATCH --cpus-per-task` allocation when
changing resources.

### Bit-for-bit compatibility test

`compare_old_new.py` runs one source JSON through the old package, the new
package in `cpu_threads=1` reference mode, and the new package in multi-CPU
mode. It compares generated `.npy` files both as NumPy arrays and as raw file
bytes. Example:

```bash
python compare_old_new.py \
    --old-pack /path/to/old/afm_package \
    --new-pack /path/to/new/afm_package \
    --config /path/to/afm_config_nm.json \
    --parallel-threads 8 \
    --no-plot
```

Run this on a small test configuration first; a production 512^3 configuration
can be computationally and memory intensive. The checker also writes
`comparison_timing.csv`. Add `--max-iter 50 --single-position` for a bounded,
equal-work 512^3 benchmark; this keeps the configured convergence tolerance
at `1e-6` (or whatever the source JSON specifies) while intentionally using a
common iteration cap for timing and output comparison.

# AFM Simulation Package — Alliance-ready release

This branch is a cleaned release of the current local AFM electrostatic
simulation code. It keeps the current movement-aware solver/zoom implementation
and the useful postprocessing/diagnostic tools, while removing generated data,
backup configurations, obsolete configuration variants, and deprecated
presimulation/Joule-heating code.

## Included configurations

The distributed configuration files are:

- `afm_config_nm.json` — canonical nanometre-coordinate AFM simulation configuration.

Additional JSON configurations can be added alongside these files.

### Current Alliance trial configuration

The distributed `afm_config_nm.json` is configured for the current FIR trial:

```text
main grid       1024 × 1024 × 1024
main residual   1e-6
zoom residual   1e-6
Vtip sweep      -1.0 V to -5.0 V, step 0.5 V
solver CPUs     35 (from JSON)
Slurm CPUs      35
memory          140G
wall time       15:00:00
```

## Run a simulation

From inside `afm_package/`, pass the JSON filename/path you want to run:

```bash
python run_all.py afm_config_nm.json
```

The positional argument is always an actual JSON file. There is no base-name
or suffix lookup.

With no argument in a normal terminal, the launcher automatically selects the
newest AFM JSON in the current working directory and runs only that file:

```bash
python run_all.py
```

For an interactive selection, enumerate every AFM JSON in the working directory
and choose one explicitly, then choose the output root:

```bash
python run_all.py --interactive
```

Interactive output is written under `<output-root>/<config-name>/`. The output
root can also be selected non-interactively with `--output-dir`:

```bash
python run_all.py afm_config_nm.json --output-dir outputs/local_afm
```

In Spyder-like IDE execution, running `run_all.py` with no arguments retains the
launcher menu; selecting **Simulation** opens the same all-config selection and
output prompts.

The Python API runs one explicit JSON at a time:

```python
from simulation.main_loop import batch_main
batch_main("afm_config_nm.json")
```

## Alliance / Slurm

`jobs/run_afm.sh` is the only job-submission script. It accepts one JSON path,
so the same `.sh` file runs every solver configuration:

```bash
# Run from the package root:
sbatch --account=rrg-hongguo-ad jobs/run_afm.sh afm_config_nm.json

# Or run from package/jobs/ (the user's current Fir layout):
cd jobs
sbatch --account=rrg-hongguo-ad run_afm.sh afm_config_nm.json

# A second configuration uses the same launcher:
sbatch --account=rrg-hongguo-ad run_afm.sh ../afm_config_case2.json
```

If the JSON argument is omitted, `afm_config_nm.json` is used. The launcher
passes the selected file directly to `run_all.py`; it never asks for input or
performs base-name lookup. Additional non-interactive `run_all.py` options may
follow the JSON, for example `--no-plot` or `--output-dir`. Both `--interactive`
and `--plot` are rejected in a batch job.

The supplied header uses the RAS account `rrg-hongguo-ad`; override it at
submission if a different account is required. The launcher also sets
`AFM_NONINTERACTIVE=1` and `MPLBACKEND=Agg`, so Python never waits for stdin or
an interactive display. It invokes Python directly after activating the venv,
which avoids execute-bit failures from a staged helper `.sh` file.

One-time Fir setup and an optional configuration/environment check are:

```bash
bash jobs/setup_afm_env.sh
bash jobs/preflight_afm.sh
```

Slurm stages the batch script in a temporary spool directory. The launcher
therefore resolves the package from `SLURM_SUBMIT_DIR` (or an explicitly set
`AFM_PACKAGE_ROOT`) rather than from `BASH_SOURCE[0]`. This is required for
both submission layouts shown above.

Simulation results use the package's conventional `outputs` directory by
default. The selected configuration name is appended beneath a job-specific
directory, for example:

```text
/home/$USER/afm_parallel/outputs/job_123456/afm_config_nm/
```

Slurm stdout and stderr do not go in this scientific-result directory. They
are written to the sibling `outputs/slurm_logs/` directory as
`<job-name>-<jobid>.out` and `<job-name>-<jobid>.err`, keeping the package root
and each configuration directory free of scheduler logs.

Historical test outputs, old scheduler logs, deployment bundles, status notes,
one-off upload helpers, and generated bytecode are retained under
`archive_old/` during cleanup. This directory is excluded from deployment;
only the live production outputs and current package source are kept in the
working tree.

To choose a different root explicitly (for example, project storage):

```bash
sbatch --account=rrg-hongguo-ad \
  --export=ALL,AFM_OUTPUT_ROOT="$PWD/outputs" \
  jobs/run_afm.sh afm_config_nm.json
```

The JSON's `output_dir` remains the local-run default when no batch override is
provided. This package-local default can be overridden only by setting
`AFM_OUTPUT_ROOT` at submission.

The canonical trial writes one 4-GiB float32 main-grid NPY per voltage and
movement position (`save_full: true`), before zoom outputs are added. The
current sweep is 9 voltages × 12 positions = 108 solves (the 7.071 nm input
spacing yields 11 intervals over the 70.71 nm diagonal), so confirm sufficient
package output capacity before submission; set `AFM_OUTPUT_ROOT` to a project-specific
location if required. `mg_max_runtime` is a per-grid-solve limit, not a limit
for the entire voltage/movement sweep.

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

The distributed FIR trial configuration uses a 1024×1024×1024 main grid with `zoom_factor: 2` and `zoom_limit: 4`. With the standard half-domain cuts (`0.25–0.75` in x/y and `0–0.5` in z), the solver produces exact nominal 2× and 4× physical-resolution levels. The zoom code computes the target sample count from `voxel_nm3` and physical zoom magnification, so the same configuration remains valid if the main grid is changed to another compatible power-of-two resolution such as 512³ or 2048³.

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

Every zoom-cut axis is optional.  If the ``cut`` object, or an individual
``x_range``, ``y_range``, or ``z_range``, is omitted or null, it defaults to
``[0.0, 1.0]`` (the full current domain).  Dielectric and gate objects follow
the same full-axis default for omitted/null ranges, and an omitted
``Vgate_val`` means ``0.0`` V.



## Memory usage logging

Each main multigrid level and each zoom level is sampled for process resident memory while its solver is running. The peak resident memory is appended to `memory_usage_log.csv` in the same output directory as the other simulation results. The CSV has exactly two columns:

```text
level resolution,memory cost(in GB)
main 32x32x32,0.123456
zoom 2x (64x64x64),0.234567
zoom 4x (128x128x128),0.456789
```

Memory tracking is loaded lazily only when `memory_tracking` is enabled.
`psutil` is included in `requirements.txt` because the canonical Fir
configuration enables it. Memory is reported in GiB (1024^3 bytes) under the
requested `memory cost(in GB)` column name.


## Physical coordinates and tip-z presimulation

All AFM geometry in the canonical JSON configurations is expressed in nanometres
relative to `coordinate_system.origin_fraction` in the main grid. The default
origin is `[0.5, 0.5, 0.0]`, i.e. the centre of the bottom XY plane. The solver
converts these physical coordinates to fractional grid coordinates internally,
using the single `voxel_nm3` entry together with `grid_resolution`. For example,
`voxel_nm3: 0.5` and the distributed `1024x1024x1024` main grid define a 512 nm
physical domain on each axis. Changing the main grid automatically changes the
physical domain without changing any geometry coordinates.

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
For the standard half-domain cuts, a 1024³ main array produces 1024³ arrays at
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
presimulation block itself is removed.

Each generated JSON is now an independent configuration. Run one explicitly,
for example:

```text
python run_all.py sim afm_config_nm_+5nm.json
```

Or use interactive mode to see all generated/configuration JSONs and select one.
No base-name lookup or implicit multi-file series execution is performed.

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
"save_cut": true,
"save_full": true,
"save_cut_reference": "movement_center",
"save_cut_box_nm": [-50, 50, -50, 50, -20, 80]
```

- `save_full=true` saves the complete final main and final zoom arrays.
- `save_cut=true` saves a physical nm box from the final main and final zoom arrays.
- `save_cut_box_nm` contains six signed offsets `[xmin, xmax, ymin, ymax, zmin, zmax]` in nm
  relative to the current physical movement center.
- The packaged movement z coordinate is 20 nm. Therefore
  `[-50, 50, -50, 50, -20, 80]` follows each movement position in x/y while
  saving the absolute z interval 0..100 nm (`20-20` through `20+80`).
  The requested box is clipped to the available physical field when it crosses a boundary.
- An omitted, null, or empty `save_cut_box_nm` selects the full represented field.
- The controls are independent. With both false, final main/zoom NPY files are not
  written, while interactive plotting can still run.
- Cut filenames preserve configuration, movement, voltage, and zoom tags and append
  the uncut source shape, for example `_cut_from_grid512x512x512.npy`.
- New movement tags are physical signed offsets from the first configured centre:
  `afm_phi_1_0nm_-1.00V.npy` at the centre,
  `afm_phi_1_0.001nm_-1.00V.npy` for a positive offset, and
  `afm_phi_1-0.001nm_-1.00V.npy` for a negative offset. This replaces rounded
  fractional `cx/cy/cz` tags, so distinct small movement steps do not alias.

For targets through 512 in every main-grid axis, the hierarchy starts at `8^3`;
larger targets below 2048 start at `64^3`. Targets at least 2048 use the dynamic
6–8-level policy: 2048, 4096 and 8192 start at `64^3`, while 16384 starts at
`128^3`, then grows independently along x/y/z until the target is reached.
## Alliance/Fir batch-job control

Use one generic `jobs/run_afm.sh` for all JSON configurations. The JSON path is
the first argument to `sbatch`, not a value encoded in a second selector file:

```bash
sbatch --account=rrg-hongguo-ad jobs/run_afm.sh afm_config_nm.json
sbatch --account=rrg-hongguo-ad jobs/run_afm.sh afm_config_case2.json
```

The header defaults to `rrg-hongguo-ad` (the RAS account used for the Fir trial)
and can be overridden with `sbatch --account=<your-account>`. The script also
works when submitted from inside `jobs/`:

```bash
cd jobs
sbatch --account=rrg-hongguo-ad run_afm.sh ../afm_config_case2.json
```

With no JSON argument, the launcher uses `afm_config_nm.json`. It runs exactly
one explicit JSON, rejects `--interactive` and `--plot`, and appends `--no-plot`
unless the caller explicitly supplies `--no-plot`.

Every Slurm job exports `AFM_JOB_OUTPUT_ROOT`, which defaults to:

```text
outputs/job_<SLURM_JOB_ID>/<config-name>/
```

Therefore simultaneous jobs cannot mix their NPY/CSV outputs even though the
JSON can keep its legacy `output_dir: "outputs"` and `output_dir_mode: "default"`.
Set `AFM_OUTPUT_ROOT` before submission only for an intentional alternate
location. The project default is package-local `outputs`; do not use scratch
unless explicitly requested. Keep batch stdout/stderr in `outputs/slurm_logs`,
separate from job numerical results. See [workspace handoff](../docs/AI_HANDOFF.md)
for current state, numerical contracts and validation; historical job/status
examples in this guide are not live observations.

### Per-level memory test

`afm_memory_test.json` and `jobs/run_memory_test.sh` now provide a direct
4096^3-to-8192^3 Fir diagnostic. It starts at 4096^3, lets that solver run for
up to 20 minutes, then advances to the 8192^3 stage. The test requests one
`cpularge_bynode_b1` node with `192` CPUs and `4096G` RAM, has a one-hour
Slurm wall-time, and ends its child at 55 minutes with an explicit
`[MEMORY-TEST][TIME-BUDGET]` marker. It uses `1e-4` residual tolerances,
disables plotting/zoom/full-field NPY saves, and flushes one-second process-RSS
samples to `memory_live_rss.csv`; those samples remain available if a level is
stopped before its conventional peak-memory row can be written.

Submit it from the package root:

```bash
sbatch --account=rrg-hongguo-ad jobs/run_memory_test.sh afm_memory_test.json
```

Or from inside `jobs/`:

```bash
sbatch --account=rrg-hongguo-ad run_memory_test.sh ../afm_memory_test.json
```

By default, results are placed under:

```text
outputs/afm_memory_test/job_<SLURM_JOB_ID>/
├── afm_memory_test_level_<N>/memory_usage_log.csv
├── afm_memory_test_level_<N>/mg_timing_log.csv
├── _logs/level_<N>.out and level_<N>.err
└── current_level.txt
```

The Slurm error file contains `[MEMORY-TEST][START]` before every allocation
and `[MEMORY-TEST][FAIL]` with the target level, grid, exit code, and detected
OOM/timeout reason when the child returns an error. If the kernel kills the
entire cgroup with SIGKILL, the last `START` marker and Slurm's OOM record are
the available failure evidence; SIGKILL itself cannot be trapped by Bash.

### Terminal behavior

```bash
python run_all.py
```

From a normal terminal, the first command runs the canonical
`afm_config_nm.json` without a menu; the second runs only the explicitly named
JSON. In Spyder-like IDE execution, `run_all.py` without arguments still opens
the legacy launcher menu, while an explicit JSON path runs only that file.

### Adding another AFM batch run

Do not copy or edit another selector script. Add the JSON beside the package
and submit it as the first argument to the same launcher. This keeps the Slurm
account, environment setup, output isolation, and non-interactive behavior in
one maintained file.
