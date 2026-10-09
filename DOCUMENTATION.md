# AFM configuration and numerical contract

Updated for the standalone lossless RAM default on October 9, 2026. This is
implemented storage behaviour, not the abandoned October 4–6 certified-refinement
upgrade. See [RAM policy](RAM_SAVING.md), [MPI instructions](README_MPI.md) and the
[configuration/job catalogue](docs/CONFIGS_AND_JOBS.md).

## Launch and inheritance

`python run_all.py CONFIG.json --no-plot --output-dir OUTPUT` runs one explicit
shared-memory configuration. `run_mpi.py CONFIG.json` runs the distributed
path under MPI; `--plan` only estimates resources. Neither is an implicit
directory-wide batch runner. No-argument terminal `run_all.py` selects the
newest AFM JSON by modification time; IDE/menu execution can prompt. Select an
explicit file for reproducibility, especially with large/diagnostic presets.

`simulation.mpi_config.load_afm_config` resolves `extends` relative to its
declaring JSON, recursively merges dictionaries and replaces rather than
concatenates lists. Keep child and base files together; a child alone does
not show the effective physical problem. After inheritance, absent storage keys
default to `memory_mode=ram_compact`, `phi_update_mode=in_place` and
`residual_accumulation=scalar`. Normal JSONs record these explicitly and disable
solver-side plotting. See [alternatives and diagnostic exceptions](RAM_SAVING.md).

## Units, axes and geometry

| Setting | Implemented interpretation |
| --- | --- |
| `grid_resolution.nx/ny/nz` | Main-grid sample counts, each at least 2 |
| `voxel_nm3` | Legacy name for the main-grid edge length in **nm**, not a volume |
| Domain extent | Axis length is `N * voxel_nm3`; endpoint-node pitch is that length divided by `N - 1` |
| `coordinate_system.origin_fraction` | Origin for `*_nm` coordinates; default `[0.5, 0.5, 0.0]` |
| `blocks_nm` / `Vgate_nm` | Dielectric blocks / fixed-voltage gates in origin-relative nanometres |
| `tip_z_nm`, `R_nm`, `r_tip_nm` | Tip dimensions in nm; pyramid `r_tip_nm` is its base half-side |
| `tip_shape` | `pyramid` default; `cone` is an explicit alternative |
| `movement.start_nm/end_nm/spacing_nm` | Physical movement path; movement-aware saving follows its current centre |
| `fixed_blocks` | Zero-based block indices that do not follow movement |

Fractional coordinates are used internally. Do not reinterpret extent as
`(N - 1) * voxel_nm3`, change array-axis order, or infer an exact physical
cylinder from raster blocks. QTCAD FEM geometry is owned by the separate
quantum workflow; no new geometry/intake contract is introduced here.

Missing/null spatial axis ranges normalize to the complete main-domain axis.
A `Vgate_nm` entry with only `"z_range_nm": [0.0, 0.0]` covers full x/y at z=0;
omitted `Vgate_val` defaults to 0 V. Explicit ranges, movement constraints and
existing block-order rules still apply. Missing/empty saved-cut boxes mean
full-field cuts. [Range tests](tests/test_config_defaults.py) define the
supported omission cases; this is not a new movement algorithm.

Generated JSONs put `cpu_threads` first and dielectric blocks last using
`ordered_config_for_json`. Preset storage/plotting defaults have changed;
physical/numerical values and job resource requests are preserved in a verified
pre-promotion backup.

## Materials, hierarchy and convergence

The equation is `div(epsilon * grad(phi)) = 0` with existing fixed masks and
Dirichlet conditions. The maintained iteration is weighted **snapshot**
Jacobi on a coarse-to-fine hierarchy, not a residual-correction V-cycle or
in-place Gauss–Seidel method. Normal fields/coefficients are float32. The RAM
default keeps one phi with bounded old-plane snapshots and reconstructs face
coefficients from an exact deduplicated cell-epsilon plane bank. `standard`
retains oriented face arrays as an explicit alternative.

The original `epsilon_material.reference_resolution` policy remains:
a high-resolution reference (512 by default) is volume-averaged for applicable
coarse cells, with source-defined rasterization for finer grids. Lossless
plane deduplication/packed masks change storage only; no reduced epsilon
precision, common finest-mesh policy or certified dielectric coarsening is
enabled. [README_MPI.md](README_MPI.md) explains bounded construction and MPI IO.

Default starts depend on the largest target axis: through 512 start 8;
larger targets below 2048 start 64. Large targets select 6–8 doubling levels:
2048 starts 64, 4096 starts 64, 8192 starts 64 and 16384 starts 128.
Explicit `initial_grid_level` / `mpi.initial_grid_level` are overrides.

`res_tol_main` is the source-defined RMS residual threshold, not a certified
maximum voltage error. Logs also record the maximum residual. Normal levels
stop by residual or `mg_max_runtime`; a legacy `max_iter` is **not** an ordinary
iteration cap. MPI `require_convergence` defaults to true and rejects an
unconverged hierarchy before scientific output. Inspect final logs even when
a shared-memory run writes an output after a deadline.

`residual_tolerance_mode` defaults to `absolute`. Explicit `relative_drive`
normalizes **all** Dirichlet drives by a common maximum magnitude, converges
that normalized problem, then rescales to physical volts. This pre-existing
mode addresses voltage-dependent float32 floors; an absolute native residual
and normalized residual are not interchangeable. Opt-in `solver_dtype:
float64` / `float64_memory_smoke` diagnostics require explicit `memory_mode:
standard` with null storage selectors; the named diagnostic presets provide
that opt-out. Fixed diagnostic
iterations are not convergence evidence.

Original shared-memory zoom remains in `simulation/zoom.py`, disabled in shipped
normal JSONs and unsupported in RAM modes. MPI rejects enabled zoom because it uses full arrays. The
cancelled upgrade's zoom removal was reverted.

## Fields, cuts and provenance

`save_full` and `save_cut` control full/cut potential NPYs. Arrays retain
`(x, y, z)` axis order and ordinary NPY layout; normal potentials stay float32.
Keep effective JSON and logs with fields. NPY alone is not a geometry manifest.

Maintained presets use `save_cut_reference: movement_center` and
`save_cut_box_nm: [-50, 50, -50, 50, -20, 80]`, offsets from the current centre.
At `(cx, cy, 20) nm`, this requests
`[cx-50,cx+50] × [cy-50,cy+50] × [0,100] nm`. Changes in centre z also move
the cut. Clipping/node selection follows the existing endpoint implementation;
saved sample count depends on resolution and need not equal 100³.

Cut names retain case, signed physical path displacement and voltage, then
append `_cut_from_gridNXxNYxNZ` for the original uncut shape. For example,
`afm_phi_1_0nm_-1.00V_cut_from_grid2048x2048x2048.npy` is the reference centre.
Sub-centinanometre offsets are not rounded into one token; collision handling
is preserved.

`output_dir_mode` supports `default`, `config`, `job_config`.
`AFM_JOB_OUTPUT_ROOT` is the authoritative batch root override and adds the
config stem. Jobs normally use `outputs/job_<id>/`; scheduler stdout/stderr
stay separately under `outputs/slurm_logs/`. Residual/timing/memory CSVs and
MPI case logs are evidence, not source files for publication.

Compare compatible fields with the memory-mapped plane-wise tool:

```text
python postprocessing/compare_mpi_npy.py REFERENCE.npy CANDIDATE.npy --atol 1e-6 --rtol 0 --report outputs/comparison.json
```

It checks shape, dtype, finite values and every element at explicit tolerances.
A field-comparison pass alone does not establish convergence or validate a
different geometry. Dated evidence is in
[MPI_PARITY_VALIDATION_20260902.md](MPI_PARITY_VALIDATION_20260902.md)
and the MPI/job guides.
