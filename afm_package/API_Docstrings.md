# Public API Reference


This release documents the public Python API for the electrostatic AFM simulator.


## Simulation

### `simulation.io_utils`

- `save_for_qtcad(results, filename, output_dir)` — Save the potential array to a .npy file for QTCAD compatibility.

- `save_phi_3d(phi, results, tag, output_dir)` — Save the full 3D potential plus fields as a compressed NPZ.

- `log_residual_csv(iteration, res_avg, res_max, csv_file, output_dir)` — Append one row to the residual convergence CSV log.

- `log_timing(level, nx, ny, nz, elapsed, logfile, output_dir)` — Log the elapsed time for one MG level to a CSV file.

- `get_sorted_config_files(base_name, directory)` — Get a numerically sorted list of config files matching base_name_*.json.

- `make_gate_mask(nx, ny, nz, gate)` — Create a boolean 3D mask for a gate region.

### `simulation.main_loop`

- `run_afm_simulation(Vtip, nx, ny, nz, tip_z, R, r_tip, damping, nu1, nu2, max_iter, tol, aspect_ratio, verbose, eps_r, eps, mg_max_runtime, blocks, Vgate, output_dir, save_all_levels, level_name_prefix, plotting_enabled)` — Run a multiresolution AFM electrostatic simulation.

- `move_voltage_gate(Vgate, gate_index, center, xrange, yrange, zrange, Vgate_val)` — Reposition a voltage gate to a new centre with given half-extents.

- `move_dielectric_block(cfg, block_index, center, xrange, yrange, zrange, eps_val)` — Reposition a dielectric block to a new centre.

- `compute_block_positions(start_center, end_center, spacing)` — Linearly interpolate positions between two centres at a given spacing.

- `apply_block_motion(cfg, block_motion_list, center)` — Apply a list of block motions relative to a centre.

- `apply_vgate_motion(Vgate_list, vgate_motion_list, center)` — Apply a list of gate motions relative to a centre.

- `precompute_relative_offsets_for_blocks(blocks, indices, center0)` — Compute relative (centre-relative) offsets for a set of blocks.

- `precompute_relative_offsets_for_vgates(vgates, indices, center0)` — Compute relative offsets for a set of voltage gates.

- `apply_relative_blocks(blocks, rel, center)` — Apply precomputed relative offsets to position blocks at a new centre.

- `apply_relative_vgates(vgates, rel, center)` — Apply precomputed relative offsets to position gates at a new centre.

- `preview_before_run(cfg, plotting_enabled)` — Display an interactive 2D preview of dielectric blocks and gates.

- `preview_tip_only(cfg, plotting_enabled)` — Display a 3-projection scatter plot of the AFM tip voxels.

- `batch_main(CONFIG_BASE_NAME, config_dir, plotting_override)` — Main loop: iterate over config files, run simulation + zoom for each.

### `simulation.materials`

- `generate_eps_level(phi_shape, blocks, reference_shape, reference)` — Build a deterministic, volume-averaged epsilon array for one solver level from the JSON-derived block distribution.
- `build_eps_reference_memmap(reference_shape, blocks)` — Create a temporary high-resolution NPY/memmap epsilon reference without keeping the entire reference resident in RAM.
- `average_reference_to_cells(reference, target_cells)` — Box-average a high-resolution epsilon field onto the requested solver-cell dimensions.
- `release_eps_reference(path, mmap)` — Flush and remove a temporary epsilon reference file.
- `generate_eps_cell(phi, blocks, reference_shape, reference)` — Backward-compatible wrapper for `generate_eps_level`.

### `simulation.plotting`

- `plot_phi_plane(phi_matrix, boundary_mask, plane, cmap, tip_mask, apex)` — Plot a 2D slice of the electrostatic potential.

- `plot_residual_plane(residual_matrix, boundary_mask, plane, vmin, vmax, tip_mask, apex)` — Plot a log-scale 2D slice of the residual (with boundary masked black).

- `plot_residual_line(residual_matrix, line)` — Plot a 1D line-out of the residual magnitude.

- `visualize_afm_results(results, x_frac, y_frac, z_frac)` — Comprehensive 6-panel visualisation of AFM simulation results.

- `combine_phi_and_residual(phi, residual, boundary_mask, plane, title_tag)` — Create a side-by-side figure of potential and residual slices.

### `simulation.runtime`

- `is_spyder_like_ide()` — Return True when executing inside a Spyder-like IDE/IPython kernel.

- `resolve_plotting_enabled(cfg)` — Resolve whether interactive plotting should be performed.

- `resolve_output_dir(base_output_dir, config_path, cfg)` — Resolve output_dir while preserving the legacy default behavior.

### `simulation.solver`

- `build_downward_pointing_tip(nx, ny, nz, tip_z, R, r_tip, aspect_ratio, verbose)` — Generate a boolean mask for a hyperbolic AFM tip geometry.

- `compute_residual_vec_unpadded(V, mask, axp, axm, ayp, aym, azp, azm, a0)` — Compute residual of the discretized Poisson equation (full matrix output).

- `compute_residual_scalars(V, mask, axp, axm, ayp, aym, azp, azm, a0)` — Compute scalar residual norms (no full matrix allocation).

- `log_residual_csv(iteration, res_avg, res_max, csv_file, output_dir)` — Append one residual measurement to the simulation residual CSV log.

- `plot_convergence(csv_file, output_dir, show)` — Plot convergence history from residual CSV.

- `mg_3d_masked(Vtip, phi, boundary_mask, damping, nu1, nu2, max_iter, tol, verbose, eps_r, eps, mg_max_runtime, output_dir, plotting_enabled)` — Multigrid solver for the 3D Poisson equation with dielectric variation.

### `simulation.zoom`

- `cut_indices(n, lo, hi)` — Deterministic crop indices for a fractional cut range on an axis.

- `shape_factor(s, n, zoom_factor)` — Zoom factor for scaling a crop of s nodes into the n-node grid.

- `run_zoom_simulation(cfg, results, V, config_idx, time_log, output_dir, movement_active, center, center0, plotting_enabled)` — Run a recursive central-crop zoom simulation on top of the main result.


## Postprocessing

### `postprocessing.capacitance_sanity_check`

- `sphere_plane_capacitance(Z_m, R_m, eps0, tol)` — Exact capacitance of a sphere of radius R_m at height Z_m above

- `parse_phi_filename(fname)` — Extract metadata from a phi .npy filename.

- `extract_phi_qd(phi, qd_xr, qd_yr, qd_zr, Lx, Ly, Lz, zoom_bounds)` — Extract potential values inside a QD region.

- `compute_stats(phi_values)` — Compute mean and max-absolute value.

- `find_qd_block(nm_cfg, default_eps)` — Find the quantum dot block in the nm config by eps_val match.

- `create_default_qd_block(Lx, Ly, Lz, dot_diameter, dot_height, dot_bottom)` — Create a default QD block for testing.

- `sanity_check_comparison_multi(phi_files, qd_block, qd_top_z_nm, Lx_nm, Ly_nm, Lz_nm, R_nm, config_dir, nm_cfg, use_max)` — Compare simulated lever arm with analytical sphere‑plane model,

- `main()` — Interactive entry point for the multi‑layer sanity check.

### `postprocessing.field_calculator`

- `compute_residual_Laplace(phi, eps_cell, boundary_mask)` — Compute the Poisson equation residual: div(eps * grad(phi)).

- `check_curl_E(phi, dx, dy, dz)` — Check that curl of the electric field is approximately zero.

- `check_dirichlet(phi, tip_voltage, backgate_voltage, tip_mask)` — Check Dirichlet boundary condition fidelity.

- `check_neumann(phi, tip_mask)` — Check homogeneous Neumann BC fidelity on all six faces.

### `postprocessing.field_lines`

- `plot_field_lines(phi, plane, coord, Lx_nm, Ly_nm, Lz_nm, crop_radius_nm, title, save_path, tip_params, field_sign, tip_buffer_cells, blocks, streamplot_density, arrow_spacing_nm, max_arrow_len_nm, min_arrow_len_nm, mag_percentile)` — Plot electric-field lines for a saved electrostatic potential.

- `interactive_main()` — Run the interactive electric-field-line plotting interface.

### `postprocessing.lever_arm_calc`

- `parse_phi_filename(fname)` — Extract metadata from a standard AFM phi filename.

- `find_qd_block(nm_cfg, default_eps)` — Find the quantum dot block in the nm config by eps_val match.

- `extract_phi_qd(phi, qd_xr, qd_yr, qd_zr, Lx, Ly, Lz, zoom_bounds)` — Extract the potential values inside a QD region.

- `compute_stats(phi_values)` — Compute mean and max-absolute value of a 1D array.

- `main()` — QD lever arm calculation entry point.

- `plot_results(results, output_dir)` — Plot lever arm (alpha) vs tip spacing, colour-coded by Vtip.

### `postprocessing.npy_utils`

- `parse_phi_filename(fname)` — Extract metadata from a phi .npy filename.

### `postprocessing.plot_npy`

- `plot_afm_from_npy(phi_file, ex_file, ey_file, ez_file, x_frac, y_frac, z_frac, axis, cmap_phi, cmap_E, cmap_comp, show, save_prefix, show_component_slices, Lx_nm, Ly_nm, Lz_nm)` — Load and plot AFM potential (and optionally field) from .npy files.

### `postprocessing.potential_map`

- `plot_potential_map(phi, qd_block, tip_center_frac, R_nm, Lx_nm, Ly_nm, Lz_nm, z_slice, save_prefix)` — Plot a 3-D electrostatic potential map or selected slice.

- `interactive_main()` — Run the interactive potential-map plotting interface.

### `postprocessing.sanity_check`

- `plot_field_slice(field, Lx, Ly, Lz, plane, coord, zoom, title)` — Plot a 2-D electric-field or potential diagnostic slice.

- `run_sanity_check(phi_file, config_file, tip_mask_file, check_type, output_dir)` — Run sanity checks: Poisson residual, curl E, Dirichlet/Neumann BCs.


### `simulation.memory`

- `MemoryTracker(interval)` — Background process-RSS sampler that records peak memory during one simulation level.
- `track_memory(interval)` — Context manager that exposes the measured `peak_gb` value after the level finishes.
- `log_memory_usage(level_resolution, memory_gb, logfile, output_dir)` — Append peak process memory usage for one main or zoom level to `memory_usage_log.csv`.


## `simulation.coordinates`

### `normalize_config(cfg)`
Converts the canonical physical-coordinate configuration into the fractional
coordinate representation consumed internally by the solver.

### `nm_to_fraction(value_nm, axis, cfg)`
Converts an origin-relative physical coordinate in nanometres to a main-grid
fraction.

### `nm_range_to_fraction(range_nm, axis, cfg)`
Converts an origin-relative physical range in nanometres to a fractional range.

## `simulation.presimulation`

### `generate_tip_offset_configs(config_path, offsets_nm=None, overwrite=True)`
Generates one JSON configuration per requested tip-z offset and names each file
with the signed physical offset, such as `base_-5nm.json` or `base_+5nm.json`.

### `run_presimulation(config_path, offsets_nm=None, overwrite=True)`
Runs tip-offset JSON generation and reports the generated files.


## Physical coordinate scaling

`simulation.coordinates.voxel_nm(cfg)` returns the configured main-grid voxel
edge length in nanometres. `simulation.coordinates.physical_domain_nm(cfg)`
derives `(Lx, Ly, Lz)` from `grid_resolution` and `voxel_nm3`. The JSON config
therefore needs only `voxel_nm3` rather than separate axis-length entries.

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



## Output API

`simulation.io_utils.save_potential_full(phi, filename, output_dir)` writes a complete
3-D potential array.

`simulation.io_utils.save_potential_physical_cut(phi, center_nm, box_offsets_nm,
field_bounds_nm, filename, output_dir)` extracts and saves a physical-nm box. The
six offsets are relative to the movement centre and are ordered
`[xmin, xmax, ymin, ymax, zmin, zmax]`; the requested interval is clipped to the
represented field on each axis.

The batch workflow uses the JSON controls `save_cut`, `save_full`, and
`save_cut_box_nm`.
