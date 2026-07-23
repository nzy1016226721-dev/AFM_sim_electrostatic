# AFM Simulation Package — Full Documentation

## Package Structure

```
afm_package/
├── run_all.py                          # Launcher entry point
├── presimulation/                      # Config & material generation
│   ├── master_presim.py                # Presimulation orchestrator
│   ├── eps_z_gen.py                    # Epsilon depth profile
│   ├── generate_json.py                # Fractional config generator
│   ├── generate_sigma_json.py          # Conductivity block generator
│   ├── json_tips_gen.py                # Tip offset sweep configs
│   └── precompute_materials.py         # High-res NPZ + grid arrays
├── simulation/                         # Core AFM simulation
│   ├── main_loop.py                    # Batch simulation orchestrator
│   ├── solver.py                       # MG Poisson solver
│   ├── materials.py                    # Material cell generation
│   ├── io_utils.py                     # File I/O helpers
│   ├── joule.py                        # Joule heating computation
│   ├── plotting.py                     # Visualisation functions
│   └── zoom.py                         # Central-crop zoom simulation
├── postprocessing/                     # Analysis & validation
│   ├── master_post.py                  # Aggregated imports
│   ├── plot_npy.py                     # NPY plotting
│   ├── integrate_power.py              # Power density integration
│   ├── sanity_check.py                 # Physical consistency checks
│   ├── lever_arm_calc.py               # QD lever arm calculator
│   └── field_calculator.py             # Field diagnostics
└── outputs/                            # Simulation output files
```

---

## Execution Flow (run_all.py)

```
run_all.py main()
│
├── [1] Presimulation
│     └── master_presim.main()
│           ├── eps_z_gen.generate_eps_profile(epsilon_N.csv, As_z.csv → eps_z.csv)
│           ├── generate_json.generate_config(afm_config_nm.json → afm_config_nm_frac.json)
│           │     └── convert_blocks_nm_to_frac() + modify_block_from_csv()
│           │     └── preview_blocks() [interactive plot]
│           ├── generate_sigma_json.generate_sigma(sigma_csv → sigma_blocks.json)
│           │     └── normalize_blocks() + preview_conductivity() [interactive plot]
│           ├── json_tips_gen.generate_tip_sweep(afm_config_nm_frac.json → afm_config_N.json)
│           ├── precompute_materials.build_eps_highres(afm_config_nm_frac.json → eps_highres.npz)
│           ├── precompute_materials.build_sigma_highres(sigma_blocks.json → sigma_highres.npz)
│           └── precompute_materials.precompute_grid_arrays(→ eps_N.npy / sigma_N.npy)
│
├── [2] Simulation
│     └── main_loop.batch_main(config_base)
│           ├── main_loop.preview_tip_only(cfg) [plot]
│           ├── main_loop.preview_before_run(cfg) [plot]
│           └── for each config:
│                 ├── compute_block_positions() → centres list
│                 ├── precompute_relative_offsets_for_blocks()
│                 ├── precompute_relative_offsets_for_vgates()
│                 └── for each centre:
│                       ├── apply_relative_blocks()
│                       ├── apply_relative_vgates()
│                       └── for each V in sweep:
│                             ├── main_loop.run_afm_simulation()
│                             │     ├── solver.build_downward_pointing_tip() → tip_mask
│                             │     ├── io_utils.make_gate_mask() → boundary_mask
│                             │     ├── materials.generate_eps_cell()
│                             │     │     ├── _load_eps_cache() [eps_highres.npz]
│                             │     │     └── _generate_eps_cell_simple() [fallback]
│                             │     ├── solver.mg_3d_masked() → phi, residual
│                             │     │     ├── neumann()
│                             │     │     ├── solve_varying_dielectric_3d_zero_rhs() [or _old]
│                             │     │     ├── compute_residual_vec_unpadded()
│                             │     │     ├── log_residual_csv()
│                             │     │     └── plot_convergence()
│                             │     └── → results dict
│                             ├── io_utils.save_for_qtcad() → afm_phi_N_V.npy
│                             ├── plotting.plot_phi_plane()
│                             ├── plotting.plot_residual_plane()
│                             └── zoom.run_zoom_simulation()
│                                   ├── solver.build_downward_pointing_tip() [local]
│                                   ├── materials.generate_eps_cell() [use_precomputed=False]
│                                   ├── solver.mg_3d_masked()
│                                   ├── materials.generate_sigma_cell() [use_precomputed=False]
│                                   ├── joule.compute_joule_heating()
│                                   ├── joule.plot_scalar_plane()
│                                   └── → afm_phi_zoom_...npy + power_density_zoom_...npy
│
└── [3] Postprocessing
      ├── plot_npy.plot_afm_from_npy(phi.npy → figure)
      ├── integrate_power.interactive_main()
      │     ├── load_array()
      │     ├── compute_power()
      │     ├── energy_per_cycle()
      │     ├── slice_plot()
      │     └── line_plot()
      ├── sanity_check.run_sanity_check()
      │     ├── _generate_eps_cell_simple() / _generate_sigma_cell_simple()
      │     ├── field_calculator.compute_residual_Laplace()
      │     ├── field_calculator.check_curl_E()
      │     ├── field_calculator.check_dirichlet()
      │     ├── field_calculator.check_neumann()
      │     ├── field_calculator.compute_current_divergence()
      │     ├── field_calculator.integrate_power()
      │     ├── plot_field_slice()
      │     └── [interactive field plotting]
      └── lever_arm_calc.main()
            ├── parse_phi_filename()
            ├── find_qd_block()
            ├── extract_phi_qd()
            ├── compute_stats()
            └── plot_results() → lever_arm_vs_spacing.png
```

---

## Cross-Module Data Flow

### Presimulation — Files Produced

| File | Produced by | Consumed by |
|---|---|---|
| `eps_z.csv` | `eps_z_gen.generate_eps_profile()` | `generate_json.generate_config()`, `generate_sigma_json.generate_sigma()` |
| `afm_config_nm_frac.json` | `generate_json.generate_config()` | `json_tips_gen.generate_tip_sweep()`, `precompute_materials.build_eps_highres()`, `simulation/main_loop.py`, `postprocessing/sanity_check.py` |
| `sigma_blocks.json` | `generate_sigma_json.generate_sigma()` | `precompute_materials.build_sigma_highres()`, `simulation/main_loop.py`, `simulation/zoom.py` |
| `afm_config_N.json` | `json_tips_gen.generate_tip_sweep()` | `simulation/main_loop.batch_main()` |
| `eps_highres.npz` | `precompute_materials.build_eps_highres()` | `materials._load_eps_cache()` |
| `sigma_highres.npz` | `precompute_materials.build_sigma_highres()` | `materials._load_sigma_cache()` |
| `eps_N.npy` / `sigma_N.npy` | `precompute_materials.precompute_grid_arrays()` | (direct use / inspection) |

### Simulation — Files Produced

| File | Produced by | Consumed by |
|---|---|---|
| `afm_phi_N_{V}V.npy` | `io_utils.save_for_qtcad()` | `postprocessing/plot_npy.py`, `postprocessing/sanity_check.py`, `postprocessing/lever_arm_calc.py` |
| `afm_phi_zoom_{mag}x_{V}V_{idx}.npy` | `zoom.run_zoom_simulation()` | `postprocessing` modules |
| `power_density_zoom_{mag}x_{V}V_{idx}.npy` | `zoom.run_zoom_simulation()` → `compute_joule_heating()` | `postprocessing/integrate_power.py` |
| `residual_history.csv` | `solver.log_residual_csv()` | `solver.plot_convergence()` |
| `mg_timing_log.csv` | `main_loop.run_afm_simulation()` | (analysis) |
| `joule_power_zoom.csv` | `io_utils.log_joule_csv()` | `postprocessing/sanity_check.py` |

---

## Global Variables & Shared State

### `solver.py`
| Variable | Type | Purpose |
|---|---|---|
| `MG_TIME` | `dict` | `{"elapsed": float}` — tracks wall-clock time of the current MG solver call |
| `RESIDUAL_CSV` | `str` | Default `"residual_history.csv"` — filename for convergence log |

### `materials.py`
| Variable | Type | Purpose |
|---|---|---|
| `_eps_cache` | `dict` or `None` or `False` | Singleton cache of `eps_highres.npz` contents; `None`=not loaded, `False`=not found |
| `_sigma_cache` | `dict` or `None` or `False` | Singleton cache of `sigma_highres.npz` contents |

### `joule.py`
| Variable | Type | Purpose |
|---|---|---|
| `E_CHARGE_C` | `float` | Elementary charge (1.602e-19 C) |
| `AS_DOPED_SI_DEFAULT_N_CM3` | `float` | Default carrier density (7.0e20 cm^-3) |
| `AS_DOPED_SI_DEFAULT_MOBILITY_CM2_VS` | `float` | Default mobility (30.0 cm^2/V/s) |
| `JOULE_SUMMARY_HEADER` | `list[str]` | CSV column headers for Joule summary |

### `precompute_materials.py`
| Variable | Type | Purpose |
|---|---|---|
| `E_CHARGE_C` | `float` | Elementary charge (local copy for `resolve_sigma_value`) |
| `AS_DOPED_SI_DEFAULT_N_CM3` | `float` | Default carrier density (local copy) |
| `AS_DOPED_SI_DEFAULT_MOBILITY_CM2_VS` | `float` | Default mobility (local copy) |

---

## All Functions — Detailed Reference

### `run_all.py` — Launcher

**`show_menu()`**
Display the main AFM simulation package launcher menu.

**`run_presimulation()`**
Launch the presimulation pipeline (config generation, material profiles).

**`run_simulation()`**
Launch the simulation pipeline: batch main loop with movement + voltage sweep.

**`run_postprocessing()`**
Launch postprocessing menu: plotting, power integration, sanity checks, lever arm.

**`main()`**
Main entry point for the AFM Simulation Package. Supports command-line mode (presim/simulation/postprocessing) and interactive menu mode.

---

### `presimulation/master_presim.py` — Presimulation Orchestrator

**`confirm(message)`**
Prompt user for a yes/no confirmation.
- *message*: str — Question to display.
- Returns: bool — True if user enters 'y', False otherwise.

**`confirm_directory(description, dir_path)`**
Prompt user to confirm or correct a directory path. Loops until the user confirms the path; creates the directory if needed.
- *description*: str — Label describing the directory.
- *dir_path*: str — Proposed directory path.
- Returns: str — Confirmed directory path, or "" if skipped.

**`main()`**
Run the full presimulation pipeline interactively. Guides the user through each presimulation step: loading a source JSON, generating epsilon depth profile, fractional config, sigma blocks, tip sweep configs, high-res NPZ files, and grid arrays.

---

### `presimulation/eps_z_gen.py` — Epsilon Depth Profile

**`load_curves(data)`**
Split paired-column data into individual (x, y) curves.
- *data*: np.ndarray — 2D array where each pair of columns is an (x, y) curve.
- Returns: list of np.ndarray — Each a (N, 2) curve array.

**`generate_eps_profile(epsilon_N_csv, As_z_csv, output_csv)`**
Generate epsilon vs depth profile from material CSV data. Reads carrier concentration vs energy and doping vs depth, interpolates epsilon(z), and saves the result.
- *epsilon_N_csv*: str — Path to epsilon-N CSV (default: epsilon_N.csv).
- *As_z_csv*: str — Path to As-z doping CSV (default: As_z.csv).
- *output_csv*: str — Output path (default: eps_z.csv).

---

### `presimulation/generate_json.py` — Fractional Config Generator

**`confirm_path(description, current_path, must_exist)`**
Prompt user to confirm or correct a file path.
- Returns: str or None.

**`convert_blocks_nm_to_frac(blocks_nm, Lx, Ly, Lz)`**
Convert nanometre-scale block coordinates to fractional coordinates.
- Returns: list of dict.

**`modify_block_from_csv(blocks_frac, Lz, csv_path)`**
Replace a dielectric block's epsilon profile with CSV data.
- Returns: list of dict.

**`preview_blocks(blocks, Lx_nm, Ly_nm, Lz_nm)`**
Interactive 2D/1D preview of dielectric block distribution with XY/XZ/YZ slice plots and line-outs.

**`generate_config(source_json, dest_json, eps_csv, interactive)`**
Generate a fractional-coordinate AFM configuration JSON from an nm-scale source.
- *source_json*: str (default: afm_config_nm.json).
- *dest_json*: str (default: afm_config_nm_frac.json).
- *eps_csv*: str or None.
- *interactive*: bool (default: True).

---

### `presimulation/generate_sigma_json.py` — Conductivity Block Generator

**`confirm_path(description, current_path, must_exist)`**
Same pattern as generate_json.confirm_path.

**`get_float(prompt, positive)`**
Prompt for a float > 0.
- Returns: float.

**`get_float_pair(prompt, low_lim, high_lim)`**
Prompt for a range pair (two floats).
- Returns: [float, float].

**`normalize_blocks(blocks, Lx, Ly, Lz)`**
Convert block coordinates to fractional, handling mixed nm/frac inputs.
- Returns: list of dict.

**`preview_conductivity(blocks_frac, Lx_nm, Ly_nm, Lz_nm)`**
Interactive 2D/1D preview of conductivity block distribution.

**`generate_sigma(source_json, sigma_csv, output_json, interactive)`**
Generate conductivity (sigma) block JSON from CSV profile.
- Returns: None.

---

### `presimulation/json_tips_gen.py` — Tip Offset Sweep

**`confirm_path(...)`**
Same pattern.

**`generate_tip_sweep(template_json, offsets_nm, output_base, interactive)`**
Generate multiple config files with varying tip z-offsets.
- *template_json*: str (default: afm_config_nm_frac.json).
- *offsets_nm*: list of float.
- *output_base*: str (default: "afm_config").
- *interactive*: bool.

---

### `presimulation/precompute_materials.py` — High-Res Material Arrays

**`build_eps_highres(config_json, output_npz)`**
Build high-resolution epsilon z-profiles from a fractional config.
- *config_json*: str (default: afm_config_nm_frac.json).
- *output_npz*: str (default: eps_highres.npz).

**`resolve_sigma_value(block)`**
Resolve conductivity from a block dict, computing sigma = e * n * mu from doping if needed.
- Returns: float (S/m).

**`build_sigma_highres(sigma_blocks_json, output_npz)`**
Build high-resolution conductivity z-profiles from sigma_blocks JSON.
- Returns: None.

**`precompute_grid_arrays(config_or_blocks, output_dir, kind, max_grid)`**
Precompute downsampled 3D material arrays for multiple power-of-two grid sizes. Saves as `eps_N.npy` or `sigma_N.npy`.
- Returns: None.

**`main()`**
CLI entry point for material precomputation steps.

---

### `simulation/solver.py` — Poisson Solver

**`build_downward_pointing_tip(nx, ny, nz, tip_z, R, r_tip, aspect_ratio, verbose)`**
Generate a boolean mask for a hyperbolic AFM tip geometry.
- Returns: (mask, z_tip, z_base).

**`compute_residual_vec_unpadded(V, mask, axp, axm, ayp, aym, azp, azm, a0)`**
Compute residual of the discretized Poisson equation (full matrix output).
- Returns: (res_mean, res_max, res_matrix).

**`compute_residual_scalars(V, mask, axp, axm, ayp, aym, azp, azm, a0)`**
Compute scalar residual norms (no full matrix allocation).
- Returns: (res_L2, res_max).

**`log_residual_csv(iteration, res_avg, res_max, csv_file, output_dir)`**
Append one row to the residual convergence CSV log.

**`plot_convergence(csv_file, output_dir)`**
Plot the convergence history from a residual CSV log.

**`mg_3d_masked(Vtip, phi, boundary_mask, ...)`**
Multigrid solver for the 3D Poisson equation with dielectric variation. Contains nested functions:
- *neumann(a)* — Apply homogeneous Neumann BC.
- *solve_varying_dielectric_3d_zero_rhs_old(...)* — Legacy SOR solver.
- *solve_varying_dielectric_3d_zero_rhs(...)* — Optimised SOR solver.
- Returns: (phi, residual).

---

### `simulation/materials.py` — Material Cell Generation

**`_load_eps_cache()`**
Load and cache the high-res epsilon NPZ data. Returns dict or False.

**`_load_sigma_cache()`**
Load and cache the high-res sigma NPZ data. Returns dict or False.

**`invalidate_material_cache()`**
Clear both epsilon and sigma caches.

**`_generate_eps_cell_simple(phi, blocks)`**
Generate epsilon cell array directly from block definitions.
- Returns: np.ndarray (Nx-1, Ny-1, Nz-1).

**`_generate_sigma_cell_simple(phi, blocks)`**
Generate conductivity cell array directly from block definitions.
- Returns: np.ndarray.

**`generate_eps_cell(phi, blocks, use_precomputed)`**
Generate epsilon on cells, using precomputed high-res data or falling back to block-based generation.
- Returns: np.ndarray.

**`generate_sigma_cell(phi, blocks, use_precomputed)`**
Generate conductivity on cells, using precomputed high-res data or falling back to block-based generation.
- Returns: np.ndarray.

---

### `simulation/io_utils.py` — File I/O

**`save_for_qtcad(results, filename, output_dir)`**
Save potential array to .npy.

**`save_phi_3d(phi, results, tag, output_dir)`**
Save 3D phi plus fields as compressed NPZ.

**`log_residual_csv(iteration, res_avg, res_max, csv_file, output_dir)`**
Append row to residual CSV.

**`log_timing(level, nx, ny, nz, elapsed, logfile, output_dir)`**
Log elapsed time per MG level.

**`log_joule_csv(config_idx, V, P_total, is_zoom, csv_file, output_dir)`**
Log Joule power to CSV.

**`get_sorted_config_files(base_name, directory)`**
Get numerically sorted list of config files.
- Returns: list[str].

**`_clamp01(a)`**
Clamp to [0, 1].

**`_range_to_indices(lo, hi, N)`**
Convert fractional range to integer grid indices.
- Returns: (int, int).

**`make_gate_mask(nx, ny, nz, gate)`**
Create bool 3D mask for a gate region.
- Returns: np.ndarray (bool).

---

### `simulation/main_loop.py` — Simulation Orchestrator

**`run_afm_simulation(Vtip, nx, ny, nz, ...)`**
Run multiresolution AFM electrostatic simulation with grid refinement.
- Returns: dict of results.

**`move_voltage_gate(Vgate, gate_index, center, xrange, yrange, zrange, Vgate_val)`**
Reposition a voltage gate.
- Returns: list (modified Vgate).

**`move_dielectric_block(cfg, block_index, center, ...)`**
Reposition a dielectric block.
- Returns: dict (modified cfg).

**`compute_block_positions(start_center, end_center, spacing)`**
Interpolate centres between start and end at given spacing.
- Returns: list of tuple.

**`apply_block_motion(cfg, block_motion_list, center)`**
Apply block motions relative to a centre.
- Returns: dict.

**`apply_vgate_motion(Vgate_list, vgate_motion_list, center)`**
Apply gate motions relative to a centre.
- Returns: list.

**`_clip01(a, b)`**
Clamp pair to [0, 1], returning (lo, hi).

**`precompute_relative_offsets_for_blocks(blocks, indices, center0)`**
Compute centre-relative offsets for mobile blocks.
- Returns: dict.

**`precompute_relative_offsets_for_vgates(vgates, indices, center0)`**
Compute centre-relative offsets for mobile gates.
- Returns: dict.

**`apply_relative_blocks(blocks, rel, center)`**
Reposition blocks using precomputed relative offsets with edge-fix.
- Returns: list.

**`apply_relative_vgates(vgates, rel, center)`**
Reposition gates using precomputed relative offsets.
- Returns: list.

**`_parse_fix_entry(v)`**
Parse a fix entry: None/empty/":" → None.

**`_apply_edge_fix(edge_value, lo_fix, hi_fix)`**
Constrain edge value within fix bounds, clamped [0,1].

**`preview_before_run(cfg)`**
Interactive 2D preview of blocks and gates; prompts to proceed or abort.
- Returns: bool (or sys.exit).

**`preview_tip_only(cfg)`**
Display a 3-projection scatter plot of the AFM tip voxels.

**`batch_main(CONFIG_BASE_NAME)`**
Main loop: iterate over config files, run simulation + zoom for each. Handles movement, voltage sweeps, saving, plotting, and cleanup.

---

### `simulation/joule.py` — Joule Heating

**`conductivity_from_carrier_density(n_cm3, mobility_cm2_v_s)`**
Compute sigma = e * n * mu.
- Returns: float (S/m).

**`resolve_sigma_value(block, default_mobility_cm2_v_s)`**
Resolve conductivity from block, computing from doping if needed.
- Returns: float.

**`joule_time_settings(cfg)`**
Extract frequency, period, time mode, and averaging scale from config.
- Returns: (frequency_hz, period_s, time_mode, average_scale).

**`joule_energy_summary(P_instantaneous, cfg)`**
Compute average power and energy per cycle.
- Returns: (P_average, E_cycle, frequency_hz, time_mode).

**`append_joule_summary(csv_path, row)`**
Append row to Joule summary CSV.

**`compute_joule_heating(phi, sigma_cell, Lx_nm, Ly_nm, Lz_nm)`**
Compute Joule power density and total power from phi and sigma.
- Returns: (power_density, P_total, Jx, Jy, Jz, Ex_cell, Ey_cell, Ez_cell).

**`plot_scalar_plane(data3d, boundary_mask, plane, cmap, label, vmin, vmax)`**
Plot a 2D slice of a 3D scalar field.
- Returns: matplotlib.figure.Figure.

---

### `simulation/plotting.py` — Visualisation

**`plot_phi_plane(phi_matrix, boundary_mask, plane, cmap)`**
Plot 2D slice of electrostatic potential.
- Returns: Figure.

**`plot_residual_plane(residual_matrix, boundary_mask, plane, vmin, vmax)`**
Plot log-scale 2D slice of residual with boundary masked black.
- Returns: Figure.

**`plot_residual_line(residual_matrix, line)`**
Plot 1D line-out of residual magnitude.
- Returns: Figure.

**`visualize_afm_results(results, x_frac, y_frac, z_frac)`**
Comprehensive 6-panel visualisation of AFM results.
- Returns: Figure.

**`combine_phi_and_residual(phi, residual, boundary_mask, plane, title_tag)`**
Side-by-side figure of potential and residual slices.

---

### `simulation/zoom.py` — Zoom Simulation

**`run_zoom_simulation(cfg, results, V, config_idx, time_log, output_dir)`**
Run recursive central-crop zoom simulation. Crops the potential, upsamples, re-solves Poisson, computes Joule heating if sigma blocks are available, saves zoomed results.
- Returns: original results dict.

---

### `postprocessing/plot_npy.py` — NPY Plotting

**`plot_afm_from_npy(phi_file, ...)`**
Load and plot AFM potential (and optionally field) from .npy files. Displays 2D slice plus line-outs (or component slices).
- Returns: Figure or None.

---

### `postprocessing/integrate_power.py` — Power Integration

**`energy_per_cycle(power_w, frequency_hz, period_s)`**
Compute energy per cycle from power and frequency/period.
- Returns: float or NaN.

**`load_array(path)`**
Load .npy array and return (arr, shape).

**`compute_power(p_dens, Lx, Ly, Lz, ...)`**
Integrate power density over a sub-region.
- Returns: (P_total, indices).

**`slice_plot(p_dens, Lx, Ly, Lz, plane, coord, zoom, region_bounds)`**
Plot 2D slice of power density.
- Returns: Figure.

**`line_plot(p_dens, Lx, Ly, Lz, axis, coord1, coord2, zoom)`**
Plot 1D line-out of power density.
- Returns: Figure.

**`get_float(prompt, default)`**
Prompt for a float with optional default.

**`get_optional_float(prompt, default)`**
Prompt for a float; returns None if blank.

**`interactive_main()`**
Interactive console tool for loading, viewing, and integrating power density.

---

### `postprocessing/sanity_check.py` — Sanity Checks

**`_generate_eps_cell_simple(phi, blocks)`**
Local copy of epsilon cell generation.

**`_generate_sigma_cell_simple(phi, blocks)`**
Local copy of sigma cell generation.

**`plot_field_slice(field, Lx, Ly, Lz, plane, coord, zoom, title)`**
Plot 2D slice of a 3D field with optional zoom.

**`run_sanity_check(phi_file, config_file, ...)`**
Run comprehensive sanity checks: Poisson residual, curl(E), Dirichlet/Neumann BCs, div(J), Joule power.

---

### `postprocessing/lever_arm_calc.py` — QD Lever Arm

**`parse_phi_filename(fname)`**
Extract metadata from AFM phi filename.
- Returns: dict or None.

**`find_qd_block(nm_cfg, default_eps)`**
Find the quantum dot block in nm config.
- Returns: (int or None, dict or None).

**`extract_phi_qd(phi, qd_xr, qd_yr, qd_zr, Lx, Ly, Lz, zoom_bounds)`**
Extract potential values inside a QD region.
- Returns: np.ndarray (1D).

**`compute_stats(phi_values)`**
Compute mean and max-absolute value.
- Returns: (avg, max_val).

**`main()`**
QD lever arm calculation entry point. Parses args, loads config, iterates over phi files, computes alpha, saves CSV, plots.

**`plot_results(results, output_dir)`**
Plot lever arm vs tip spacing colour-coded by Vtip.
- Returns: None.

---

### `postprocessing/field_calculator.py` — Field Diagnostics

**`compute_residual_Laplace(phi, eps_cell, boundary_mask)`**
Compute Poisson residual: div(eps * grad(phi)).
- Returns: (L2, max_res, resid_full).

**`check_curl_E(phi, dx, dy, dz)`**
Check curl(E) ≈ 0 via central differences.
- Returns: (mx, my, mz, mag, curl_full).

**`check_dirichlet(phi, tip_voltage, backgate_voltage, tip_mask)`**
Check Dirichlet BC fidelity.
- Returns: (err_bot, tip_err, max_phi, min_phi).

**`check_neumann(phi, tip_mask)`**
Check Neumann BC fidelity on all six faces.
- Returns: (e0, e1, e2, e3, ez1).

**`cell_center_to_node(sigma_cell)`**
Interpolate cell-centred data to nodes via 8-point averaging.
- Returns: np.ndarray.

**`compute_current_divergence(phi, sigma_cell, dx, dy, dz)`**
Compute div(J) = div(sigma * E).
- Returns: (max_div, rms_div, full).

**`integrate_power(phi, sigma_cell, dx, dy, dz)`**
Compute total Joule power P = int(sigma * |E|^2 dV).
- Returns: float (W).

---

## Module Dependency Graph

```
run_all.py
  ├── presimulation/master_presim.py
  │     ├── eps_z_gen.py
  │     ├── generate_json.py
  │     ├── generate_sigma_json.py
  │     ├── json_tips_gen.py
  │     └── precompute_materials.py
  └── simulation/main_loop.py
        ├── solver.py
        ├── materials.py
        │     └── eps_highres.npz, sigma_highres.npz
        ├── io_utils.py
        ├── plotting.py
        └── zoom.py
              ├── solver.py
              ├── materials.py
              └── joule.py
                    └── plotting.py (via plot_scalar_plane)
```

Postprocessing modules are called independently (not chained):
```
run_all.run_postprocessing()
  ├── plot_npy.py
  ├── integrate_power.py
  ├── sanity_check.py
  │     └── field_calculator.py
  └── lever_arm_calc.py
```
