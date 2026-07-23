# API Reference — Extracted from `afm next.py`

_Generated: 2025-11-30 14:51:05.592995_

## build_downward_pointing_tip

**Signature:** `build_downward_pointing_tip(nx, ny, nz, tip_z=0.2, R=0.05, r_tip=0.15, aspect_ratio=2.0, verbose=True)`

**Docstring (existing):**

```text
Construct a downward-pointing AFM tip with a hyperbolic shape.
The aspect_ratio now correctly controls the asymptote angle (opening).
The tip points downward (toward low Z), with base at Z=1.

Parameters
----------
nx, ny, nz : int
    Grid size.
tip_z : float
    Tip position (fraction of Z in [0,1]).
R : float
    Radius of curvature at the vertex.
r_tip : float
    Maximum base radius.
aspect_ratio : float
    Controls the asymptote slope (tanθ = aspect_ratio).
verbose : bool
    Print geometry diagnostics.

Returns
-------
mask : ndarray(bool)
    Tip mask volume.
z_tip, z_base : float
    Tip and base coordinates.
```



## mg_3d_masked

**Signature:** `mg_3d_masked(Vtip, phi, boundary_mask, damping=0.8, nu1=2, nu2=2, max_iter=50, tol=1e-06, verbose=True, eps_r=None, eps=False, mg_max_runtime=None)`

**Docstring (existing):**

```text
Simplified 3D multigrid solver with masked boundary conditions

Parameters:
- phi: Initial solution and boundary conditions (3D array)
- boundary_mask: Boundary mask, True indicates fixed values (3D array)
- damping: SOR damping factor
- nu1, nu2: Pre- and post-smoothing iterations
- max_iter: Maximum number of iterations
- tol: Convergence tolerance
- verbose: Whether to print debug information

Returns:
- phi: Solution
- res_matrix
```



## run_afm_simulation

**Signature:** `run_afm_simulation(Vtip=5, nx=32, ny=32, nz=32, tip_z=0.2, R=0.05, r_tip=0.15, damping=0.8, nu1=2, nu2=2, max_iter=1000, tol=0.0001, aspect_ratio=2.0, verbose=True, eps_r=None, eps=True, mg_max_runtime=None, blocks=None, Vgate=None)`

**Docstring (existing):**

```text
Multiresolution AFM simulation using convolution solver.
Starts from 8x8x8, doubles each level with np.kron() until reaching nx,ny,nz.
```



## generate_eps_cell

**Signature:** `generate_eps_cell(phi, blocks=None)`

**Docstring (existing):**

```text
Generate 3D eps_cell array (Nx-1,Ny-1,Nz-1) based on fractional block definitions.

Parameters
----------
phi : ndarray (Nx,Ny,Nz)
    3D voltage grid. Shape is used to determine domain size.
blocks : dict or list of dict, optional
    Dielectric blocks to insert. Each dict must contain:
        'eps_val' : float
        'x_range' : (float,float) in [0,1], fraction of Nx
        'y_range' : (float,float) in [0,1], fraction of Ny
        'z_range' : (float,float) in [0,1], fraction of Nz
    Example of a single block:
        {"eps_val": 4.0,
         "x_range": (0.25, 0.5),
         "y_range": (0.25, 0.5),
         "z_range": (0.25, 0.5)}
    You may also pass a list of such dicts.

Returns
-------
eps_cell : ndarray, shape (Nx-1,Ny-1,Nz-1)
    Dielectric distribution, default 1.0 everywhere.
```



## generate_v_gate

**Signature:** `generate_v_gate(nx, ny, nz, Vgate_list, verbose=True)`

**Docstring (existing):**

```text
Generate boolean mask(s) and voltage array for conductive Vgate regions.

Parameters
----------
nx, ny, nz : int
    Grid dimensions.
Vgate_list : list of dict
    Each dict should contain:
        {
          "Vgate_val": float,
          "x_range": [x1_frac, x2_frac],
          "y_range": [y1_frac, y2_frac],
          "z_range": [z1_frac, z2_frac]
        }
verbose : bool
    Print gate geometry diagnostics.

Returns
-------
Vgate_mask : ndarray(bool)
    True where the gate conductor is located.
Vgate_val : float
    Gate voltage values at masked positions.
```



## visualize_afm_results

**Signature:** `visualize_afm_results(results, x_frac=0.5, y_frac=0.5, z_frac=0.3)`

**Docstring (existing):**

```text
Visualize the correct AFM simulation results (downward pointing tip)
with sampling position directly specified by fractional coordinates.

Parameters
----------
results : dict
    AFM simulation result dictionary.
x_frac, y_frac, z_frac : float
    Fractional positions (0–1) for sampling and slicing.
```



## plot_residual_line

**Signature:** `plot_residual_line(residual_matrix, line)`

**Docstring (existing):**

```text
Plot residual magnitude along a line through the 3D box.

Parameters
----------
residual_matrix : ndarray
    3D residual array (e.g., from AFM solver).
line : tuple
    (x_frac, y_frac, z_frac), where exactly one coordinate = None.
    Example: (0.5, 0.5, None) → residual vs z at x=y=0.5.
```



## plot_residual_plane

**Signature:** `plot_residual_plane(residual_matrix, boundary_mask=None, plane=(True, True, 0.5))`

**Docstring (existing):**

```text
Plot a residual heatmap on a specified 2D plane.

Parameters
----------
residual_matrix : ndarray
    3D residual magnitude array.
boundary_mask : ndarray, optional
    Boolean mask of fixed (Dirichlet) nodes. These are shown in black.
plane : tuple
    (x, y, z) format, where two values are True and one is a float coordinate.
    Example: (True, True, 0.5) → XY plane at z=0.5
             (True, 0.4, True) → XZ plane at y=0.4
             (0.5, True, True) → YZ plane at x=0.5
```



## plot_phi_plane

**Signature:** `plot_phi_plane(phi_matrix, boundary_mask=None, plane=(True, True, 0.5), cmap='RdBu_r')`

**Docstring (existing):**

```text
Plot a potential (φ) heatmap on a specified 2D plane, showing boundary mask in black.

Parameters
----------
phi_matrix : ndarray
    3D potential distribution (φ).
boundary_mask : ndarray, optional
    Boolean mask of fixed (Dirichlet) nodes. Displayed as black.
plane : tuple
    (x, y, z) format, where two values are True and one is a float coordinate.
    Example: (True, True, 0.5) → XY plane at z=0.5
             (True, 0.4, True) → XZ plane at y=0.4
             (0.5, True, True) → YZ plane at x=0.5
cmap : str
    Colormap for potential visualization (default 'RdBu_r').
```



## move_voltage_gate

**Signature:** `move_voltage_gate(Vgate, gate_index, center, xrange, yrange, zrange, Vgate_val=None)`

**Docstring (suggested template):**

```python
"""
move_voltage_gate

Parameters
----------
Vgate : type
    Description.
gate_index : type
    Description.
center : type
    Description.
xrange : type
    Description.
yrange : type
    Description.
zrange : type
    Description.
Vgate_val : type
    Description.

Returns
-------
type
    Description.

Notes
-----
Add method-specific notes here.

Examples
--------
>>> # example usage
>>> move_voltage_gate(Vgate, gate_index, center, xrange, yrange, zrange, Vgate_val=None)
"""
```



## move_dielectric_block

**Signature:** `move_dielectric_block(cfg, block_index, center, xrange, yrange, zrange, eps_val=None)`

**Docstring (suggested template):**

```python
"""
move_dielectric_block

Parameters
----------
cfg : type
    Description.
block_index : type
    Description.
center : type
    Description.
xrange : type
    Description.
yrange : type
    Description.
zrange : type
    Description.
eps_val : type
    Description.

Returns
-------
type
    Description.

Notes
-----
Add method-specific notes here.

Examples
--------
>>> # example usage
>>> move_dielectric_block(cfg, block_index, center, xrange, yrange, zrange, eps_val=None)
"""
```



## compute_block_positions

**Signature:** `compute_block_positions(start_center, end_center, spacing)`

**Docstring (suggested template):**

```python
"""
compute_block_positions

Parameters
----------
start_center : type
    Description.
end_center : type
    Description.
spacing : type
    Description.

Returns
-------
type
    Description.

Notes
-----
Add method-specific notes here.

Examples
--------
>>> # example usage
>>> compute_block_positions(start_center, end_center, spacing)
"""
```



## apply_block_motion

**Signature:** `apply_block_motion(cfg, block_motion_list, center)`

**Docstring (existing):**

```text
Move arbitrary number of dielectric blocks according to the motion list.

block_motion_list format:
[
    {"index": int,
     "extent": [xneg, xpos, yneg, ypos, zneg, zpos],
     "eps_val": optional float}
]
```



## apply_vgate_motion

**Signature:** `apply_vgate_motion(Vgate_list, vgate_motion_list, center)`

**Docstring (existing):**

```text
Move arbitrary number of Vgate blocks.
vgate_motion_list format:
[
    {"index": int,
     "extent": [xneg, xpos, yneg, ypos, zneg, zpos],
     "Vgate_val": optional float}
]
```



## _clamp01

**Signature:** `_clamp01(a)`

**Docstring (suggested template):**

```python
"""
_clamp01

Parameters
----------
a : type
    Description.

Returns
-------
type
    Description.

Notes
-----
Add method-specific notes here.

Examples
--------
>>> # example usage
>>> _clamp01(a)
"""
```



## _range_to_indices

**Signature:** `_range_to_indices(lo, hi, N)`

**Docstring (existing):**

```text
Convert fractional [lo, hi] in [0,1] to integer inclusive indices for 0..N-1.
Assumes lo<=hi after clamping. Returns (i0, i1) inclusive.
```



## make_gate_mask

**Signature:** `make_gate_mask(nx, ny, nz, gate)`

**Docstring (existing):**

```text
Build a boolean mask for one gate from fractional ranges in gate dict.
gate = {"Vgate_val": float, "x_range":[x0,x1], "y_range":[y0,y1], "z_range":[z0,z1]}
```



## _clip01

**Signature:** `_clip01(a, b)`

**Docstring (existing):**

```text
Return (low, high) clipped to [0,1] and ordered.
```



## precompute_relative_offsets_for_blocks

**Signature:** `precompute_relative_offsets_for_blocks(blocks, indices, center0)`

**Docstring (existing):**

```text
From the current JSON-defined ranges, compute per-block offsets
relative to the initial reference center (center0).
Returns: dict { idx: {"x":[dx1,dx2], "y":[dy1,dy2], "z":[dz1,dz2], "eps":float} }
```



## precompute_relative_offsets_for_vgates

**Signature:** `precompute_relative_offsets_for_vgates(vgates, indices, center0)`

**Docstring (existing):**

```text
Same idea for voltage gates. Returns:
dict { idx: {"x":[dx1,dx2], "y":[dy1,dy2], "z":[dz1,dz2], "V":float} }
```



## apply_relative_blocks1

**Signature:** `apply_relative_blocks1(blocks, rel, center)`

**Docstring (existing):**

```text
Move any number of dielectric blocks using precomputed relative offsets.
Mutates the 'blocks' list in-place and returns it.
```



## apply_relative_vgates1

**Signature:** `apply_relative_vgates1(vgates, rel, center)`

**Docstring (existing):**

```text
Move any number of Vgate regions using precomputed relative offsets.
Mutates the 'vgates' list in-place and returns it.
```



## parse_fix_entry

**Signature:** `parse_fix_entry(v)`

**Docstring (existing):**

```text
Parse one of the four fix entries.
Accepts '', ':', None → means no constraint.
Otherwise must be a float value.
```



## apply_edge_fix

**Signature:** `apply_edge_fix(edge_value, lo_fix, hi_fix)`

**Docstring (existing):**

```text
Apply [lo_fix, hi_fix] constraint to a single edge.
If lo_fix exists → edge_value = max(edge_value, lo_fix)
If hi_fix exists → edge_value = min(edge_value, hi_fix)
Returns clipped to [0,1].
```



## apply_relative_blocks

**Signature:** `apply_relative_blocks(blocks, rel, center)`

**Docstring (suggested template):**

```python
"""
apply_relative_blocks

Parameters
----------
blocks : type
    Description.
rel : type
    Description.
center : type
    Description.

Returns
-------
type
    Description.

Notes
-----
Add method-specific notes here.

Examples
--------
>>> # example usage
>>> apply_relative_blocks(blocks, rel, center)
"""
```



## apply_relative_vgates

**Signature:** `apply_relative_vgates(vgates, rel, center)`

**Docstring (suggested template):**

```python
"""
apply_relative_vgates

Parameters
----------
vgates : type
    Description.
rel : type
    Description.
center : type
    Description.

Returns
-------
type
    Description.

Notes
-----
Add method-specific notes here.

Examples
--------
>>> # example usage
>>> apply_relative_vgates(vgates, rel, center)
"""
```



## preview_before_run

**Signature:** `preview_before_run(cfg)`

**Docstring (existing):**

```text
Preview dielectric blocks + gates using max-intensity projection (MIP)
on XY/XZ/YZ, with TWO-RANGE COLOR MAPPING:

   Range 1:  ε < 25    → viridis
   Range 2:  ε ≥ 25    → inferno_r (REVERSED)

Shows:
   • Left-side colorbar: 0–25 (viridis)
   • Right-side colorbar: 25–ε_max (inferno_r)
Gate-overlay plots have **no** colorbars.
```



## preview_tip_only

**Signature:** `preview_tip_only(cfg)`

**Docstring (existing):**

```text
Tip-only preview using the REAL geometry (build_downward_pointing_tip).
Shows three projections (all as orange scatter, fractional coords):
    • XY:  (x, y) of voxels with z >= tip_z
    • XZ:  (x, z) of voxels with z >= tip_z
    • YZ:  (y, z) of voxels with z >= tip_z
```



## combine_phi_and_residual

**Signature:** `combine_phi_and_residual(phi, residual, boundary_mask, plane, title_tag='')`

**Docstring (existing):**

```text
Wrap existing plot_phi_plane and plot_residual_plane into a single figure
WITHOUT modifying the original functions.
```



## save_phi_3d

**Signature:** `save_phi_3d(phi, results, tag='')`

**Docstring (existing):**

```text
Saves the full 3D phi array and metadata in a .npz archive.
Use `tag` to uniquely identify sweep conditions (cx, cy, cz, Vtip, etc).
```


