# Local post-run potential plotting

The plotting-integration branch is adapted onto the maintained lossless RAM
solver. Use these tools **locally after MPI completion and download**, not on
compute nodes or inside an MPI launcher. Slurm/MPI environment checks apply to
the compatibility functions, new menu entries and existing local CLI. A local
headless session may select Matplotlib Agg; no backend or Qt/IPython state is
forced. Simulation/MPI entry points do not automatically import these tools.

## Coordinates and inputs

Retain each NPY with its matching `.npy.coords.json` receipt. Coordinates are
the actual original solver nodes, in nanometres; phi is a real floating-point
xyz array in volts. Selected lines/planes are detached copies, not full-volume
copies. The canonical reader is `local_visualization.data.PotentialData`.

- Receipts, including source indices/bounds when present, are validated.
- Optional `config_path` validates the recorded grid, origin, cut and movement.
  It never overrides receipt coordinates. Wrong but same-shaped JSONs fail.
- With no receipt, supply explicit first/last retained-node `bounds_nm`, or
  explicitly reconstruct a receipt using the completed run's exact config:
  `python local_post.py coordinates FIELD.npy --config EXACT.json --center X Y Z`.
  Reconstruction is a separate, deliberate write; ordinary plotting is read-only.
- Explicit bounds must agree with any existing receipt. Legacy cut-bin bounds,
  pixel edges and node bounds are different and must not be interchanged.
- Large full grids (any axis >=2048) remain rejected by the compatibility
  reader unless `allow_full=True`. Prefer a saved cut. This plotting safeguard
  does not change solver resource controls or MPI behavior.

## Local launchers and compatibility calls

`local_post.py` remains the primary CLI for listing, sanity checks, lines,
planes, arbitrary xyz profiles and drawn profiles. Examples:

```text
python local_post.py list --directory outputs/downloaded
python local_post.py sanity outputs/downloaded/FIELD.npy
python local_post.py planes outputs/downloaded/FIELD.npy --plane xy --at 20 --out outputs/figures
python local_post.py profile outputs/downloaded/FIELD.npy --start -10 0 20 --end 10 0 30 --out outputs/figures
python run_all.py post
```

In the post menu, option7 provides comparative lines or planes; option8 provides
a two-click profile on a displayed plane. File discovery prefers cuts over
matching full fields, uses natural ordering and accepts inclusive index ranges.
The JSON prompt is optional with valid receipts. The menu and local CLI preserve
existing PNGs and choose a collision-free suffix.

Historical function names remain available:

```python
import matplotlib.pyplot as plt
from postprocessing.bulk_potential_plotter import plot_potential_lines
from postprocessing.potential_plot_data import load_potential

cut = "outputs/downloaded/afm_phi_1_0nm_-1.00V_cut_from_grid512x512x512.npy"
with load_potential(cut) as volume:
    positions_nm, potential_V = volume.line("z", (0.0, 0.0))

fig = plot_potential_lines([cut], axis="z", fixed_nm=(0.0, 0.0))
try:
    plt.show()
finally:
    plt.close(fig)
```

Passing the matching JSON as the second plot argument remains supported.
`bounds_nm` and `center_nm` are optional explicit inputs. The old line/plane
coordinate order, transposed imshow plane order and positional config argument
are retained. Context-manage readers or call `close()`; selected data remain
valid after close. Reader methods reject use after close. Direct access to
`volume.array` is only valid while its context is open.

## Explicit compatibility changes

- A JSON alone no longer supplies implicit coordinates for a missing receipt.
- `PotentialData.bounds_nm` retains legacy crop-bin bounds when full receipt
  provenance is present; `node_bounds_nm` explicitly exposes retained-node
  bounds. Without source provenance, bounds are the explicit node bounds.
- `single_plane_plotter` still returns three values: figure, axes and now a
  plain metadata dictionary, not a live mapped field. Plotting functions return
  caller-owned figures with all input mappings already closed.
- The shared 2-D profile sampler accumulates float64 bilinear samples, capped
  at100000 points, from float32 input. The existing xyz profile remains
  trilinear. Neither changes stored phi, solver precision or residuals.
- Two-click handlers disconnect on figure close; selections outside retained
  node centres are handled without an uncaught callback error.

See the [integration/provenance record](PLOTTING_MERGE_20261009.md) and
[RAM defaults](../RAM_SAVING.md). Local sanity/profile outputs are not dielectric
residual certificates, large-MPI validation or QTCAD results.
