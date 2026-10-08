# MPI-compatible potential plotting

These plotting utilities port the useful line/plane and interactive profile ideas
from the QoL branch to the maintained standalone MPI package. They are
postprocessing tools only: there are **no changes** to the solver, MPI ranks,
materials, convergence logic, geometry generation, Slurm launchers, or zoom mode.

## Input requirements

- Use a saved `.npy` potential output from `run_mpi.py` or `run_all.py`.
- Supply the **matching JSON configuration**, including any `extends`
  inheritance chain. Physical coordinates are origin-relative nanometres.
- On large grids, configure `save_cut: true`, `save_full: false` and
  `save_cut_box_nm` before solving. The plotting code does not generate cuts
  retrospectively, request fields from MPI ranks, or allocate a global field.
- Files are opened using `np.load(..., mmap_mode="r")`. By default the reader
  rejects full grids with any axis at least 2048; cuts are accepted. A full
  grid can be deliberately overridden by Python callers with `allow_full=True`,
  but the interactive menu does not bypass the safeguard.
- Saved-cut locations depend on the movement centre and cut definition. Output
  filenames must match a unique movement point in the supplied configuration;
  mismatched cut shapes/configurations are rejected.
- Potential samples use the original solver-node physical coordinates.
  `save_cut_box_nm` describes voxel-bin selection; it is not the exact
  coordinate range of the selected solver nodes.

## Graphical postprocessing

From the package directory, run `python run_all.py post` in a graphical
environment and choose:

- **7:** comparative line plots, or XY/XZ/YZ potential slices for selected
  files. The file list automatically prefers cuts over a matching full field.
- **8:** interactive two-click potential profile along a displayed 2D slice.

The interactive line tool uses the existing Matplotlib backend and requires
a graphical environment; it does not force Qt, IPython, or a display on HPC
compute nodes.

## Programmatic plotting (also works headlessly)

```python
from postprocessing.bulk_potential_plotter import (
    plot_potential_lines, plot_potential_planes
)

config = "afm_config_nm_mpi_trial_2048.json"
cut = "outputs/job_123/config_name/afm_phi_1_0nm_-1.00V_cut_from_grid2048x2048x2048.npy"

fig = plot_potential_lines([cut], config, axis="z", fixed_nm=(0.0, 0.0))
fig.savefig("potential_line.png")
figs = plot_potential_planes([cut], config, plane="xy", at_nm=20.0)
figs[0].savefig("potential_xy.png")
```

In a headless job select Matplotlib's `Agg` backend **before** importing
plotting modules. Each line/plane must lie within the saved cut. The default
multi-file call expects all files to use the same effective config; use
separate calls when configurations differ.

## Validation

The added `tests/test_mpi_plotting.py` covers physical-cut geometry,
noncubic dimensions, movement offsets, filename interpretation, array
memory mapping, line/plane extraction, interactive event sampling and
large-full-field rejection. Run:

```text
python -m pytest -q tests/test_mpi_plotting.py
python -m pytest -q tests -k "not test_float64_diagnostic_runs_only_requested_iterations"
```

This validation does not replace existing serial/MPI numerical parity gates
or demonstrate interactive GUI operation on an HPC compute node.
