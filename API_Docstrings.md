# Standalone AFM module/API map

Updated October 9, 2026. Source docstrings/implementations define signatures.
This map links the maintained modules instead of duplicating long argument
lists. See [README.md](README.md) and [numerical conventions](DOCUMENTATION.md).

| Module | Responsibility / useful symbols |
| --- | --- |
| [run_all.py](run_all.py) | Explicit shared-memory JSON launch and separate interactive menus |
| [run_mpi.py](run_mpi.py) | Distributed entry point and `--plan` estimates |
| [main_loop.py](simulation/main_loop.py) | `batch_main`, `run_afm_simulation`, `default_initial_grid_level`; cases/hierarchy |
| [coordinates.py](simulation/coordinates.py) | `normalize_config`, `physical_domain_nm`, `nm_to_fraction`, `ordered_config_for_json` |
| [config_defaults.py](simulation/config_defaults.py) | Lightweight effective JSON storage/plotting defaults, applied after inheritance |
| [runtime.py](simulation/runtime.py) | Config discovery, plotting and output-path resolution |
| [presimulation.py](simulation/presimulation.py) | Existing configuration-generation helpers |
| [materials.py](simulation/materials.py) | `generate_eps_level`, `build_eps_reference_memmap`, `average_reference_to_cells`, `release_eps_reference` |
| [solver.py](simulation/solver.py) | `mg_3d_masked`, `build_downward_pointing_tip`, scalar/vector residual routines |
| [parallel.py](simulation/parallel.py) | CPU-thread configuration; snapshot-preserving face-based Jacobi |
| [ram_first.py](simulation/ram_first.py) | In-place/buffered cell-based snapshot Jacobi and scalar/row/array residual selectors |
| [compact_storage.py](simulation/compact_storage.py) | Exact epsilon-plane bank and packed/predicate masks, bounded plane construction |
| [compact_solver.py](simulation/compact_solver.py) | Shared compact orchestration using unchanged indexed numerical kernels |
| [materials_bounded.py](simulation/materials_bounded.py) | Bounded exact coarse-reference averaging and fine local rasterization |
| [numerics.py](simulation/numerics.py) | Shared tip/dtype/tolerance/diagnostic policies |
| [io_utils.py](simulation/io_utils.py) | Movement suffixes, grid-tagged cut names, NPY and CSV saving |
| [output_coordinates.py](simulation/output_coordinates.py) | Actual-node coordinate sidecars for saved fields/cuts |
| [memory.py](simulation/memory.py) | `MemoryTracker` observations, not kernel allocation reservations |
| [plotting.py](simulation/plotting.py) | Shared-memory plotting |
| [zoom.py](simulation/zoom.py) | Restored legacy shared-memory zoom; not MPI zoom |
| [mpi_config.py](simulation/mpi_config.py) | `load_afm_config`, `load_mpi_config`; inheritance/validation |
| [mpi_domain.py](simulation/mpi_domain.py) | `DomainDecomposition`, `DistributedField`, `solve_distributed_level`; halos/kernels/plans |
| [mpi_main.py](simulation/mpi_main.py) | `batch_main_mpi`; coordinated physical cases and hierarchy |
| [mpi_io.py](simulation/mpi_io.py) | `save_distributed_npy`, `save_distributed_physical_cut` |
| [compare_mpi_npy.py](postprocessing/compare_mpi_npy.py) | `compare_npy_fields`; memory-mapped plane-wise checks |
| [npy_utils.py](postprocessing/npy_utils.py) | Saved-array inspection helpers |
| [field_calculator.py](postprocessing/field_calculator.py) | Electric-field analysis |
| [sanity_check.py](postprocessing/sanity_check.py) | Saved-field checks |
| [plot_npy.py](postprocessing/plot_npy.py) | Potential slices |
| [field_lines.py](postprocessing/field_lines.py) | Interactive field-line visualization |
| [potential_map.py](postprocessing/potential_map.py) | Potential-map visualization |
| [lever_arm_calc.py](postprocessing/lever_arm_calc.py) | QD lever-arm postprocessing |
| [capacitance_sanity_check.py](postprocessing/capacitance_sanity_check.py) | Capacitance checks |
| [local_post.py](local_post.py) | Eric's separate local-only downloaded-field visualization/sanity CLI |

JSON/physical-config entry points now select the [RAM-saving defaults](RAM_SAVING.md).
Direct low-level numerical APIs retain `standard` defaults for compatibility;
pass explicit RAM mode/selectors when calling those APIs without normalization.

Small regressions cover range defaults, movement/cut naming, kernels/threads,
MPI decomposition, runtime policy, tip geometry and saved-field comparison.
They do not establish large-grid convergence. Standalone, legacy and
combined/hybrid numerical copies remain independent; importing an old package
to fill a missing module is unsupported.
