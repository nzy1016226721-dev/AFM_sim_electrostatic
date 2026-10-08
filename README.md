# AFM electrostatic solver — shared-memory and MPI

A self-contained, source-only distribution of the maintained standalone
`afm_parallel` package: solver, distributed-grid modules, post-processing,
configs, Slurm launchers and small regressions. The October 2026 experimental
upgrade was cancelled; this publication introduces no new numerical solver
or experimental default.

## Start here

- [Configuration, numerical conventions and output contract](DOCUMENTATION.md)
- [Local/Alliance environment and launch instructions](README_ALLIANCE.md)
- [Distributed solver and dated parity evidence](README_MPI.md)
- [Configuration and job catalogue](docs/CONFIGS_AND_JOBS.md)
- [Module/API navigation](API_Docstrings.md)
- [MPI-compatible potential plotting](docs/MPI_PLOTTING.md)
- [October 7 source consolidation](docs/PACKAGE_CONSOLIDATION_20261007.md)
- [Cancelled upgrade status](UPGRADE_ABORT_20261006.md)

`simulation/` and `postprocessing/` are required beside `run_all.py` and
`run_mpi.py`. Do not substitute the old serial `afm_package` or old release
ZIP: those implementations differ.

## Local quick start

Run from this package root using a compatible Python environment:

```text
python -m pip install -r requirements.txt
python run_all.py --help
python run_all.py tests/data/afm_config_mpi_smoke.json --no-plot --output-dir outputs/local_smoke
```

The last command is a tiny shared-memory smoke, not the canonical large sweep.
`run_all.py` uses shared memory even when the fixture contains an MPI section.
Choose an explicit JSON: no-argument terminal discovery selects the newest
AFM JSON by modification time, potentially a large/diagnostic preset.

For distributed MPI, use [requirements-mpi.txt](requirements-mpi.txt) and a
compatible MPI runtime. Planning does not launch ranks or allocate the grid:

```text
python run_mpi.py tests/data/afm_config_mpi_smoke.json --plan --ranks 2 --ranks-per-node 1 --node-memory-gib 4
mpiexec -n 2 python run_mpi.py tests/data/afm_config_mpi_smoke.json --no-plot --output-dir outputs/mpi_smoke
```

The second command is a real numerical run. On Alliance, use the site modules
and scripts in [README_ALLIANCE.md](README_ALLIANCE.md), not an unrelated MPI
installation.

## Tests and evidence boundaries

```text
python -m pip install pytest
python -m pytest -q tests -k "not test_float64_diagnostic_runs_only_requested_iterations" --basetemp outputs/pytest_small
```

This selection follows the restored package's float32 testing policy. Retained
opt-in float64 memory-smoke settings/tests are historical diagnostics, not
production defaults. Choose a fresh writable `--basetemp` if an older directory
is protected or contains evidence to preserve.

Small tests/imports/planning are not proof of large-grid convergence, memory
safety or inter-node parity. Dated benchmark guides distinguish historical
results from prepared studies. Outputs, archives, caches, environments and
legacy ZIPs are excluded. AFM checks require no QTCAD software or licence.

## Layout and provenance

```text
run_all.py / run_mpi.py    shared-memory / distributed entry points
simulation/              geometry, materials, iteration, MPI and saving
postprocessing/          field analysis, NPY comparison and plotting
tests/                   small regressions and JSON fixtures
jobs/                    setup, preflight, runs and comparison scripts
docs/                    current catalogue and consolidation record
afm_*.json               explicit canonical/production/diagnostic presets
docs/SOURCE_MANIFEST.json SHA-256 source-payload inventory
```

Independent legacy/combined/hybrid packages are not bundled or synchronized.
Keep effective JSON, provenance and residual logs with fields transferred to
quantum workflows. Source manifest hashes are integrity checks, not numerical
certificates or an environment lockfile.
