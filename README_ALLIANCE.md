# Environment and Alliance launch guide

For the standalone parallel package, not the old serial `afm_package`.
Start with [README.md](README.md), the
[numerical contract](DOCUMENTATION.md) and the
[configuration/job catalogue](docs/CONFIGS_AND_JOBS.md). The examples below
are not authorization to launch a large campaign. This publication did not
submit, cancel, change or monitor Fir jobs.

## Local Python

The local JSON default is now lossless `ram_compact`/`in_place`/`scalar`, with
solver-side plotting disabled. See [RAM policy, temporary-disk needs and local
post-run visualization](RAM_SAVING.md). Existing thread/rank requests and runtime
limits are preserved; recomputation can make the solve substantially slower.
This local promotion has not been deployed or numerically validated on Fir.

Use an isolated compatible Python environment with
[requirements.txt](requirements.txt): NumPy, SciPy, Matplotlib, Numba, psutil.
MPI additionally needs [requirements-mpi.txt](requirements-mpi.txt) and a
compatible native MPI runtime. AFM requires no QTCAD import or licence.

```text
python -m pip install -r requirements.txt
python run_all.py tests/data/afm_config_mpi_smoke.json --no-plot --output-dir outputs/local_smoke
python run_mpi.py tests/data/afm_config_mpi_smoke.json --plan --ranks 2 --ranks-per-node 1 --node-memory-gib 4
```

Select explicit JSONs rather than modification-time discovery. Imports/planning
are not numerical convergence or memory-residency evidence. Thread settings
are not proof of actual core placement.

## One-time Alliance setup

Copy the complete tree including `simulation`, `postprocessing`, `jobs` and
`tests/data`; retain `local_post.py`/`local_visualization` on the visualization
machine and download coordinate sidecars with NPYs. From its root on the login node:

```bash
bash jobs/setup_afm_env.sh
# For distributed MPI instead:
bash jobs/setup_afm_mpi_env.sh
```

The scripts load site `python`/`scipy-stack` (and `mpi4py` for MPI), create
or reuse `${AFM_VENV:-$HOME/afm_env}` with system site packages and install
Numba/psutil from the Alliance wheelhouse using `--no-index`. Override
`AFM_PYTHON_MODULE`, `AFM_SCIPY_MODULE`, `AFM_MPI4PY_MODULE`, `AFM_VENV`,
`AFM_WHEELHOUSE` only with explicit site-compatible choices. Review environment
destinations before setup. Compute-job scripts require an existing environment
and never install packages during the job. Keep mpi4py/launcher/site MPI matched.

```bash
bash jobs/preflight_afm.sh tests/data/afm_config_mpi_smoke.json
bash jobs/preflight_afm_mpi.sh tests/data/afm_config_mpi_smoke.json 2 1 4
```

Preflight checks imports/configuration and MPI memory estimates, not a completed
runtime test or scheduler availability.

## Shared-memory jobs

[jobs/run_afm.sh](jobs/run_afm.sh) accepts an explicit JSON and optional output
root and refuses execution outside Slurm. Match `cpu_threads` to requested
CPUs; presets have deliberately different resources.

```bash
mkdir -p outputs/slurm_logs
sbatch --cpus-per-task=1 --mem=4G --time=00:10:00 \
  --output=outputs/slurm_logs/%x-%j.out \
  --error=outputs/slurm_logs/%x-%j.err \
  jobs/run_afm.sh tests/data/afm_config_mpi_smoke.json
```

This overrides the generic script's larger default request for the one-thread
fixture. Select your own permitted account/partition. Historical site defaults
in the bundled scripts are not portable resource guarantees.

## Distributed jobs

`run_mpi.py` decomposes one global problem, not independent full-grid copies.
The process-grid product equals ranks; `cpu_threads` equals CPUs per rank.
See [README_MPI.md](README_MPI.md) for numerical/output constraints.

```bash
python run_mpi.py tests/data/afm_config_mpi_smoke.json \
  --plan --ranks 2 --ranks-per-node 1 --node-memory-gib 4
mkdir -p outputs/slurm_logs
sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=1 \
  --mem-per-cpu=4G --time=00:10:00 \
  --output=outputs/slurm_logs/%x-%j.out \
  --error=outputs/slurm_logs/%x-%j.err \
  jobs/run_afm_mpi.sh tests/data/afm_config_mpi_smoke.json
```

The launcher enforces rank/thread layout, its default 4-GiB-per-CPU policy,
and estimated/runtime per-node memory checks. A planner pass is not a RAM
reservation. Tiny smoke success does not establish 16384³ feasibility.

## Paths and acceptance

Submit from the package root or set `AFM_PACKAGE_ROOT`; scripts do not treat
Slurm's spool directory as source. Bundled scheduler defaults historically
reference `/home/nizy/afm_parallel`. Override `--output`/`--error` for other
installations and **create those directories before submission**, because
Slurm opens logs before the shell starts.

The launch scripts/runtime use `AFM_OUTPUT_ROOT` / `AFM_JOB_OUTPUT_ROOT`
for package-local science paths, normally
`outputs/job_<id>/<config-stem>/`. Scheduler logs live separately under
`outputs/slurm_logs/`. Do not default to scratch or overwrite evidence to
clean the source tree. Jobs are headless/noninteractive; BLAS/OpenMP ceilings
limit nested teams while Numba uses the configured threads.

Acceptance needs a terminal successful job, inspected stderr, final residuals,
timings and a loadable field. Imports, `sbatch --test-only` and estimates only
validate preparation. Parity also needs compatible converged physical cases.
Dated job IDs, timings, upload status and headroom in retained study notes are
history, not live queue/resource measurements. No large/cluster study was
rerun for this source publication.
