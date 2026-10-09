# Distributed-memory MPI AFM solver

Start with the [package guide](README.md) and
[configuration contract](DOCUMENTATION.md). This guide retains dated
benchmark/deployment history; job-state, upload and resource statements in
those sections are not live status. The October 9 local package now uses the
[lossless RAM default](RAM_SAVING.md). The October 7 publication and older
benchmarks describe the pre-promotion implementation; the October 9 release
and its validation are distinguished in the [RAM evidence summary](docs/RAM_VALIDATION_20261009.md).

`run_mpi.py` splits **one global electrostatic grid** over a three-dimensional
Cartesian MPI communicator. It is not a voltage-sweep task farm: every MPI
rank cooperates on the same voltage and movement position, exchanges one-node
halos with neighbouring subdomains, and participates in global residual
reductions. The numerical kernels remain hybrid MPI + Numba, with
`cpu_threads` threads inside each rank.

This is the package path intended to remove the single-node 4 TiB ceiling for
a 16384³ level. The original `run_all.py` and `jobs/run_afm.sh` remain the
serial/shared-memory path.

## What is distributed

- Potential, masks, epsilon and refinement fields are rank-local with halos;
  no rank creates a full 16384³ volume. The RAM default retains one phi plus
  bounded old-plane snapshots, packed/predicate masks and a lossless float32
  epsilon-plane bank, reconstructing faces on demand. `standard` retains the
  original face/buffer representation.
- Both paths reproduce the existing `epsilon_material.reference_resolution`
  policy (512-cell reference by default). RAM modes construct rank-local
  material planes with bounded exact reference-cell averaging below that
  resolution, then direct local rasterization for finer levels. `standard`
  retains its bounded rank-zero compatibility staging. Neither changes
  dielectric precision or the discretized physical problem.
- Residual sum/count/max values are reduced globally with MPI.
- Coarse-to-fine trilinear prolongation reads the local coarse block plus
  exchanged halos; it does not gather or write a temporary global field.
- `save_full=true` writes one valid C-order float32 `.npy`; RAM modes stream
  bounded rank slabs rather than packing another complete local block.
- RAM-mode physical cuts use bounded collective MPI-IO slabs, with no root
  volume gather. `standard` retains its root gather and the associated
  `mpi.max_cut_gather_gib` size check. Coordinate sidecars describe actual
  retained nodes. The 16384 template saves a small cut and disables
  the roughly 16 TiB full-field output.

The distributed path currently supports the main grid only. It rejects
`zoom_simulation.enabled=true` because the existing zoom implementation is a
full-array SciPy operation. Voltage and movement cases are executed
sequentially, with all ranks participating in each case.

`mpi.require_convergence` defaults to `true`. A time-limited unconverged level
stops the hierarchy before a finer grid or scientific output is written. Normal
solves no longer terminate by `max_iter`; fixed-iteration diagnostic runs are
explicitly separate and must not be treated as converged production results.

For targets with largest axis at least2048, both serial and MPI select a
dynamic start giving6–8 doubling levels:2048 starts64,4096 starts64,8192 starts64,
and16384 starts128. Smaller targets above512 start64; targets through512 start8.
An explicit serial `initial_grid_level` or `mpi.initial_grid_level` remains an
intentional diagnostic override. The older fixed512 policy is superseded.

## Numerical consistency gate

The promoted storage implementation passed shared 512/1024 original-byte and
complete-native-history replay and real two-rank small-grid parity. The installed
version is rechecked in the [installed-validation summary](docs/RAM_VALIDATION_20261009.md).
Large-grid MPI memory/performance and inter-node Fir results remain unvalidated;
small MPI private memory can increase from setup/JIT overhead. Preserve existing
resource/preflight controls and review runtime budgets for the slower RAM path.

Convergence of two independent jobs is necessary but is not, by itself,
evidence that their discretizations match. Auditing the distributed stencil
found four independent parity requirements: serial-compatible volume-averaged
dielectric coefficients, the same float32 coefficient/update order, the same
ten-iteration convergence cadence, and coarse-to-fine interpolation with the
same coordinate and eight-corner accumulation order as SciPy. Physical-endpoint
mapping was also changed so no zero-weight term can read an uninitialised outer
halo.

The following historical local validation was stricter than a tolerance check:

- strict one-thread serial, low-memory eight-thread serial, and 8-rank MPI
  produced a bitwise-identical 128³ production-geometry field;
- a no-zoom 800³ run compared all 512,000,000 float32 values and was also
  bitwise identical (`max_abs_diff=0`, `rms_abs_diff=0`);
- all levels in the 800³ hierarchy had identical iteration counts and global
  residuals, including 690 iterations and `9.957622957734e-7` at 800³;
- the Python regression suite then contained47 passing tests; the2026-09-18
  license-free rerun passed61 tests (see workspace VALIDATION.md).

The reproducible 800³ inputs and launcher are
`afm_config_nm_local_comparison_{serial,mpi}_800.json` and
`jobs/run_local_mpi_serial_comparison_800.ps1`; its evidence is under
`archive_old/outputs/local_mpi_comparison_800/`. These are local multi-process results, not
an inter-node Fir result. A newly uploaded large configuration must still pass
a representative Fir comparison before being described as cluster-validated.

## Historical cleanup and retained artifacts

The paths and job-state statements below describe the earlier development
workspace, not files included in this source-only Git checkout. The current
packaging boundary is recorded in
[the consolidation record](docs/PACKAGE_CONSOLIDATION_20261007.md).

Generated benchmark trees, historical Fir job outputs and scheduler logs,
one-off transfer bundles, status notes, upload helpers, and Python bytecode
are moved (not deleted) to `archive_old/`. The live 2048³ production result
under `outputs/job_57603857/`, the package source/configuration files, and the
current pending-job paths remain in place. `archive_old/` is excluded by
`.gitignore` and should not be included in a new deployment bundle.

## Alliance environment

Run once on the login node:

```bash
cd /home/nizy/afm_parallel
bash jobs/setup_afm_mpi_env.sh
bash jobs/preflight_afm_mpi.sh \
  afm_config_nm_mpi_16384_template.json 4096 16 768
```

The setup loads the Alliance `python`, `scipy-stack`, and `mpi4py` modules,
then reuses the required `${AFM_VENV:-$HOME/afm_env}` virtual environment.
`requirements-mpi.txt` records the portable Python dependency set; the module
is preferred on Alliance so mpi4py is built against the site MPI stack.

## 16384³ planning result

The packaged template inherits the established 4096 physical geometry, but is
deliberately a **single voltage at one position**. Widen the sweep only after
one case finishes and its residual/output are inspected.

With no explicit start override, its first distributed level is128³ under the
current dynamic policy. The sequence is128³,256³,512³,1024³,2048³,4096³,8192³,
16384³ (eight levels); the final-grid operator is unchanged.

The no-allocation planner currently reports:

```text
global grid                 16384 x 16384 x 16384
one float32 field           16.00 TiB
MPI ranks / process grid    4096 / 16 x 16 x 16
ranks per node              16
threads per rank            12
nodes / CPUs per node       256 / 192
memory request              4 GiB per CPU = 768 GiB per node
worst estimated node peak   465.88 GiB
sum of rank peak estimates  116.47 TiB
80% node-memory check       PASS (614.40 GiB allowed per node)
```

This is a conservative array-count estimate, **not a completed 16384³
benchmark or a guarantee that a 256-node/768-GiB-per-node request is available
on Fir or permitted by the allocation**. Re-run the planner for the exact
partition/node type. The job repeats the check against physical node RAM
before allocating the final layout.

## Submission

One generic `.sh` runs any MPI JSON. Resource flags remain explicit because
Slurm reads `#SBATCH` options before Python can read the JSON:

```bash
cd /home/nizy/afm_parallel
sbatch \
  --nodes=256 \
  --ntasks=4096 \
  --ntasks-per-node=16 \
  --cpus-per-task=12 \
  --mem-per-cpu=4G \
  jobs/run_afm_mpi.sh afm_config_nm_mpi_16384_template.json
```

The launcher enforces:

- `cpu_threads == SLURM_CPUS_PER_TASK`;
- exactly 4 GiB requested per CPU unless
  `AFM_ENFORCE_4G_PER_CPU=0` is explicitly exported;
- JSON process-grid product equals `SLURM_NTASKS`;
- planner and runtime per-node memory checks pass;
- an existing virtual environment and importable site mpi4py are present.

Output remains package-local by default:

```text
/home/nizy/afm_parallel/outputs/job_<SLURM_JOB_ID>/
```

Each case has rank-safe residual/timing/memory logs under `mpi_logs/`. Batch
stdout and stderr are stored separately from science data under:

```text
/home/nizy/afm_parallel/outputs/slurm_logs/<job-name>-<JOB_ID>.out
/home/nizy/afm_parallel/outputs/slurm_logs/<job-name>-<JOB_ID>.err
```

This sibling directory keeps scheduler logs out of both the package root and
the per-job configuration-result directory.

## Two-rank smoke test before production

The package ships a small MPI smoke configuration. A local MS-MPI validation
completed the 8³ -> 16³ hierarchy with two ranks, wrote and reloaded the MPI-IO
`.npy`, and converged both levels. The one-rank and two-rank distributed fields
were bit-for-bit identical (`max_abs_diff=0.0`); with the same material
discretization, the original serial solver also matched the two-rank field
bit-for-bit. That historical suite had47 passing tests;61 passed in the
2026-09-18 license-free rerun.

That evidence validates local rank decomposition, but it is not an inter-node
Fir result and it does not validate 16384³ runtime or memory. Run this
inexpensive two-node test before requesting the production allocation:

```bash
sbatch \
  --nodes=2 \
  --ntasks=2 \
  --ntasks-per-node=1 \
  --cpus-per-task=1 \
  --mem-per-cpu=4G \
  --time=00:10:00 \
  jobs/run_afm_mpi.sh tests/data/afm_config_mpi_smoke.json
```

Success requires a terminal `COMPLETED` Slurm state, empty batch stderr, a
`MPI configuration processed successfully` marker, convergence/timing/memory
CSVs, and a loadable 16³ full `.npy`. Static imports or `sbatch --test-only`
are setup checks, not MPI runtime proof.

## 2048³ MPI parity trial

`afm_config_nm_mpi_trial_2048.json` inherits the established 2048³ production
physics, material model, tolerance, and per-level time limit. It restricts the
work to the first physical position `(0, 0, 20) nm` and first voltage `-1 V`.
The companion `jobs/run_afm_mpi_trial_2048.sh` requests 64 ranks in a 4 x 4 x
4 grid over two nodes, with 32 ranks x 2 threads x 4 GiB per CPU on each node.
This is 64 CPUs and 256 GiB per node, passes the planner's 80% memory-safety
check, and preserves the Alliance 4 GiB-per-core charge ratio.
It writes only the same physical NPY cut as the production configuration, so a
completed serial production first-case cut can be compared without handling a
32-GiB full 2048³ field.

Movement-aware output uses physical displacement from the first configured
centre: `0nm` for that centre, `_0.001nm` for a positive offset, and
`-0.001nm` for a negative offset. The one-position parity trial therefore
writes the reference-centre token `0nm`.

```bash
cd /home/nizy/afm_parallel
sbatch --test-only jobs/run_afm_mpi_trial_2048.sh
sbatch jobs/run_afm_mpi_trial_2048.sh
```

After a terminal MPI job, compare its cut against the matching first serial
production cut. The comparator memory-maps arrays and uses one first-axis
plane at a time, so it is safe for both cuts and full fields:

```bash
python postprocessing/compare_mpi_npy.py \
  /path/to/serial_first_case_cut.npy \
  /home/nizy/afm_parallel/outputs/job_<MPI_JOB_ID>/afm_config_nm_mpi_trial_2048/afm_phi_1_0nm_-1.00V_cut_from_grid2048x2048x2048.npy \
  --atol 2e-5 --rtol 2e-5 \
  --report /home/nizy/afm_parallel/outputs/job_<MPI_JOB_ID>/mpi_serial_comparison.json
```

It returns exit code zero only when shape, dtype, finite values, and every
element satisfy `abs(MPI - serial) <= atol + rtol * abs(serial)`. The JSON
report records maximum, RMS, and maximum relative differences.

Historical Fir evidence from the intermediate correction: MPI job 57783896
completed a 2048³ solve on two nodes in 1:52, and comparison job 57783897
passed all 8,000,000 cut values at `atol=rtol=2e-5` (maximum absolute
difference 2.16961e-5 V; RMS 1.31789e-5 V). Both batch stderr files were empty.
That run predates the exact arithmetic/interpolation parity changes above and
must not be presented as validation of the current revision.

### Current-code one-hour 2048³ serial/MPI gate

The current exact-parity revision has a new cut-only gate with clearly
separated configurations:

- `afm_config_nm_serial_mpi_comparison_serial_2048.json`: one-node,
  96-thread shared-memory reference;
- `afm_config_nm_serial_mpi_comparison_mpi_2048.json`: 64 MPI ranks in a
  4x4x4 process grid, with two threads per rank over two nodes.

Both inherit the same physics, 64³ initial level, centre position, -1 V bias,
and 1e-5 diagnostic solver tolerance. The looser solver residual is confined
to this bounded parity test; numerical acceptance remains `atol=1e-6,
rtol=0`, and a separate zero-tolerance report records whether the cuts are
bitwise identical. Neither solver writes the 32 GiB full 2048³ field.

`jobs/submit_serial_mpi_comparison_2048.sh` performs imports, effective-JSON
validation, the MPI memory plan, and three `sbatch --test-only` checks before
submitting the two independent one-hour solver jobs. A small comparison job is
submitted with an `afterok` dependency on both. It independently rejects an
unconverged source field before reading the two cut NPYs.

```bash
cd /home/nizy/afm_parallel
bash jobs/submit_serial_mpi_comparison_2048.sh
```

The serial request is one node, 96 CPUs, and 384 GiB. The MPI request is two
nodes, 64 ranks x 2 CPUs, and 512 GiB total. Every request retains the 4 GiB
per CPU accounting ratio and writes scheduler stdout/stderr only under
`outputs/slurm_logs/`.

## Prepared 4096³ decomposition-parity trial

`jobs/run_afm_mpi_comparison_4096.sh` is included in the complete source
distribution. Its earlier preparation note did not establish a completed
runtime test. In one one-hour allocation it runs the same one-position,
one-voltage 4096³ case with 8x8x8 and 16x8x4 Cartesian decompositions, saves the
same 201³ physical cut from each, and fails unless the cuts are bitwise
identical. The deliberately different internal boundaries test halo exchange
and refinement independently of one particular partition.

The request is 16 nodes, 512 one-thread ranks, and 4 GiB per CPU: 512 CPUs and
2 TiB total, maintaining the 4 GiB/core charge ratio. Both layouts pass the
no-allocation planner at 128 GiB/node: worst estimated node peaks are 116.94
and 117.10 GiB. This is a plan, not completed 4096³ runtime evidence.
