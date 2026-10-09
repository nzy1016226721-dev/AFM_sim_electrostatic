# RAM-saving default: standalone AFM

As of October 9, 2026, both JSON launchers default to the tested lossless RAM
path. This is a storage upgrade, not the cancelled V-cycle/certified-refinement
upgrade. Combined/hybrid AFM copies and quantum packages remain independent.

## Configuration

Normal shipped JSONs explicitly record:

```json
{
  "memory_mode": "ram_compact",
  "phi_update_mode": "in_place",
  "residual_accumulation": "scalar",
  "plotting": {"enabled": false}
}
```

Missing keys receive these defaults after `extends` is resolved. Physical dict
normalization and generated tip-offset JSONs use the same policy. CPU threads
remain first and dielectric blocks last. Existing geometry, voltage sweeps,
tolerances, CPU/rank requests, cuts and runtime controls are preserved.

| Option | Meaning |
| --- | --- |
| `memory_mode: ram_compact` | Lossless float32 epsilon-plane deduplication, packed/predicate masks, bounded construction and MPI saving |
| `phi_update_mode: in_place` | One full phi volume; per-worker old-plane snapshots preserve Jacobi reads |
| `residual_accumulation: scalar` | Native RMS and maximum reductions without a full residual or row array |
| `phi_update_mode: buffered` | Explicit higher-RAM alternative retaining the second phi volume |
| `residual_accumulation: row_array` / `array` | Explicit alternatives retaining row sums / a full residual volume |
| `memory_mode: ram_first` | Earlier dense cell-epsilon RAM path; defaults to in-place/row-array when selectors are omitted |
| `memory_mode: standard` | Original face-based implementation and legacy plotting/float64 diagnostic path |

To opt out, set `memory_mode` to `standard` and set both `phi_update_mode` and
`residual_accumulation` to `null`, especially in a child inheriting RAM settings.
The two explicitly named float64 memory-smoke JSONs already do this; they remain
diagnostic exceptions. RAM modes require float32. Low-level numerical function
defaults remain `standard` for API compatibility; configuration entry points
select the RAM default.

## Numerical and memory contract

Phi and epsilon stay float32. Deduplication compares exact plane bytes, not
rounded values or a quantized palette. Distinct coarse cell averages remain
distinct. The existing 512-cell reference/coarsening policy, fine direct raster,
block/gate/tip precedence, fixed/Neumann restoration order, snapshot Jacobi,
residual arithmetic/reduction order, rank ownership and halos are unchanged.
The mean convergence statistic remains the native **RMS**, not a signed mean
or a pointwise voltage-error certificate.

MPI reconstructs coefficients locally and streams bounded slabs for NPY/cut
saving. Potentials retain C-order `(x, y, z)` NPY, physical volts, movement/
voltage names, original-grid cut tags and coordinate sidecars.

The plane bank uses temporary file-backed storage: provide a writable temporary
directory and sufficient disk. Mapped pages still count toward RSS. An all-unique
material can compress poorly. Recomputing faces and indexing packed masks costs
substantial runtime. Existing production preflights, memory-test controls and
runtime limits are retained; the earlier unguarded trial was a one-case experiment,
not a change to maintained launcher safeguards.

## Local use and post-run visualization

From the package directory, using a compatible installed environment:

```text
python run_all.py configs/ram_compact_64_local.json --no-plot --output-dir outputs/check64
python run_all.py configs/ram_compact_512_local.json --no-plot --output-dir outputs/run512
python run_all.py configs/ram_compact_1024_local.json --no-plot --output-dir outputs/run1024
```

These examples retain the 256-nm domain with 4/0.5/0.25-nm voxel settings,
minus-one voltage, absolute native tolerance 1e-6, eight threads and a movement-
centred cut requesting z=0..100 nm at centre z=20 nm. They launch actual solves.
The large examples retain a 7200-second runtime control; they do not reproduce
the old until-finish-or-crash experiment. For MPI use matching software and an
appropriate process grid/thread count. Plan first: the conservative all-unique
epsilon estimate is not a measured peak or guaranteed admission. Large MPI/Fir
validation has not been performed for this upgrade.

Solver-side plotting stays off in RAM modes, including in an IDE. After MPI
has finished and output plus `.coords.json` sidecars are downloaded, use Eric's
local-only tools, separate from distributed computation:

```text
python local_post.py --help
python local_post.py list --directory outputs/downloaded
python local_post.py sanity outputs/downloaded/FIELD.npy
python local_post.py planes outputs/downloaded/FIELD.npy --plane xy --at 20 --electric-field --out outputs/figures
python local_post.py lines outputs/downloaded/FIELD.npy --axis z --first 0 --second 0 --out outputs/figures
```

These read saved fields; they do not alter outputs or launch MPI/QTCAD. Retain
actual coordinate sidecars when moving cuts. Nominal 100-nm requests snap/clip
to existing nodes and need not span exactly 100 nm.

The adapted plotting merge also provides local post-menu options7/8 and the
historical plotting function names, backed by the same reader/renderers.
Optional JSONs validate saved coordinates instead of replacing them; a missing
receipt needs explicit retained-node bounds. Selected data are copied before
mapping close and repeated figure saves preserve existing files. See
[local plotting contracts and compatibility changes](docs/MPI_PLOTTING.md).

## Evidence and recovery

The sealed physical 512/1024 trials matched every original float32 NPY byte
and complete residual history. At 1024, sampled private memory fell from
10.679 to 4.814 GiB (54.92%) and RSS from 10.117 to 4.953 GiB (51.04%), with about
3.20x end-to-end runtime. At 512 private/RSS reductions were 48.28%/26.36%; the
original 512 measurement is the matching prefix of the 1024 reference, not a
separately timed save. Small two-rank private memory increased 4.76% from
setup/JIT overhead, so this is not a universal MPI-memory-win claim.

See the self-contained [validation and provenance summary](docs/RAM_VALIDATION_20261009.md)
for measured peaks, native norms, exact field hashes, installation checks and
limitations. The development workspace also retains the full ordinary Markdown
reports `docs/AFM_RAM_DEFAULT_PROMOTION_20261009.md` and
`docs/AFM_LOSSLESS_MEMORY_20261008.md`; their numerical receipts and large NPYs
are intentionally not bundled in this source-only checkout.

In that workspace, the standalone before promotion is byte-backed-up at
`archive_old/afm_ram_default_20261009/standalone_before/`; `before.json` records
every hash. A separate publication backup retains the complete post-promotion
source and all GitHub branches before upload. Recover reviewed files into an
inactive checkout, preserving later work and numerical evidence. Source
publication does not deploy to Fir or synchronize combined/hybrid engines.
