# Source consolidation — October 7, 2026

## Scope and preserved implementation

The user requested local consolidation/removal of stale duplicates followed
by upload to GitHub branch `afm-parallelized-mpi`. Its original head was
`6c54cf6a2556f2bd5f03436165c07a3fca16b2bc`. That 94-file tree contained the
parallel entry points/configs but omitted `simulation/`, `postprocessing/`,
`jobs/` and `tests/`. Its embedded legacy serial package and v15 ZIP cannot
substitute for those missing maintained modules.

The local modules were already complete; they are consolidated into one
self-contained source payload, not recovered by mixing old/new solvers.
All retained numerical modules, root JSON configurations, regression tests,
requirements and executable scripts remain byte-identical to the restored
local implementation. No convergence setting, voltage, array precision,
material discretization, MPI/halo behaviour or zoom policy changes here.
The [October upgrade cancellation](../UPGRADE_ABORT_20261006.md) still stands.

## Cleanup and documentation

Four active local files were retired into a checksum-verified local archive:

| Retired path | Reason |
| --- | --- |
| `compare_old_new.py` | Superseded serial-reference harness; `max_iter` no longer bounds normal solves |
| `run_comparison_test.py` | Old zoom/−1/−9 campaign with a retired default JSON and hard-coded workspace paths |
| `CONSOLIDATION.md` | August notice incorrectly claimed only one JSON/job wrapper |
| `jobs/UPLOAD_MANIFEST_RELATIVE_F64_20260909.txt` | Historical manual upload list, not a current release inventory |

Old embedded `afm_package/`, its 20 bytecode/cache files and the v15 ZIP are
removed from the active Git distribution, not copied into the maintained
package. Their exact original Git bytes and the full 125-file local starting
inventory are preserved locally; the old Git commit also remains in history.
The independent legacy package in the development workspace is untouched.

`README.md` is the package-local entry point. `DOCUMENTATION.md`,
`README_ALLIANCE.md` and `API_Docstrings.md` now focus on distinct contracts
rather than repeating stale installation/API prose. `README_MPI.md` keeps
dated numerical evidence while clarifying current packaging versus historical
upload/job states. The [catalogue](CONFIGS_AND_JOBS.md) makes the 27 root JSONs
and 34 shell scripts discoverable. Distinct study variants are retained, not
deduplicated by filename resemblance.

`.gitattributes` pins LF text checkout endings, with one explicit exception for
the original CRLF `simulation/plotting.py` so numerical source stays byte-for-byte
unchanged. Source hashes must match the local payload. `.gitignore` excludes generated results, caches, archives,
environments and old ZIP/embedded-serial payloads. `docs/SOURCE_MANIFEST.json`
lists published files and SHA-256 hashes, excluding itself to avoid recursive
hashing. It is an integrity inventory, not a numerical certificate or dependency
version lock. Shell scripts retain executable Git modes.

An initial validation caught the root-level manifest being selected by a
regression that deliberately loads every root JSON as a solver config.
The manifest was moved into `docs/`; solver, presets and tests were not
changed to accommodate it. The failed check and resource report are preserved
alongside the subsequent validation rather than overwritten.

## Local preservation and validation records

In the development workspace, the recoverable archive is
`D:\afm_pack_v1\afm_parallel\archive_old\source_consolidation_20261007\`.
`local_before/` contains the 125 original active files and `github_before/`
the 94 original remote files. `retired_active/` contains the four moved files;
the before/after receipts retain source fingerprints and the dirty main
checkout roster. The archive and numerical outputs are deliberately not
Git dependencies or part of the uploaded payload.

Fresh publication checks are recorded under
`D:\afm_pack_v1\afm_parallel\outputs\validation\source_consolidation_20261007\`.
The workspace publication record documents the final test outcomes, push
receipt and verified commit. Validation covers syntax, inheritance, local
links, shell parsing, import/CLI planning and the selected small float32
regression; it does not claim a fresh large-grid or inter-node MPI campaign.

The corrected isolated checkout passed **60 tests, one opt-in float64 diagnostic
deselected**. Checks passed for 39 Python files, 33 configuration/fixture JSONs
(plus the source manifest), 34 shell scripts, one PowerShell script, 27 inherited
root configurations, 130 local Markdown links and 23 documented API symbols.
All 109 retained implementation/config/test/executable-script/dependency files
matched the before-cleanup SHA-256 fingerprints; 1,798 independent numerical
source files and the dirty main checkout roster were unchanged.

The successful bounded check completed in 7.156 s under a 2-GiB Windows Job
Object private-commit cap and exact 500,000,000-byte free-RAM reserve. Kernel
private peak was 0.157600 GiB, sampled tree RSS 0.178612 GiB, and minimum
sampled system-available RAM 6.615658 GiB. No guard stop occurred; sampling was
complete. These measurements describe only the small publication checks,
not large-grid solver memory. Threads/affinity were child-only; no persistent
environment change was made.

Raw Git whitespace checks report three inherited warnings in unchanged
`jobs/setup_afm_mpi_env.sh`, `simulation/main_loop.py` and
`tests/test_numerics.py`. They are preserved deliberately rather than altering
original implementation bytes; no new documentation whitespace issue remains.

The main dirty checkout is preserved; publication uses a separate checkout of
the existing remote branch and a normal fast-forward push. No forced update,
combined/hybrid synchronization, licensed QTCAD run, Fir action or experimental
upgrade promotion is included.
