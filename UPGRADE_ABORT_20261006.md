# October AFM upgrade cancelled; original numerical source restored

The October 4–6, 2026 upgrade was scrapped at the user's request on October 6
(Toronto). Standalone was restored to its original 124-file pre-implementation
snapshot; a fresh selected small regression passed 60 tests with one opt-in
float64 memory diagnostic deselected. Original shared-memory zoom and preset
values were restored. No new solver, memory-first preset, certified-refinement
mode or new geometry/field contract was promoted. Earlier maintenance remained.

The later [October 7 consolidation](docs/PACKAGE_CONSOLIDATION_20261007.md)
changes documentation/packaging only: superseded helpers are archived and
missing source directories are published. Solver/configuration bytes remain
unchanged; the whole active file set is no longer literally the old 124-file
snapshot. The current source manifest lists this distribution. This does not
resume the cancelled upgrade.

In the development workspace, findings, retired additions, failed/completed/
interrupted attempts and source are in
`D:\afm_pack_v1\archive_old\afm_upgrade_aborted_20261006\` and the package-local
`archive_old/afm_upgrade_aborted_20261006/`. The workspace narrative is
`D:\afm_pack_v1\docs\AFM_UPGRADE_ABORT_20261006.md`. These are local-only archives,
not missing dependencies of this Git distribution.

Do not resume old controllers, reset consumed attempts or relabel experimental
certificates as current-default validation. Existing fields/raw outputs remain
unchanged. Any future numerical upgrade needs a new request. Independent
combined/hybrid/QTCAD numerical packages were not synchronized here.
