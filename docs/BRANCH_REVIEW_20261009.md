# GitHub branch review — October 9, 2026

## Scope and snapshot

The user requested comparison of GitHub branches, cleanup recommendations and
upload of the newly tested local standalone pack. The upload target is the
maintained [`afm-parallelized-mpi`](https://github.com/nzy1016226721-dev/AFM_sim_electrostatic/tree/afm-parallelized-mpi)
branch. This review does not authorize branch deletion, changing the repository
default branch, or overwriting a separate solver/plotting lineage.

The live fetch/API snapshot at 19:45 UTC contained five branches, no tags and no
open pull requests. `master` was the default; none of the five branches was
protected. Exact before-upload heads and content counts:

| Branch | Head | Tracked files | Finding and recommendation |
| --- | --- | --- | --- |
| `afm-parallelized-mpi` | `7ec0e647d35332362d73f4bcf75cc56f550abefe` | 126 | Maintained standalone; publish the tested RAM-default source here using a normal fast-forward |
| `afm-enhanced-pack` | `adb70e47049c6cffce9e2f91e5b640364e5da734` | 53 | Fully ancestral to maintained MPI; zero unique commits, two commits behind. Safe retirement candidate after human confirmation; no merge needed |
| `afm-mpi-qol-plotting-integration` | `2be34adda9ea3948f82287a4b5cfa26b62631986` | 132 | One unique commit directly on the old maintained head; retain pending an explicit consolidation decision |
| `afm-additional-postprocessing-options-and-QoL` | `4db9c60f3d071e20f6f6947d4005b70aea18a72b` | 81 | Three commits unique versus maintained MPI, two beyond master; retain Eric's original source/provenance |
| `master` | `3d1aa0804fb0d840c9b0b6799905164c99dcea70` | 52 | Legacy default with one unique commit versus maintained MPI; retain, rather than replace it with a different package |

Counts/ancestry describe the frozen before-upload snapshot. Publication advances
only the maintained MPI branch; sibling heads remain as recorded. A later clone
must compare live heads again before any cleanup.

## Content comparison, not just commit counts

`afm-enhanced-pack` is already reachable through the maintained branch's history.
Its tree still contains 20 bytecode/cache files and an old v15 ZIP. Those are
historical artifacts, not missing active files to copy into the new release.
Deleting only this branch reference would not erase its commits, because the
maintained branch still reaches them. No delete was performed by this review.

The plotting-integration commit changes nine paths: four new plot/data modules,
nine new plotting tests, a plotting guide, source manifest, README link and
`run_all.py` menu choices 7/8. It provides comparative planes/lines, two-click
profiles, cut-preferred discovery and config/filename-based cut-coordinate
reconstruction. It changes no numerical solver modules. These interfaces are
**not byte/API-identical** to the RAM release's separate `local_post.py` and
`local_visualization/`, even where features overlap. The RAM tools explicitly
reject MPI/Slurm execution, use retained-node sidecars/bounds, support bounded
read-only sanity and optional small conductivity checks, and keep plotting
outside numerical launchers. A blind merge would duplicate plotters and restore
different menu/API behavior. Recommendation: retain this feature branch for now;
if consolidation is requested, adapt its menu/API calls to the local-only tools
and run dedicated compatibility/coordinate tests before retiring it. Its
unmerged commit is not silently discarded or described as integrated.

Eric's original branch contains older simulation/presimulation code, demo/UI
state and 42 bytecode/cache files, as well as the plotting source and notes.
The RAM release contains credited, tested **adaptations**, not a whole-branch
merge or a claim that every change is equivalent. Original Python/notes bytes
are locally archived. Preserve the branch as source provenance until the human
chooses an archival tag or another explicit long-term reference. Mixing its old
solver into maintained MPI would reverse current numerical maintenance.

The one master-only commit, `3d1aa08`, edits
`afm_package/presimulation/master_presim.py` (308 added/57 removed lines).
It is legacy presimulation/menu work, not a missing MPI storage fix. The master
and MPI trees have different layouts and purposes. Leave master/default alone;
changing the default to the maintained solver is a separate repository decision.

## Publication and recovery

The published payload is the current standalone source manifest, not the dirty
workspace root or other packages. Numerical/configuration/job files remain
exactly the tested local promotion; this publication adds self-contained RAM
validation/branch documentation. [RAM evidence](RAM_VALIDATION_20261009.md)
distinguishes large sealed trials, installed tests and unexecuted large MPI work.
No stale caches, ZIPs, numerical outputs, licences or environments are shipped.
Shell launchers retain executable Git modes and source bytes are hash-verified.

In the development workspace, `archive_old/afm_github_publication_20261009/`
contains a verified `github_all_before.bundle`, all 151 pre-publication local
source files, root-document snapshots and checksum receipts. The bundle stores
all five original heads/history and does not depend on branch names surviving.
Comparison, test and independent fresh-clone receipts live under
`afm_parallel_trials/github_publication_20261009/outputs/`; the ordinary report is
`docs/AFM_GITHUB_PUBLICATION_20261009.md`. These local recovery/evidence paths
are deliberately outside the source distribution, not broken checkout links.

No branch deletion, cross-branch merge, default-branch change, force push,
combined/hybrid numerical change or Fir deployment is included. Cleanup remains
an explicit follow-up decision; the requested upload preserves sibling work.
