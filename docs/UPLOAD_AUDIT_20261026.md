# Source upload audit — October 26, 2026

Publication update: after this audit the user explicitly requested direct
publication to `afm-parallelized-mpi`. The initial preparation-only authority
and review-branch recommendation below describe the completed audit stage.
See [the publication record](PUBLICATION_20261026.md) for the subsequent
authorization and source-preservation boundary. Scientific/default gates stay
in force; branch publication does not close them.

## Destination and authority

Proposed destination is
[AFM_sim_electrostatic / afm-parallelized-mpi](https://github.com/nzy1016226721-dev/AFM_sim_electrostatic/tree/afm-parallelized-mpi),
freshly pinned to `8c4819d6f2ac2797774303502a740a50dd9858c0`.
This task prepares an unpublished local candidate. It does not authorize a
commit, push, direct merge, branch deletion, cluster deployment or new job.
A subsequent publication should use a review branch such as
`codex/afm-consolidation-20261026` and recheck the remote head before acting.

## Payload boundary

The payload is the maintained standalone package, rooted here: solver,
configuration/inheritance fixtures, shared/MPI entry points, tests, launch
templates, Eric's adapted local-only visualization, ordinary agent/handoff
Markdown and portable validation/source-integrity guidance. Existing numerical,
JSON and job bytes are preserved from the local before snapshot.

The local audit explicitly inventories eligible files instead of trusting the
larger workspace's ignore rules. It excludes output directories, archives,
caches, environments, large arrays, operational controllers/SSH journals,
credentials/licences, vendor software and independent legacy/combined/hybrid
packages. These exclusions are preserved local evidence, not deleted content.
Historical launchers remain templates: inspect accounts, paths, resources and
site modules before any real submission. Their presence is not job authority.

## Integrity and review

The development workspace retains a create-once backup of all pre-consolidation
standalone files, their SHA-256 inventory and root Git/index/independent-engine
fingerprints. A separate candidate directory, ZIP and isolated preview index
provide the exact prospective diff against the pinned public commit. That
index is outside the development Git directory and creates no commit or ref.

`docs/SOURCE_MANIFEST.json` inventories this candidate except itself. Its
October 9 predecessor is retained byte-for-byte in
`docs/history/SOURCE_MANIFEST_20261009.json`; it remains a historical manifest,
not an assertion that old hashes describe newer code. Run
`python tools/verify_source.py` from a fresh checkout to verify current bytes.
The outer archive hash and prospective tree ID live in the external local
receipt to avoid a self-referential manifest.

The audit checks AST/JSON/shell syntax, config inheritance, storage/default
exceptions, portable Markdown links, filename case collisions, file types and
sizes, known credential/token patterns, exact archive member hashes and shell
executable modes. The credential scan is bounded pattern matching, not a
formal security audit or proof that arbitrary private data cannot exist.
Two independent code-review axes inspect repository standards and the user's
consolidation/upload-preparation scope. Detailed results and exact commands
are retained with the local audit receipts.

## Qualification and publication gate

Read [current state](CURRENT_STATE.md), [stopping semantics](STOPPING_MODES.md)
and [validation limits](VALIDATION_20261026.md). The RAM-saving default is
retained. New stopping detectors remain opt-in; absolute/relative semantics,
phi precision and physical presets are not retuned by this consolidation.
Small regression success is not a large MPI or field-accuracy certification.

The prepared snapshot is suitable for source review only after its recorded
static/integrity/tests checks pass and material review findings are resolved
or explicitly held. Direct production merge/default changes require the
separate qualification evidence described above. Before publication, review
the exact file list, retained limitations, branch destination and authority;
then independently verify a fresh resulting checkout. No current cluster or
licence status is part of this source-only audit.

## Completed checks and carried warnings

The current [validation record](VALIDATION_20261026.md) includes 265 selected
regressions, four genuine local two-rank shared/MPI pairs and all 36 effective
configurations. Numerical/config/job/plotting bytes from the before snapshot
are unchanged. The current manifest, archive and isolated index carry those
same tested source bytes; documentation/evidence changes are resealed separately.

The candidate retains the local `* -text` Git attribute policy to preserve
source fingerprints byte-for-byte across checkouts. This is an explicit
metadata change from the published branch's extension-based normalization;
shell files remain LF and their executable index/ZIP modes are audited.
The raw Git whitespace check carries exactly two pre-existing EOF blank-line
warnings in `stopping.py` and `stopping_stats.py`. They are accepted only while
those source bytes remain unchanged, not a blanket whitespace waiver.

The scope review's initial numeric-versus-bit equality finding was corrected:
the final replay compares uint32 views/direct payload bytes, coordinates and
native histories. The standards review's optional gate-hardening recommendation
was also implemented: unaccepted payload, preservation or diff findings make
the audit tool return failure. No unresolved material review finding remains.
