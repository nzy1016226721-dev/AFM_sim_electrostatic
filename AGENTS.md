# Standalone AFM handoff

This repository root is the maintained standalone AFM package. Read
[current state](docs/CURRENT_STATE.md) before changing behavior. In the larger
development workspace, also follow its root AGENTS.md; bundled and hybrid
engines have separate release and coordination rules.

- **Numerics or configuration:** read [the numerical contract](DOCUMENTATION.md),
  [storage policy](RAM_SAVING.md) and [stopping controls](docs/STOPPING_MODES.md).
  Preserve float32 production fields, snapshot Jacobi, material ordering,
  ownership/halos, residual definitions and coordinate-aware output contracts.
- **Visualization:** read [local plotting contracts](docs/MPI_PLOTTING.md).
  Eric's adapted tools inspect downloaded results locally after MPI finishes.
- **Validation:** use [the current validation record](docs/VALIDATION_20261026.md).
  Report actual tests and terminal numerical outcomes separately. An early stop,
  diagnostic count or deadline is not strict convergence or a phi-error bound.
- **Publication:** follow [the upload audit](docs/UPLOAD_AUDIT_20261026.md).
  Its preparation-stage authority was followed by the user's direct-publication
  instruction recorded in [publication provenance](docs/PUBLICATION_20261026.md).
  Preserve source provenance and historical manifests; retain agent Markdown.
  Outputs, environments, credentials, frozen operational studies and other
  packages remain outside this standalone source payload.

Check Git status before editing and preserve unrelated work. Inspect existing
code and tests before replacing behavior. Finish with relevant small tests,
an inspected diff, a verified source manifest and updated evidence/limitations.
Commit, push, cluster access, job changes and licensed calculations require
authority for those actions; preparing an audit grants none of them.
