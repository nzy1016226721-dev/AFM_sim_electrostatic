# October 26 standalone AFM publication

The user explicitly authorized: **"upload to repo and update
`afm-parallelized-mpi` directly"**, superseding the earlier preparation-only
scope and review-branch recommendation for this publication. Destination is
[AFM_sim_electrostatic / afm-parallelized-mpi](https://github.com/nzy1016226721-dev/AFM_sim_electrostatic/tree/afm-parallelized-mpi).
The release's actual commit and parent are recorded by Git history; no commit
hash is embedded in its own hashed payload.

## Audited source and publication metadata

The preceding public commit is `8c4819d6f2ac2797774303502a740a50dd9858c0`.
The reviewed 173-file source candidate has Git tree
`054ae42f57acaaeaa410c72233a56fb6245f728b` and source-ZIP SHA-256
`78699c5564357d761b5e1507f75534cd15b56532ffd3247c8c500f616085ab68`.
That candidate, archive and all preparation receipts remain preserved locally.

This publication adds this record and refreshes publication-status Markdown
and the source manifest. It preserves every audited numerical/config/job/
plotting/test byte and the source verifier. The payload has 174 source files;
the manifest inventories every member except itself. Run
`python tools/verify_source.py` from the standalone root to verify a checkout.

## Validation and unchanged limits

The [validation record](VALIDATION_20261026.md) covers 265 selected tests, four
genuine local two-rank shared/MPI pairs, exact full/cut payload and coordinate
replay, ownership/internal halos, native histories and configuration checks.
Publication-only documentation does not require a new numerical solve. The
publication readback checks the committed Git tree and source bytes against
the prepared payload, including shell executable modes.

RAM-saving remains the normal JSON default: lossless float32 material storage,
snapshot-preserving in-place phi and scalar RMS/maximum reductions. New
stopping policies remain opt-in. This source publication does not promote
stagnation/precision thresholds, change physical presets, certify microvolt
phi accuracy, resolve older failed cluster gates or synchronize hybrid/bundled
engines. It performs no Fir/job/licensed-calculation action.

The [upload audit](UPLOAD_AUDIT_20261026.md) is retained as a dated preparation
record. Its earlier unpublished/review-branch wording belongs to that stage;
this subsequent direct-publication instruction changes the Git action only.
Preserve historical numerical outcomes, source seals and their limitations.
