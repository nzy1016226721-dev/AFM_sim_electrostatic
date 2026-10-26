# AFM validation and scientific limits — October 26, 2026

## Candidate checks

The broad maintained selection passed **265 tests**, with one opt-in float64
diagnostic deselected, in **83.17 seconds**, from the isolated candidate root.
The environment was Python 3.12.7, NumPy 1.26.4, Numba 0.63.1 and pytest 7.4.4.
Numba was limited to eight and nested OMP/MKL/OpenBLAS/BLIS pools to one, for
these child processes only. This does not claim eight-core utilization.

The genuine local MPI audit also passed four shared/two-rank pairs on 8³→16³
hierarchies: compact RMS, compact both norms, standard both norms, and compact
both+cut+QD/history/precision observation. Splits exercise x, y and z. All eight
full/cut pairs (16,960 values) match float32 payload bits, including signed
zeros; eight coordinate sidecars match and cuts equal their full-field crops.
Independent reconstruction verifies 16,384 uniquely owned nodes and 2,048
internal-face halo values. All levels converge. The 624 native-history row
pairs match iteration IDs and agree within 5e-12 relative (maximum absolute
difference 4.882760862301438e-13), allowing MPI's 12-digit CSV formatting;
histories are not claimed byte-identical. Full NPY headers can differ.

The MPI environment was the existing Python 3.12.14 / NumPy 2.3.5 / Numba 0.67.0 /
mpi4py 4.1.2 / Intel MPI 2021.18.1 Windows runtime. One analysis-harness sidecar
assumption initially failed after successful solves; corrected readback reused
those original results. It did not trigger a numerical retry or solver edit.
All 36 configs resolve with CPU first, dielectric blocks last and zoom disabled;
34 use normal RAM defaults and two preserve explicit float64 diagnostic opt-outs.
The source verifier passes intact input and rejects changed/missing/extra files.
See [compact machine-readable evidence](evidence/UPLOAD_CHECKS_20261026.json).

The source-only upload audit rechecks a clean local copy, separate from the
dirty development repository. Its receipts record static Python/JSON/shell
checks, config inheritance/defaults, portable links, source/archive hashes,
the selected small regression suite and CLI planning. Exact outcomes are
recorded in the completed upload audit. These checks execute no large-grid
campaign, Fir operation or licensed calculation.

From the standalone root, a compatible environment can run:

```text
python tools/verify_source.py
python -m pytest -q tests -k "not test_float64_diagnostic_runs_only_requested_iterations" -p no:cacheprovider --basetemp outputs/pytest_fresh
python run_all.py --help
python run_mpi.py tests/data/afm_config_mpi_smoke.json --plan --ranks 2 --ranks-per-node 1 --node-memory-gib 4
```

Create the output parent and use a fresh basetemp; pytest may replace an
existing basetemp. The excluded opt-in float64 smoke is a historical diagnostic,
not production policy. Production phi remains float32. Child-only thread/cache
settings used by this audit do not change an installed environment or OS policy.

## Dated numerical evidence

These observations summarize retained campaigns; they are not new solves or
current remote status. Original controls, logs, arrays and source seals remain
in the development workspace. The October 9 detailed record is retained in
[RAM validation](RAM_VALIDATION_20261009.md); later qualifications below take
precedence over its then-unexecuted cluster/port statements.

| Comparison | Observed agreement | RAM/time evidence and limits |
| --- | --- | --- |
| Shared 512/1024 compact-storage campaign | Full float32 fields and complete histories equal their RAM-first references; checkpoints also equal original-standard small fields | At 1024 private commit 10.679→4.814 GiB, sampled RSS 10.117→4.953 GiB; 1189.27→3805.20 s, about 3.20x. This is additional compact saving over RAM-first, not a measured full standard→compact 1024 pair |
| Genuine small two-rank RAM replay | 36 uneven/noncubic diagnostics plus physical64; 111 full/cut NPYs and potential halos agree | Small-rank private memory increased 4.76% from setup/JIT overhead; no universal RAM-win claim |
| Matched 4096 MPI campaign | 128 ranks/four nodes; saved 200-cube cuts and native histories equal, 30 finest updates, RMS 8.915873465480e-7 | Worst sampled node RSS 348.746→90.695 GiB (73.994% lower); numerical step 122.295→234.949 s (1.921x) |
| 12000 MPI campaign, different allocations | 200-cube cuts byte-identical; 2498 rows have equal iteration/max/count, RMS differs by at most 2.69709e-13 | Sum of individual-node RSS peaks 32.647→7.962628 TiB (75.61% lower), not simultaneous peak; wall time 7.792x, summed rank CPU 22.56% lower, allocated numerical core-hours 29.86% higher |

The 12000 standard run used 96 nodes and compact 16; different layouts/dates
prevent a same-allocation speedup claim. Its full current source/receipt readback
has 18 inaccessible original paths. The compact batch's original strict
shutdown-observation failure is retained. Exact readable cut agreement neither
repairs those holes nor retrospectively promotes the strict admission outcome.
The matched cut SHA-256 is
`38ec718b07b82f41bdf0b321cc0649e6a2356816081fbab4448af52a68b100bd`.

## Stopping and phi convergence

All ten original Fir stopping pilots reached 2048. RMS-only passed at 100
finest updates, RMS 9.75426497e-7 and maximum 7.27415085e-4. The both-norm case
reached 2261 updates, RMS 2.653689e-7 and maximum 2.336502e-5, then its solver
time limit. Their retained cut differs by up to 11.5966 mV and the eps15 QD
scope by 2.76228 mV. They are different stopping endpoints, not exact-error
references. Precision-evaluation observations were substantially more costly;
detector thresholds remain opt-in rather than production defaults.

The later four 1024 plateau cases retained finite cuts at their roughly
47.6-hour finest-stage time limits; they did not pass requested BOTH 1e-10.
Full-history replay covered 77,638 checks. Long histories and a repeated
float32 maximum do not certify an ultimate precision floor.

Matched-residual 256/1nm and 256/2nm centered/+50nm cases all passed their
achieved 512-reference RMS/max targets, not 1e-10. After coordinate-aware
interpolation, QD RMS differences versus the time-limited 512/0.5nm reference
were 0.143–0.321 mV for 1nm and 1.804–2.372 mV for 2nm. Domain extent and
rasterization also changed, so this does not isolate spacing error. Extra
relaxation on the same 2nm problem shifted QD phi by 3.247/6.818 microvolts RMS
for center/+50nm. Algebraic stopping, domain/mesh convergence and downstream
quantum observables require separate evidence.

## Failed, cancelled and unexecuted work

- The October 4–6 V-cycle/certified-refinement upgrade was cancelled/reverted;
  its failures and bounded attempts remain archived. It is not shipped here.
- A later eight-rank Fir confirmation qualifier failed its dtype/finite-field
  assertion before compact mode; its four dependent confirmations were never
  submitted. Other study qualifiers passing do not erase that failed release's
  unresolved assertion. This publication audit does not rerun it.
- The complete requested stopping/spacing/domain matrix through 12000 has not
  been completed. Large observation windows include time-limit nonpasses.
- No completed 16384 numerical comparison is established by the retained reports.
  Later job status is not inferred from older queue observations.
- The offline diminishing-returns control and a fresh hybrid stopping release
  remain future work. No authoritative-cylinder/coarsening upgrade is enabled.

No residual threshold above establishes microvolt maximum potential accuracy.
Equivalence of RAM representations answers a different question from whether
the discretized field or a QTCAD observable has physically converged.
