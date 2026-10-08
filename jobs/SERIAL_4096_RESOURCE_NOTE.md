# 4096 serial comparison resource adjustment — 2026-09-07

Historical scheduler/resource observations, not live status. Retained during
October 7, 2026 source consolidation; no Fir action was taken. See the
[current package catalogue](../docs/CONFIGS_AND_JOBS.md).

Job 57850909 was updated in place from 6000 GiB and
`cpularge_bynode_b5` to 2560 GiB and `cpularge_bynode_b1`.
It retains one node, one task, 192 threads, a three-hour wall-time limit,
its original job ID and submission timestamp, and comparison dependency
57894385. No solver code or scientific configuration was changed.

The live Fir controller reported only two b5 large-memory nodes, one drained
and the other occupied. The b1 partition supports up to three hours and
includes all eight large-memory nodes, with two idle at inspection.
The partition change expands eligibility; a shorter queue is not guaranteed.

Completed shared-memory 2048 job 57843357 recorded 211.730965 GiB in
`memory_usage_log.csv`; Slurm recorded 222511192 KiB (212.20 GiB).
Doubling each dimension gives an eightfold extrapolation of about
1694–1698 GiB. The 2560 GiB request leaves approximately 51% above that
estimate. This remains an extrapolation, not a measured 4096 peak.
The local and remote solver.py and main_loop.py hashes matched at inspection.

2560 GiB / 4 GiB per core corresponds to 640 memory-based core equivalents.
This shared-memory run cannot use 640 physical CPUs: each eligible node has
192. Its request keeps all 192 CPUs, and the controller changed requested
billing TRES from 1500000 to 640000. Actual allocated billing can differ on
whole-node partitions; inspect AllocTRES after scheduling.

The modified wrapper passed remote `bash -n`, its local/remote SHA256 matched,
and `sbatch --test-only --partition=cpularge_bynode_b1 --mem=2560G` succeeded.
The old wrapper is retained locally and on Fir in
`archive_old/jobs/memory_request_20260907/`.
