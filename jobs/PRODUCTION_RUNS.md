# Prepared Fir production runs

Historical preparation/resource note retained October 7, 2026. Job states,
headroom and submission prerequisites below describe that study, not current
Fir availability or a new validation. See the
[current package catalogue](../docs/CONFIGS_AND_JOBS.md).

These two configurations preserve the complete nanometre-coordinate physics,
the `-1.0` through `-5.0 V` sweep, and the diagonal motion specification from
`afm_config_nm.json`.  The source step `7.071 nm` produces 12 movement centres
with the current source algorithm; together with nine voltages, each job runs
108 solver cases.

| Run | Configuration | Launcher | Allocation | Output |
| --- | --- | --- | --- | --- |
| 2048 cubed | `afm_config_nm_production_2048.json` | `run_afm_nm_2048.sh` | 96 CPUs, 384 GiB, 24 h, `cpubase_bycore_b5` | `outputs/job_<jobid>/afm_config_nm_production_2048/` |
| 4096 cubed | `afm_config_nm_production_4096.json` | `run_afm_nm_4096.sh` | 192 CPUs, 4 TiB, 4 d, `cpularge_bynode_b5` | `outputs/job_<jobid>/afm_config_nm_production_4096/` |

Submit from the package root:

```bash
sbatch jobs/run_afm_nm_2048.sh
sbatch jobs/run_afm_nm_4096.sh
```

The 4096 cubed job is deliberately prepared but must not be submitted until
the direct 4096 cubed to 8192 cubed memory trial has returned useful runtime
memory evidence.

## Large-grid hierarchy and stopping policy

Large targets use a dynamic coarse-grid start so their doubling hierarchy has
six to eight levels: 2048 cubed starts at 64 cubed (six levels), 4096 cubed at
64 cubed (seven), 8192 cubed at 64 cubed (eight), and 16384 cubed at 128 cubed
(eight). A legacy `max_iter` value is never a solver stopping criterion. Each
level stops only when it reaches the configured residual tolerance or its
wall-clock deadline (`mg_max_runtime`, with the Slurm allocation as the outer
deadline).

## Output policy

Both jobs retain the template's physical 100 nm movement-centred NPY cuts,
residual/timing logs, and memory tracking.  They intentionally set
`save_full: false`: a 108-case production sweep would otherwise write roughly
3.4 TiB of full 2048 cubed fields or 27 TiB of full 4096 cubed fields, while
the package's Fir home filesystem currently has only about 48 GiB available.
For a QTCAD full-field input, prepare a separate selected-position/voltage
rerun rather than changing this broad production sweep.

Slurm stdout and stderr are kept out of both the package root and the
per-job science directory. They are written to the sibling
`outputs/slurm_logs/<job-name>-<jobid>.out` and `.err` paths.

The 2048 cubed request follows the 4 GiB per CPU policy exactly.  The 4096
cubed request is an unavoidable one-node topology exception: its conservative
4 TiB memory request requires Fir's 192-CPU high-memory node, whereas literal
4 GiB-per-CPU pairing would require 1024 CPUs and cannot be used by this
single-process Numba solver.
