#!/usr/bin/env bash
# Per-level AFM memory test for Fir/Alliance.
#
# One invocation tests the levels listed in afm_memory_test.json in order.  Each
# target level is run in a separate srun child so an OOM or solver failure can be
# attributed to that level.  The test stops at the first failure by default;
# set AFM_MEMORY_CONTINUE_AFTER_FAILURE=1 only when deliberately testing past a
# failed level.
#
# Submit from the package root or jobs directory:
#   sbatch jobs/run_memory_test.sh afm_memory_test.json
#   sbatch run_memory_test.sh ../afm_memory_test.json
#
#SBATCH --job-name=afm-memory
# Fir's cpularge nodes provide 192 CPUs and 6 TB RAM. The focused 8192^3
# diagnostic uses 4 TB to measure the 2048^3/4096^3 stages; an in-memory 8192^3
# solution is expected to exceed one node and is therefore a bounded trial.
# A single non-MPI process cannot spread its arrays across nodes.
#SBATCH --partition=cpularge_bynode_b1
#SBATCH --cpus-per-task=192
#SBATCH --mem=4096G
#SBATCH --time=01:00:00
#SBATCH --account=rrg-hongguo-ad
#SBATCH --output=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home/nizy/afm_parallel/outputs/slurm_logs/%x-%j.err

set -euo pipefail

# Keep diagnostics and the JSON's solver thread count visibly aligned with the
# Slurm request above. The literal #SBATCH values are required by sbatch; these
# labels are used only for validation/logging after allocation.
REQUESTED_CPU_COUNT=192
REQUESTED_MEMORY_LABEL="4096G"
# Stop the sole child five minutes before Slurm's one-hour limit. This yields a
# level-aware failure marker and avoids an uninformative scheduler TIMEOUT.
CHILD_DEADLINE_SECONDS="${AFM_MEMORY_CHILD_DEADLINE_SECONDS:-3300}"

current_level="not-started"
trap 'rc=$?; echo "[MEMORY-TEST][SIGNAL] target_level=${current_level} received termination signal (exit=${rc})" >&2; exit "${rc}"' TERM INT HUP

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this memory test with sbatch, not by executing it directly." >&2
    exit 2
fi

if ! [[ "${CHILD_DEADLINE_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: AFM_MEMORY_CHILD_DEADLINE_SECONDS must be a positive integer." >&2
    exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
    echo "ERROR: GNU timeout is required for the memory-test safety deadline." >&2
    exit 2
fi

# Slurm may execute a staged copy of this file from a spool directory.  Resolve
# the package from SLURM_SUBMIT_DIR, never from BASH_SOURCE[0].
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
SUBMIT_DIR="$(cd -- "${SUBMIT_DIR}" && pwd)"

if [[ -n "${AFM_PACKAGE_ROOT:-}" ]]; then
    PROJECT_ROOT="${AFM_PACKAGE_ROOT}"
    if [[ "${PROJECT_ROOT}" != /* ]]; then
        PROJECT_ROOT="${SUBMIT_DIR}/${PROJECT_ROOT}"
    fi
    PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
elif [[ -f "${SUBMIT_DIR}/run_all.py" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../run_all.py" ]]; then
    PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
    echo "ERROR: could not locate run_all.py from SLURM_SUBMIT_DIR=${SUBMIT_DIR}." >&2
    echo "Submit from the package root or jobs directory, or set AFM_PACKAGE_ROOT." >&2
    exit 2
fi

if [[ ! -f "${PROJECT_ROOT}/run_all.py" || ! -d "${PROJECT_ROOT}/simulation" ]]; then
    echo "ERROR: invalid AFM package root: ${PROJECT_ROOT}" >&2
    exit 2
fi

CONFIG_ARG="${1:-${AFM_CONFIG:-afm_memory_test.json}}"
if [[ "${CONFIG_ARG}" == -* ]]; then
    echo "ERROR: the first argument must be a JSON configuration filename/path." >&2
    echo "Usage: sbatch run_memory_test.sh [afm_memory_test.json]" >&2
    exit 2
fi

resolve_config() {
    local requested="$1"
    local candidate
    if [[ "${requested}" = /* ]]; then
        candidate="${requested}"
        [[ -f "${candidate}" ]] && {
            printf '%s\n' "$(cd -- "$(dirname -- "${candidate}")" && pwd)/$(basename -- "${candidate}")"
            return 0
        }
    else
        for candidate in "${PROJECT_ROOT}/${requested}" "${SUBMIT_DIR}/${requested}"; do
            if [[ -f "${candidate}" ]]; then
                printf '%s\n' "$(cd -- "$(dirname -- "${candidate}")" && pwd)/$(basename -- "${candidate}")"
                return 0
            fi
        done
    fi
    return 1
}

if ! CONFIG_PATH="$(resolve_config "${CONFIG_ARG}")"; then
    echo "ERROR: JSON configuration not found: ${CONFIG_ARG}" >&2
    echo "Package root: ${PROJECT_ROOT}" >&2
    exit 2
fi
case "${CONFIG_PATH}" in
    *.json|*.JSON) ;;
    *) echo "ERROR: configuration must be a .json file: ${CONFIG_PATH}" >&2; exit 2 ;;
esac

cd -- "${PROJECT_ROOT}"

if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: the Environment Modules command is unavailable." >&2
    exit 2
fi
module load "${AFM_PYTHON_MODULE:-python}"
module load "${AFM_SCIPY_MODULE:-scipy-stack}"

AFM_VENV="${AFM_VENV:-${HOME}/afm_env}"
if [[ ! -f "${AFM_VENV}/bin/activate" ]]; then
    echo "ERROR: AFM virtual environment not found: ${AFM_VENV}" >&2
    echo "Run bash jobs/setup_afm_env.sh once on Fir, or export AFM_VENV=/path/to/venv." >&2
    exit 2
fi
source "${AFM_VENV}/bin/activate"

python_bin="${PYTHON_BIN:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${python_bin}" >&2
    exit 2
fi

# Keep results in the package's conventional outputs directory by default.
# AFM_OUTPUT_ROOT is an explicit opt-in override for a project or other path.
if [[ -n "${AFM_OUTPUT_ROOT:-}" ]]; then
    output_base="${AFM_OUTPUT_ROOT%/}"
else
    output_base="${PROJECT_ROOT}/outputs/afm_memory_test"
fi
export AFM_JOB_OUTPUT_ROOT="${output_base}/job_${SLURM_JOB_ID}"
if ! mkdir -p "${AFM_JOB_OUTPUT_ROOT}"; then
    echo "ERROR: cannot create memory-test output directory: ${AFM_JOB_OUTPUT_ROOT}" >&2
    exit 2
fi

export AFM_NONINTERACTIVE=1
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${REQUESTED_CPU_COUNT}}"

echo "============================================================"
echo "AFM per-level memory test"
echo "SLURM_JOB_ID     = ${SLURM_JOB_ID}"
echo "SLURM_SUBMIT_DIR = ${SUBMIT_DIR}"
echo "Package root     = ${PROJECT_ROOT}"
echo "Base config      = ${CONFIG_PATH}"
echo "Output root      = ${AFM_JOB_OUTPUT_ROOT}"
echo "CPU allocation   = ${SLURM_CPUS_PER_TASK:-${REQUESTED_CPU_COUNT}}"
echo "Memory request   = ${REQUESTED_MEMORY_LABEL}"
echo "Child deadline   = ${CHILD_DEADLINE_SECONDS} s (pre-emptive)"
echo "Python           = $(command -v "${python_bin}")"
"${python_bin}" --version
"${python_bin}" -c 'import numpy, scipy, numba, psutil; print("NumPy", numpy.__version__, "SciPy", scipy.__version__, "Numba", numba.__version__, "psutil", psutil.__version__)'
echo "============================================================"

# Validate the memory-test policy before allocating any grid.  The validator
# emits only one level per line so mapfile receives a clean Bash array.
levels_text="$("${python_bin}" - "${CONFIG_PATH}" "${SLURM_CPUS_PER_TASK:-${REQUESTED_CPU_COUNT}}" <<'PY'
import json
import math
import sys

path = sys.argv[1]
slurm_cpus = int(sys.argv[2])
with open(path, "r", encoding="utf-8") as handle:
    cfg = json.load(handle)

test = cfg.get("memory_test")
if not isinstance(test, dict) or not test.get("enabled", False):
    raise SystemExit("memory_test.enabled must be true")

levels = test.get("levels")
if not isinstance(levels, list) or not levels:
    raise SystemExit("memory_test.levels must be a non-empty list")
levels = [int(level) for level in levels]
max_allowed = min(16384, int(test.get("max_level", 16384)))
if max_allowed < 1024:
    raise SystemExit("memory_test.max_level must be at least 1024")
if any(level < 1024 or level > max_allowed for level in levels):
    raise SystemExit(f"memory-test levels must be between 1024 and {max_allowed}")
if any(level & (level - 1) for level in levels):
    raise SystemExit("memory-test levels must be powers of two")
if levels != sorted(set(levels)):
    raise SystemExit("memory-test levels must be unique and in ascending order")

threads = int(cfg.get("cpu_threads", 0))
if threads < 1 or threads > slurm_cpus:
    raise SystemExit(f"cpu_threads={threads} exceeds Slurm CPUs={slurm_cpus}")
expected_tolerance = float(test.get("expected_tolerance", 1e-5))
expected_runtime = float(test.get("expected_runtime_limit_seconds", 120.0))
for tolerance_key in ("res_tol_main", "res_tol_zoom"):
    if not math.isclose(float(cfg.get(tolerance_key, 0.0)), expected_tolerance, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"{tolerance_key} must equal memory_test.expected_tolerance={expected_tolerance}")
if not math.isclose(float(cfg.get("mg_max_runtime", 0.0)), expected_runtime, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit(
        "mg_max_runtime must equal "
        f"memory_test.expected_runtime_limit_seconds={expected_runtime}"
    )
grid = cfg.get("grid_resolution", {})
if any(int(grid.get(axis, 0)) < 2 for axis in ("nx", "ny", "nz")):
    raise SystemExit("grid_resolution must contain dimensions >= 2")
initial_level = int(cfg.get("initial_grid_level", 0))
if initial_level < 2 or initial_level > min(levels):
    raise SystemExit("initial_grid_level must be between 2 and the first target level")
if int(test.get("initial_grid_level", initial_level)) != initial_level:
    raise SystemExit("initial_grid_level must match memory_test.initial_grid_level")
live_interval = float(cfg.get("memory_live_interval_seconds", 0.0))
if live_interval <= 0.0:
    raise SystemExit("memory_live_interval_seconds must be positive for the memory test")

print("\n".join(str(level) for level in levels))
PY
)"
mapfile -t LEVELS <<< "${levels_text}"
if [[ ${#LEVELS[@]} -eq 0 || -z "${LEVELS[0]}" ]]; then
    echo "ERROR: no memory-test levels were selected." >&2
    exit 2
fi

solver_runtime_limit="$("${python_bin}" - "${CONFIG_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = json.load(handle)
print(cfg["memory_test"]["expected_runtime_limit_seconds"])
PY
)"

CONFIG_TMP_DIR="${AFM_JOB_OUTPUT_ROOT}/_configs"
LOG_DIR="${AFM_JOB_OUTPUT_ROOT}/_logs"
mkdir -p "${CONFIG_TMP_DIR}" "${LOG_DIR}"

stop_on_failure="$("${python_bin}" - "${CONFIG_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = json.load(handle)
print("1" if cfg.get("memory_test", {}).get("stop_on_failure", True) else "0")
PY
)"
if [[ "${AFM_MEMORY_CONTINUE_AFTER_FAILURE:-0}" == "1" ]]; then
    stop_on_failure=0
fi
had_failure=0

for level in "${LEVELS[@]}"; do
    current_level="${level}"
    printf '%s\n' "${level}" > "${AFM_JOB_OUTPUT_ROOT}/current_level.txt"
    level_config="${CONFIG_TMP_DIR}/afm_memory_test_level_${level}.json"
    "${python_bin}" - "${CONFIG_PATH}" "${level}" "${level_config}" <<'PY'
import copy
import json
import sys

source_path, level_text, destination = sys.argv[1:]
level = int(level_text)
with open(source_path, "r", encoding="utf-8") as handle:
    cfg = json.load(handle)

cfg = copy.deepcopy(cfg)
cfg["grid_resolution"] = {"nx": level, "ny": level, "nz": level}
cfg["save_full"] = False
cfg["save_cut"] = False
cfg["save_all_levels"] = False
cfg["memory_tracking"] = True
cfg["plotting"] = {"enabled": False, "disable_in_non_ide": True}
cfg.setdefault("zoom_simulation", {})["enabled"] = False
cfg.setdefault("memory_test", {})["active_level"] = level

with open(destination, "w", encoding="utf-8") as handle:
    json.dump(cfg, handle, indent=2)
    handle.write("\n")
PY

    level_dir="${AFM_JOB_OUTPUT_ROOT}/afm_memory_test_level_${level}"
    level_stdout="${LOG_DIR}/level_${level}.out"
    level_stderr="${LOG_DIR}/level_${level}.err"

    # This marker is emitted before allocation.  If the cgroup kills the whole
    # step with SIGKILL, the last START marker still identifies the level.
    echo "[MEMORY-TEST][START] target_level=${level} grid=${level}^3 cpu_threads=${SLURM_CPUS_PER_TASK:-${REQUESTED_CPU_COUNT}} mem=${REQUESTED_MEMORY_LABEL}" >&2
    echo "[MEMORY-TEST][START] output=${level_dir}" >&2

    set +e
    timeout --signal=TERM --kill-after=60s "${CHILD_DEADLINE_SECONDS}s" \
        srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-${REQUESTED_CPU_COUNT}}" \
        "${python_bin}" run_all.py "${level_config}" --no-plot \
        2> >(tee "${level_stderr}" >&2) | tee "${level_stdout}"
    status=${PIPESTATUS[0]}
    set -e

    oom_detail=""
    if oom_detail="$(grep -Ei 'oom|out[ -]?of[ -]?memory|oom-kill|out_of_memory|cannot allocate memory|memoryerror' "${level_stdout}" "${level_stderr}" | tail -n 1)"; then
        :
    fi

    if [[ ${status} -ne 0 || -n "${oom_detail}" ]]; then
        had_failure=1
        reason="solver/step failure"
        failure_status="${status}"
        if [[ -n "${oom_detail}" ]]; then
            reason="out-of-memory reported by Slurm or the process"
            [[ ${failure_status} -eq 0 ]] && failure_status=137
        elif [[ ${status} -eq 124 ]]; then
            reason="pre-emptive child deadline reached after ${CHILD_DEADLINE_SECONDS} s"
            echo "[MEMORY-TEST][TIME-BUDGET] target_level=${level} deadline_seconds=${CHILD_DEADLINE_SECONDS}" >&2
        elif [[ ${status} -eq 137 || ${status} -eq 9 ]]; then
            reason="process killed by signal ${status} (likely out-of-memory)"
        fi
        echo "[MEMORY-TEST][FAIL] target_level=${level} grid=${level}^3 exit_code=${status} reason=${reason}" >&2
        [[ -n "${oom_detail}" ]] && echo "[MEMORY-TEST][DETAIL] ${oom_detail}" >&2
        echo "[MEMORY-TEST][FAIL] level stdout=${level_stdout}" >&2
        echo "[MEMORY-TEST][FAIL] level stderr=${level_stderr}" >&2
        if [[ ${stop_on_failure} -eq 1 ]]; then
            exit "${failure_status:-1}"
        fi
        continue
    fi

    timeout_line=""
    if timeout_line="$(grep -E 'MG aborted early|NOT converged after' "${level_stdout}" | tail -n 1)"; then
        had_failure=1
        echo "[MEMORY-TEST][FAIL] target_level=${level} grid=${level}^3 reason=solver timing/convergence limit (${solver_runtime_limit} s per level)" >&2
        echo "[MEMORY-TEST][DETAIL] ${timeout_line}" >&2
        echo "[MEMORY-TEST][FAIL] level stdout=${level_stdout}" >&2
        if [[ ${stop_on_failure} -eq 1 ]]; then
            exit 3
        fi
        continue
    fi

    if [[ -f "${level_dir}/memory_usage_log.csv" ]]; then
        echo "[MEMORY-TEST][PASS] target_level=${level} grid=${level}^3 memory_log=${level_dir}/memory_usage_log.csv" >&2
    else
        echo "[MEMORY-TEST][WARN] target_level=${level} completed but memory_usage_log.csv was not found at ${level_dir}" >&2
    fi
done

echo "[MEMORY-TEST][COMPLETE] tested levels: ${LEVELS[*]}" >&2
echo "[MEMORY-TEST][COMPLETE] results: ${AFM_JOB_OUTPUT_ROOT}" >&2
if [[ ${had_failure} -ne 0 ]]; then
    echo "[MEMORY-TEST][COMPLETE] one or more levels failed; see the level logs." >&2
    exit 1
fi
