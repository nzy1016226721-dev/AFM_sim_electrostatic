#!/bin/bash
#SBATCH --job-name=afm
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --time=12:00:00
## Set your Alliance project/account here if required:
## #SBATCH --account=def-yourproject

set -euo pipefail

# Alliance Python/scientific stack. On a specific cluster, choose compatible
# versions with `module avail python` and `module avail scipy-stack`.
module load python
module load scipy-stack

# Keep NumPy/SciPy/BLAS single-threaded for the current single-thread calculation.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR}"

# Examples:
#   sbatch jobs/run_afm.sh afm_config_nm.json
#   sbatch jobs/run_afm.sh /path/to/afm_config_nm.json
#   sbatch jobs/run_afm.sh --no-plot afm_config_nm.json
#   sbatch jobs/run_afm.sh --plot afm_config_nm.json
python run_all.py "$@"
