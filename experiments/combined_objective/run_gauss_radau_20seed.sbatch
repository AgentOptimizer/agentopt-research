#!/bin/bash
#SBATCH --job-name=gauss-radau-20seed
#SBATCH --account=torch_pr_790_general
#SBATCH --array=0-159%16
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/combined_objective/results/gauss_radau_8bench_20seed_independent/slurm/%A_%a.out
#SBATCH --error=experiments/combined_objective/results/gauss_radau_8bench_20seed_independent/slurm/%A_%a.err

set -euo pipefail
cd /scratch/yh6415/project/agentopt-research

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl-gauss-radau-${SLURM_JOB_ID}"
export AGENTOPT_RADIAL_GITTINS_CACHE_DIR="${PWD}/experiments/combined_objective/results/gauss_radau_8bench_20seed_independent/boundary_cache"

.venv-plot/bin/python -u \
  experiments/combined_objective/run_gauss_radau_slurm_task.py \
  --output-root experiments/combined_objective/results/gauss_radau_8bench_20seed_independent
