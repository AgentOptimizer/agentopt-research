#!/bin/bash
#SBATCH --array=0-159
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/combined_objective/results/gittins_ablation_v2_8bench_20seed/slurm/%x_%A_%a.out
#SBATCH --error=experiments/combined_objective/results/gittins_ablation_v2_8bench_20seed/slurm/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-${PWD}}"

readonly CONFIGURATION="${1:?usage: run_gittins_ablation_v2_20seed.sbatch CONFIGURATION}"
readonly OUTPUT_ROOT="${PWD}/experiments/combined_objective/results/gittins_ablation_v2_8bench_20seed"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl-gittins-ablation-v2-${SLURM_JOB_ID}"
export AGENTOPT_RADIAL_GITTINS_CACHE_DIR="${OUTPUT_ROOT}/boundary_cache"

"${AGENTOPT_PYTHON:-.venv/bin/python}" -u \
  experiments/combined_objective/run_gittins_ablation_v2_slurm_task.py \
  --configuration "${CONFIGURATION}" \
  --output-root "${OUTPUT_ROOT}"
