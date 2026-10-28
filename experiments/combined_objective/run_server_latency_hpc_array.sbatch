#!/bin/bash
#SBATCH --job-name=three-server-latency
#SBATCH --account=torch_pr_790_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yh6415/project/agentopt-research/analysis/server_latency_paper_preview_20261028/slurm/%A_%a.out
#SBATCH --error=/scratch/yh6415/project/agentopt-research/analysis/server_latency_paper_preview_20261028/slurm/%A_%a.err

set -euo pipefail

REPO=/scratch/yh6415/project/agentopt-research-server-latency
PYTHON=/scratch/yh6415/project/agentopt-research-three-objective/.venv/bin/python
RESULTS=${RESULTS_ROOT:-/scratch/yh6415/project/agentopt-research/analysis/server_latency_paper_full_20261028/runs}
TASK_TMP=/scratch/yh6415/project/agentopt-research/analysis/server_latency_paper_preview_20261028/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${TASK_TMP}/tmp" "${TASK_TMP}/mpl" "${TASK_TMP}/cache"
export TMPDIR="${TASK_TMP}/tmp"
export MPLCONFIGDIR="${TASK_TMP}/mpl"
export XDG_CACHE_HOME="${TASK_TMP}/cache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1
cd "${REPO}"
exec "${PYTHON}" -u -m experiments.combined_objective.run_server_latency_hpc_task \
  --kind "${KIND}" --task-id "${SLURM_ARRAY_TASK_ID}" --output-root "${RESULTS}" \
  --quality-prior-variance-mode "${QUALITY_PRIOR_VARIANCE_MODE:-warm}"
