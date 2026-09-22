#!/usr/bin/env bash
#SBATCH --account=torch_pr_790_general

set -euo pipefail

readonly ROOT=/scratch/yh6415/project/agentopt-research
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/mpl-eval-budget-${SLURM_JOB_ID}"

cd "$ROOT"
exec env PYTHONPATH=src:. "$ROOT/.venv-plot/bin/python" -u \
  experiments/combined_objective/run_eval_budget_baseline_task.py \
  --task-id "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" \
  "$@"
