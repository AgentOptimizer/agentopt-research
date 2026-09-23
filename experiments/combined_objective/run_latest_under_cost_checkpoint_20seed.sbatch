#!/usr/bin/env bash
#SBATCH --account=torch_pr_790_general
#SBATCH --job-name=usd-latest-under
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yh6415/project/agentopt-research/analysis/usd_cost_checkpoints_latest_under_20seed/logs/%A_%a.out
#SBATCH --error=/scratch/yh6415/project/agentopt-research/analysis/usd_cost_checkpoints_latest_under_20seed/logs/%A_%a.err

set -euo pipefail

readonly ROOT=/scratch/yh6415/project/agentopt-research
readonly MANIFEST="${1:?usage: run_latest_under_cost_checkpoint_20seed.sbatch MANIFEST}"
readonly OUTPUT_ROOT="$ROOT/analysis/usd_cost_checkpoints_latest_under_20seed/runs"
readonly TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
readonly LINE_NUMBER=$((TASK_ID + 2))

IFS=$'\t' read -r method dataset seed < <(sed -n "${LINE_NUMBER}p" "$MANIFEST")
if [[ -z "${method:-}" || -z "${dataset:-}" || -z "${seed:-}" ]]; then
  echo "No manifest row for task $TASK_ID in $MANIFEST" >&2
  exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$ROOT"
exec env PYTHONPATH=src:. "$ROOT/.venv-plot/bin/python" -u \
  experiments/combined_objective/run_cost_checkpoint_20seed_task.py \
  --method "$method" \
  --dataset "$dataset" \
  --seed "$seed" \
  --output-root "$OUTPUT_ROOT"
