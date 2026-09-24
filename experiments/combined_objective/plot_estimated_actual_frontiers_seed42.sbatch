#!/usr/bin/env bash
#SBATCH --account=torch_pr_790_general
#SBATCH --job-name=plot-est-actual
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yh6415/project/agentopt-research/analysis/estimated_actual_frontiers_seed42/logs/plot_%j.out
#SBATCH --error=/scratch/yh6415/project/agentopt-research/analysis/estimated_actual_frontiers_seed42/logs/plot_%j.err

set -euo pipefail

readonly ROOT=/scratch/yh6415/project/agentopt-research
readonly OUTPUT_ROOT="$ROOT/analysis/estimated_actual_frontiers_seed42"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/agentopt-mpl-${SLURM_JOB_ID:-local}"

mkdir -p "$OUTPUT_ROOT/logs" "$MPLCONFIGDIR"
cd "$ROOT"
exec env PYTHONPATH=src:. "$ROOT/.venv-plot/bin/python" -u \
  experiments/combined_objective/plot_estimated_actual_frontiers_seed42.py \
  --results "$OUTPUT_ROOT" \
  --targets 10 30
