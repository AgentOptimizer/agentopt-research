#!/usr/bin/env bash
#SBATCH --account=torch_pr_790_general
#SBATCH --job-name=est-actual-s42
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yh6415/project/agentopt-research/analysis/estimated_actual_frontiers_seed42/logs/%A_%a.out
#SBATCH --error=/scratch/yh6415/project/agentopt-research/analysis/estimated_actual_frontiers_seed42/logs/%A_%a.err

set -euo pipefail

readonly ROOT=/scratch/yh6415/project/agentopt-research
readonly OUTPUT_ROOT="$ROOT/analysis/estimated_actual_frontiers_seed42"
readonly TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
readonly METHODS=(radial_gittins ege_sh ape_k qnehvi random_configurations random_questions)
readonly DATASETS=(mathqa stackoverflow)
readonly METHOD="${METHODS[$((TASK_ID % 6))]}"
readonly DATASET="${DATASETS[$((TASK_ID / 6))]}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/agentopt-mpl-${SLURM_JOB_ID:-local}"

mkdir -p "$OUTPUT_ROOT/logs" "$MPLCONFIGDIR"
cd "$ROOT"
exec env PYTHONPATH=src:. "$ROOT/.venv-plot/bin/python" -u \
  experiments/combined_objective/run_estimated_actual_frontiers_seed42.py \
  --dataset "$DATASET" \
  --method "$METHOD" \
  --seed 42 \
  --output-root "$OUTPUT_ROOT"
