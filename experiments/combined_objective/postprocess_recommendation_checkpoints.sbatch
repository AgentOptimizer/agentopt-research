#!/usr/bin/env bash
#SBATCH --account=torch_pr_790_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

readonly ROOT=/scratch/yh6415/project/agentopt-research
cd "$ROOT"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
env PYTHONPATH=src:. "$ROOT/.venv-plot/bin/python" \
  experiments/combined_objective/build_recommendation_checkpoint_index.py
env PYTHONPATH=src:. "$ROOT/.venv-plot/bin/python" \
  experiments/combined_objective/plot_baseline_20seed_frequency_frontiers.py
env PYTHONPATH=src:. "$ROOT/.venv-plot/bin/python" \
  experiments/combined_objective/plot_20seed_eight_dataset_curves.py
