#!/usr/bin/env bash
#SBATCH --account=torch_pr_790_general
#SBATCH --job-name=complete-20seed-finalize
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yh6415/project/agentopt-research/analysis/complete_20seed_runs/logs/%A_finalize.out
#SBATCH --error=/scratch/yh6415/project/agentopt-research/analysis/complete_20seed_runs/logs/%A_finalize.err

set -euo pipefail
readonly ROOT=/scratch/yh6415/project/agentopt-research
cd "$ROOT"
exec python experiments/combined_objective/organize_20seed_results.py --require-complete
