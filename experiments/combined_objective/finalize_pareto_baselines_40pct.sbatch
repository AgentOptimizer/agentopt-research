#!/usr/bin/env bash
#SBATCH --job-name=pareto-base-2x6
#SBATCH --account=torch_pr_790_general
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/paper_20seed_method_comparison/logs/pareto-base-2x6-%j.out
#SBATCH --error=analysis/paper_20seed_method_comparison/logs/pareto-base-2x6-%j.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$repo_root"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export PYTHONPATH="src:."
mkdir -p "$MPLCONFIGDIR"
.venv-plot/bin/python experiments/combined_objective/aggregate_pareto_baselines_40pct.py
.venv-plot/bin/python experiments/combined_objective/plot_paper_20seed_comparison.py

