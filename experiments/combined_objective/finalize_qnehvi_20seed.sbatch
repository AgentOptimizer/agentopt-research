#!/usr/bin/env bash
#SBATCH --job-name=qnehvi-plot
#SBATCH --account=torch_pr_790_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=analysis/paper_20seed_method_comparison/logs/qnehvi-plot-%j.out
#SBATCH --error=analysis/paper_20seed_method_comparison/logs/qnehvi-plot-%j.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$repo_root"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export PYTHONPATH="src:."
mkdir -p "$MPLCONFIGDIR"
.venv-plot/bin/python experiments/combined_objective/aggregate_qnehvi_20seed.py
.venv-plot/bin/python experiments/combined_objective/plot_paper_20seed_comparison.py

