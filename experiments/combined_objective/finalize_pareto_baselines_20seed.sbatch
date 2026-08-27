#!/usr/bin/env bash
#SBATCH --job-name=pareto-baseline-plot
#SBATCH --account=torch_pr_790_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=analysis/paper_20seed_method_comparison/logs/pareto-baseline-plot-%j.out
#SBATCH --error=analysis/paper_20seed_method_comparison/logs/pareto-baseline-plot-%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$repo_root"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export PYTHONPATH="src:."
mkdir -p "$MPLCONFIGDIR"

.venv-plot/bin/python experiments/combined_objective/aggregate_pareto_baselines_20seed.py
.venv-plot/bin/python experiments/combined_objective/plot_pareto_identification_baselines.py \
  --from-csv analysis/paper_20seed_method_comparison/pareto_baselines/cost_trajectory.csv \
  --seeds 20 \
  --gittins-trajectory analysis/vs/radial_gittins_post_stop/cost_trajectory.csv \
  --outdir analysis/paper_20seed_method_comparison/pareto_baselines
.venv-plot/bin/python experiments/combined_objective/plot_paper_20seed_comparison.py
