#!/usr/bin/env bash
#SBATCH --job-name=pareto-base-10pct
#SBATCH --account=torch_pr_790_general
#SBATCH --array=0-79
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/paper_20seed_method_comparison/logs/pareto-base-10pct-%A_%a.out
#SBATCH --error=analysis/paper_20seed_method_comparison/logs/pareto-base-10pct-%A_%a.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$repo_root"
seeds=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 59 60 61 62)
methods=(ege_sh ape_k)
benchmarks=(hotpotqa mathqa)
task_id="${SLURM_ARRAY_TASK_ID}"
seed="${seeds[$((task_id % 20))]}"
method="${methods[$(((task_id / 20) % 2))]}"
benchmark="${benchmarks[$((task_id / 40))]}"
outdir="analysis/paper_20seed_method_comparison/pareto_baselines_10pct/raw/${benchmark}/${method}/seed-${seed}"
export PYTHONPATH="src:."
mkdir -p "$outdir"
.venv-plot/bin/python experiments/combined_objective/offline_pareto_baselines.py \
  --pickle "experiments/data/lookup/${benchmark}_lookup.pkl" \
  --methods "$method" --seeds 1 --base-seed "$seed" \
  --batch-size 4 --budget-fraction 0.1 --outdir "$outdir"

