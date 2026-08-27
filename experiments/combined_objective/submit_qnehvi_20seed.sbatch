#!/usr/bin/env bash
#SBATCH --job-name=qnehvi-pareto
#SBATCH --account=torch_pr_790_general
#SBATCH --array=0-79
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=analysis/paper_20seed_method_comparison/logs/qnehvi-%A_%a.out
#SBATCH --error=analysis/paper_20seed_method_comparison/logs/qnehvi-%A_%a.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$repo_root"

seeds=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 59 60 61 62)
benchmarks=(hotpotqa mathqa)
task_id="${SLURM_ARRAY_TASK_ID}"
mode_index=$((task_id / 40))
within_mode=$((task_id % 40))
benchmark="${benchmarks[$((within_mode / 20))]}"
seed="${seeds[$((within_mode % 20))]}"
if (( mode_index == 0 )); then
  mode="full"
  budget="1.0"
else
  mode="10pct"
  budget="0.1"
fi
outdir="analysis/paper_20seed_method_comparison/qnehvi/${mode}/raw/${benchmark}/seed-${seed}"

export PYTHONPATH="src:."
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$outdir"
.venv-plot/bin/python experiments/combined_objective/offline_pareto_baselines.py \
  --pickle "experiments/data/lookup/${benchmark}_lookup.pkl" \
  --methods qnehvi --seeds 1 --base-seed "$seed" \
  --batch-size 4 --budget-fraction "$budget" \
  --qnehvi-mc-samples 64 --qnehvi-refit-every 32 \
  --outdir "$outdir"

