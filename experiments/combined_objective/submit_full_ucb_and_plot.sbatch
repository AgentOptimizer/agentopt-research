#!/usr/bin/env bash
#SBATCH --job-name=ucb-full-20seed
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=analysis/paper_20seed_method_comparison/logs/ucb-full-%j.out
#SBATCH --error=analysis/paper_20seed_method_comparison/logs/ucb-full-%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$repo_root"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export PYTHONPATH="src:."
mkdir -p "$MPLCONFIGDIR"

.venv-plot/bin/python \
  experiments/combined_objective/run_ucb_20seed_comparison.py \
  --outdir analysis/paper_20seed_method_comparison \
  --workers "${SLURM_CPUS_PER_TASK:-16}" \
  --benchmarks hotpotqa mathqa gpqa bfcl \
  --force

.venv-plot/bin/python \
  experiments/combined_objective/plot_paper_20seed_comparison.py \
  --skip-distances

.venv-plot/bin/python \
  experiments/combined_objective/plot_paper_20seed_comparison.py \
  --all-benchmarks \
  --skip-distances

echo "Full-budget UCB data and paper figures updated successfully."
