#!/bin/bash
#SBATCH --job-name=qa-random-sheets
#SBATCH --array=0-39
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/vs/random_search_20seeds/slurm/%A_%a.out
#SBATCH --error=analysis/vs/random_search_20seeds/slurm/%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

benchmarks=(hotpotqa mathqa)
benchmark_index=$((SLURM_ARRAY_TASK_ID / 20))
seed=$((42 + SLURM_ARRAY_TASK_ID % 20))
benchmark="${benchmarks[$benchmark_index]}"
version="${RANDOM_VERSION:-random_configurations}"
start_seconds=$SECONDS

MPLCONFIGDIR=/tmp/mpl-qa-random-${SLURM_JOB_ID} \
  "${AGENTOPT_PYTHON:-.venv-plot/bin/python}" -u \
  experiments/combined_objective/plot/plot_multiobjective_random_search.py \
  --outdir experiments/combined_objective/results/qa_random_20seeds \
  --benchmarks "$benchmark" --seed "$seed" \
  --contact-sheet-only --version "$version"

elapsed_seconds=$((SECONDS - start_seconds))
mkdir -p analysis/vs/random_search_20seeds/walltime
printf 'benchmark=%s version=%s seed=%s job_id=%s walltime_seconds=%s\n' \
  "$benchmark" "$version" "$seed" "$SLURM_JOB_ID" "$elapsed_seconds" \
  > "analysis/vs/random_search_20seeds/walltime/${benchmark}_${version}_seed-${seed}.txt"
