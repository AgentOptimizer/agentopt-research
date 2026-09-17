#!/bin/bash
#SBATCH --job-name=gauss-radau-aggregate
#SBATCH --account=torch_pr_790_general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/combined_objective/results/gauss_radau_8bench_20seed_independent/slurm/%j_aggregate.out
#SBATCH --error=experiments/combined_objective/results/gauss_radau_8bench_20seed_independent/slurm/%j_aggregate.err

set -euo pipefail
cd /scratch/yh6415/project/agentopt-research
: "${PARENT_ARRAY_JOB_ID:?PARENT_ARRAY_JOB_ID must be exported by the submission command}"

.venv-plot/bin/python -u \
  experiments/combined_objective/aggregate_gauss_radau_20seed.py \
  experiments/combined_objective/results/gauss_radau_8bench_20seed_independent \
  --slurm-array-job-id "${PARENT_ARRAY_JOB_ID}"
