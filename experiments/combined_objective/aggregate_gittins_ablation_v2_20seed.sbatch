#!/bin/bash
#SBATCH --job-name=gittins-ablation-v2-aggregate
#SBATCH --account=torch_pr_790_general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/combined_objective/results/gittins_ablation_v2_8bench_20seed/slurm/%j_aggregate.out
#SBATCH --error=experiments/combined_objective/results/gittins_ablation_v2_8bench_20seed/slurm/%j_aggregate.err

set -euo pipefail
cd /scratch/yh6415/project/agentopt-research
: "${PARENT_ARRAY_JOB_IDS:?PARENT_ARRAY_JOB_IDS must be exported by the submission command}"

IFS=':' read -r -a JOB_IDS <<< "${PARENT_ARRAY_JOB_IDS}"
.venv-plot/bin/python -u \
  experiments/combined_objective/aggregate_gittins_ablation_v2_20seed.py \
  experiments/combined_objective/results/gittins_ablation_v2_8bench_20seed \
  --slurm-array-job-ids "${JOB_IDS[@]}"
