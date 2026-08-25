#!/bin/bash
#SBATCH --job-name=gittins-cover-129
#SBATCH --account=torch_pr_790_general
#SBATCH --partition=cs
#SBATCH --array=0-199
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=analysis/vs/gittins_eta_pareto_coverage_grid129/slurm/%A_%a.out
#SBATCH --error=analysis/vs/gittins_eta_pareto_coverage_grid129/slurm/%A_%a.err

set -euo pipefail
cd /scratch/yh6415/project/agentopt-research
mkdir -p analysis/vs/gittins_eta_pareto_coverage_grid129/slurm
repo_root=/scratch/yh6415/project/agentopt-research
export PYTHONPATH="$repo_root/.venv/lib64/python3.12/site-packages:$repo_root/.venv/lib/python3.12/site-packages:$repo_root/src:$repo_root"

etas=(0.1 0.2 0.4 0.7 1.0)
benchmarks=(hotpotqa mathqa)
task=${SLURM_ARRAY_TASK_ID}
seed_offset=$((task % 20))
eta_index=$(((task / 20) % 5))
benchmark_index=$((task / 100))

/usr/bin/python3.12 experiments/combined_objective/run_radial_gittins_eta_sweep.py \
  --outdir analysis/vs/gittins_eta_pareto_coverage_grid129 \
  --benchmarks "${benchmarks[$benchmark_index]}" \
  --etas "${etas[$eta_index]}" \
  --first-seed $((42 + seed_offset)) \
  --seeds 1 \
  --grid-size 129 \
  --boundary-z-padding-min 2 \
  --continue-after-stop \
  --overwrite
