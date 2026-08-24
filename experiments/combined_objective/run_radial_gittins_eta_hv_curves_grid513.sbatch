#!/usr/bin/env bash
#SBATCH --job-name=gittins-eta-513
#SBATCH --account=torch_pr_790_general
#SBATCH --partition=cs
#SBATCH --array=0-199
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=analysis/vs/gittins_eta_hv_curves_grid513/slurm/%A_%a.out
#SBATCH --error=analysis/vs/gittins_eta_hv_curves_grid513/slurm/%A_%a.err

set -euo pipefail

repo_root=/scratch/yh6415/project/agentopt-research
etas=(0.1 0.2 0.4 0.7 1.0)
benchmarks=(hotpotqa mathqa)
benchmark_index=$((SLURM_ARRAY_TASK_ID / 100))
within_benchmark=$((SLURM_ARRAY_TASK_ID % 100))
eta_index=$((within_benchmark / 20))
seed=$((42 + within_benchmark % 20))

cd "$repo_root"
export MPLCONFIGDIR=/tmp/matplotlib-agentopt-"$SLURM_JOB_ID"
export PYTHONPATH="$repo_root/.venv/lib64/python3.12/site-packages:$repo_root/.venv/lib/python3.12/site-packages:$repo_root/src:$repo_root"
/usr/bin/python3.12 experiments/combined_objective/run_radial_gittins_eta_sweep.py \
  --outdir analysis/vs/gittins_eta_hv_curves_grid513 \
  --benchmarks "${benchmarks[$benchmark_index]}" \
  --etas "${etas[$eta_index]}" \
  --first-seed "$seed" \
  --seeds 1 \
  --continue-after-stop
