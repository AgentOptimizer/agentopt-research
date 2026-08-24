#!/usr/bin/env bash
#SBATCH --job-name=plot-gittins-eta-hv
#SBATCH --account=torch_pr_790_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=analysis/vs/gittins_eta_hv_curves/slurm/plot_%j.out
#SBATCH --error=analysis/vs/gittins_eta_hv_curves/slurm/plot_%j.err

set -euo pipefail

repo_root=/scratch/yh6415/project/agentopt-research
cd "$repo_root"
export MPLCONFIGDIR=/tmp/matplotlib-agentopt-"$SLURM_JOB_ID"
export PYTHONPATH="$repo_root/.venv/lib64/python3.12/site-packages:$repo_root/.venv/lib/python3.12/site-packages:$repo_root/src:$repo_root"
/usr/bin/python3.12 experiments/combined_objective/plot_radial_gittins_eta_hv_curves.py \
  --indir analysis/vs/gittins_eta_hv_curves
