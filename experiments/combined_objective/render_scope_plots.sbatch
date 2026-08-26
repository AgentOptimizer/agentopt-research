#!/bin/bash
#SBATCH --job-name=scope-render
#SBATCH --account=torch_pr_790_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/vs/scope/slurm/%j_render.out
#SBATCH --error=analysis/vs/scope/slurm/%j_render.err

set -euo pipefail
cd /scratch/yh6415/project/agentopt-research
MPLCONFIGDIR=/tmp/mpl-scope-${SLURM_JOB_ID} \
  .venv-plot/bin/python -u experiments/combined_objective/plot/plot_scope_radial_vs_random.py \
  --scope-dir data/scope --outdir analysis/vs/scope --render-only \
  --seed 42 --seeds 1 --batch-size 4 --grid-size 129 --eta 1.0 --x-axis cost
