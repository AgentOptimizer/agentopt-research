#!/bin/bash
#SBATCH --job-name=gittins-ablation-plot
#SBATCH --account=torch_pr_790_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/combined_objective/results/gittins_ablation_8bench_20seed/slurm/%j_plot.out
#SBATCH --error=experiments/combined_objective/results/gittins_ablation_8bench_20seed/slurm/%j_plot.err

set -euo pipefail
cd /scratch/yh6415/project/agentopt-research

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl-gittins-ablation-plot-${SLURM_JOB_ID}"

.venv-plot/bin/python -u \
  experiments/combined_objective/plot_gittins_ablation_20seed.py \
  --strict "$@"
