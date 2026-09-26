#!/bin/bash
#SBATCH --job-name=gittins-ablation-v2-plot
#SBATCH --account=torch_pr_790_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/combined_objective/results/gittins_ablation_v2_8bench_20seed/slurm/%j_plot.out
#SBATCH --error=experiments/combined_objective/results/gittins_ablation_v2_8bench_20seed/slurm/%j_plot.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${AGENTOPT_PYTHON:-.venv/bin/python}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl-gittins-ablation-v2-plot-${SLURM_JOB_ID}"

plot_args=()
if (( $# > 0 )); then
  plot_args=(--families "$@")
fi

"${PYTHON_BIN}" -u \
  experiments/combined_objective/plot_gittins_ablation_v2_20seed.py \
  "${plot_args[@]}"
