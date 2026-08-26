#!/bin/bash
#SBATCH --job-name=scope-gittins
#SBATCH --account=torch_pr_790_general
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=analysis/vs/scope/slurm/%A_%a.out
#SBATCH --error=analysis/vs/scope/slurm/%A_%a.err

set -euo pipefail
cd /scratch/yh6415/project/agentopt-research

benchmarks=(
  bing_querylogs
  bird_dev
  bird_mini_dev
  restaurant_test
  restaurant_valid
  stackoverflow
)
benchmark="${benchmarks[$SLURM_ARRAY_TASK_ID]}"
z_padding_extra="${SCOPE_Z_PADDING_EXTRA:-0.0}"
start_seconds=$SECONDS

MPLCONFIGDIR=/tmp/mpl-scope-${SLURM_JOB_ID} \
  .venv-plot/bin/python -u experiments/combined_objective/plot/plot_scope_radial_vs_random.py \
  --scope-dir data/scope --outdir analysis/vs/scope --benchmark "$benchmark" \
  --seed 42 --seeds 1 --batch-size 4 --grid-size 129 --eta 1.0 \
  --z-padding-extra "$z_padding_extra" --x-axis cost

elapsed_seconds=$((SECONDS - start_seconds))
printf 'benchmark=%s job_id=%s walltime_seconds=%s\n' \
  "$benchmark" "$SLURM_JOB_ID" "$elapsed_seconds" \
  > "analysis/vs/scope/cache/${benchmark}_job_walltime.txt"
