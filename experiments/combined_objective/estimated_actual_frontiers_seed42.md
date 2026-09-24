# Seed-42 estimated-versus-actual Pareto appendix

This experiment compares six methods on MathQA and Stack Overflow:

1. two-direction Gauss-Radau Radial Gittins;
2. EGE-SH;
3. APE-k;
4. qNEHVI;
5. random configurations;
6. random questions.

The two paper figures freeze the first completed policy update that reaches or
crosses 10% and 30% of exhaustive realized USD search cost. Each figure has two
dataset subfigures side by side. Within each dataset, six method rows show the
estimated coordinates and complete-matrix actual coordinates of the same
recommended configuration IDs.

Every recommendation-membership change is saved in a self-contained compressed
JSON artifact. Each frame contains arm IDs, model names, sample counts, online
sample means, complete-matrix means, and estimated/actual cumulative spend, so
an animation can be rendered without replaying a selector. The generated
`animation_manifest.json` indexes all 12 artifacts.

Radial Gittins uses its internal cost estimate: realized warm-up spend followed
by frozen per-arm warm-up mean prices. The other methods have no internal
search-cost predictor, so their diagnostic estimate is causal: before each
pull, use that arm's observed mean cost, fall back to the pooled observed mean
for an unseen arm, and initialize the very first pull at realized cost. The CSV
exports identify the estimator used in every row.

Run all 12 cells in parallel and render both figures after success:

```bash
array_job=$(sbatch --parsable \
  experiments/combined_objective/run_estimated_actual_frontiers_seed42.sbatch)
sbatch --dependency="afterok:$array_job" \
  experiments/combined_objective/plot_estimated_actual_frontiers_seed42.sbatch
```

To run or redraw directly:

```bash
PYTHONPATH=src:. .venv-plot/bin/python \
  experiments/combined_objective/run_estimated_actual_frontiers_seed42.py \
  --dataset mathqa --method ege_sh --seed 42

PYTHONPATH=src:. .venv-plot/bin/python \
  experiments/combined_objective/plot_estimated_actual_frontiers_seed42.py \
  --targets 10 30
```

Outputs are under `analysis/estimated_actual_frontiers_seed42/`.
