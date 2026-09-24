# Estimated vs. actual Pareto values at 10% actual search spend

This diagnostic uses the **first available post-warm-start state** for which
actual cumulative API spend reaches 10% of the exhaustive cost on the same
complete question intersection. A batch can cross the threshold, so the reported actual
percentage can be slightly above 10%. This is a post-run checkpoint; it does
not change the policy or impose a stopping rule.
The runner keeps its existing `finite_lcb` recommendation rule and does not
change the Gittins acquisition target.

For configuration `a`, the run freezes the warm-start per-question price
estimate `m_a` after its four-question warm batch. At the checkpoint:

```text
estimated spend = actual warm-start spend
                + sum_over_adaptive_batches(batch_size_j * m_arm_j)
actual spend    = sum of realized charges for all evaluated cells
full spend      = sum of realized charges for the complete matrix

estimated search cost % = 100 * estimated spend / full spend
actual search cost %    = 100 * actual spend / full spend
```

The two percentages share the offline exhaustive denominator, isolating error
in predicted spend along the realized allocation path. The estimated percentage
is **conditional on that path**, not a forecast of the final number of batches
before the run starts. Do not substitute the Gittins effective pull cost: it
includes the search-cost multiplier and numerical cost binning.

The two Pareto panels use the same recommended configuration IDs at the exact
checkpoint. "Estimated" coordinates are the observed per-configuration sample
means at that point; "Actual" coordinates are the complete-matrix means used
only for evaluation. Selection itself follows the run's finite-LCB rule, so
estimated points are drawn individually rather than joined into a Pareto line.

## Run

From the repository root, after installing the project dependencies:

```bash
python -m experiments.combined_objective.run_two_direction_ablation \
  --pairs gauss_radau_accuracy_endpoint \
  --benchmarks stackoverflow bird_dev restaurant_valid

python -m experiments.combined_objective.plot.plot_estimated_actual_10pct \
  --pair gauss_radau_accuracy_endpoint \
  --benchmarks stackoverflow bird_dev restaurant_valid \
  --results experiments/combined_objective/results/two_direction_finite_lcb_raw_mean_estimated_actual_seed42_independent
```

The run writes `search_cost_checkpoint_10pct` into each benchmark's
`result.json` and adds estimated/actual checkpoint percentages to `summary.json`.
The plot command writes a paired PNG and PDF next to each `result.json`.
The example uses the three checked-in `data/scope/` benchmarks. Add
`hotpotqa mathqa` to both `--benchmarks` lists after placing their local lookup
pickles in `experiments/data/lookup/`.
Use a fresh `--outdir` for new experiments. Older compact `result.json` files
do not contain the full pull sequence needed to reconstruct this checkpoint.
