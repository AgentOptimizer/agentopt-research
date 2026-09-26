# Experiment outputs

The current paper method is **CC-Gittins (two-axis)**. Its reference
configuration is G0 in [`gittins_ablation_v2.py`](../gittins_ablation_v2.py).
See the [experiment guide](../../README.md) for protocols, seeds, execution,
validation, and plotting commands.

## Output locations

- Canonical G0-G9 runs:
  `gittins_ablation_v2_8bench_20seed/<configuration>/seed-<seed>/`.
- Quick two-axis runs:
  `two_axis_finite_lcb_raw_mean_seed42_independent/exact_axes/<benchmark>/result.json`.
- Exploratory three-objective comparison:
  `three_objective_20seed/`, with summaries and plots under `comparison/`.

The anonymous source archive contains this guide, not historical run outputs,
caches, or logs. Regenerate outputs using the included frozen matrices. Use a
fresh output directory when changing inputs, seeds, or protocol parameters.

## Interpreting and comparing runs

The observation budget caps evaluated configuration-question cells. Reported
search cost is actual evaluation USD divided by exhaustive evaluation USD;
it is neither the observation fraction nor elapsed time. The policy can stop
before the observation cap. An early-stopped run has no new observations at
larger budget checkpoints; consult each comparison's carry-forward convention.

Compare runs only when matrix hashes, arm ordering, and full-data truth agree.
Finite-LCB recommendations use observed data. Full-matrix Pareto sets and
hypervolume values are offline diagnostics and do not enter acquisition.
Recommendations can contain uncompleted configurations; selected, evaluated,
completed, and recommended are different events.

The canonical collector checks protocol settings, truth, ordering, and native
task status. It expects eight benchmarks and seeds 42-61 for each of the ten
configurations. Quick-run outputs are not automatically part of this grid.
Generated task metadata retains parameters, file hashes, software versions,
and timing. Console logs or third-party tools can still contain local paths.
