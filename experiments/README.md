# Benchmark experiments

The current paper method is **CC-Gittins (two-axis)**. Its canonical definition
is G0 in [`gittins_ablation_v2.py`](combined_objective/gittins_ablation_v2.py).
The experiment runner defaults to the exact deployment and quality axes,
`((0, 1), (1, 0))`, in that order. Gauss-Radau and the earlier direction grids
are retired.

The current protocol uses raw mean USD costs, real per-arm continuation costs,
1:1 round-robin scheduling, independent per-direction eta decay, and finite-LCB
recommendations with beta=1. Each arm has an independent question order,
including its four-question warm batch. The axes share observations. Eta starts
at 1 and halves on a direction stop. The DP grid is 129 in each dimension.

## Setup and data

Run all commands below from the repository root with Python 3.10+:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

HotpotQA and MathQA use the committed complete 100-configuration × 200-question
matrices in `data/hotpotqa/` and `data/mathqa/`. Other protocol benchmarks use
`data/scope/{restaurant_test,stackoverflow,bird_dev,restaurant_valid,bing_querylogs,bird_mini_dev}/`.
A full eight-benchmark run uses all of these committed directories. See
[`data/README.md`](../data/README.md) for input formats.

The algorithm reads frozen matrices; these commands make no model API calls.
Replay results and DP caches default to `experiments/combined_objective/results/`;
the ablation plotters have separate output paths, described below.
Do not mix results from different matrix versions: saved input hashes, model
ordering, and full-data truth must agree across each comparison.

## Run the current method

```bash
.venv/bin/python -m experiments.combined_objective.run_two_direction_ablation \
  --benchmarks hotpotqa mathqa
```

This defaults to `exact_axes`, seed 42, and an observation-count budget of 100%.
The budget limits the number of evaluated configuration-question cells; it is
not a dollar-spend target, and the policy can stop before exhausting it.
Reported search-cost percentages and the 10% checkpoint use actual USD instead.
Output:
`experiments/combined_objective/results/two_axis_finite_lcb_raw_mean_seed42_independent/exact_axes/<benchmark>/result.json`.
Use a fresh `--outdir` when changing the seed, protocol, or input data.
This quick-run directory is separate from the canonical ablation grid. Use the
task wrapper below to produce G0–G9 results with the labels and metadata expected
by the collector.

The file includes finite-LCB membership checkpoints, raw truth for offline
scoring, direction eta events, timing, and the estimated/actual dollar-spend
checkpoint. The latter is unavailable when a run stops before 10%; that does
not invalidate the run. See [checkpoint semantics](combined_objective/estimated_actual_10pct.md).

```bash
.venv/bin/python -m experiments.combined_objective.plot.plot_two_direction_ablation
.venv/bin/python -m experiments.combined_objective.plot.plot_estimated_actual_10pct \
  --benchmarks hotpotqa mathqa
```

If you changed the runner's `--outdir`, pass the same directory as `--results`
to both plotters. The estimated/actual plot requires a non-null
`search_cost_checkpoint_10pct`; restrict its `--benchmarks` to runs that reached
10% actual spend. An earlier stop remains valid for the other diagnostics.

## Three-objective latency pilot

The separate latency pilot uses the complete MathQA and HotpotQA matrices and
the fixed direction order **Q, (Q+L)/2, L, D**, with coordinates `(Q, L, D)`:
`(1,0,0)`, `(0.5,0.5,0)`, `(0,1,0)`, `(0,0,1)`. Q maximizes accuracy; L
minimizes mean latency in seconds; D minimizes mean deployment cost in USD.
The midpoint denotes a radial direction in normalized Q–L reward space, using
the existing radial Gittins utility; it is not an arithmetic average of
accuracy and seconds. Each endpoint uses the existing scalar DP.

```bash
.venv/bin/python -m experiments.combined_objective.run_three_objective_gittins \
  --benchmarks mathqa hotpotqa --seed 42
```

This exploratory run keeps the two-objective paper protocol above unchanged.
It uses independent per-arm question orders, four-question warm and adaptive
batches, shared observations across all objectives, round-robin scheduling,
independent eta decay from 1 by 0.5, and a 129-point DP grid. Latency and USD
posteriors are fitted in raw mean units. Their affine reward scales and noise
estimates are frozen from the warm observations. All four directions use
expected **USD** continuation costs; latency is an objective, not search spend.

Recommendations use the three-dimensional finite-test conservative frontier:
`(Q_mean - std, L_mean + std, D_mean + std)`. Full-matrix values are used only
for offline scoring. Precision and recall compare with the true 3D frontier;
hypervolume uses `(Q, R_L/(R_L+mean_L), R_D/(R_D+mean_D))`, with positive
full-data median scales for evaluation only and a zero reference point.
The four acquisition directions need not recover every 3D Pareto configuration.

Results, per-batch recommendations, direction traces, input/source hashes,
and plots are written under
`combined_objective/results/three_objective_ql_midpoint_d/seed_42/`.
The 100% observation budget is an upper limit; the policy can stop earlier.
Saved 5%/10%/20%/30%/50% checkpoints use the latest completed batch at or below
the target fraction of exhaustive USD spend, and are unavailable if the run
never reaches the target. To regenerate figures from saved results:

```bash
.venv/bin/python -m experiments.combined_objective.run_three_objective_gittins --plot-only
```

Use a fresh `--outdir` for changed inputs, code, seeds, or protocol parameters.

For a matched 20-seed comparison (42–61) on both QA datasets:

```bash
.venv/bin/python -m experiments.combined_objective.run_three_objective_gittins_multiseed
.venv/bin/python -m experiments.combined_objective.run_three_objective_gittins_multiseed \
  --variant axes_only
.venv/bin/python -m experiments.combined_objective.compare_three_objective_baselines \
  --axes-root experiments/combined_objective/results/three_objective_axes_20seed
```

The batch runner stores compact per-batch recommendation/metric trajectories
under `combined_objective/results/three_objective_20seed/` and can reuse the
validated existing seed-42 pilot. The `axes_only` variant uses exactly **Q, L, D**
in that order and saves separately under `three_objective_axes_20seed/`; it never
reuses the four-direction seed-42 pilot. All three directions use scalar DP.
Paired runs share warm observations and calibration, question orders, batch
sizes, USD penalties, eta schedules, and recommendation rules. Removing the
midpoint changes the round-robin visits from four directions to three.
The comparison verifies the paired protocol and calibration, and adds 20 seeds
each of:

- `random_questions`: a nested shared random question prefix evaluated on every
  configuration, recommending its empirical three-objective Pareto set.
- `random_configurations`: a nested random configuration prefix, evaluating
  every question for each chosen configuration and recommending its empirical
  three-objective Pareto set.

The included methods use the same actual-USD budget checkpoints and full-data
evaluation coordinates. A checkpoint takes the latest completed unit at or
below its budget; it never interpolates or includes a future unit. An early
policy stop retains its final recommendation at larger budgets without adding
spend. The comparison covers the methods' original recommendation rules:
CC-Gittins finite LCB versus empirical means for the random baselines.

Primary plots show normalized HV, GD, and IGD, with means and 10–90% seed
percentile bands for every method. A separate plot expands small HV regrets.
GD/IGD retain the repository convention of filtering the returned set to its
nondominated subset in true evaluation coordinates before taking mean nearest
Euclidean distances. GD can therefore be zero despite missing true-front
regions; read it together with IGD and HV. Figures, checkpoints, and a report
are saved under `three_objective_20seed/comparison/`. The four-direction Gittins
is orange; the Q/L/D-only variant is blue with a dashed line. Omit `--axes-root`
to reproduce the original comparison without the axes-only variant.

To add the Pareto identification baselines in the same three-objective space,
run EGE-SH, EGE-SR, APE-k, and qNEHVI with matching seeds, then pass their
saved directory to the comparison:

```bash
.venv/bin/python -m experiments.combined_objective.run_three_objective_pareto_baselines \
  --benchmarks mathqa hotpotqa --seeds $(seq 42 61)
.venv/bin/python -m experiments.combined_objective.compare_three_objective_baselines \
  --axes-root experiments/combined_objective/results/three_objective_axes_20seed \
  --pareto-root experiments/combined_objective/results/three_objective_pareto_baselines
```

The Pareto runner reuses the existing EGE/APE/qNEHVI sampling rules with
`(Q,L,D)` observations. It recomputes recommendation checkpoints after the
run, using the same three-dimensional HV, GD, IGD, precision, and recall as
Gittins and random search; oracle diagnostics never feed acquisition. Search
spend is the sum of observed USD costs;
comparison checkpoints take the latest completed pull at or below each actual
USD target. Full-data metric scales are evaluation-only and never enter
acquisition. The observation fraction remains a cap on evaluated matrix cells,
so use its default `1.0` for the full nested cost trajectory. APE-k retains the
existing fixed-budget sampler: its paper's `k` stopping condition is not
implemented. Runs are saved under `three_objective_pareto_baselines/seed_<n>/`
with input and source hashes; changed protocols require a new output directory.
For a smaller run, pass the same method names to `--methods` on the runner and
`--pareto-methods` on the comparison command.

## Retained ablations

The single source of truth is `gittins_ablation_v2.CONFIGURATIONS` and `FAMILIES`.
Only these ten configurations belong to the current experiment protocol:

| ID | Configuration | Change from G0 |
|---|---|---|
| G0 | `g0_current_gittins` | Exact axes, real costs, eta decay, round-robin |
| G1 | `g1_q_only_real_cost` | Quality axis only |
| G2 | `g2_q_only_unit_cost` | Quality axis, unit acquisition costs |
| G3 | `g3_two_axis_unit_cost` | Two axes, unit acquisition costs |
| G4 | `g4_d_only_real_cost` | Deployment axis only |
| G5 | `g5_axes_midpoint_real_cost` | Axes plus `(0.5, 0.5)` |
| G6 | `g6_five_directions_real_cost` | Five evenly spaced directions |
| G7 | `g7_fixed_eta_no_stop` | Fixed eta; force continuation without stopping |
| G8 | `g8_q_then_d` | Quality axis until its eta floor, then deployment |
| G9 | `g9_d_then_q` | Deployment axis until its eta floor, then quality |

The four comparison families are `cost_mechanism`, `directions`, `continuation`,
and `scheduler`. Unit costs change acquisition penalties; recorded evaluation
spend remains actual USD. Sequential schedulers revisit an axis after shared
observations make it eligible again.

For an individual canonical run, the task wrapper fixes every protocol option:

```bash
.venv/bin/python -m experiments.combined_objective.run_gittins_ablation_v2_slurm_task \
  --configuration g0_current_gittins --task-id 20 --dry-run
```

Remove `--dry-run` to execute. Each configuration has 160 tasks: eight benchmarks
in `BENCHMARKS` order × seeds 42–61; task 20 is HotpotQA, seed 42. G0 can run
natively. Other configurations validate their dataset against G0, so produce G0
first when starting without prior results. `--output-root` selects a separate
experiment root.

For Slurm, submit from the repository root after creating the log directory:

```bash
mkdir -p experiments/combined_objective/results/gittins_ablation_v2_8bench_20seed/slurm
sbatch experiments/combined_objective/run_gittins_ablation_v2_20seed.sbatch g0_current_gittins
```

After G0 finishes, submit the same script for each retained G1–G9 configuration.
Set your cluster account with `sbatch --account=...`; set `AGENTOPT_PYTHON` if the
Python environment is elsewhere. Existing historical `g2_exact_axes` results
can serve as G0 when no native G0 result exists. This is a compatibility path for
saved two-axis results, not an additional algorithm.

## Validate and plot the ablation grid

```bash
.venv/bin/python -m experiments.combined_objective.aggregate_gittins_ablation_v2_20seed
.venv/bin/python -m experiments.combined_objective.plot_gittins_ablation_v2_20seed
.venv/bin/python -m experiments.combined_objective.plot_gittins_direction_runtime_v2_20seed
```

The collector expects 1,600 runs and checks protocol fields, full-data truth,
arm ordering, total exhaustive USD cost, and successful native-task metadata.
It requires a G0 seed-42 truth reference for each benchmark before it can report
missing or invalid runs. It does not compare saved input hashes, so keep all
configurations on the same input version. Use `--families directions` on the ablation
plotter to render just a completed family. Plots show the same reciprocal mean
USD evaluation space as the paper; it is distinct from the raw-cost acquisition
model. Runtime plots compare the two-, three-, and five-direction policies.

The collector writes `aggregate/` under its results root. The ablation plotters
default to `analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_figures/gittins_g2_main_figures/ablation/`,
with runtime figures under `runtime/`. The `g2` directory name preserves the
historical paper output location; the current control is G0.

For a separate experiment directory, pass that same directory to every reader:

```bash
.venv/bin/python -m experiments.combined_objective.aggregate_gittins_ablation_v2_20seed \
  path/to/results
.venv/bin/python -m experiments.combined_objective.plot_gittins_ablation_v2_20seed \
  --results-root path/to/results --output path/to/figures
.venv/bin/python -m experiments.combined_objective.plot_gittins_direction_runtime_v2_20seed \
  --results-root path/to/results --output-dir path/to/figures/runtime
```

## Baselines and shared libraries

Pareto baselines remain in `offline_pareto_baselines.py` and
`offline_multiobjective_random_search.py`; accuracy-only baselines remain under
`single_objective/`. The replay engine and recommendation/plot helpers retain
library support needed by these comparisons and their regression tests. The
old scheduler, eta, and recommendation experiment commands have been removed.

Output locations and interpretation are described in
[results/README.md](combined_objective/results/README.md). The anonymous archive
includes current source and complete replay matrices, not historical results
or local lookup pickles. See the [reviewer instructions](../REPRODUCIBILITY.md)
for a bounded smoke test and the scope of the tested environment.
