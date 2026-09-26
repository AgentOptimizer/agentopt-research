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
A full eight-benchmark run requires those directories locally. See
[`data/README.md`](../data/README.md) for input formats.

The algorithm reads frozen matrices; these commands make no model API calls.
Results and DP caches are written under `experiments/combined_objective/results/`.
Do not mix results from different matrix versions: saved input hashes, model
ordering, and full-data truth must agree across each comparison.

## Run the current method

```bash
.venv/bin/python -m experiments.combined_objective.run_two_direction_ablation \
  --benchmarks hotpotqa mathqa
```

This defaults to `exact_axes`, seed 42, and the full observation budget. Output:
`experiments/combined_objective/results/two_axis_finite_lcb_raw_mean_seed42_independent/exact_axes/<benchmark>/result.json`.
Use a fresh `--outdir` when changing the seed, protocol, or input data.

The file includes finite-LCB membership checkpoints, raw truth for offline
scoring, direction eta events, timing, and the estimated/actual dollar-spend
checkpoint. The latter is unavailable when a run stops before 10%; that does
not invalidate the run. See [checkpoint semantics](combined_objective/estimated_actual_10pct.md).

```bash
.venv/bin/python -m experiments.combined_objective.plot.plot_two_direction_ablation
.venv/bin/python -m experiments.combined_objective.plot.plot_estimated_actual_10pct \
  --benchmarks hotpotqa mathqa
```

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

The collector validates all 1,600 runs, checks configuration and input agreement,
and reports missing/invalid results. Use `--families directions` on the ablation
plotter to render just a completed family. Plots show the same reciprocal mean
USD evaluation space as the paper; it is distinct from the raw-cost acquisition
model. Runtime plots compare the two-, three-, and five-direction policies.

## Baselines and shared libraries

Pareto baselines remain in `offline_pareto_baselines.py` and
`offline_multiobjective_random_search.py`; accuracy-only baselines remain under
`single_objective/`. The replay engine and recommendation/plot helpers retain
library support needed by these comparisons and their regression tests. The
old scheduler, eta, and recommendation experiment commands have been removed.

Historical results are indexed in [results/README.md](combined_objective/results/README.md).
They do not define current defaults; reproduce retired protocols from their
corresponding Git versions.
