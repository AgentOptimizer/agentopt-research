# CC-Gittins: anonymous supplementary code

Offline model-selection algorithms and frozen benchmark matrices for accuracy,
deployment cost, latency, and token usage. The current paper method is
**Cost-Coupled Gittins (CC-Gittins, two-axis)**.

## Current method

G0 uses the exact deployment and quality axes, `((0, 1), (1, 0))`, with real
per-arm continuation costs, 1:1 round-robin scheduling, independent per-direction
eta decay, and finite-LCB recommendations. Gauss-Radau is retired.

The canonical G0–G9 configurations and four ablation families are defined in
[`gittins_ablation_v2.py`](experiments/combined_objective/gittins_ablation_v2.py).
Start with the [reviewer instructions](REPRODUCIBILITY.md) for the tested
environment, a small offline replay, and the contents of the anonymous ZIP.
See the [experiment guide](experiments/README.md) for full runs, Slurm submission,
validation, and plots. Use its experiment runner for the paper protocol; the
lower-level replay APIs also support historical settings.

## Setup and quick start

Requires Python 3.10+. Run from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -c constraints-tested.txt -e '.[radial,plots]' pytest
.venv/bin/python -m pytest

# Current two-axis method, seed 42, on the committed complete QA matrices
.venv/bin/python -m experiments.combined_objective.run_two_direction_ablation \
  --benchmarks hotpotqa mathqa
```

No model API calls or local lookup pickles are needed for this replay.
Results go to
`experiments/combined_objective/results/two_axis_finite_lcb_raw_mean_seed42_independent/exact_axes/<benchmark>/result.json`.
The full observation budget is an upper limit; the policy may stop earlier.
For the canonical eight-benchmark × 20-seed ablation grid, use the task wrapper
in the [experiment guide](experiments/README.md#retained-ablations).

An exploratory three-objective run adds measured latency on MathQA and
HotpotQA, using directions Q, (Q+L)/2, L, D:

```bash
.venv/bin/python -m experiments.combined_objective.run_three_objective_gittins \
  --benchmarks mathqa hotpotqa --seed 42
```

See the [latency pilot protocol](experiments/README.md#three-objective-latency-pilot)
for objective scaling, three-dimensional recommendations, and output plots.

The Python import name `agentopt` is retained for compatibility.
`import agentopt` needs only `numpy` and `pydantic`. The `dev` extra installs
SciPy, JAX, Matplotlib, and pytest for the current replay and tests. Other extras
cover Bayesian baselines (`bayesian`) and optional dependencies; see
[pyproject.toml](pyproject.toml). Legacy scalar-Gittins source modules are
excluded from the anonymous ZIP and are not needed by CC-Gittins.

## Data

| Benchmark | Current replay input | Configurations × questions |
|---|---|---|
| HotpotQA | [`data/hotpotqa/`](data/hotpotqa/) | 100 × 200 (10 planners × 10 solvers) |
| MathQA | [`data/mathqa/`](data/mathqa/) | 100 × 200 (10 answer models × 10 critics) |
| Six SCOPE benchmarks | [`data/scope/`](data/scope/) | Varies by benchmark |

Each QA directory contains six complete per-question CSV matrices: accuracy,
USD cost, latency in seconds, input tokens, output tokens, and total tokens,
plus metadata. These matrices have one row per configuration and one column
per question. See [data/README.md](data/README.md) for formats and regeneration.

The anonymous ZIP includes the complete QA and six SCOPE datasets. Sparse
duplicates, legacy lookup pickles, and historical results are omitted.
See [code and data notes](DATA_AND_CODE_NOTES.md) for provenance and scope.

## Repository layout

```text
data/                          # Committed QA and SCOPE response matrices
src/agentopt/
├── model_selection/           # Gittins, Pareto identification, and baseline algorithms
├── base_models.py             # Shared types
└── model_price.py / .json      # Pricing table
experiments/
├── combined_objective/        # Current two-axis runner, ablations, and Pareto baselines
├── single_objective/          # Accuracy-only baseline replays
└── data/                      # Local legacy lookup pickles and aggregated CSVs
```

This repository focuses on offline research; live proxy and evaluation harnesses
are stubbed or omitted. Generated results and plots are local outputs. The
[results guide](experiments/combined_objective/results/README.md)
describes output locations and interpretation.
