# AgentOpt Research

Offline model-selection algorithms and frozen benchmark matrices for accuracy,
deployment cost, latency, and token usage. The current paper method is
**Cost-Coupled Gittins (CC-Gittins, two-axis)**.

## Current method

G0 uses the exact deployment and quality axes, `((0, 1), (1, 0))`, with real
per-arm continuation costs, 1:1 round-robin scheduling, independent per-direction
eta decay, and finite-LCB recommendations. Gauss-Radau is retired.

The canonical G0–G9 configurations and four ablation families are defined in
[`gittins_ablation_v2.py`](experiments/combined_objective/gittins_ablation_v2.py).
See the [experiment guide](experiments/README.md) for full runs, Slurm submission,
validation, and plots. Use its experiment runner for the paper protocol; the
lower-level replay APIs also support historical settings.

## Setup and quick start

Requires Python 3.10+. Run from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
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

`import agentopt` needs only `numpy` and `pydantic`. The `dev` extra installs
SciPy, JAX, Matplotlib, and pytest for the current replay and tests. Other extras
cover scalar Gittins (`gittins`), Bayesian baselines (`bayesian`), and the full
optional dependency set (`all`); see [pyproject.toml](pyproject.toml).

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

Legacy aggregated GPQA/BFCL summaries and lookup pickles live locally under
`experiments/data/`, which is gitignored. The aggregated CSV columns are
`Rank`, `Model`, `Accuracy`, `Server Latency (s)`, `Wall Latency (s)`, and
`Cost ($)`; that schema does not apply to the per-question matrices above.

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
[historical results index](experiments/combined_objective/results/README.md)
records earlier protocols separately from the current experiment guide.
