# AgentOpt Research

Model-selection **algorithms** plus frozen benchmark results (**accuracy**, **cost**, **latency**) across 9 LLMs × 4 benchmarks.

## What's here

```
src/agentopt/
├── model_selection/            # brute_force, random_search, matrix_ucb, gittins, bayesian
├── base_models.py              # Shared types
└── model_price.py / .json      # Pricing table

experiments/
├── data/                       # Shared lookup pickles + brute-force CSVs
├── combined_objective/         # Multi-objective replays + their results/
└── single_objective/           # Accuracy-only baseline + its results/
```

Proxy / daemon / live eval harness are stubbed or omitted; use the main `agentopt` package for online runs.

## Setup

Requires Python 3.10+. The offline sims evaluate `X | Y` annotations at runtime,
so 3.9 fails on import.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"   # radial JAX/SciPy + plots + pytest
.venv/bin/python -m pytest          # no PYTHONPATH needed
```

`import agentopt` itself needs only `numpy` and `pydantic`. Heavier selector
stacks are extras, matching the try/except guards in `agentopt.model_selection`:
`radial` (scipy), `radial-jax` (scipy + jax batched boundary builds),
`plots` (matplotlib), `gittins` (jax, jaxtyping, torch), `bayesian`
(botorch, gpytorch), `dotenv`, and `all`.

## Quick access: per-combo metrics

Use the brute-force CSVs (no dependencies):

| Benchmark | File | Combos |
|-----------|------|--------|
| GPQA | `experiments/data/brute_force/gpqa.csv` | 9 |
| BFCL | `experiments/data/brute_force/bfcl.csv` | 9 |
| HotpotQA | `experiments/data/brute_force/hotpotqa.csv` | 81 (planner+solver) |
| MathQA | `experiments/data/brute_force/mathqa.csv` | 81 (answer+critic) |

Columns: `Rank`, `Model`, `Accuracy`, `Server Latency (s)`, `Wall Latency (s)`, `Cost ($)`.

## Per-sample matrices (for selector algorithms)

```text
experiments/data/lookup/{gpqa,bfcl,hotpotqa,mathqa}_lookup.pkl
```

Each pickle: `{model_names, datapoints, table[combo][idx] → SampleResult(score, latency_seconds, cost, …)}`.

See `experiments/README.md` for offline simulation usage.
