# AgentOpt Research

Model-selection **algorithms** plus frozen benchmark results (**accuracy**, **cost**, **latency**, and token usage). The committed HotpotQA and MathQA matrices contain a complete 10-model workflow grid.

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
| HotpotQA | `data/hotpotqa/` | 100 (10 planners x 10 solvers) |
| MathQA | `data/mathqa/` | 100 (10 answer models x 10 critics) |

Columns: `Rank`, `Model`, `Accuracy`, `Server Latency (s)`, `Wall Latency (s)`, `Cost ($)`.

## Per-sample matrices (for selector algorithms)

The committed HotpotQA and MathQA per-sample matrices live in `data/hotpotqa/`
and `data/mathqa/`. Each directory contains accuracy, USD cost, input-token,
output-token, and total-token CSVs for 100 configurations x 200 questions.

See `experiments/README.md` for offline simulation usage.
