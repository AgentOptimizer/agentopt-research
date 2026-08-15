# AgentOpt Research

Model-selection **algorithms** plus frozen benchmark results (**accuracy**, **cost**, **latency**) across 9 LLMs × 4 benchmarks.

## What's here

```
src/agentopt/
├── model_selection/            # brute_force, random_search, matrix_ucb, bayesian
├── base_models.py              # Shared types
└── model_price.py / .json      # Pricing table

experiments/
├── results/cache_db_results/   # Per-sample lookup tables (*.pkl)
├── cached_results/             # Aggregated CSVs (per combo accuracy/cost/latency)
├── offline_selector_sim_v2.py  # Offline accuracy-only selector replay
└── combined_objective/         # Offline J = acc − λ·cost − λ·lat sims
```

Proxy / daemon / live eval harness are stubbed or omitted; use the main `agentopt` package for online runs.

## Quick access: per-combo metrics

Use the brute-force CSVs (no dependencies):

| Benchmark | File | Combos |
|-----------|------|--------|
| GPQA | `experiments/cached_results/gpqa/brute_force_results.csv` | 9 |
| BFCL | `experiments/cached_results/bfcl/brute_force_results.csv` | 9 |
| HotpotQA | `experiments/cached_results/hotpotqa/brute_force_results.csv` | 81 (planner+solver) |
| MathQA | `experiments/cached_results/mathqa/brute_force_results.csv` | 81 (answer+critic) |

Columns: `Rank`, `Model`, `Accuracy`, `Server Latency (s)`, `Wall Latency (s)`, `Cost ($)`.

## Per-sample matrices (for selector algorithms)

```text
experiments/results/cache_db_results/{gpqa,bfcl,hotpotqa,mathqa}_lookup.pkl
```

Each pickle: `{model_names, datapoints, table[combo][idx] → SampleResult(score, latency_seconds, cost, …)}`.

See `experiments/README.md` for offline simulation usage.
