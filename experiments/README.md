# Benchmark Experiments

Frozen results for 9 LLM configurations across 4 benchmarks. No API calls required.

Selector **algorithm implementations** live in `../src/agentopt/model_selection/`. The offline sims below re-implement the same decision logic against pickle lookup tables.

## Data

### Aggregated (per model combination)

`cached_results/{gpqa,bfcl,hotpotqa,mathqa}/brute_force_results.csv`

| Column | Meaning |
|--------|---------|
| Accuracy | Task success rate |
| Server Latency (s) | Mean server-side latency |
| Wall Latency (s) | Mean wall-clock latency |
| Cost ($) | Total / mean cost for the combo |

Also: `selector_results.csv`, multiobjective LaTeX tables, ablation/sweep CSVs.

### Per-sample lookup tables

`results/cache_db_results/*_lookup.pkl`

| File | Combos × samples |
|------|------------------|
| `gpqa_lookup.pkl` | 9 × 198 |
| `bfcl_lookup.pkl` | 9 × 200 |
| `hotpotqa_lookup.pkl` | 81 × 200 |
| `mathqa_lookup.pkl` | 81 × 200 |

Schema: `model_names`, `datapoints`, `table[combo][dp_idx] → SampleResult(score, latency_seconds, cost, tokens…)`.

## Offline selector simulation

Pickles under `results/cache_db_results/` are **gitignored** — keep them locally.

```bash
# Pure accuracy (v2)
python offline_selector_sim_v2.py \
    --pickle results/cache_db_results/gpqa_lookup.pkl \
    --selectors all --seeds 50

# Combined objective (v3): J = acc − λ_cost·NormCost − λ_latency·NormLatency
python combined_objective/offline_selector_sim_v3.py \
    --pickle results/cache_db_results/gpqa_lookup.pkl \
    --selectors all --seeds 50 \
    --lambda-cost 0.1 --lambda-latency 0.1

cd combined_objective && python run_all_selectors.py
```

Scripts auto-add `../src` to `PYTHONPATH` so `agentopt.model_selection` imports work.
If a pickle is missing, they exit with a pointer to the local data directory.

Available selectors: `brute_force` (v3), `random_search`, `matrix_ucb`, `bayesian_optimization`.

## Benchmarks

| Benchmark | Samples | Combos | Architecture |
|-----------|---------|--------|--------------|
| GPQA | 198 | 9 | Direct QA |
| BFCL | 200 | 9 | Multi-turn function calling |
| HotpotQA | 200 | 81 | Planner + Solver |
| MathQA | 200 | 81 | Answer + Critic |

## Models

Claude 3 Haiku, Claude Haiku 4.5, Claude Opus 4.6, gpt-oss-20b, gpt-oss-120b, Kimi K2.5, Ministral 3 8B, Qwen3 32B, Qwen3 Next 80B A3B

## Dependencies

```bash
pip install numpy botorch gpytorch   # Bayesian Optimization selector only
```

CSV inspection needs only Python / pandas. Pickle loading needs stdlib (+ the `SampleResult` class in the sim scripts).
